# /usr/bin/env python
# coding: utf-8

"""
Merged pipeline script: 
1) Load & preprocess the dataset 
2) Compute baseline predictions & distribution heads
3) Train a cross-encoder model on the same data
4) Evaluate final model
"""

import argparse
import collections
import logging
import os
import random
import re
import shutil
from datetime import datetime
from typing import Callable

import numpy as np
import pandas as pd
import torch

from datasets import Dataset, load_dataset
from sklearn.model_selection import train_test_split

# ─────────────── GLOBAL SEED CONTROL ───────────────
# SEEDS = [12345, 223 , 555, 333, 123, 555, 888, 8982, 12, 13]           # multi‑run ensemble
SEEDS = [12345, 223 , 555, 333, 123]           # multi‑run ensemble
# SEEDS = [14, 224]
def set_global_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Sentence-Transformers / CrossEncoder
from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CrossEncoderCorrelationEvaluator
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer
from sentence_transformers.cross_encoder.training_args import CrossEncoderTrainingArguments

# Custom modules (place them in your PYTHONPATH or same folder)
from preprocess.preprc_utils import load_dataset as sts_load_dataset
from preprocess.krippendorff import alpha_per_item
from evaluate_ord.evaluate_ord import EMD, JSD
from evaluate_ord.evaluation_utils import CrossEncoderML2Reg
from training.custom_st_losses import KLDivergenceLoss, JSDLoss, OrdinalLogLoss
from training.custom_st_evaluators import (
    CrossEncoderSoftDistributionEvaluator,
    CrossEncoderSoftDataCollator,
)

logging.basicConfig(
    format="%(asctime)s - %(message)s", 
    datefmt="%Y-%m-%d %H:%M:%S", 
    level=logging.INFO
)

# ────────────────────────────────────────────────────────────────────────────────
# Section A. HELPER FUNCTIONS
# ────────────────────────────────────────────────────────────────────────────────
def assign_tercile(value, quartiles):
    """Map a numeric value to {1,2,3} based on tercile breakpoints."""
    if value <= quartiles[0]:
        return 1
    if value <= quartiles[1]:
        return 2
    return 3


def scores_to_distribution(scores, n_classes=6):
    """Convert a list of discrete labels into an empirical distribution."""
    counts = np.bincount(scores, minlength=n_classes)
    return counts / counts.sum()

LABELS = np.arange(6)  # 0..5

def trunc_normal_head(s, sigma):
    """Gaussian-like smoothing around s (mapped to [0..5])."""
    s_scaled = s * (LABELS.shape[0] - 1)
    probs = np.exp(-((LABELS - s_scaled) ** 2) / (2 * sigma ** 2))
    return probs / probs.sum()

# ── Post‑hoc temperature scaling for the Soft model ────────────────────────────
def soft_logits_to_probs(logits, T=1.0):
    """
    logits : np.ndarray shape (6,)
    T      : positive scalar temperature
    Returns a temperature‑scaled probability vector.
    """
    z = logits / T
    z -= z.max()  # numerical stability
    p = np.exp(z)
    return p / p.sum()


def _add_avg_row_col(table: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of table with row and column averages appended."""
    table_with_row_avg = table.copy()
    table_with_row_avg["avg"] = table_with_row_avg.mean(axis=1, skipna=True)

    avg_row = table_with_row_avg.mean(axis=0, skipna=True)
    avg_row.name = "avg"

    return pd.concat([table_with_row_avg, avg_row.to_frame().T])


def accumulate_slice_metrics(destination, slice_rows):
    """
    Accumulate per-slice metrics in-place. `destination` is expected to be a
    defaultdict that maps (mode, tercile) -> metric totals.
    """
    for row in slice_rows:
        key = (row["mode"], row["tercile"])
        entry = destination[key]
        entry["n"] += 1
        for metric in ("jsd", "emd_no", "emd_with"):
            entry[metric] += row[metric]


def average_slice_metrics(accumulator):
    """Convert an accumulator produced by `accumulate_slice_metrics` into averages."""
    averaged = []
    for (mode, tercile), totals in accumulator.items():
        if totals["n"] == 0:
            continue
        averaged.append({
            "mode": mode,
            "tercile": tercile,
            "jsd": totals["jsd"] / totals["n"],
            "emd_no": totals["emd_no"] / totals["n"],
            "emd_with": totals["emd_with"] / totals["n"],
        })
    return averaged


def build_dataset(df: pd.DataFrame, label_fn: Callable) -> Dataset:
    """
    Create a HuggingFace Dataset with `sentence1`, `sentence2`, and `label`
    columns using the provided label transformation.
    """
    data = df[["sentence1", "sentence2", "score_list"]].copy()
    data["label"] = data["score_list"].apply(label_fn)
    return Dataset.from_pandas(data[["sentence1", "sentence2", "label"]].reset_index(drop=True))


def create_metrics_summary(results_list, system_name):
    if not results_list:
        logging.warning("No results available for %s; skipping summary.", system_name)
        return

    results_df = pd.DataFrame(results_list)
    print(f"\n=== Slice-wise results ({system_name}) ===")
    for metric, title in (
        ("jsd", "JSD"),
        ("emd_no", "EMD (without distance matrix)"),
        ("emd_with", "EMD (with distance matrix)"),
    ):
        table = results_df.pivot(index="mode", columns="tercile", values=metric)
        table = _add_avg_row_col(table)
        print(f"\n{title}:")
        print(table)


_NDCG_RE = re.compile(r"^(Nano[A-Za-z0-9]+)_R100_ndcg@10$")

def extract_ndcg_per_dataset(metrics_dict):
    """
    Pull the per-dataset nDCG@10 from the evaluator output.

    Returns
    -------
    pd.Series  index = dataset name,  values = nDCG@10
    """
    pairs = {
        _NDCG_RE.match(k).group(1): v
        for k, v in metrics_dict.items()
        if _NDCG_RE.match(k)
    }
    return pd.Series(pairs).sort_index()


def permutation_pvalue(diff_vec):
    """
    Two-sided paired permutation test on the mean of diff_vec.
    Exactly 2^n permutations (n ≤ 8 → 256).
    """
    n = len(diff_vec)
    observed = diff_vec.mean()
    if observed == 0:            # edge case
        return 1.0
    more_extreme = 0
    for mask in range(1 << n):
        flipped = np.where([(mask >> i) & 1 for i in range(n)],
                           diff_vec, -diff_vec)
        if flipped.mean() >= observed:
            more_extreme += 1
    return more_extreme / (1 << n)


def normalize_distance_matrix(matrix: np.ndarray) -> np.ndarray:
    """Normalize a distance matrix to [0, 1], warning if the range is zero."""
    dist_min, dist_max = matrix.min(), matrix.max()
    if dist_max > dist_min:
        return (matrix - dist_min) / (dist_max - dist_min)

    logging.warning("Distance matrix has zero range. Skipping normalization.")
    return matrix

# ────────────────────────────────────────────────────────────────────────────────
# Section B. DATA LOADING & PREPROCESS
# ────────────────────────────────────────────────────────────────────────────────
def load_and_preprocess_data(exp_dir, dataset_path="data/text.clean"):
    """
    Loads the STS dataset, computes Krippendorff's alpha per item, 
    assigns terciles for agreement, stratifies, and does a 60/20/20 split.
    Returns train/val/test dataframes, plus the computed distance matrix 
    (Krippendorff's alpha distance).
    """
    logging.info(f"Loading dataset from: {dataset_path}")
    df_full = sts_load_dataset(dataset_path)
    df_full["mode"] = df_full["mode"].fillna(-1)

    # Calculate Krippendorff's alpha
    logging.info("Calculating Krippendorff's alpha and distance matrix...")
    reliability_data = np.array(df_full['score_list'].tolist()).transpose()
    per_item_agreement, distance_matrix = alpha_per_item(
        reliability_data=reliability_data,
        level_of_measurement='ordinal'
    )
    df_full['per_item_agreement'] = per_item_agreement

    # Assign each per_item_agreement to a tercile
    terciles = np.percentile(df_full["per_item_agreement"], [33.33, 66.67])
    df_full['tercile'] = df_full['per_item_agreement'].apply(lambda x: assign_tercile(x, terciles))
    df_full['strata'] = df_full['mode'].astype(str) + "_" + df_full['tercile'].astype(str)

    distance_matrix = normalize_distance_matrix(distance_matrix)

    # 60/20/20 split with label×agreement stratification
    train_val, test = train_test_split(
        df_full, test_size=0.20, stratify=df_full['strata'], random_state=42
    )
    train, val = train_test_split(
        train_val, test_size=0.25, stratify=train_val['strata'], random_state=42
    )

    # Print and save the split sizes by strata
    logging.info(f"Train size: {len(train)}, Val size: {len(val)}, Test size: {len(test)}")
    splits_file = os.path.join(exp_dir, "splits_stats.txt")
    split_stats = {
        "Train": train["strata"].value_counts().sort_index(),
        "Val": val["strata"].value_counts().sort_index(),
        "Test": test["strata"].value_counts().sort_index(),
    }
    with open(splits_file, "w", encoding="utf-8") as f:
        f.write("\n\n".join(f"{name} split by strata:\n{stats.to_string()}" for name, stats in split_stats.items()))
    logging.info(f"Saved split statistics to {splits_file}")
    for name, stats in split_stats.items():
        print(f"{name} split by strata:\n{stats.to_string()}\n")

    logging.info("Full dataset statistics:")
    logging.info(df_full["strata"].value_counts().sort_index())
    logging.info(df_full["mode"].value_counts().sort_index())
    logging.info(df_full["tercile"].value_counts().sort_index())
    logging.info(df_full["per_item_agreement"].describe())


    # Also save the datasets as CSV files
    train.to_csv(os.path.join(exp_dir, "train.csv"), index=False)
    val.to_csv(os.path.join(exp_dir, "val.csv"), index=False)
    test.to_csv(os.path.join(exp_dir, "test.csv"), index=False)

    return train, val, test, distance_matrix


# ────────────────────────────────────────────────────────────────────────────────
# Section C. BASELINE PREDICTIONS (Pre-trained CrossEncoder)
# ────────────────────────────────────────────────────────────────────────────────
def run_baseline_experiment(pretrained_ce, train, val, test, distance_matrix):
    """
    1) Loads a pre-trained CrossEncoder ("cross-encoder/stsb-roberta-large") that
       outputs scalar similarity in [0..5].
    2) Predicts a scalar for train/val/test.
    3) Tunes the TruncNormal head per (mode, tercile) on the validation split.
    4) Evaluates JSD and EMD on the test split using the TruncNormal head.
    """

    logging.info("Running BASELINE experiment with CrossEncoder('cross-encoder/stsb-roberta-large')")
    pretrained_ce.eval()
    

    @torch.no_grad()
    def batch_predict(df_chunk, batch_size=32):
        pairs = list(zip(df_chunk["sentence1"], df_chunk["sentence2"]))
        preds = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            preds.extend(pretrained_ce.predict(batch))
        return np.array(preds)

    # Predict scalar similarity in [0..5]
    for split, frame in [("train", train), ("val", val), ("test", test)]:
        frame["pred_scalar"] = batch_predict(frame)

    # Param search
    jsd_calc = JSD(convert_logits=False)
    def grid_search_per_strata(val_df, head_fn, grid):
        best_param_map = {}
        for (lbl, q), group_df in val_df.groupby(["mode", "tercile"]):
            best_param, best_score = None, float("inf")
            y_true = np.stack(group_df["score_list"].apply(lambda x: scores_to_distribution(x, n_classes=6)).values)
            for param in grid:
                y_pred = np.stack(group_df["pred_scalar"].apply(lambda s: head_fn(s, param)))
                score = jsd_calc((y_pred, y_true), support_size=6)["jsd"]
                if score < best_score:
                    best_param, best_score = param, score
            best_param_map[(lbl, q)] = best_param
        return best_param_map

    sigma_grid = np.logspace(-1.3, 0, 10)  # ~ 0.05 … 1.0
    best_sigma_by_strata = grid_search_per_strata(val, trunc_normal_head, sigma_grid)

    # Final evaluation on test
    test_true = np.stack(test["score_list"].apply(lambda x: scores_to_distribution(x, n_classes=6)).values)
    
    emd_no = EMD(convert_logits=False)
    emd_with = EMD(distance_matrix=distance_matrix, convert_logits=False)

    global_scores = {}
        
    # Handle TruncNormal with strata-based sigma
    test_pred_tn = test.apply(
        lambda row: trunc_normal_head(
            row["pred_scalar"],
            best_sigma_by_strata[(row["mode"], row["tercile"])]
        ),
        axis=1
    )
    test_pred_tn = np.stack(test_pred_tn.values)
    global_scores["TruncNormal"] = {
        "jsd": jsd_calc((test_pred_tn, test_true), support_size=6)["jsd"],
        "emd_no": emd_no((test_pred_tn, test_true), support_size=6)["emd"],
        "emd_with": emd_with((test_pred_tn, test_true), support_size=6)["emd"]
    }
    slice_scores = {"TruncNormal": []}
    for (label, q), slice_df in test.groupby(["mode", "tercile"]):
        true_slice = np.stack(slice_df["score_list"].apply(lambda x: scores_to_distribution(x, n_classes=6)).values)
        pred_slice_tn = np.stack(slice_df.apply(
            lambda row: trunc_normal_head(row["pred_scalar"], best_sigma_by_strata[(label, q)]),
            axis=1,
        ))
        slice_scores["TruncNormal"].append({
            "mode": label,
            "tercile": q,
            "jsd": jsd_calc((pred_slice_tn, true_slice), support_size=6)["jsd"],
            "emd_no": emd_no((pred_slice_tn, true_slice), support_size=6)["emd"],
            "emd_with": emd_with((pred_slice_tn, true_slice), support_size=6)["emd"],
        })
        
    
    # Print results summary
    print("\n=== BASELINE RESULTS ===")
    print("JSD (TruncNormal):", global_scores["TruncNormal"]["jsd"])
    print("EMD (TruncNormal):", global_scores["TruncNormal"]["emd_with"])
    print("EMD_no (TruncNormal):", global_scores["TruncNormal"]["emd_no"])
    


    del pretrained_ce
    # Empty CUDA cache
    torch.cuda.empty_cache()

    # # Call the helper function for each set of slice-wise results
    # print_score_table("Slice-wise Δ-JSD", slice_scores_jsd)
    # print_score_table("Slice-wise EMD (no distance matrix)", slice_scores_emd_no)
    # print_score_table("Slice-wise EMD (with distance matrix)", slice_scores_emd_with)

    # Return a dictionary of results for further analysis if needed
    return global_scores, slice_scores


# ────────────────────────────────────────────────────────────────────────────────
# Section D. TRAINING a CrossEncoder (distribution-based loss)
# ────────────────────────────────────────────────────────────────────────────────
def train_distribution_model(train_df, 
                             val_df, 
                             distance_matrix, 
                             models_dir,
                             seed,
                             loss_name="OrdinalLogLoss"):
    """
    Train a CrossEncoder model with 6 output labels, using a distribution-based 
    loss (e.g. JSDLoss). Returns the trained model + the final output_dir path.
    Stores all outputs under `base_dir`.
    """
    set_global_seed(seed)
    label_fn = lambda scores: scores_to_distribution(scores, n_classes=6)
    train_dataset = build_dataset(train_df, label_fn)
    eval_dataset = build_dataset(val_df, label_fn)

    # Create CrossEncoder with 6 output units
    model_name = "roberta-large"
    model = CrossEncoder(model_name, num_labels=6, max_length=256)
    model.dist_matrix = distance_matrix
    
    # Choose your distribution-based loss
    # loss = JSDLoss(model)
    # loss = KLDivergenceLoss(model)
    # loss = CrossEntropyLoss(model)
    # loss = OrdinalLogLoss(model)
    if loss_name == "JSDLoss":
        loss = JSDLoss(model)
    elif loss_name == "KLDivergenceLoss":
        loss = KLDivergenceLoss(model)
    elif loss_name == "OrdinalLogLoss":
        loss = OrdinalLogLoss(model)
    else:
        raise ValueError(f"Unknown loss_name: {loss_name}")
    

    # Create distribution-based evaluator
    soft_evaluator_dev = CrossEncoderSoftDistributionEvaluator(
        sentence_pairs=list(zip(eval_dataset["sentence1"], eval_dataset["sentence2"])),
        labels=eval_dataset["label"],
        name="sts15-soft-eval",
        distance_matrix=distance_matrix,
    )

    # Setup training hyperparams
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(models_dir, f"training_ce_soft_sts15-{timestamp}")
    short_model_name = model_name if "/" not in model_name else model_name.split("/")[-1]
    run_name = f"reranker-{short_model_name}-sts"

    args = CrossEncoderTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=4,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        warmup_ratio=0.1,
        fp16=False,  # or True if your GPU supports it
        bf16=True,   # if your GPU supports BF16
        eval_strategy="steps",
        eval_steps=80,
        save_strategy="steps",
        save_steps=80,
        save_total_limit=1,  # Only keep the best model checkpoint
        logging_steps=20,
        run_name=run_name,
        seed=seed,
        data_seed=seed,
        load_best_model_at_end=True,
    )

    trainer = CrossEncoderTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=soft_evaluator_dev,
        data_collator=CrossEncoderSoftDataCollator(tokenize_fn=model.tokenizer),
    )

    logging.info("Starting model training...")
    trainer.train()
    logging.info("Training complete.")

    final_output_dir = os.path.join(output_dir, "final")
    model.save_pretrained(final_output_dir)
    logging.info("Saved trained model to: %s", final_output_dir)

    try:
        for item in os.listdir(output_dir):
            checkpoint_dir = os.path.join(output_dir, item)
            if os.path.isdir(checkpoint_dir) and item.startswith("checkpoint-"):
                shutil.rmtree(checkpoint_dir)
                logging.info("Cleaned up checkpoint directory: %s", checkpoint_dir)
    except Exception as e:
        logging.warning("Error cleaning up checkpoint files: %s", e)

    return model, final_output_dir


# ────────────────────────────────────────────────────────────────────────────────
# Section E. EVALUATE the newly trained model
# ────────────────────────────────────────────────────────────────────────────────
def evaluate_trained_model_soft(model, test_df, distance_matrix, eval_dir):
    """
    Evaluate the newly trained CrossEncoder on the test split, 
    including slice-wise metrics by strata if you like.
    """
    test_df = test_df.copy()
    label_fn = lambda scores: scores_to_distribution(scores, n_classes=6)
    test_dataset = build_dataset(test_df, label_fn)

    global_scores = {}

    soft_evaluator_test = CrossEncoderSoftDistributionEvaluator(
        sentence_pairs=list(zip(test_dataset["sentence1"], test_dataset["sentence2"])),
        labels=test_dataset["label"],
        name="sts15-soft-test",
        distance_matrix=distance_matrix,
    )
    logging.info("Evaluating final trained model on TEST set...")
    metrics = soft_evaluator_test(model, output_path=eval_dir)
    print("\n=== TRAINED MODEL RESULTS (Test set) ===")
    print(f"JSD: {metrics.get('jsd'):.4f}, EMD: {metrics.get('emd_with'):.4f}, EMD_noDist: {metrics.get('emd_no'):.4f}")
    # Save the metrics to the results dictionary
    global_scores["trained_model"] = {
        "jsd": metrics.get("jsd"),
        "emd_no": metrics.get("emd_no"),
        "emd_with": metrics.get("emd_with")
    }
    slice_scores = {"trained_model": []}

    if "mode" in test_df.columns and "tercile" in test_df.columns:
        test_df["strata"] = test_df["mode"].astype(str) + "_" + test_df["tercile"].astype(str)

        for strata_value, group in test_df.groupby("strata"):
            slice_eval_data = group[["sentence1", "sentence2", "score_list"]].copy()
            slice_eval_data["label"] = slice_eval_data["score_list"].apply(
                lambda x: scores_to_distribution(x, n_classes=6)
            )
            slice_evaluator = CrossEncoderSoftDistributionEvaluator(
                sentence_pairs=list(zip(slice_eval_data["sentence1"], slice_eval_data["sentence2"])),
                labels=slice_eval_data["label"],
                name=f"sts15-soft-test-strata-{strata_value}",
                distance_matrix=distance_matrix,
            )
            logging.info(f"Evaluating on slice: {strata_value} (n={len(group)})")
            slice_metrics = slice_evaluator(model, output_path=eval_dir)
            slice_scores["trained_model"].append({
                "mode": group["mode"].values[0],
                "tercile": group["tercile"].values[0],
                "jsd": slice_metrics.get("jsd"),
                "emd_no": slice_metrics.get("emd_no"),
                "emd_with": slice_metrics.get("emd_with")
            })

    else:
        logging.info("No 'mode' / 'tercile' columns found in test_df for slice-based reporting.")

    return global_scores, slice_scores



def train_regression_model(train_df, val_df, models_dir, seed):
    """
    Train a CrossEncoder regression model on the training data.
    """
    set_global_seed(seed)
    label_fn = lambda scores: np.mean(scores) / 5
    train_dataset = build_dataset(train_df, label_fn)
    eval_dataset = build_dataset(val_df, label_fn)

    # Create CrossEncoder with 6 output units
    model_name = "roberta-large"
    model = CrossEncoder(model_name, num_labels=1, max_length=256)
    
    
    # Choose your distribution-based loss
    # loss = JSDLoss(model)
    # loss = KLDivergenceLoss(model)
    # loss = CrossEntropyLoss(model)
    loss = BinaryCrossEntropyLoss(model)

    # Create distribution-based evaluator
    eval_evaluator = CrossEncoderCorrelationEvaluator(
        sentence_pairs=list(zip(eval_dataset["sentence1"], eval_dataset["sentence2"])),
        scores=eval_dataset["label"],
        name="sts15-regression-eval",
    )

    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(models_dir, f"training_ce_hard_sts15-{timestamp}")
    short_model_name = model_name if "/" not in model_name else model_name.split("/")[-1]
    run_name = f"reranker-{short_model_name}-sts"

    
    eval_evaluator(model)
    args = CrossEncoderTrainingArguments(
        # Required parameter:
        output_dir=output_dir,
        # Optional training parameters:
        num_train_epochs=4,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        warmup_ratio=0.1,
        fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=True,  # Set to True if you have a GPU that supports BF16
        # Optional tracking/debugging parameters:
        eval_strategy="steps",
        eval_steps=80,
        save_strategy="steps",  # Keep steps so best model is saved during training
        save_steps=80,
        save_total_limit=1,    # Only keep the best model checkpoint
        logging_steps=20,
        run_name=run_name,  # Will be used in W&B if `wandb` is installed
        seed=seed,
        data_seed=seed,
        load_best_model_at_end=True,  # Load best model at end of training
    )

    trainer = CrossEncoderTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=eval_evaluator,
    )
    trainer.train()
    logging.info("Training complete.")
    
    # Return best model but clean up checkpoint files
    best_model = trainer.model
    try:
        shutil.rmtree(output_dir)
        logging.info("Cleaned up checkpoint directory: %s", output_dir)
    except Exception as e:
        logging.warning("Error cleaning up checkpoint files: %s", e)

    return best_model 


# ────────────────────────────────────────────────────────────────────────────────
# Section F. MAIN
# ────────────────────────────────────────────────────────────────────────────────
def main(base_dir):
    # prepare experiment folder
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir    = os.path.join(base_dir, f"exp-{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    models_dir = os.path.join(exp_dir, "trained_models")
    os.makedirs(models_dir, exist_ok=True)
    eval_dir   = os.path.join(exp_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)

    # 1) Load data & do stratified splits
    train_df, val_df, test_df, distance_matrix = load_and_preprocess_data(exp_dir=exp_dir)

    # Recompute distance matrix for the train + val split
    train_val_df = pd.concat([train_df, val_df])
    reliability_data = np.array(train_val_df['score_list'].tolist()).transpose()
    _, distance_matrix_test = alpha_per_item(
        reliability_data=reliability_data,
        level_of_measurement='ordinal'
    )

    # Normalize distance_matrix_test to [0,1]
    distance_matrix_test = normalize_distance_matrix(distance_matrix_test)
    

    ordinal_loss_name = "OrdinalLogLoss"

    # ──── Multi-seed evaluation accumulators ────
    # ndcg_soft_runs, ndcg_hard_runs = [], []
    hard_tn_runs = []
    soft_calib_runs = []
    soft_calib_slice_last = []
    trained_global_runs = []      # soft global JSD/EMD
    trained_slice_dict  = collections.defaultdict(lambda: {"jsd":0.0,
                                                           "emd_no":0.0,
                                                           "emd_with":0.0,
                                                           "n":0})
    # ---- HARD per-slice accumulators
    hard_tn_slice_dict = collections.defaultdict(lambda: {"jsd": 0.0,
                                                          "emd_no": 0.0,
                                                          "emd_with": 0.0,
                                                          "n": 0})
    soft_stsb_corr_runs  = []
    soft_sts15_corr_runs = []
    # Correlation accumulators for Hard and Soft-Calibrated models
    hard_corr_runs = []
    softcal_corr_runs = []

    # Prepare evaluators and dataset for baseline and test
    stsb_test_dataset = load_dataset("sentence-transformers/stsb", split="test")
    stsb_test_evaluator = CrossEncoderCorrelationEvaluator(
        sentence_pairs=list(zip(stsb_test_dataset["sentence1"], stsb_test_dataset["sentence2"])),
        scores=stsb_test_dataset["score"],
        name="stsb-regression-test",
    )
    gold = test_df['score_list'].apply(lambda xs: np.mean(xs)/5).tolist()
    pairs = list(zip(test_df["sentence1"], test_df["sentence2"]))
    sts15_corr_evaluator = CrossEncoderCorrelationEvaluator(
        sentence_pairs=pairs,
        scores=gold,
        name="sts15-regression-test",
    )
    # nanobeir_evaluator = CrossEncoderNanoBEIREvaluator(
    #     show_progress_bar=True,
    # )

    for seed in SEEDS:
        set_global_seed(seed)

        # ---- SOFT model ----
        trained_model_soft, final_dir = train_distribution_model(
            train_df, val_df, distance_matrix_test,
            models_dir, seed=seed, loss_name=ordinal_loss_name
        )

        trained_eval_dir = os.path.join(eval_dir, f"ce-soft-train-{ordinal_loss_name}-seed{seed}")
        os.makedirs(trained_eval_dir, exist_ok=True)
        model_results_all, model_results_slice = evaluate_trained_model_soft(trained_model_soft, test_df, distance_matrix, trained_eval_dir)

        # ---- Aggregate soft global and slice metrics
        trained_global_runs.append(model_results_all["trained_model"])
        accumulate_slice_metrics(trained_slice_dict, model_results_slice["trained_model"])

        reg_model = CrossEncoderML2Reg(model_name_or_path=final_dir, eval_bs=64)
        reg_model.model     = trained_model_soft.model
        reg_model.tokenizer = trained_model_soft.tokenizer
        reg_model.to("cuda")
        stsb_res = stsb_test_evaluator(reg_model, output_path=trained_eval_dir)
        sts15_res = sts15_corr_evaluator(reg_model, output_path=trained_eval_dir)
        soft_stsb_corr_runs.append(stsb_res)
        soft_sts15_corr_runs.append(sts15_res)
        # reg_nano_metrics = nanobeir_evaluator(reg_model, output_path=trained_eval_dir)
        # ndcg_soft_runs.append(extract_ndcg_per_dataset(reg_nano_metrics))

        # ---------- Post‑hoc calibration on SOFT model (δ‑aware TS) ----------
        @torch.no_grad()
        def batch_predict_soft_logits(df_chunk, batch_size=32):
            pairs = list(zip(df_chunk["sentence1"], df_chunk["sentence2"]))
            logits = []
            for i in range(0, len(pairs), batch_size):
                batch = pairs[i : i + batch_size]
                logits.extend(trained_model_soft.predict(batch))   # shape (n,6)
            return np.asarray(logits)

        # Attach logits to VAL / TEST
        for split, frame in [("val", val_df), ("test", test_df)]:
            frame[f"soft_logits_seed{seed}"] = list(batch_predict_soft_logits(frame))

        # Grid‑search temperature per (mode, tercile) on VAL (minimises δ‑EMD)
        T_grid = np.logspace(-1.3, 0.7, 15)   # ~0.05 … 5.0
        jsd_calc = JSD(convert_logits=False)
        emd_with = EMD(distance_matrix=distance_matrix, convert_logits=False)
        best_T_soft = {}
        for (lbl, q), g in val_df.groupby(["mode", "tercile"]):
            y_true = np.stack(g["score_list"].apply(lambda x: scores_to_distribution(x,6)))
            best_T, best_loss = None, float("inf")
            for T in T_grid:
                probs = np.stack(g[f"soft_logits_seed{seed}"].apply(lambda l: soft_logits_to_probs(l, T)))
                loss = emd_with((probs, y_true), support_size=6)["emd"]
                if loss < best_loss:
                    best_loss, best_T = loss, T
            best_T_soft[(lbl, q)] = best_T

        # Evaluate calibrated probs on TEST
        test_true = np.stack(test_df["score_list"].apply(lambda x: scores_to_distribution(x,6)))
        probs_calib = np.stack(test_df.apply(
            lambda row: soft_logits_to_probs(row[f"soft_logits_seed{seed}"],
                                             best_T_soft[(row["mode"], row["tercile"])]),
            axis=1
        ))
        soft_calib_runs.append({
            "jsd": jsd_calc((probs_calib, test_true),6)["jsd"],
            "emd_no": EMD(convert_logits=False)((probs_calib, test_true),6)["emd"],
            "emd_with": emd_with((probs_calib, test_true),6)["emd"]
        })
        # Collect calibrated soft correlations (STS‑15 only, expectation of probs)
        exp_score = (probs_calib @ LABELS) / 5.0
        sts15_calib_p = np.corrcoef(exp_score, gold)[0,1]
        sts15_calib_s = pd.Series(exp_score).corr(pd.Series(gold), method="spearman")
        # For STS‑B, expectation unchanged; reuse stsb_res
        softcal_corr_runs.append({
            "stsb_p": stsb_res["stsb-regression-test_pearson"], "stsb_s": stsb_res["stsb-regression-test_spearman"],
            "sts15_p": sts15_calib_p,        "sts15_s": sts15_calib_s
        })
        # store slice-wise metrics for the final seed only
        if seed == SEEDS[-1]:
            for (lbl, q), g in test_df.groupby(["mode", "tercile"]):
                true_slice = np.stack(g["score_list"].apply(lambda x: scores_to_distribution(x, 6)))
                pred_slice = np.stack(g.apply(
                    lambda row: soft_logits_to_probs(
                        row[f"soft_logits_seed{seed}"],
                        best_T_soft[(lbl, q)]
                    ), axis=1
                ))
                soft_calib_slice_last.append({
                    "mode": lbl,
                    "tercile": q,
                    "jsd": jsd_calc((pred_slice, true_slice), 6)["jsd"],
                    "emd_no": EMD(convert_logits=False)((pred_slice, true_slice), 6)["emd"],
                    "emd_with": emd_with((pred_slice, true_slice), 6)["emd"]
                })

        # Clean up
        del reg_model
        del trained_model_soft
        torch.cuda.empty_cache()

        # ---- HARD model ----
        trained_model_hard = train_regression_model(
            train_df, val_df, models_dir, seed=seed
        )
        trained_hard_eval_dir = os.path.join(eval_dir, f"ce-hard-train-seed{seed}")
        os.makedirs(trained_hard_eval_dir, exist_ok=True)
        # Capture hard model correlations
        stsb_hard = stsb_test_evaluator(trained_model_hard, output_path=trained_hard_eval_dir)
        sts15_hard = sts15_corr_evaluator(trained_model_hard, output_path=trained_hard_eval_dir)
        hard_corr_runs.append({
            "stsb_p": stsb_hard["stsb-regression-test_pearson"], "stsb_s": stsb_hard["stsb-regression-test_spearman"],
            "sts15_p": sts15_hard["sts15-regression-test_pearson"], "sts15_s": sts15_hard["sts15-regression-test_spearman"]
        })
        # hard_nano_metrics = nanobeir_evaluator(trained_model_hard, output_path=trained_hard_eval_dir)
        # ndcg_hard_runs.append(extract_ndcg_per_dataset(hard_nano_metrics))

        # ---------- Calibration metrics on HARD model ----------
        
        @torch.no_grad()
        def batch_predict_hard(df_chunk, batch_size=32):
            pairs = list(zip(df_chunk["sentence1"], df_chunk["sentence2"]))
            scores = []
            for i in range(0, len(pairs), batch_size):
                batch = pairs[i : i + batch_size]
                scores.extend(trained_model_hard.predict(batch))
            return np.array(scores)

        # attach scalar predictions
        for split, frame in [("val", val_df), ("test", test_df)]:
            frame[f"hard_pred_scalar_seed{seed}"] = batch_predict_hard(frame)

        # grid‑search σ per (mode, tercile) on VAL
        sigma_grid = np.logspace(-1.3, 0, 10)
        jsd_calc = JSD(convert_logits=False)
        best_sigma_hard = {}
        for (lbl, q), g in val_df.groupby(["mode", "tercile"]):
            y_true = np.stack(g["score_list"].apply(lambda x: scores_to_distribution(x,6)))
            best_s, best_jsd = None, float("inf")
            for sig in sigma_grid:
                y_pred = np.stack(g[f"hard_pred_scalar_seed{seed}"].apply(lambda s: trunc_normal_head(s, sig)))
                score = jsd_calc((y_pred, y_true), support_size=6)["jsd"]
                if score < best_jsd:
                    best_jsd, best_s = score, sig
            best_sigma_hard[(lbl, q)] = best_s

        test_true = np.stack(test_df["score_list"].apply(lambda x: scores_to_distribution(x,6)))
        emd_no   = EMD(convert_logits=False)
        emd_with = EMD(distance_matrix=distance_matrix, convert_logits=False)

        # TruncNormal metrics (heteroscedastic)
        y_pred_tn = np.stack(test_df.apply(
            lambda row: trunc_normal_head(row[f"hard_pred_scalar_seed{seed}"],
                                          best_sigma_hard[(row["mode"], row["tercile"])]),
            axis=1
        ))
        hard_tn_runs.append({
            "jsd": jsd_calc((y_pred_tn, test_true),6)["jsd"],
            "emd_no": emd_no((y_pred_tn, test_true),6)["emd"],
            "emd_with": emd_with((y_pred_tn, test_true),6)["emd"]
        })

        # --- per‑slice evaluation for the HARD model ---
        hard_tn_slice_run = []
        for (lbl, q), g in test_df.groupby(["mode", "tercile"]):
            true_slice = np.stack(g["score_list"].apply(lambda x: scores_to_distribution(x, 6)))

            # TruncNormal head
            pred_tn_slice = np.stack(g.apply(
                lambda row: trunc_normal_head(row[f"hard_pred_scalar_seed{seed}"],
                                              best_sigma_hard[(lbl, q)]),
                axis=1
            ))
            tn_metrics = {
                "mode": lbl,
                "tercile": q,
                "jsd": jsd_calc((pred_tn_slice, true_slice), 6)["jsd"],
                "emd_no": emd_no((pred_tn_slice, true_slice), 6)["emd"],
                "emd_with": emd_with((pred_tn_slice, true_slice), 6)["emd"]
            }
            hard_tn_slice_run.append(tn_metrics)

        accumulate_slice_metrics(hard_tn_slice_dict, hard_tn_slice_run)


    # After the loop: average per-dataset metrics but save stdev 
    # ndcg_soft = pd.concat(ndcg_soft_runs, axis=1).mean(axis=1)
    # ndcg_soft_stdev = pd.concat(ndcg_soft_runs, axis=1).std(axis=1)
    # ndcg_hard = pd.concat(ndcg_hard_runs, axis=1).mean(axis=1)
    # ndcg_hard_stdev = pd.concat(ndcg_hard_runs, axis=1).std(axis=1)
    hard_tn_avg    = pd.DataFrame(hard_tn_runs).mean().to_dict()
    soft_calib_avg = pd.DataFrame(soft_calib_runs).mean().to_dict()

    # ---- aggregate Soft global
    soft_global_avg = pd.DataFrame(trained_global_runs).mean().to_dict()

    # ---- aggregate per-slice
    soft_slice_avg = average_slice_metrics(trained_slice_dict)
    hard_tn_slice_avg = average_slice_metrics(hard_tn_slice_dict)

    
    # Save model and baseline results to CSV in evaluation/calibration folder
    calib_dir = os.path.join(eval_dir, "calibration")
    os.makedirs(calib_dir, exist_ok=True)
    
    # Optionally save to CSV alongside other results
    pd.DataFrame(hard_tn_slice_avg).to_csv(os.path.join(calib_dir, "hard_truncnormal_slice_avg.csv"), index=False)

    # ---- aggregate STSB / STS15 correlations
    def _avg_corr(runs, key):
        return np.mean([r[key] for r in runs])
        
    def _std_corr(runs, key):
        return np.std([r[key] for r in runs])
        
    soft_stsb_avg_pearson   = _avg_corr(soft_stsb_corr_runs, "stsb-regression-test_pearson")
    soft_sts15_avg_pearson  = _avg_corr(soft_sts15_corr_runs, "sts15-regression-test_pearson")
    soft_stsb_avg_spearman  = _avg_corr(soft_stsb_corr_runs, "stsb-regression-test_spearman")
    soft_sts15_avg_spearman = _avg_corr(soft_sts15_corr_runs, "sts15-regression-test_spearman")
    
    # Calculate standard deviations
    soft_stsb_std_pearson   = _std_corr(soft_stsb_corr_runs, "stsb-regression-test_pearson")
    soft_sts15_std_pearson  = _std_corr(soft_sts15_corr_runs, "sts15-regression-test_pearson")
    soft_stsb_std_spearman  = _std_corr(soft_stsb_corr_runs, "stsb-regression-test_spearman")
    soft_sts15_std_spearman = _std_corr(soft_sts15_corr_runs, "sts15-regression-test_spearman")
    
    # Aggregate correlations for Hard and Soft-Calibrated models
    hard_avg_stsb_p  = _avg_corr(hard_corr_runs, "stsb_p")
    hard_avg_stsb_s  = _avg_corr(hard_corr_runs, "stsb_s")
    hard_avg_sts15_p = _avg_corr(hard_corr_runs, "sts15_p")
    hard_avg_sts15_s = _avg_corr(hard_corr_runs, "sts15_s")
    
    # Standard deviations for Hard model
    hard_std_stsb_p  = _std_corr(hard_corr_runs, "stsb_p")
    hard_std_stsb_s  = _std_corr(hard_corr_runs, "stsb_s")
    hard_std_sts15_p = _std_corr(hard_corr_runs, "sts15_p")
    hard_std_sts15_s = _std_corr(hard_corr_runs, "sts15_s")

    softc_avg_stsb_p  = _avg_corr(softcal_corr_runs, "stsb_p")
    softc_avg_stsb_s  = _avg_corr(softcal_corr_runs, "stsb_s")  # Fixed - was incorrectly using "sts15_s"
    softc_avg_sts15_p = _avg_corr(softcal_corr_runs, "sts15_p")
    softc_avg_sts15_s = _avg_corr(softcal_corr_runs, "sts15_s")
    
    # Standard deviations for Soft-Calibrated model
    softc_std_stsb_p  = _std_corr(softcal_corr_runs, "stsb_p")
    softc_std_stsb_s  = _std_corr(softcal_corr_runs, "stsb_s")  # Fixed - was incorrectly using "sts15_s"
    softc_std_sts15_p = _std_corr(softcal_corr_runs, "sts15_p")
    softc_std_sts15_s = _std_corr(softcal_corr_runs, "sts15_s")

    # Baseline (single run)
    pretrained_ce = CrossEncoder("cross-encoder/stsb-roberta-large", max_length=256)
    pretrained_ce.eval()
    pretrained_ce.to("cuda")
    pretrained_eval_dir = os.path.join(eval_dir, f"ce-baseline")
    os.makedirs(pretrained_eval_dir, exist_ok=True)
    baseline_results_all, baseline_results_slice = run_baseline_experiment(pretrained_ce, train_df, val_df, test_df, distance_matrix)
    pretrained_stsb_results = stsb_test_evaluator(pretrained_ce, output_path=pretrained_eval_dir)
    pretrained_sts15_results = sts15_corr_evaluator(pretrained_ce, output_path=pretrained_eval_dir)
    # baseline_nano_metrics = nanobeir_evaluator(pretrained_ce, output_path=pretrained_eval_dir)
    # ndcg_base = extract_ndcg_per_dataset(baseline_nano_metrics)

    # Permutation p-value calculations
    # diff = (ndcg_soft - ndcg_base)
    # p_val = permutation_pvalue(diff.values)
    # print("\n=== NanoBEIR macro nDCG@10 (Soft - Pretrained) ===")
    # print(pd.DataFrame({
    #     "Soft": ndcg_soft,
    #     "Soft_stdev": ndcg_soft_stdev,
    #     "Baseline": ndcg_base,
    #     "Δ": diff
    # }).round(4))
    # print(f"\nMean Δ = {diff.mean():.4f},  permutation p-value = {p_val:.4f}")

    # diff = (ndcg_soft - ndcg_hard)
    # p_val = permutation_pvalue(diff.values)
    # print("\n=== NanoBEIR macro nDCG@10 (Soft - Hard) ===")
    # print(pd.DataFrame({
    #     "Soft": ndcg_soft,
    #     "Soft_stdev": ndcg_soft_stdev,
    #     "Hard": ndcg_hard,
    #     "Hard_stdev": ndcg_hard_stdev,
    #     "Δ": diff
    # }).round(4))
    # print(f"\nMean Δ = {diff.mean():.4f},  permutation p-value = {p_val:.4f}")
    # print(f"\nAveraged over {len(SEEDS)} seeds")

    # The rest: keep previous logic for metrics summary and CSV saving
    # Now use seed-averaged soft model metrics
    print("\n=== Calibration Comparison ===")
    print("JSD (Soft):", soft_global_avg["jsd"])
    print("JSD (Soft‑Calibrated):", soft_calib_avg["jsd"])
    print("EMD (Soft):", soft_global_avg["emd_with"])
    print("EMD (Soft‑Calibrated):", soft_calib_avg["emd_with"])
    print("EMD_no (Soft):", soft_global_avg["emd_no"])
    print("EMD_no (Soft‑Calibrated):", soft_calib_avg["emd_no"])
    print("JSD (Hard TruncNormal):", hard_tn_avg["jsd"])
    print("EMD (Hard TruncNormal):", hard_tn_avg["emd_with"])
    print("EMD_no (Hard TruncNormal):", hard_tn_avg["emd_no"])
    print("JSD (Baseline TruncNormal):", baseline_results_all["TruncNormal"]["jsd"])
    print("EMD (Baseline TruncNormal):", baseline_results_all["TruncNormal"]["emd_with"])
    print("EMD_no (Baseline TruncNormal):", baseline_results_all["TruncNormal"]["emd_no"])


    create_metrics_summary(baseline_results_slice["TruncNormal"], "TruncNormal")
    create_metrics_summary(soft_slice_avg, "trained_model_avg")
    create_metrics_summary(soft_calib_slice_last, "Soft-Calibrated")
    create_metrics_summary(hard_tn_slice_avg, "Hard-Calibrated TruncNormal")
    
    
    print(f"\n=== STS Correlations (avg±stdev over {len(SEEDS)} seeds) ===")
    print(f"\nPretrained  — STS-B r={pretrained_stsb_results['stsb-regression-test_pearson']:.4f}  ρ={pretrained_stsb_results['stsb-regression-test_spearman']:.4f}   |  STS-15 r={pretrained_sts15_results['sts15-regression-test_pearson']:.4f}  ρ={pretrained_sts15_results['sts15-regression-test_spearman']:.4f}")
    print(f"\nSoft Model  — STS-B r={soft_stsb_avg_pearson:.4f}±{soft_stsb_std_pearson:.4f}  ρ={soft_stsb_avg_spearman:.4f}±{soft_stsb_std_spearman:.4f}   |  STS-15 r={soft_sts15_avg_pearson:.4f}±{soft_sts15_std_pearson:.4f}  ρ={soft_sts15_avg_spearman:.4f}±{soft_sts15_std_spearman:.4f}")
    print(f"Soft‑Calibrated — STS-B r={softc_avg_stsb_p:.4f}±{softc_std_stsb_p:.4f}  ρ={softc_avg_stsb_s:.4f}±{softc_std_stsb_s:.4f}   |  STS-15 r={softc_avg_sts15_p:.4f}±{softc_std_sts15_p:.4f}  ρ={softc_avg_sts15_s:.4f}±{softc_std_sts15_s:.4f}")
    print(f"\nHard Model  — STS-B r={hard_avg_stsb_p:.4f}±{hard_std_stsb_p:.4f}  ρ={hard_avg_stsb_s:.4f}±{hard_std_stsb_s:.4f}   |  STS-15 r={hard_avg_sts15_p:.4f}±{hard_std_sts15_p:.4f}  ρ={hard_avg_sts15_s:.4f}±{hard_std_sts15_s:.4f}")

    # ────────────────────────────────────────────────────────────────────────────────
    # Calculate and print per-strata Spearman correlations
    # ────────────────────────────────────────────────────────────────────────────────
    print("\n=== Per-strata RMSE for STS-15 ===")
    
    def print_strata_metric_table(metric_df, model_name):
        """Print a nicely formatted table of per-strata metrics."""
        print(f"\n--- {model_name} ---")
        
        # Pivot to get mode as rows and tercile as columns
        pivot = metric_df.pivot(index="mode", columns="tercile", values="rmse")
        
        # Add row averages
        pivot["avg"] = pivot.mean(axis=1)
        
        # Add column averages (including the avg column)
        avg_row = pivot.mean()
        avg_row.name = "avg"
        pivot = pd.concat([pivot, avg_row.to_frame().T])
        
        # Format the table
        print(pivot.round(4))
    
    # Calculate per-strata RMSE for the pretrained model
    pretrained_strata = []
    for (mode, tercile), group in test_df.groupby(["mode", "tercile"]):
        gold = group['score_list'].apply(lambda xs: np.mean(xs)/5).tolist()
        preds = group["pred_scalar"].tolist()
        if len(preds) >= 5:  # Skip if too few samples
            rmse = np.sqrt(np.mean([(p - g)**2 for p, g in zip(preds, gold)]))
            pretrained_strata.append({
                "mode": mode,
                "tercile": tercile,
                "n": len(preds),
                "rmse": rmse
            })
    print_strata_metric_table(pd.DataFrame(pretrained_strata), "Pretrained Model")
    
    # Calculate per-strata RMSE for the hard model (averaged over all seeds)
    hard_strata_acc = collections.defaultdict(lambda: {"rmse_sum": 0.0, "count": 0})
    for seed in SEEDS:
        for (mode, tercile), group in test_df.groupby(["mode", "tercile"]):
            gold = group['score_list'].apply(lambda xs: np.mean(xs)/5).tolist()
            preds = group[f"hard_pred_scalar_seed{seed}"].tolist()
            if len(preds) >= 5:
                rmse = np.sqrt(np.mean([(p - g)**2 for p, g in zip(preds, gold)]))
                hard_strata_acc[(mode, tercile)]["rmse_sum"] += rmse
                hard_strata_acc[(mode, tercile)]["count"] += 1
    
    hard_strata = []
    for (mode, tercile), data in hard_strata_acc.items():
        if data["count"] > 0:
            hard_strata.append({
                "mode": mode,
                "tercile": tercile,
                "n": len(test_df[(test_df["mode"] == mode) & (test_df["tercile"] == tercile)]),
                "rmse": data["rmse_sum"] / data["count"]
            })
    print_strata_metric_table(pd.DataFrame(hard_strata), "Hard Model (avg over seeds)")
    
    # Calculate per-strata RMSE for the soft model (averaged over all seeds)
    soft_strata_acc = collections.defaultdict(lambda: {"rmse_sum": 0.0, "count": 0})
    for seed in SEEDS:
        for (mode, tercile), group in test_df.groupby(["mode", "tercile"]):
            gold = group['score_list'].apply(lambda xs: np.mean(xs)/5).tolist()
            # Calculate expected values from logits
            preds = [(soft_logits_to_probs(logits) @ LABELS) / 5.0 
                    for logits in group[f"soft_logits_seed{seed}"]]
            if len(preds) >= 5:
                rmse = np.sqrt(np.mean([(p - g)**2 for p, g in zip(preds, gold)]))
                soft_strata_acc[(mode, tercile)]["rmse_sum"] += rmse
                soft_strata_acc[(mode, tercile)]["count"] += 1
    
    soft_strata = []
    for (mode, tercile), data in soft_strata_acc.items():
        if data["count"] > 0:
            soft_strata.append({
                "mode": mode,
                "tercile": tercile,
                "n": len(test_df[(test_df["mode"] == mode) & (test_df["tercile"] == tercile)]),
                "rmse": data["rmse_sum"] / data["count"]
            })
    print_strata_metric_table(pd.DataFrame(soft_strata), "Soft Model (avg over seeds)")
    
    # Calculate per-strata RMSE for the soft-calibrated model (averaged over all seeds)
    softcal_strata_acc = collections.defaultdict(lambda: {"rmse_sum": 0.0, "count": 0})
    for seed in SEEDS:
        for (mode, tercile), group in test_df.groupby(["mode", "tercile"]):
            if (mode, tercile) not in best_T_soft:
                continue  # Skip if we don't have calibration parameters for this seed
                
            gold = group['score_list'].apply(lambda xs: np.mean(xs)/5).tolist()
            # Calculate calibrated expected values
            T = best_T_soft[(mode, tercile)]
            preds = []
            for logits in group[f"soft_logits_seed{seed}"]:
                probs = soft_logits_to_probs(logits, T)
                preds.append((probs @ LABELS) / 5.0)
            if len(preds) >= 5:
                rmse = np.sqrt(np.mean([(p - g)**2 for p, g in zip(preds, gold)]))
                softcal_strata_acc[(mode, tercile)]["rmse_sum"] += rmse
                softcal_strata_acc[(mode, tercile)]["count"] += 1
    
    softcal_strata = []
    for (mode, tercile), data in softcal_strata_acc.items():
        if data["count"] > 0:
            softcal_strata.append({
                "mode": mode,
                "tercile": tercile,
                "n": len(test_df[(test_df["mode"] == mode) & (test_df["tercile"] == tercile)]),
                "rmse": data["rmse_sum"] / data["count"]
            })
    print_strata_metric_table(pd.DataFrame(softcal_strata), "Soft-Calibrated Model (avg over seeds)")
    
    # Save per-strata correlation results to CSV
    pd.DataFrame(pretrained_strata).to_csv(os.path.join(calib_dir, "pretrained_strata_corr.csv"), index=False)
    pd.DataFrame(hard_strata).to_csv(os.path.join(calib_dir, "hard_strata_corr.csv"), index=False)
    pd.DataFrame(soft_strata).to_csv(os.path.join(calib_dir, "soft_strata_corr.csv"), index=False)
    pd.DataFrame(softcal_strata).to_csv(os.path.join(calib_dir, "softcal_strata_corr.csv"), index=False)

    # Save trained model results
    # Combine all results into a single DataFrame and save to one file
    combined_results = {
        "trained_model": model_results_all["trained_model"],
        "baseline_truncnormal": baseline_results_all["TruncNormal"],
        "hard_truncnormal_avg": hard_tn_avg,
    }
    combined_df = pd.DataFrame(combined_results)
    combined_df.to_csv(os.path.join(calib_dir, "all_results.csv"), index=True)

    # Save trained model slice results
    pd.DataFrame(model_results_slice["trained_model"]).to_csv(os.path.join(calib_dir, "trained_model_slices.csv"), index=False)
    # Save baseline TruncNormal slice results
    pd.DataFrame(baseline_results_slice["TruncNormal"]).to_csv(os.path.join(calib_dir, "baseline_truncnormal_slices.csv"), index=False)
    # Save hard model slice results
    pd.DataFrame(hard_tn_runs).to_csv(os.path.join(calib_dir, "hard_truncnormal_slices.csv"), index=False)
    
    # Save average results for hard model
    df_hard_tn_avg = pd.DataFrame([hard_tn_avg])
    df_hard_tn_avg.to_csv(os.path.join(calib_dir, "hard_truncnormal_avg.csv"), index=True)
    # Save average results for soft
    df_trained_model_avg = pd.DataFrame([model_results_all["trained_model"]])
    df_trained_model_avg.to_csv(os.path.join(calib_dir, "trained_model_avg.csv"), index=True)
    
    

    def compute_model_differences(trained_results, comparison_results, comparison_name, use_slice_avg=False):
        """
        Compute differences between trained model and comparison model metrics.
        
        Parameters:
        - trained_results: List of dicts with metrics from trained model
        - comparison_results: List of dicts with metrics from comparison model
        - comparison_name: Name of the comparison model for display
        - use_slice_avg: If True, comparison_results is a slice average list
        """
        diff_results = []
        
        for row in trained_results:
            mode = row["mode"]
            tercile = row["tercile"]
            
            # Find matching row in comparison results
            if use_slice_avg:
                # For slice averages (pre-aggregated results)
                matching_row = next((r for r in comparison_results 
                                    if r["mode"] == mode and r["tercile"] == tercile), None)
                if not matching_row:
                    continue
            else:
                # For baseline results
                matching_row = [r for r in comparison_results 
                               if r["mode"] == mode and r["tercile"] == tercile][0]
            
            # Calculate differences
            diff_results.append({
                "mode": mode,
                "tercile": tercile,
                "jsd_diff": row["jsd"] - matching_row["jsd"],
                "emd_no_diff": row["emd_no"] - matching_row["emd_no"],
                "emd_with_diff": row["emd_with"] - matching_row["emd_with"]
            })
        
        # Create and format DataFrame
        diff_df = pd.DataFrame(diff_results)
        diff_df = diff_df.pivot(index="mode", columns="tercile", 
                                values=["emd_with_diff"])
                            #    values=["jsd_diff", "emd_no_diff", "emd_with_diff"])
        
        # Add row averages
        # for metric in ["jsd_diff", "emd_no_diff", "emd_with_diff"]:
        for metric in ["emd_with_diff"]:
            diff_df[(metric, "avg")] = diff_df[metric].mean(axis=1)
        
        # Add column averages
        avg_row = pd.DataFrame({
            (metric, tercile): diff_df[metric][tercile].mean()
            # for metric in ["jsd_diff", "emd_no_diff", "emd_with_diff"]
            for metric in ["emd_with_diff"]
            for tercile in diff_df[metric].columns
        }, index=["avg"])
        
        diff_df = pd.concat([diff_df, avg_row])

        # Multiply by 100 
        diff_df = diff_df * 100
        diff_df = diff_df.round(2)
        
        # Print results
        print(f"\n=== Difference between Trained Model and {comparison_name} ===")
        print(diff_df)
        
        return diff_df

    # Calculate differences for all comparison models
    compute_model_differences(
        model_results_slice["trained_model"], 
        baseline_results_slice["TruncNormal"], 
        "Baseline (TruncNormal)"
    )

    compute_model_differences(
        model_results_slice["trained_model"], 
        hard_tn_slice_avg, 
        "Hard Model (TruncNormal)", 
        use_slice_avg=True
    )






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run STS pipeline")
    parser.add_argument(
        "--base_dir", 
        type=str, 
        default="output", 
        help="Base experiment directory for all outputs"
    )
    args = parser.parse_args()
    main(args.base_dir)
