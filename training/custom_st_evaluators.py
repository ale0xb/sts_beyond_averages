from __future__ import annotations

import csv
import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from evaluate_ord.evaluate_ord import JSD, EMD
from sentence_transformers import InputExample
from sentence_transformers.evaluation.SentenceEvaluator import SentenceEvaluator

import torch
from sentence_transformers.cross_encoder.data_collator import CrossEncoderDataCollator

import numpy as np

from collections.abc import Collection

if TYPE_CHECKING:
    from sentence_transformers.cross_encoder.CrossEncoder import CrossEncoder

logger = logging.getLogger(__name__)

class CrossEncoderSoftDistributionEvaluator(SentenceEvaluator):
    """
    Evaluator for CrossEncoder models that predicts *distributional* targets.
    Computes the average JSD (Jensen-Shannon Divergence) and EMD (Earth-Mover's Distance)
    between predicted and gold distributions.

    Args:
        sentence_pairs: list of [sentence1, sentence2] pairs.
        labels: list/array of gold distributions (e.g., shape [N,6])
        name: identifier for output/logging.
        batch_size: evaluation batch size.
        show_progress_bar: show tqdm progress bar if enabled.
        write_csv: append results to a CSV file.
        jsd_metric: instance of your JSD class.
        emd_metric: instance of your EMD class.
    """

    def __init__(
        self,
        sentence_pairs: list[list[str]],
        labels: list[list[float]],
        *,
        name: str = "",
        batch_size: int = 32,
        show_progress_bar: bool = True,
        write_csv: bool = True,
        jsd_metric=JSD(),
        emd_no_dist_metric=EMD(),
        distance_matrix=None,
        **kwargs,
    ):
        self.sentence_pairs = sentence_pairs
        self.labels = np.asarray(labels)
        self.name = name
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar
        self.csv_file = "CrossEncoderSoftDistributionEvaluator" + ("_" + name if name else "") + "_results.csv"
        self.write_csv = write_csv
        self.jsd_metric = jsd_metric
        self.emd_metric = EMD(distance_matrix=distance_matrix)
        self.emd_no_dist_metric = emd_no_dist_metric
        if distance_matrix is not None:
            self.emd_metric = EMD(distance_matrix=distance_matrix)

        self.primary_metric = "emd"
        self.csv_headers = ["epoch", "steps", "JSD", "EMD_with", "EMD_no"]
        
        


    @classmethod
    def from_input_examples(cls, examples: list[InputExample], **kwargs):
        sentence_pairs = []
        scores = []

        for example in examples:
            sentence_pairs.append(example.texts)
            scores.append(example.label)
        return cls(sentence_pairs, scores, **kwargs)
    
    def __call__(
        self, model: "CrossEncoder", output_path: str = None, epoch: int = -1, steps: int = -1
    ) -> dict[str, float]:
        # Logging setup
        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps"
        else:
            out_txt = ""

        logger.info(f"CrossEncoderSoftDistributionEvaluator: Evaluating the model on {self.name} dataset{out_txt}:")

        # Predict logits from model
        pred_logits = model.predict(
            self.sentence_pairs,
            convert_to_numpy=True,
            show_progress_bar=self.show_progress_bar,
            batch_size=self.batch_size,
        )  # shape: (N, 6)

        # For metric classes, pass logits and gold label distributions
        eval_preds = (pred_logits, self.labels)

        jsd_result = self.jsd_metric(eval_preds) if self.jsd_metric is not None else {}
        emd_result = self.emd_metric(eval_preds) if self.emd_metric is not None else {}
        emd_no_dist_result = self.emd_no_dist_metric(eval_preds) if self.emd_no_dist_metric is not None else {}

        jsd = jsd_result.get("jsd", np.nan)
        emd = emd_result.get("emd", np.nan)
        emd_no = emd_no_dist_result.get("emd", np.nan)

        logger.info(f"JSD:                {jsd:.4f}")
        logger.info(f"EMD:                {emd:.4f}")
        logger.info(f"EMD (no distance):   {emd_no:.4f}")

        metrics = {
            "jsd": jsd,
            "emd_no": emd_no,
            "emd_with": emd,
        }

        # Write results to CSV
        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(csv_path, mode="a" if output_file_exists else "w", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)
                writer.writerow([epoch, steps, jsd, emd, emd_no])

        return metrics

    # (Optional) API compatibility with SentenceEvaluator
    def prefix_name_to_metrics(self, metrics, name):
        if name:
            return {f"{name}_{k}": v for k, v in metrics.items()}
        return metrics
    

class CrossEncoderSoftDataCollator(CrossEncoderDataCollator):
    """Collator for a CrossEncoder model.
    This encodes the text columns to {column}_input_ids and {column}_attention_mask columns.
    This works with the two text dataset that is used as the example in the training overview:
    https://www.sbert.net/docs/sentence_transformer/training_overview.html

    It is important that the columns are in the expected order. For example, if your dataset has columns
    "answer", "question" in that order, then the MultipleNegativesRankingLoss will consider
    "answer" as the anchor and "question" as the positive, and it will (unexpectedly) optimize for
    "given the answer, what is the question?".
    """

    tokenize_fn: Callable
    valid_label_columns: list[str] = field(default_factory=lambda: ["label", "labels", "score", "scores"])
    _warned_columns: set[tuple[str]] = field(default_factory=set, init=False, repr=False)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        column_names = list(features[0].keys())

        # We should always be able to return a loss, label or not:
        batch = {}

        if "dataset_name" in column_names:
            column_names.remove("dataset_name")
            batch["dataset_name"] = features[0]["dataset_name"]

        # Extract the label column if it exists
        for label_column in self.valid_label_columns:
            if label_column in column_names:
                # If the label column is a list/tuple/collection, we create a list of tensors
                if isinstance(features[0][label_column], Collection):
                    batch["label"] = torch.stack([torch.tensor(row[label_column]) for row in features])
                else:
                    # Otherwise, if it's e.g. single values, we create a tensor
                    batch["label"] = torch.tensor([row[label_column] for row in features])
                column_names.remove(label_column)
                break

        for column_name in column_names:
            # If the prompt length has been set, we should add it to the batch
            if column_name.endswith("_prompt_length") and column_name[: -len("_prompt_length")] in column_names:
                batch[column_name] = torch.tensor([row[column_name] for row in features], dtype=torch.int)
                continue

            batch[column_name] = [row[column_name] for row in features]

        return batch