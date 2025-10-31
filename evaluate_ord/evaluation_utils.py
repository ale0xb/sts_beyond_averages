import os
import csv

import numpy as np
import torch

from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CrossEncoderClassificationEvaluator
from sentence_transformers.evaluation import BinaryClassificationEvaluator
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
)

from evaluate_ord.evaluate_ord import project_expected_distance


class NLICECustomRegressionEvaluator(CrossEncoderClassificationEvaluator):
    """
    Inherits all CSV/logging from CrossEncoderClassificationEvaluator (for test set),
    but first finds the best F1 threshold on a held-out dev set and then applies that
    fixed threshold when evaluating on the test set.
    """

    def __init__(
        self,
        dev_pairs: list[list[str]],
        dev_labels: list[int],
        test_pairs: list[list[str]],
        test_labels: list[int],
        *,
        name: str = "",
        batch_size: int = 32,
        show_progress_bar: bool | None = None,
        write_csv: bool = True,
    ):
        # Initialize parent on the TEST set
        super().__init__(
            sentence_pairs=test_pairs,
            labels=test_labels,
            name=name,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            write_csv=write_csv,
        )

        # Store dev for threshold fitting
        if len(dev_pairs) != len(dev_labels):
            raise ValueError("dev_pairs and dev_labels must have the same length")
        self.dev_pairs = dev_pairs
        self.dev_labels = np.asarray(dev_labels)

    def __call__(
        self,
        model: CrossEncoder,
        output_path: str = None,
        epoch: int = -1,
        steps: int = -1
    ) -> dict[str, float]:
        # Predict on dev to find best F1 threshold
        dev_scores = model.predict(
            self.dev_pairs,
            convert_to_numpy=True,
            show_progress_bar=self.show_progress_bar,
        )
        # Reuse the same static helper for F1 threshold
        best_f1, _, _, best_threshold = BinaryClassificationEvaluator.find_best_f1_and_threshold(
            dev_scores, self.dev_labels, high_score_more_similar=True
        )

        # Predict on TEST and binarize at the fixed threshold
        test_scores = model.predict(
            self.sentence_pairs,
            convert_to_numpy=True,
            show_progress_bar=self.show_progress_bar,
        )
        test_preds = (test_scores >= best_threshold).astype(int)

        # Compute test metrics
        acc  = accuracy_score(self.labels, test_preds)
        prec = precision_score(self.labels, test_preds, zero_division=0)
        rec  = recall_score(self.labels, test_preds, zero_division=0)
        f1   = f1_score(self.labels, test_preds, zero_division=0)
        ap   = average_precision_score(self.labels, test_scores)

        # Prepare metrics dict
        metrics = {
            "threshold": best_threshold,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "average_precision": ap,
        }

        # Write CSV via parent logic
        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            file_exists = os.path.isfile(csv_path)
            with open(csv_path, mode="a" if file_exists else "w", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not file_exists:
                    # reuse parent's headers for num_labels=1
                    writer.writerow([
                        "epoch", "steps",
                        "Accuracy", "Accuracy_Threshold",
                        "F1", "F1_Threshold",
                        "Precision", "Recall", "Average_Precision"
                    ])
                writer.writerow([
                    epoch, steps,
                    acc, best_threshold,
                    f1, best_threshold,
                    prec, rec, ap
                ])

        # Name-prefix and model-card storage as in parent
        metrics = self.prefix_name_to_metrics(metrics, self.name)
        self.store_metrics_in_model_card_data(model, metrics, epoch, steps)
        return metrics
    

class CrossEncoderML2Reg(CrossEncoder):
    def __init__(self, model_name_or_path, eval_bs=16, **kwargs):
        super().__init__(model_name_or_path=model_name_or_path, **kwargs)
        self.batch_size = eval_bs

    def predict(self, sentence_pairs, **kwargs):
        logits = super().predict(sentence_pairs, batch_size=self.batch_size, **kwargs)

        # Softmax
        exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
       
        # Label mean and normalize
        labels = np.arange(probs.shape[1])                
        exp_labels = (probs * labels).sum(axis=1)
        return exp_labels / (probs.shape[1] - 1) 


class CrossEncoderML2DistanceWeightedReg(CrossEncoder):
    def __init__(self, model_name_or_path, distance_matrix, eval_bs=16,  **kwargs):
        super().__init__(model_name_or_path=model_name_or_path, **kwargs)
        self.distance_matrix = distance_matrix
        self.batch_size = eval_bs

    def predict(self, sentence_pairs, **kwargs):
        logits = super().predict(sentence_pairs, batch_size=self.batch_size, **kwargs)

        # Softmax
        exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
       
        # Label mean and normalize
        labels = np.arange(probs.shape[1])                
        
        scores = project_expected_distance(
            predictions=probs, labels=labels, distance_matrix=self.distance_matrix
        )

        return scores


def ce_hf_to_st_model(
        model: AutoModelForSequenceClassification, model_id: str, eval_bs: int = 16
) -> CrossEncoderML2Reg:
    """
    Monkey patch the Hugging Face model to use it in the Sentence Transformers format.
    This is a workaround for the fact that the CrossEncoder class does not natively
    support instantiated model conversion.
    """
    dummy = CrossEncoderML2Reg(
        model_name_or_path=model_id,
        trust_remote_code=True,
        num_labels=model.config.num_labels,
        device='cpu',
        eval_bs=eval_bs
    )
    # Monkey‑patch in real model & tokenizer
    dummy.model = model.cpu().eval()
    
    torch.cuda.empty_cache()
    
    dummy.to(device='cuda')
    dummy.tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    return dummy


def ce_hf_to_st_model_distance_weighted(
        model: AutoModelForSequenceClassification,
        model_id: str,
        distance_matrix: np.ndarray,
        eval_bs: int = 16,
) -> CrossEncoderML2DistanceWeightedReg:
    """
    Monkey patch the Hugging Face model to use it in the Sentence Transformers format.
    This is a workaround for the fact that the CrossEncoder class does not natively
    support instantiated model conversion.
    """
    dummy = CrossEncoderML2DistanceWeightedReg(
        model_name_or_path=model_id,
        trust_remote_code=True,
        num_labels=model.config.num_labels,
        distance_matrix=distance_matrix,
        device='cpu',
        eval_bs=eval_bs,
    )
    # Monkey‑patch in real model & tokenizer
    dummy.model = model.cpu().eval()
    
    torch.cuda.empty_cache()
    
    dummy.to(device='cuda')
    dummy.tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    return dummy