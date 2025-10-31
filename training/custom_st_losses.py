from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from sentence_transformers.cross_encoder.CrossEncoder import CrossEncoder
from sentence_transformers.util import fullname

class KLDivergenceLoss(nn.Module):
    def __init__(
        self,
        model: CrossEncoder,
        activation_fn: nn.Module = nn.LogSoftmax(dim=-1),
        reduction: str = "batchmean",
        **kwargs,
    ) -> None:
        """
        Computes the KL-Divergence loss between the model's predicted distribution (log-probabilities) and 
        the target label distribution. Expects the CrossEncoder to be initialized with num_labels=6.

        Args:
            model (:class:`~sentence_transformers.cross_encoder.CrossEncoder`): A CrossEncoder model to be trained.
            activation_fn (:class:`~torch.nn.Module`): Activation function applied to the logits before computing the loss.
                Should produce log-probabilities (default: LogSoftmax).
            reduction (str): Specifies the reduction to apply to the output (default: "batchmean").
            **kwargs: Additional keyword arguments for nn.KLDivLoss.

        Inputs:
            +-----------------------------+-----------------------------------+-------------------------------+
            | Texts                       | Labels                            | Number of Model Output Labels |
            +=============================+===================================+===============================+
            | (sentence_A, sentence_B)    | Soft distribution over 6 classes  | 6                             |
            +-----------------------------+-----------------------------------+-------------------------------+
        """
        super().__init__()
        self.model = model
        self.activation_fn = activation_fn
        self.kl_div_loss = nn.KLDivLoss(reduction=reduction, **kwargs)

        if not isinstance(self.model, CrossEncoder):
            raise ValueError(
                f"{self.__class__.__name__} expects a model of type CrossEncoder, "
                f"but got a model of type {type(self.model)}."
            )

        if self.model.num_labels != 6:
            raise ValueError(
                f"{self.__class__.__name__} expects a model with 6 output labels, "
                f"but got a model with {self.model.num_labels} output labels."
            )

    def forward(self, inputs: list[list[str]], labels: Tensor) -> Tensor:
        if len(inputs) != 2:
            raise ValueError(
                f"{self.__class__.__name__} expects a dataset with two non-label columns, but got a dataset with {len(inputs)} columns."
            )

        pairs = list(zip(inputs[0], inputs[1]))
        tokens = self.model.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        tokens.to(self.model.device)
        logits = self.model(**tokens)[0]  # shape: [batch_size, 6]
        log_probs = self.activation_fn(logits)  # Log-softmax for KLDivLoss

        # KLDivLoss expects input as log-probabilities and target as probabilities
        if labels.shape != log_probs.shape:
            raise ValueError(
                f"Shape mismatch: log_probs {log_probs.shape}, labels {labels.shape}"
            )

        loss = self.kl_div_loss(log_probs, labels.float().to(log_probs.device))
        return loss

    def get_config_dict(self):
        return {
            "activation_fn": fullname(self.activation_fn),
            "reduction": self.kl_div_loss.reduction,
        }

class JSDLoss(nn.Module):
    def __init__(
        self,
        model: CrossEncoder,
        activation_fn: nn.Module = nn.Identity(),
    ) -> None:
        """
        Computes the Jensen-Shannon Divergence loss between the model's predicted distribution and
        the target label distribution. Expects the CrossEncoder to be initialized with num_labels=6.

        Args:
            model (:class:`~sentence_transformers.cross_encoder.CrossEncoder`): A CrossEncoder model to be trained.
            activation_fn (:class:`~torch.nn.Module`): Activation function applied to logits before softmax (default: nn.Identity).
        """
        super().__init__()
        self.model = model
        self.activation_fn = activation_fn

        if not isinstance(self.model, CrossEncoder):
            raise ValueError(
                f"{self.__class__.__name__} expects a model of type CrossEncoder, "
                f"but got a model of type {type(self.model)}."
            )

        if self.model.num_labels != 6:
            raise ValueError(
                f"{self.__class__.__name__} expects a model with 6 output labels, "
                f"but got a model with {self.model.num_labels} output labels."
            )

    def forward(self, inputs: list[list[str]], labels: Tensor) -> Tensor:
        if len(inputs) != 2:
            raise ValueError(
                f"{self.__class__.__name__} expects a dataset with two non-label columns, but got a dataset with {len(inputs)} columns."
            )

        pairs = list(zip(inputs[0], inputs[1]))
        tokens = self.model.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        tokens.to(self.model.device)
        logits = self.model(**tokens)[0]  # shape: [batch_size, 6]
        logits = self.activation_fn(logits)

        preds = logits.softmax(dim=-1)    # Model prediction probabilities
        targets = labels.float().to(logits.device)  # Ensure float, [batch_size, 6]

        mean = 0.5 * (targets + preds)
        trg_div_mean = torch.clamp_min(targets / mean, 1e-9)
        prd_div_mean = torch.clamp_min(preds / mean, 1e-9)
        kl_trg_mean = (targets * trg_div_mean.log2()).sum(dim=-1)
        kl_prd_mean = (preds * prd_div_mean.log2()).sum(dim=-1)
        loss = 0.5 * (kl_trg_mean + kl_prd_mean).mean()
        return loss

    def get_config_dict(self):
        return {
            "activation_fn": fullname(self.activation_fn),
        }
    
class WassersteinLoss(nn.Module):
    """1-Wasserstein (a.k.a. Earth-Mover) loss for ordinal/discrete distributions."""
    def __init__(self,
                 model: CrossEncoder,
                 p: int = 1,
                 reduction: str = "mean") -> None:
        super().__init__()
        if model.num_labels != 6:
            raise ValueError("Require num_labels=6")
        self.model = model
        self.p = p
        self.reduction = reduction

        # Pre-compute label values on the device for the expectation trick (optional)
        self.register_buffer("values", torch.arange(model.num_labels).float())

    def forward(self,
                inputs: list[list[str]],
                labels: torch.Tensor) -> torch.Tensor:
        # 1. Forward pass through the Cross-Encoder
        a, b = inputs
        toks = self.model.tokenizer(list(zip(a, b)),
                                    padding=True, truncation=True,
                                    return_tensors="pt").to(self.model.device)
        logits = self.model(**toks).logits          # [B, 6]
        probs  = logits.softmax(-1)

        # 2. CDFs and per-example ℓ¹ / ℓ² distance
        cdf_p = probs.cumsum(-1)
        cdf_q = labels.to(probs).cumsum(-1)
        dist  = (cdf_p - cdf_q).abs()
        if self.p == 2:
            dist = dist.pow(2)
        loss = dist.sum(-1) / probs.size(-1)        # normalise by #bins

        return loss.mean() if self.reduction == "mean" else loss.sum()
    
class OrdinalLogLoss(nn.Module):
    """
    Ordinal-/distance-aware negative log-likelihood introduced in
    Castagnos et al. (2022).  For α=1 it reduces to the OLL-1 variant from the
    original repo; larger α puts more weight on near-misses.

    Expected label format
    ---------------------
    `labels` is a probability distribution over the 6 ordered classes  
    (shape ``[batch, 6]``).  The loss uses the **distance matrix stored in
    ``model.module.dist_matrix``** – this should be a square tensor
    ``[6, 6]`` computed once (e.g. with Krippendorff’s ordinal distance) and
    normalised to the range [0, 1].
    """

    def __init__(
        self,
        model: CrossEncoder,
        alpha: float = 1.5,
        reduction: str = "mean",
        eps: float = 1e-7,
    ) -> None:
        super().__init__()

        if not isinstance(model, CrossEncoder):
            raise ValueError(
                f"{self.__class__.__name__} expects a CrossEncoder, got {type(model)}"
            )
        if model.num_labels != 6:
            raise ValueError(
                f"{self.__class__.__name__} expects model.num_labels == 6, "
                f"got {model.num_labels}"
            )
        if reduction not in {"mean", "sum"}:
            raise ValueError("reduction must be 'mean' or 'sum'")

        self.model = model
        self.alpha = float(alpha)
        self.reduction = reduction
        self.eps = eps

        # -----------------------------------------------------------------
        # distance matrix  D  (shape [K, K]) comes from the wrapped model
        # -----------------------------------------------------------------
        dist = getattr(getattr(model, "module", model), "dist_matrix", None)
        if dist is None:
            raise ValueError(
                f"{self.__class__.__name__} expects `model.module.dist_matrix` "
                "to be defined."
            )
        if dist.shape != (model.num_labels, model.num_labels):
            raise ValueError(
                f"dist_matrix shape {dist.shape} is not ({model.num_labels}, "
                f"{model.num_labels})"
            )
        # keep a non‑trainable copy on the correct device
        self.register_buffer("dist_matrix", torch.tensor(dist).float())

    # --------------------------------------------------------------------- #
    #                            forward pass                               #
    # --------------------------------------------------------------------- #
    def forward(self, inputs: list[list[str]], labels: Tensor) -> Tensor:
        if len(inputs) != 2:
            raise ValueError(
                f"{self.__class__.__name__} expects two non-label columns, "
                f"got {len(inputs)}"
            )

        # --- encode the pair of sentences ---------------------------------
        pairs = list(zip(inputs[0], inputs[1]))
        tokens = self.model.tokenizer(
            pairs, padding=True, truncation=True, return_tensors="pt"
        ).to(self.model.device)

        logits = self.model(**tokens).logits           # [B, 6]
        probs = F.softmax(logits, dim=-1)              # model prediction

        # --- compute distance weights  d(y,i)^alpha  ---------------------
        if labels.shape != probs.shape:
            raise ValueError(
                f"Shape mismatch: probs {probs.shape}, labels {labels.shape}"
            )
        labels = labels.to(probs)

        # expected distance to each predicted class, given soft gold labels
        # [B, K] · [K, K] -> [B, K]
        expected_dist = torch.matmul(labels, torch.tensor(self.dist_matrix).float())

        weight = expected_dist.pow(self.alpha)      # d(y,i)^α

        # --- Ordinal Log-Loss  −d·log(1 − p) -----------------------------
        loss_terms = -torch.log(torch.clamp(1.0 - probs, min=self.eps)) * weight
        loss = loss_terms.sum(dim=-1)        # sum over classes

        if self.reduction == "mean":
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss

    # --------------------------------------------------------------------- #
    #                  allow Sentence-Transformers to save config           #
    # --------------------------------------------------------------------- #
    def get_config_dict(self):
        return {
            "alpha": self.alpha,
            "reduction": self.reduction,
            "eps": self.eps,
        }
        
