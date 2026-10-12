"""Losses used by the affinity fine-tuning task.

Regression labels are optional.  Missing labels are excluded from both the
numerator and denominator instead of being replaced by the model prediction.
"""

from typing import Any, Literal, Optional

import torch
from torch import Tensor
from torch.nn import functional as F


def _zero_loss(reference: Tensor) -> Tensor:
    """Return a differentiable zero on the same device as ``reference``."""
    return reference.sum() * 0.0


def _target_tensor(
    records: list[Any],
    attribute: str,
    prediction: Tensor,
    multiplicity: int = 1,
) -> Tensor:
    """Build a target tensor without treating a valid numeric zero as missing."""
    values: list[float] = []
    for record in records:
        affinity = getattr(record, "affinity", None)
        value = getattr(affinity, attribute, None) if affinity is not None else None
        values.append(float("nan") if value is None else float(value))

    targets = torch.as_tensor(
        values,
        device=prediction.device,
        dtype=prediction.dtype,
    )
    if multiplicity > 1:
        targets = targets.repeat_interleave(multiplicity)
    if targets.numel() != prediction.numel():
        raise ValueError(
            f"Target/prediction size mismatch for {attribute}: "
            f"{targets.numel()} targets vs {prediction.numel()} predictions."
        )
    return targets.reshape_as(prediction)


def huber_loss(y: Tensor, y_true: Tensor, delta: float = 0.5) -> tuple[Tensor, Tensor]:
    """Masked Huber loss; non-finite targets do not enter the mean."""
    valid_mask = torch.isfinite(y_true)
    if not torch.any(valid_mask):
        return _zero_loss(y), y_true
    return F.huber_loss(y[valid_mask], y_true[valid_mask], delta=delta), y_true


def mse_loss(y: Tensor, y_true: Tensor) -> tuple[Tensor, Tensor]:
    """Masked mean squared error."""
    valid_mask = torch.isfinite(y_true)
    if not torch.any(valid_mask):
        return _zero_loss(y), y_true
    return F.mse_loss(y[valid_mask], y_true[valid_mask]), y_true


def focal_loss(
    prob: Tensor,
    targets: Tensor,
    gamma: float = 1.0,
    alpha: Optional[float] = None,
) -> Tensor:
    """Binary focal loss on probabilities.

    If supplied, ``alpha`` is the standard positive-class weight and must be in
    [0, 1]; negative examples receive weight ``1 - alpha``.  ``None`` applies no
    class weighting.
    """
    valid_mask = torch.isfinite(targets)
    if not torch.any(valid_mask):
        return _zero_loss(prob)

    # BCE on probabilities is explicitly disallowed under CUDA autocast.
    # Convert the (possibly ensembled) probability to a float32 logit and use
    # the autocast-safe logits implementation. This preserves the probability-
    # ensemble definition while retaining a stable gradient.
    prob = prob[valid_mask].float().clamp(min=1e-7, max=1.0 - 1e-7)
    targets = targets[valid_mask].float()
    logits = torch.logit(prob)
    p_t = targets * prob + (1.0 - targets) * (1.0 - prob)
    loss = (1.0 - p_t).pow(gamma) * F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    if alpha is not None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("focal_alpha must be between 0 and 1.")
        alpha_t = targets * alpha + (1.0 - targets) * (1.0 - alpha)
        loss = alpha_t * loss
    return loss.mean()


def bce_loss(prob: Tensor, targets: Tensor) -> Tensor:
    """Masked binary cross entropy on probabilities."""
    valid_mask = torch.isfinite(targets)
    if not torch.any(valid_mask):
        return _zero_loss(prob)
    prob = prob[valid_mask].float().clamp(min=1e-7, max=1.0 - 1e-7)
    targets = targets[valid_mask].float()
    return F.binary_cross_entropy_with_logits(
        torch.logit(prob),
        targets,
    )


def affinity_loss(
    model_out: dict[str, Tensor],
    feats: dict[str, Any],
    multiplicity: int = 1,
    *,
    regression_enabled: bool = True,
    classification_enabled: bool = True,
    regression_loss: Literal["huber", "mse"] = "huber",
    classification_loss: Literal["focal", "bce"] = "focal",
    lambda_reg: float = 1.0,
    lambda_cls: float = 1.0,
    huber_delta: float = 0.5,
    focal_gamma: float = 1.0,
    focal_alpha: Optional[float] = None,
) -> dict[str, Any]:
    """Compute a configurable joint affinity objective.

    The total is ``lambda_reg * L_reg + lambda_cls * L_cls``.  Either task can
    be disabled for the single-task ablations, but at least one must be active.
    """
    if not regression_enabled and not classification_enabled:
        raise ValueError("At least one affinity task must be enabled.")
    if lambda_reg < 0 or lambda_cls < 0:
        raise ValueError("Affinity loss weights must be non-negative.")

    records = feats["record"]
    reg_pred = model_out["affinity_pred_value"]
    cls_pred = model_out["affinity_probability_binary"]
    reg_targets = _target_tensor(records, "true_affinity", reg_pred, multiplicity)
    cls_targets = _target_tensor(records, "true_class", cls_pred, multiplicity)

    reg_valid = torch.isfinite(reg_targets)
    cls_valid = torch.isfinite(cls_targets)

    if regression_enabled:
        if regression_loss == "huber":
            reg_value, _ = huber_loss(reg_pred, reg_targets, delta=huber_delta)
        elif regression_loss == "mse":
            reg_value, _ = mse_loss(reg_pred, reg_targets)
        else:
            raise ValueError(f"Unsupported regression loss: {regression_loss}")
    else:
        reg_value = _zero_loss(reg_pred)

    if classification_enabled:
        if classification_loss == "focal":
            cls_value = focal_loss(
                cls_pred,
                cls_targets,
                gamma=focal_gamma,
                alpha=focal_alpha,
            )
        elif classification_loss == "bce":
            cls_value = bce_loss(cls_pred, cls_targets)
        else:
            raise ValueError(f"Unsupported classification loss: {classification_loss}")
    else:
        cls_value = _zero_loss(cls_pred)

    total = lambda_reg * reg_value + lambda_cls * cls_value
    return {
        "loss": total,
        "loss_breakdown": {
            "affinity_pred_value_loss": reg_value,
            "affinity_probability_binary_loss": cls_value,
            "rel_affinity_pred_value": reg_targets,
            "rel_affinity_probability_binary": cls_targets,
            "regression_valid_count": reg_valid.sum(),
            "classification_valid_count": cls_valid.sum(),
        },
    }
