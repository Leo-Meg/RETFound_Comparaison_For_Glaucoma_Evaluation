"""Metric calculation for 3-class glaucoma grading."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from .config import CLASS_NAMES, NUM_CLASSES


def _safe_float(value: float) -> float | None:
    if value is None or math.isnan(float(value)):
        return None
    return float(value)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    num_classes: int = NUM_CLASSES,
) -> tuple[dict[str, float | None], dict[str, Any]]:
    """Compute global and per-class metrics."""

    labels = list(range(num_classes))
    y_true_onehot = label_binarize(y_true, classes=labels)

    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)

    try:
        auroc_macro = roc_auc_score(
            y_true_onehot, y_prob, multi_class="ovr", average="macro"
        )
    except ValueError:
        auroc_macro = float("nan")

    try:
        average_precision_macro = average_precision_score(
            y_true_onehot, y_prob, average="macro"
        )
    except ValueError:
        average_precision_macro = float("nan")

    if math.isnan(auroc_macro):
        composite = float("nan")
    else:
        composite = (f1_macro + auroc_macro + kappa) / 3

    global_metrics = {
        "n_images": float(len(y_true)),
        "accuracy": _safe_float(accuracy),
        "auroc_macro_ovr": _safe_float(auroc_macro),
        "f1_macro": _safe_float(f1_macro),
        "cohen_kappa": _safe_float(kappa),
        "precision_macro": _safe_float(precision_macro),
        "recall_macro": _safe_float(recall_macro),
        "average_precision_macro": _safe_float(average_precision_macro),
        "composite_f1_auc_kappa": _safe_float(composite),
    }

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=list(CLASS_NAMES),
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    details = {
        "classification_report": report,
        "confusion_matrix": cm,
        "class_names": list(CLASS_NAMES),
    }
    return global_metrics, details
