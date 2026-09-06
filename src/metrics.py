"""Metric definitions shared by the manuscript analysis."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import auc, precision_recall_curve


def trapezoidal_pr_auc(y_true, probability) -> float:
    """Return PR-AUC by trapezoidal integration with recall on the x-axis."""
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    if y_true.ndim != 1 or probability.ndim != 1 or len(y_true) != len(probability):
        raise ValueError("y_true and probability must be one-dimensional and equal length")
    if len(np.unique(y_true)) < 2:
        raise ValueError("PR-AUC requires both outcome classes")
    if not np.isfinite(probability).all():
        raise ValueError("probability contains non-finite values")
    precision, recall, _ = precision_recall_curve(y_true, probability)
    return float(auc(recall, precision))
