import numpy as np
from sklearn.metrics import auc, average_precision_score, precision_recall_curve

from src.metrics import trapezoidal_pr_auc


def test_trapezoidal_definition_matches_curve_auc_and_not_average_precision():
    y_true = np.array([0, 1, 0, 1, 1, 0])
    probability = np.array([0.05, 0.55, 0.45, 0.90, 0.35, 0.20])
    precision, recall, _ = precision_recall_curve(y_true, probability)
    observed = trapezoidal_pr_auc(y_true, probability)
    assert np.isclose(observed, auc(recall, precision))
    assert not np.isclose(observed, average_precision_score(y_true, probability))
