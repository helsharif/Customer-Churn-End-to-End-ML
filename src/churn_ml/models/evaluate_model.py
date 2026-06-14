"""Model evaluation helpers."""

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_proba: pd.Series | None = None,
) -> dict[str, float]:
    """Calculate common binary classification metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    if y_proba is not None:
        metrics["pr_auc"] = average_precision_score(y_true, y_proba)
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)

    return metrics
