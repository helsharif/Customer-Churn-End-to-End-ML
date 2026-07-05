"""Model evaluation helpers."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
    PrecisionRecallDisplay,
    precision_score,
    recall_score,
    RocCurveDisplay,
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
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    if y_proba is not None:
        metrics["pr_auc"] = average_precision_score(y_true, y_proba)
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)

    return metrics


def threshold_metrics(y_true: pd.Series, y_proba: pd.Series, threshold: float) -> dict[str, float]:
    """Calculate classification metrics at a supplied probability threshold."""
    y_pred = (pd.Series(y_proba) >= threshold).astype(int)
    return classification_metrics(y_true, y_pred, y_proba)


def tune_threshold(
    y_true: pd.Series,
    y_proba: pd.Series,
    min_precision: float = 0.45,
) -> float:
    """Pick a validation threshold that maximizes F1 while meeting minimum precision when possible."""
    candidates = [round(value, 3) for value in pd.Series(y_proba).quantile([i / 100 for i in range(5, 96)]).unique()]
    scored: list[tuple[float, float, float]] = []
    for threshold in candidates:
        y_pred = (pd.Series(y_proba) >= threshold).astype(int)
        precision = precision_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        scored.append((threshold, precision, f1))

    eligible = [row for row in scored if row[1] >= min_precision]
    selected_pool = eligible or scored
    return max(selected_pool, key=lambda row: (row[2], row[1]))[0]


def save_evaluation_plots(
    y_true: pd.Series,
    y_proba: pd.Series,
    threshold: float,
    output_dir: Path,
    prefix: str,
) -> dict[str, Path]:
    """Save ROC, precision-recall, and confusion-matrix plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    y_pred = (pd.Series(y_proba) >= threshold).astype(int)
    paths: dict[str, Path] = {}

    roc_path = output_dir / f"{prefix}_roc_curve.png"
    RocCurveDisplay.from_predictions(y_true, y_proba)
    plt.title(f"{prefix} ROC Curve")
    plt.tight_layout()
    plt.savefig(roc_path, dpi=160)
    plt.close()
    paths["roc_curve"] = roc_path

    pr_path = output_dir / f"{prefix}_precision_recall_curve.png"
    PrecisionRecallDisplay.from_predictions(y_true, y_proba)
    plt.title(f"{prefix} Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(pr_path, dpi=160)
    plt.close()
    paths["precision_recall_curve"] = pr_path

    cm_path = output_dir / f"{prefix}_confusion_matrix.png"
    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred)).plot(values_format="d")
    plt.title(f"{prefix} Confusion Matrix at threshold {threshold:.3f}")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=160)
    plt.close()
    paths["confusion_matrix"] = cm_path

    return paths
