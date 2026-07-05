"""Lightweight local drift detection for portfolio monitoring."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from churn_ml.config import METRICS_DIR, TRAINING_BASELINE_PATH


def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """Calculate population stability index for numeric distributions."""
    expected_numeric = pd.to_numeric(expected, errors="coerce").dropna()
    actual_numeric = pd.to_numeric(actual, errors="coerce").dropna()
    if expected_numeric.empty or actual_numeric.empty:
        return 0.0
    quantiles = np.linspace(0, 1, bins + 1)
    breakpoints = np.unique(expected_numeric.quantile(quantiles).to_numpy())
    if len(breakpoints) < 3:
        return 0.0
    expected_counts = pd.cut(expected_numeric, breakpoints, include_lowest=True).value_counts(normalize=True)
    actual_counts = pd.cut(actual_numeric, breakpoints, include_lowest=True).value_counts(normalize=True)
    aligned = pd.concat([expected_counts, actual_counts], axis=1).fillna(0.0001)
    aligned.columns = ["expected", "actual"]
    return float(((aligned["actual"] - aligned["expected"]) * np.log(aligned["actual"] / aligned["expected"])).sum())


def category_shift(expected: pd.Series, actual: pd.Series) -> float:
    """Return total variation distance between category distributions."""
    expected_freq = expected.astype("string").fillna("<missing>").value_counts(normalize=True)
    actual_freq = actual.astype("string").fillna("<missing>").value_counts(normalize=True)
    aligned = pd.concat([expected_freq, actual_freq], axis=1).fillna(0.0)
    aligned.columns = ["expected", "actual"]
    return float((aligned["expected"] - aligned["actual"]).abs().sum() / 2)


def build_drift_report(
    baseline_path: Path = TRAINING_BASELINE_PATH,
    current_path: Path | None = None,
    output_path: Path = METRICS_DIR / "drift_report.json",
) -> dict[str, object]:
    """Compare current feature distributions with the training baseline."""
    if not baseline_path.exists():
        raise FileNotFoundError(f"Training baseline not found: {baseline_path}. Run training first.")
    baseline = pd.read_parquet(baseline_path)
    current = pd.read_parquet(current_path) if current_path else baseline.sample(frac=0.2, random_state=42)

    report: dict[str, object] = {"baseline_rows": len(baseline), "current_rows": len(current), "features": {}}
    for column in baseline.columns.intersection(current.columns):
        if column == "churn_value":
            continue
        if pd.api.types.is_numeric_dtype(baseline[column]):
            ks_stat, p_value = ks_2samp(
                pd.to_numeric(baseline[column], errors="coerce").dropna(),
                pd.to_numeric(current[column], errors="coerce").dropna(),
            )
            report["features"][column] = {
                "type": "numeric",
                "psi": psi(baseline[column], current[column]),
                "ks_statistic": float(ks_stat),
                "ks_p_value": float(p_value),
            }
        else:
            report["features"][column] = {
                "type": "categorical",
                "category_shift": category_shift(baseline[column], current[column]),
            }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    build_drift_report()
