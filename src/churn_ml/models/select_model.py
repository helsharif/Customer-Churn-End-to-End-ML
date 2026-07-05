"""Model selection helpers."""

import pandas as pd

from churn_ml.config import PRIMARY_METRIC


def select_best_model(results: dict[str, dict[str, float]], primary_metric: str = PRIMARY_METRIC) -> str:
    """Select the model with the highest validation metric."""
    if not results:
        raise ValueError("No model results were supplied.")
    missing = [name for name, metrics in results.items() if primary_metric not in metrics]
    if missing:
        raise ValueError(f"Primary metric {primary_metric!r} missing for: {missing}")
    return max(results, key=lambda model_name: results[model_name][primary_metric])


def results_to_frame(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Convert model metrics to a sorted dataframe."""
    return (
        pd.DataFrame.from_dict(results, orient="index")
        .rename_axis("model")
        .reset_index()
        .sort_values(PRIMARY_METRIC, ascending=False)
    )
