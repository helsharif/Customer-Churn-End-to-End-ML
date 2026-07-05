"""Prediction helpers for persisted churn model bundles."""

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from churn_ml.config import PRODUCTION_MODEL_PATH
from churn_ml.features.build_features import build_features


def predict_churn(model, features: pd.DataFrame):
    """Generate churn predictions from a fitted model pipeline."""
    return model.predict(features)


def load_model_bundle(model_path: str | Path = PRODUCTION_MODEL_PATH) -> dict[str, Any]:
    """Load the persisted production model bundle."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Production model artifact not found: {path}")
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError(f"Invalid model bundle at {path}")
    return bundle


def predict_from_bundle(bundle: dict[str, Any], raw_features: pd.DataFrame) -> pd.DataFrame:
    """Return prediction labels and probabilities for raw request features."""
    model = bundle["model"]
    threshold = float(bundle.get("threshold", 0.5))
    expected_columns = bundle.get("feature_columns")
    features = build_features(raw_features)
    if expected_columns:
        missing = sorted(set(expected_columns).difference(features.columns))
        if missing:
            raise ValueError(f"Missing required feature columns: {missing}")
        features = features[expected_columns]

    probabilities = pd.Series(model.predict_proba(features)[:, 1], index=features.index)
    predictions = (probabilities >= threshold).astype(int)
    return pd.DataFrame(
        {
            "prediction": predictions.map({1: "Churn", 0: "No Churn"}),
            "churn_probability": probabilities,
            "threshold": threshold,
            "model_name": bundle.get("model_name", "unknown"),
            "run_id": bundle.get("run_id", "local"),
        }
    )
