"""Inference pipeline shared by FastAPI and tests."""

from __future__ import annotations

import os
from typing import Any

import pandas as pd

from churn_ml.config import PRODUCTION_MODEL_PATH
from churn_ml.models.predict_model import load_model_bundle, predict_from_bundle


class ChurnInferencePipeline:
    """Load a model once and serve predictions for raw feature records."""

    def __init__(self, model_path=None) -> None:
        self.model_path = model_path or os.getenv("CHURN_MODEL_PATH") or PRODUCTION_MODEL_PATH
        self.bundle: dict[str, Any] | None = None

    def load(self) -> None:
        """Load the persisted model bundle."""
        self.bundle = load_model_bundle(self.model_path) if self.model_path else load_model_bundle()

    @property
    def is_loaded(self) -> bool:
        """Return whether the model bundle is loaded."""
        return self.bundle is not None

    def predict_one(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Predict churn for one customer payload."""
        if self.bundle is None:
            self.load()
        assert self.bundle is not None
        predictions = predict_from_bundle(self.bundle, pd.DataFrame([payload]))
        return predictions.iloc[0].to_dict()
