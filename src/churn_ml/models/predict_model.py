"""Prediction helpers for future inference workflows."""

import pandas as pd


def predict_churn(model, features: pd.DataFrame):
    """Generate churn predictions from a fitted model pipeline."""
    return model.predict(features)
