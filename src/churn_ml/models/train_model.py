"""Model training utilities for notebook-derived churn models."""

import json
from pathlib import Path

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from churn_ml.config import (
    ARTIFACTS_DIR,
    RANDOM_SEED,
    STANDARDIZED_TARGET_COLUMN,
)
from churn_ml.data.preprocess import build_preprocessor, get_feature_columns


def train_baseline_logistic_regression(
    df: pd.DataFrame,
    target_column: str = STANDARDIZED_TARGET_COLUMN,
) -> Pipeline:
    """Train a simple preprocessing + logistic regression baseline pipeline."""
    if target_column not in df.columns:
        raise ValueError(f"Target column not found: {target_column}")

    y = df[target_column]
    feature_columns = get_feature_columns(df, target_column=target_column)
    X = df[feature_columns]

    model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X, scale_numeric=True)),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    model.fit(X, y)
    return model


def build_candidate_models(X_train: pd.DataFrame) -> dict[str, Pipeline]:
    """Create the logistic-regression and XGBoost candidates used by the workflow."""
    models: dict[str, Pipeline] = {
        "logistic_regression": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X_train, scale_numeric=True)),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
    }

    xgboost_model = _build_xgboost_candidate(X_train)
    if xgboost_model is None:
        raise ImportError("XGBoost is required for the production model-comparison workflow.")
    models["xgboost"] = xgboost_model

    return models


def fit_model(model: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Fit and return a sklearn pipeline."""
    return model.fit(X_train, y_train)


def predict_proba(model: BaseEstimator, X: pd.DataFrame) -> pd.Series:
    """Return churn probabilities from a classifier that supports probability scoring."""
    if not hasattr(model, "predict_proba"):
        raise TypeError("Model does not support predict_proba.")
    return pd.Series(model.predict_proba(X)[:, 1], index=X.index)


def _build_xgboost_candidate(X_train: pd.DataFrame) -> Pipeline | None:
    """Build an XGBoost candidate when the dependency is available."""
    try:
        from xgboost import XGBClassifier
    except Exception:
        return None

    params_path = ARTIFACTS_DIR / "models" / "xgboost_optuna_best_params.json"
    params = _load_xgboost_params(params_path)
    classifier = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        random_state=RANDOM_SEED,
        n_jobs=2,
        **params,
    )
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train, scale_numeric=False)),
            ("classifier", classifier),
        ]
    )


def _load_xgboost_params(params_path: Path) -> dict[str, object]:
    if not params_path.exists():
        return {
            "n_estimators": 250,
            "max_depth": 3,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_lambda": 1.0,
        }
    with params_path.open("r", encoding="utf-8") as file:
        params = json.load(file)
    params.pop("cv_pr_auc", None)
    params.pop("objective", None)
    params.pop("eval_metric", None)
    params.pop("scale_pos_weight_multiplier", None)
    return params
