"""Starter baseline model training utilities."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn_ml.config import STANDARDIZED_LEAKAGE_AND_ID_COLUMNS, STANDARDIZED_TARGET_COLUMN


def train_baseline_logistic_regression(
    df: pd.DataFrame,
    target_column: str = STANDARDIZED_TARGET_COLUMN,
) -> Pipeline:
    """Train a simple preprocessing + logistic regression baseline pipeline."""
    if target_column not in df.columns:
        raise ValueError(f"Target column not found: {target_column}")

    y = df[target_column]
    excluded = set(STANDARDIZED_LEAKAGE_AND_ID_COLUMNS)
    excluded.discard(target_column)
    feature_columns = [column for column in df.columns if column not in excluded | {target_column}]
    X = df[feature_columns]

    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number", "bool"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    model.fit(X, y)
    return model
