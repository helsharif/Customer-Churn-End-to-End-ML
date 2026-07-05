"""Leakage-safe preprocessing utilities."""

from __future__ import annotations

import inspect

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn_ml.config import (
    FEATURE_COLUMNS_TO_DROP,
    RANDOM_SEED,
    STANDARDIZED_TARGET_COLUMN,
    TEST_SIZE,
    VALIDATION_SIZE,
)


def get_feature_columns(df: pd.DataFrame, target_column: str = STANDARDIZED_TARGET_COLUMN) -> list[str]:
    """Return model feature columns after excluding IDs, target, and leakage fields."""
    excluded = set(FEATURE_COLUMNS_TO_DROP) | {target_column}
    return [column for column in df.columns if column not in excluded]


def split_train_validation_test(
    df: pd.DataFrame,
    target_column: str = STANDARDIZED_TARGET_COLUMN,
    random_state: int = RANDOM_SEED,
    validation_size: float = VALIDATION_SIZE,
    test_size: float = TEST_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create stratified train, validation, and test splits."""
    if target_column not in df.columns:
        raise ValueError(f"Target column not found: {target_column}")

    train_validation_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[target_column],
        random_state=random_state,
    )
    validation_fraction = validation_size / (1 - test_size)
    train_df, validation_df = train_test_split(
        train_validation_df,
        test_size=validation_fraction,
        stratify=train_validation_df[target_column],
        random_state=random_state,
    )
    return train_df.reset_index(drop=True), validation_df.reset_index(drop=True), test_df.reset_index(drop=True)


def build_preprocessor(X: pd.DataFrame, scale_numeric: bool = True) -> ColumnTransformer:
    """Build a sklearn preprocessor fitted later on the training split only."""
    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [column for column in X.columns if column not in numeric_features]

    numeric_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    encoder_kwargs: dict[str, object] = {"handle_unknown": "ignore"}
    if "sparse_output" in inspect.signature(OneHotEncoder).parameters:
        encoder_kwargs["sparse_output"] = False
    else:
        encoder_kwargs["sparse"] = False

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(**encoder_kwargs)),
        ]
    )

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_features:
        transformers.append(("numeric", Pipeline(steps=numeric_steps), numeric_features))
    if categorical_features:
        transformers.append(("categorical", categorical_pipeline, categorical_features))

    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)


def split_xy(
    df: pd.DataFrame,
    target_column: str = STANDARDIZED_TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a model-ready dataframe into feature matrix and target series."""
    feature_columns = get_feature_columns(df, target_column=target_column)
    return df[feature_columns], df[target_column].astype(int)
