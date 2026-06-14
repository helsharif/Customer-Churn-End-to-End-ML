"""Lightweight cleaning helpers for the telco churn dataset."""

import re

import numpy as np
import pandas as pd

from churn_ml.config import STANDARDIZED_LEAKAGE_AND_ID_COLUMNS


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with snake_case column names."""
    cleaned = df.copy()
    cleaned.columns = [_to_snake_case(column) for column in cleaned.columns]
    return cleaned


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize obvious blank values and coerce known numeric columns."""
    cleaned = df.copy()
    cleaned = cleaned.replace(r"^\s*$", np.nan, regex=True)

    for column in ["total_charges", "monthly_charges", "tenure_months", "cltv"]:
        if column in cleaned.columns:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    return cleaned


def drop_leakage_columns(
    df: pd.DataFrame,
    keep_target: bool = True,
    leakage_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Drop ID, outcome-equivalent, and post-outcome columns."""
    cleaned = df.copy()
    columns_to_drop = set(leakage_columns or STANDARDIZED_LEAKAGE_AND_ID_COLUMNS)

    if keep_target:
        columns_to_drop.discard("churn_value")

    existing_columns = [column for column in columns_to_drop if column in cleaned.columns]
    return cleaned.drop(columns=existing_columns)


def clean_churn_data(df: pd.DataFrame, keep_target: bool = True) -> pd.DataFrame:
    """Run the starter cleaning flow for EDA or baseline modeling."""
    cleaned = standardize_column_names(df)
    cleaned = handle_missing_values(cleaned)
    cleaned = drop_leakage_columns(cleaned, keep_target=keep_target)
    return cleaned


def _to_snake_case(column_name: str) -> str:
    column_name = column_name.strip()
    column_name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", column_name)
    column_name = re.sub(r"[^0-9a-zA-Z]+", "_", column_name)
    return column_name.strip("_").lower()
