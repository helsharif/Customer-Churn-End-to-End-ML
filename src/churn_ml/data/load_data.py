"""Utilities for loading the raw telco churn dataset."""

from pathlib import Path

import pandas as pd

from churn_ml.config import EXPECTED_RAW_COLUMNS, RAW_DATA_PATH


def load_raw_data(csv_path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw churn CSV and validate the expected starter schema."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found: {path}")

    df = pd.read_csv(path)
    missing_columns = EXPECTED_RAW_COLUMNS.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Raw data is missing expected columns: {missing}")

    return df
