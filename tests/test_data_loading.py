from pathlib import Path

import pandas as pd
import pytest

from churn_ml.config import EXPECTED_RAW_COLUMNS, RAW_DATA_PATH
from churn_ml.data.load_data import load_raw_data


def test_raw_data_path_points_to_expected_location():
    assert RAW_DATA_PATH == Path("data/raw/Telco_customer_churn.csv").resolve()


@pytest.mark.skipif(not RAW_DATA_PATH.exists(), reason="Raw Kaggle data is not committed to the repo.")
def test_load_raw_data_local_file():
    df = load_raw_data(RAW_DATA_PATH)

    assert not df.empty
    assert EXPECTED_RAW_COLUMNS.issubset(df.columns)


def test_load_raw_data_raises_for_missing_file(tmp_path):
    missing_file = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError):
        load_raw_data(missing_file)


def test_load_raw_data_validates_expected_columns(tmp_path):
    incomplete_file = tmp_path / "incomplete.csv"
    pd.DataFrame({"CustomerID": ["0001"]}).to_csv(incomplete_file, index=False)

    with pytest.raises(ValueError, match="missing expected columns"):
        load_raw_data(incomplete_file)
