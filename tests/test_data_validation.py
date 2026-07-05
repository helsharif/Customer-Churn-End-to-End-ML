import pandas as pd

from churn_ml.config import EXPECTED_RAW_COLUMNS
from churn_ml.data.validate_data import validate_raw_data


def _valid_raw_frame() -> pd.DataFrame:
    data = {column: ["placeholder"] for column in EXPECTED_RAW_COLUMNS}
    data.update(
        {
            "CustomerID": ["0001"],
            "Count": [1],
            "Zip Code": [90001],
            "Latitude": [34.0],
            "Longitude": [-118.0],
            "Senior Citizen": ["No"],
            "Tenure Months": [12],
            "Monthly Charges": [70.0],
            "Total Charges": ["840.0"],
            "Churn Value": [0],
            "Churn Score": [10],
            "CLTV": [3000],
            "Gender": ["Female"],
            "Partner": ["No"],
            "Dependents": ["No"],
            "Phone Service": ["Yes"],
            "Multiple Lines": ["No"],
            "Internet Service": ["DSL"],
            "Online Security": ["No"],
            "Online Backup": ["No"],
            "Device Protection": ["No"],
            "Tech Support": ["No"],
            "Streaming TV": ["No"],
            "Streaming Movies": ["No"],
            "Contract": ["Month-to-month"],
            "Paperless Billing": ["Yes"],
            "Payment Method": ["Electronic check"],
        }
    )
    return pd.DataFrame(data)


def test_validate_raw_data_accepts_expected_schema():
    report = validate_raw_data(_valid_raw_frame())

    assert report.is_valid


def test_validate_raw_data_flags_invalid_target():
    df = _valid_raw_frame()
    df["Churn Value"] = [2]

    report = validate_raw_data(df)

    assert not report.is_valid
    assert any("Invalid target values" in error for error in report.errors)
