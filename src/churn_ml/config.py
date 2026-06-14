"""Project configuration and shared paths."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

RAW_DATA_PATH = RAW_DATA_DIR / "Telco_customer_churn.csv"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "telco_customer_churn_clean.csv"

TARGET_COLUMN = "Churn Value"
STANDARDIZED_TARGET_COLUMN = "churn_value"

EXPECTED_RAW_COLUMNS = {
    "CustomerID",
    "Count",
    "Country",
    "State",
    "City",
    "Zip Code",
    "Lat Long",
    "Latitude",
    "Longitude",
    "Gender",
    "Senior Citizen",
    "Partner",
    "Dependents",
    "Tenure Months",
    "Phone Service",
    "Multiple Lines",
    "Internet Service",
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Contract",
    "Paperless Billing",
    "Payment Method",
    "Monthly Charges",
    "Total Charges",
    "Churn Label",
    "Churn Value",
    "Churn Score",
    "CLTV",
    "Churn Reason",
}

LEAKAGE_AND_ID_COLUMNS = [
    "CustomerID",
    "Count",
    "Lat Long",
    "Latitude",
    "Longitude",
    "Churn Label",
    "Churn Value",
    "Churn Score",
    "Churn Reason",
    "Churn Category",
]

STANDARDIZED_LEAKAGE_AND_ID_COLUMNS = [
    "customer_id",
    "count",
    "lat_long",
    "latitude",
    "longitude",
    "churn_label",
    "churn_value",
    "churn_score",
    "churn_reason",
    "churn_category",
]
