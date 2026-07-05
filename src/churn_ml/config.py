"""Project configuration and shared paths."""

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

RAW_DATA_PATH = RAW_DATA_DIR / "Telco_customer_churn.csv"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "telco_customer_churn_clean.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"
LOGS_DIR = PROJECT_ROOT / "logs"
MONITORING_DIR = PROJECT_ROOT / "monitoring"
MLFLOW_TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
MLFLOW_EXPERIMENT_NAME = "telco-churn-production-pipeline"
PRODUCTION_MODEL_PATH = MODELS_DIR / "production_model.joblib"
TRAINING_BASELINE_PATH = ARTIFACTS_DIR / "monitoring" / "training_baseline.parquet"

TARGET_COLUMN = "Churn Value"
STANDARDIZED_TARGET_COLUMN = "churn_value"
CUSTOMER_ID_COLUMN = "CustomerID"
STANDARDIZED_CUSTOMER_ID_COLUMN = "customer_id"
RANDOM_SEED = 42
TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15
PRIMARY_METRIC = "pr_auc"
DEFAULT_THRESHOLD = 0.5
MIN_PRECISION_FOR_THRESHOLD = 0.45

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

RAW_NUMERIC_COLUMNS = [
    "Tenure Months",
    "Monthly Charges",
    "Total Charges",
    "CLTV",
]

STANDARDIZED_NUMERIC_COLUMNS = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "cltv",
]

RAW_CATEGORICAL_LEVELS = {
    "Gender": ["Female", "Male"],
    "Senior Citizen": ["No", "Yes"],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "Phone Service": ["No", "Yes"],
    "Multiple Lines": ["No", "No phone service", "Yes"],
    "Internet Service": ["DSL", "Fiber optic", "No"],
    "Online Security": ["No", "No internet service", "Yes"],
    "Online Backup": ["No", "No internet service", "Yes"],
    "Device Protection": ["No", "No internet service", "Yes"],
    "Tech Support": ["No", "No internet service", "Yes"],
    "Streaming TV": ["No", "No internet service", "Yes"],
    "Streaming Movies": ["No", "No internet service", "Yes"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "Paperless Billing": ["No", "Yes"],
    "Payment Method": [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ],
}

FEATURE_COLUMNS_TO_DROP = [
    column for column in STANDARDIZED_LEAKAGE_AND_ID_COLUMNS if column != STANDARDIZED_TARGET_COLUMN
]


@dataclass(frozen=True)
class RuntimeSettings:
    """Environment-configurable runtime settings."""

    api_model_path: Path = PRODUCTION_MODEL_PATH
    api_base_url: str = "http://127.0.0.1:8000"
    inference_log_path: Path = LOGS_DIR / "inference.jsonl"


def ensure_project_directories() -> None:
    """Create local output directories used by training, serving, and monitoring."""
    for path in [
        INTERIM_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        FIGURES_DIR,
        METRICS_DIR,
        LOGS_DIR,
        ARTIFACTS_DIR / "monitoring",
    ]:
        path.mkdir(parents=True, exist_ok=True)
