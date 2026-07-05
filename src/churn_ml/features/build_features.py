"""Feature engineering for the Telco Customer Churn workflow.

The features here extend the notebook logic without replacing it. The notebooks
already normalize known raw-data issues and preserve a `Total Charges` missing
indicator for XGBoost; this module packages those choices and adds a few
transparent, inference-time-safe business features.
"""

import numpy as np
import pandas as pd

from churn_ml.data.clean_data import handle_missing_values, standardize_column_names

SERVICE_COLUMNS = [
    "phone_service",
    "multiple_lines",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
]


FEATURE_RATIONALE = {
    "total_charges_missing": "Flags the known blank Total Charges records from the source data.",
    "tenure_group": "Buckets tenure into relationship stages that often align with churn risk.",
    "avg_monthly_total_charges": "Approximates historical monthly spend from cumulative charges.",
    "service_count": "Counts subscribed services to represent product embeddedness.",
    "has_internet_addon": "Flags customers with at least one internet add-on service.",
    "is_month_to_month": "Month-to-month contracts usually have lower switching friction.",
    "uses_electronic_check": "Electronic-check payment has been a high-risk billing segment in the notebooks.",
    "fiber_month_to_month": "Combines fiber service with short-term contract risk.",
    "senior_without_dependents": "Captures a simple household interaction available at inference time.",
}


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build leakage-safe features available both during training and inference."""
    features = standardize_column_names(df)
    features = handle_missing_values(features)

    if "total_charges" in features.columns:
        features["total_charges_missing"] = features["total_charges"].isna().astype(int)

    if "tenure_months" in features.columns:
        features["tenure_group"] = pd.cut(
            features["tenure_months"].fillna(-1),
            bins=[-2, 0, 12, 24, 48, 72, np.inf],
            labels=["missing_or_zero", "0_12", "13_24", "25_48", "49_72", "73_plus"],
        ).astype("object")

    if {"total_charges", "tenure_months"}.issubset(features.columns):
        denominator = features["tenure_months"].replace(0, np.nan)
        features["avg_monthly_total_charges"] = features["total_charges"] / denominator

    yes_columns = [column for column in SERVICE_COLUMNS if column in features.columns]
    if yes_columns:
        features["service_count"] = features[yes_columns].apply(
            lambda row: sum(str(value).strip().lower() == "yes" for value in row),
            axis=1,
        )

    internet_addons = [
        column
        for column in [
            "online_security",
            "online_backup",
            "device_protection",
            "tech_support",
            "streaming_tv",
            "streaming_movies",
        ]
        if column in features.columns
    ]
    if internet_addons:
        features["has_internet_addon"] = features[internet_addons].apply(
            lambda row: int(any(str(value).strip().lower() == "yes" for value in row)),
            axis=1,
        )

    if "contract" in features.columns:
        features["is_month_to_month"] = (
            features["contract"].astype(str).str.casefold().eq("month-to-month").astype(int)
        )

    if "payment_method" in features.columns:
        features["uses_electronic_check"] = (
            features["payment_method"].astype(str).str.casefold().eq("electronic check").astype(int)
        )

    if {"internet_service", "contract"}.issubset(features.columns):
        features["fiber_month_to_month"] = (
            features["internet_service"].astype(str).str.casefold().eq("fiber optic")
            & features["contract"].astype(str).str.casefold().eq("month-to-month")
        ).astype(int)

    if {"senior_citizen", "dependents"}.issubset(features.columns):
        features["senior_without_dependents"] = (
            features["senior_citizen"].astype(str).str.casefold().eq("yes")
            & features["dependents"].astype(str).str.casefold().eq("no")
        ).astype(int)

    return features
