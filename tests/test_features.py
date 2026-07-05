import pandas as pd

from churn_ml.features.build_features import build_features


def test_build_features_adds_missingness_and_business_features():
    df = pd.DataFrame(
        {
            "Tenure Months": [0, 12],
            "Total Charges": [" ", "1200"],
            "Monthly Charges": [70.0, 100.0],
            "Phone Service": ["Yes", "Yes"],
            "Multiple Lines": ["No", "Yes"],
            "Online Security": ["No", "Yes"],
            "Online Backup": ["No", "No"],
            "Device Protection": ["No", "Yes"],
            "Tech Support": ["No", "Yes"],
            "Streaming TV": ["No", "Yes"],
            "Streaming Movies": ["No", "Yes"],
            "Contract": ["Month-to-month", "Two year"],
            "Payment Method": ["Electronic check", "Credit card (automatic)"],
            "Internet Service": ["Fiber optic", "DSL"],
            "Senior Citizen": ["Yes", "No"],
            "Dependents": ["No", "Yes"],
        }
    )

    result = build_features(df)

    assert result.loc[0, "total_charges_missing"] == 1
    assert result.loc[0, "is_month_to_month"] == 1
    assert result.loc[0, "uses_electronic_check"] == 1
    assert result.loc[0, "fiber_month_to_month"] == 1
    assert result.loc[1, "service_count"] > result.loc[0, "service_count"]
