import pandas as pd

from churn_ml.data.clean_data import clean_churn_data


def test_clean_churn_data_drops_default_non_feature_columns():
    df = pd.DataFrame(
        {
            "CustomerID": ["0001"],
            "Count": [1],
            "Lat Long": ["34.0, -118.0"],
            "Latitude": [34.0],
            "Longitude": [-118.0],
            "Churn Label": ["No"],
            "Churn Value": [0],
            "Churn Score": [10],
            "Churn Reason": [None],
            "Monthly Charges": [50.0],
        }
    )

    clean_df = clean_churn_data(df)

    assert "churn_value" in clean_df.columns
    assert "monthly_charges" in clean_df.columns
    assert "lat_long" not in clean_df.columns
    assert "latitude" not in clean_df.columns
    assert "longitude" not in clean_df.columns
    assert "customer_id" not in clean_df.columns
    assert "churn_score" not in clean_df.columns
