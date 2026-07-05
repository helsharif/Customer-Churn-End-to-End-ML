import pandas as pd

from churn_ml.data.preprocess import get_feature_columns, split_train_validation_test


def test_get_feature_columns_excludes_target_and_leakage():
    df = pd.DataFrame(
        {
            "customer_id": ["1"],
            "churn_value": [0],
            "churn_score": [50],
            "monthly_charges": [70.0],
        }
    )

    assert get_feature_columns(df) == ["monthly_charges"]


def test_split_train_validation_test_preserves_rows():
    df = pd.DataFrame(
        {
            "monthly_charges": range(100),
            "churn_value": [0, 1] * 50,
        }
    )

    train_df, validation_df, test_df = split_train_validation_test(df)

    assert len(train_df) + len(validation_df) + len(test_df) == len(df)
    assert set(train_df["churn_value"].unique()) == {0, 1}
    assert set(validation_df["churn_value"].unique()) == {0, 1}
    assert set(test_df["churn_value"].unique()) == {0, 1}
