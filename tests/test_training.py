import pandas as pd

from churn_ml.models.train_model import build_candidate_models


def test_build_candidate_models_includes_logistic_regression_and_xgboost_only():
    X = pd.DataFrame(
        {
            "monthly_charges": [70.0, 80.0, 30.0, 20.0],
            "contract": ["Month-to-month", "One year", "Two year", "Month-to-month"],
        }
    )

    models = build_candidate_models(X)

    assert list(models) == ["logistic_regression", "xgboost"]
