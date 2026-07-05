from fastapi.testclient import TestClient

from churn_ml.api.dependencies import get_inference_pipeline
from churn_ml.api.main import app


class FakePipeline:
    bundle = {
        "model_name": "fake_model",
        "run_id": "fake-run",
        "threshold": 0.4,
        "primary_metric": "pr_auc",
    }

    @property
    def is_loaded(self):
        return True

    def predict_one(self, payload):
        return {
            "prediction": "Churn",
            "churn_probability": 0.72,
            "threshold": 0.4,
            "model_name": "fake_model",
            "run_id": "fake-run",
        }


def test_predict_endpoint_returns_prediction():
    app.dependency_overrides[get_inference_pipeline] = lambda: FakePipeline()
    client = TestClient(app)

    response = client.post("/predict", json={})

    app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.json()["prediction"] == "Churn"
    assert response.json()["churn_probability"] == 0.72
