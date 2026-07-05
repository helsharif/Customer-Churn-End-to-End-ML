"""FastAPI application for churn model serving."""

import time

from fastapi import Depends, FastAPI, HTTPException

from churn_ml.api.dependencies import get_inference_pipeline, inference_pipeline
from churn_ml.api.schemas import CustomerFeatures, ModelInfoResponse, PredictionResponse
from churn_ml.monitoring.logging import write_inference_log
from churn_ml.pipelines.inference_pipeline import ChurnInferencePipeline

app = FastAPI(
    title="Customer Churn End-to-End ML API",
    version="0.1.0",
    description="Prediction API for the Telco Customer Churn ML project.",
)


@app.on_event("startup")
def load_model_on_startup() -> None:
    """Load the model bundle once at service startup when available."""
    try:
        inference_pipeline.load()
    except FileNotFoundError:
        pass


@app.get("/health")
def health_check() -> dict[str, str]:
    """Return service health status."""
    return {"status": "ok"}


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info(
    pipeline: ChurnInferencePipeline = Depends(get_inference_pipeline),
) -> ModelInfoResponse:
    """Return metadata for the currently loaded model."""
    if not pipeline.is_loaded:
        return ModelInfoResponse(status="model_not_loaded")
    assert pipeline.bundle is not None
    return ModelInfoResponse(
        status="loaded",
        model_name=pipeline.bundle.get("model_name"),
        run_id=pipeline.bundle.get("run_id"),
        threshold=pipeline.bundle.get("threshold"),
        primary_metric=pipeline.bundle.get("primary_metric"),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(
    features: CustomerFeatures,
    pipeline: ChurnInferencePipeline = Depends(get_inference_pipeline),
) -> PredictionResponse:
    """Predict whether a customer is likely to churn."""
    started = time.perf_counter()
    try:
        payload = features.model_dump() if hasattr(features, "model_dump") else features.dict()
        prediction = pipeline.predict_one(payload)
    except FileNotFoundError as exc:
        latency_ms = (time.perf_counter() - started) * 1000
        write_inference_log(prediction=None, latency_ms=latency_ms, status="error", error=str(exc))
        raise HTTPException(status_code=503, detail="Model artifact not found. Run training first.") from exc
    except Exception as exc:
        latency_ms = (time.perf_counter() - started) * 1000
        write_inference_log(prediction=None, latency_ms=latency_ms, status="error", error=str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    latency_ms = (time.perf_counter() - started) * 1000
    write_inference_log(prediction=prediction, latency_ms=latency_ms, status="ok")
    return PredictionResponse(**prediction)
