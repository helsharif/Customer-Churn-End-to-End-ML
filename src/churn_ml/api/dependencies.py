"""FastAPI dependencies for model serving."""

from churn_ml.pipelines.inference_pipeline import ChurnInferencePipeline

inference_pipeline = ChurnInferencePipeline()


def get_inference_pipeline() -> ChurnInferencePipeline:
    """Return the singleton inference pipeline."""
    return inference_pipeline
