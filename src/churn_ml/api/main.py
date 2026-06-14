"""Minimal FastAPI app for service health checks."""

from fastapi import FastAPI

app = FastAPI(
    title="Customer Churn End-to-End ML API",
    version="0.1.0",
    description="Starter API for the customer churn ML project.",
)


@app.get("/health")
def health_check() -> dict[str, str]:
    """Return service health status."""
    return {"status": "ok"}
