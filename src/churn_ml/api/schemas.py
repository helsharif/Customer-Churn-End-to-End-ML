"""Pydantic schemas for the churn prediction API."""

from pydantic import BaseModel, Field


class CustomerFeatures(BaseModel):
    """Raw, inference-time customer fields accepted by the API."""

    gender: str = Field(default="Female")
    senior_citizen: str = Field(default="No")
    partner: str = Field(default="No")
    dependents: str = Field(default="No")
    tenure_months: int = Field(default=12, ge=0, le=100)
    phone_service: str = Field(default="Yes")
    multiple_lines: str = Field(default="No")
    internet_service: str = Field(default="Fiber optic")
    online_security: str = Field(default="No")
    online_backup: str = Field(default="No")
    device_protection: str = Field(default="No")
    tech_support: str = Field(default="No")
    streaming_tv: str = Field(default="No")
    streaming_movies: str = Field(default="No")
    contract: str = Field(default="Month-to-month")
    paperless_billing: str = Field(default="Yes")
    payment_method: str = Field(default="Electronic check")
    monthly_charges: float = Field(default=75.0, ge=0)
    total_charges: float | None = Field(default=900.0, ge=0)
    cltv: float | None = Field(default=3000.0, ge=0)
    country: str | None = Field(default="United States")
    state: str | None = Field(default="California")
    city: str | None = Field(default="Los Angeles")
    zip_code: int | None = Field(default=90001)


class PredictionResponse(BaseModel):
    """Prediction response returned by the API."""

    prediction: str
    churn_probability: float
    threshold: float
    model_name: str
    run_id: str


class ModelInfoResponse(BaseModel):
    """Current loaded model metadata."""

    status: str
    model_name: str | None = None
    run_id: str | None = None
    threshold: float | None = None
    primary_metric: str | None = None
