"""Structured JSONL inference logging."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from churn_ml.config import LOGS_DIR


def write_inference_log(
    *,
    prediction: dict[str, Any] | None,
    latency_ms: float,
    status: str,
    error: str | None = None,
    log_path: Path = LOGS_DIR / "inference.jsonl",
) -> None:
    """Append anonymized inference metadata to a JSON Lines log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "request_id": str(uuid.uuid4()),
        "status": status,
        "latency_ms": round(latency_ms, 3),
        "error": error,
    }
    if prediction:
        record.update(
            {
                "model_name": prediction.get("model_name"),
                "run_id": prediction.get("run_id"),
                "prediction": prediction.get("prediction"),
                "churn_probability": prediction.get("churn_probability"),
                "threshold": prediction.get("threshold"),
            }
        )
    with log_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")
