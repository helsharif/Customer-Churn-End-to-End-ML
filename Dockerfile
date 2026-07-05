FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    CHURN_MODEL_PATH=/app/artifacts/models/production_model.joblib \
    CHURN_API_BASE_URL=http://127.0.0.1:8000

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-hf.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements-hf.txt

COPY README.md pyproject.toml ./
COPY config ./config
COPY data/raw ./data/raw
COPY artifacts/models/xgboost_optuna_best_params.json ./artifacts/models/xgboost_optuna_best_params.json
COPY src ./src
COPY scripts ./scripts

RUN mkdir -p artifacts/models artifacts/monitoring reports/figures reports/metrics logs \
    && python scripts/run_training.py

EXPOSE 7860

CMD ["sh", "scripts/start_hf_space.sh"]
