.PHONY: install test lint train api ui mlflow docker-up docker-down drift

install:
	python -m pip install --upgrade pip
	python -m pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check src tests scripts

train:
	python scripts/run_training.py

api:
	uvicorn churn_ml.api.main:app --reload

ui:
	streamlit run src/churn_ml/ui/app.py

mlflow:
	mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

docker-up:
	docker compose up --build

docker-down:
	docker compose down

drift:
	python -m churn_ml.monitoring.drift_detection
