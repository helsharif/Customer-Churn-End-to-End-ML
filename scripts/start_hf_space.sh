#!/usr/bin/env sh
set -eu

export CHURN_API_BASE_URL="${CHURN_API_BASE_URL:-http://127.0.0.1:8000}"
export CHURN_MODEL_PATH="${CHURN_MODEL_PATH:-/app/artifacts/models/production_model.joblib}"
export PYTHONPATH="${PYTHONPATH:-/app/src}"

if [ ! -f "$CHURN_MODEL_PATH" ]; then
  echo "Production model not found at $CHURN_MODEL_PATH. Running training pipeline..."
  python scripts/run_training.py
fi

echo "Starting FastAPI backend on http://127.0.0.1:8000"
uvicorn churn_ml.api.main:app --host 127.0.0.1 --port 8000 &

echo "Waiting for FastAPI health check..."
python - <<'PY'
import time
import urllib.request

deadline = time.time() + 60
while time.time() < deadline:
    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2) as response:
            if response.status == 200:
                print("FastAPI health check passed.")
                raise SystemExit(0)
    except Exception:
        time.sleep(1)

raise SystemExit("FastAPI did not become healthy within 60 seconds.")
PY

echo "Starting Streamlit frontend on http://0.0.0.0:${PORT:-7860}"
python -m streamlit run src/churn_ml/ui/app.py \
  --server.address 0.0.0.0 \
  --server.port "${PORT:-7860}" \
  --server.headless true
