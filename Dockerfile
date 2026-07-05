FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md requirements.txt ./
COPY src ./src
COPY scripts ./scripts

RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir -e .

EXPOSE 8000

CMD ["uvicorn", "churn_ml.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
