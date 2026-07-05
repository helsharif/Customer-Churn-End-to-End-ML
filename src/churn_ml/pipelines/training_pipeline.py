"""End-to-end training pipeline for the Telco churn project."""

from __future__ import annotations

import json
import logging
from typing import Any

import joblib
import pandas as pd

from churn_ml.config import (
    FIGURES_DIR,
    METRICS_DIR,
    MIN_PRECISION_FOR_THRESHOLD,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    PRIMARY_METRIC,
    PRODUCTION_MODEL_PATH,
    RAW_DATA_PATH,
    STANDARDIZED_TARGET_COLUMN,
    TRAINING_BASELINE_PATH,
    ensure_project_directories,
)
from churn_ml.data.clean_data import clean_churn_data
from churn_ml.data.load_data import load_raw_data
from churn_ml.data.preprocess import get_feature_columns, split_train_validation_test, split_xy
from churn_ml.data.validate_data import validate_raw_data
from churn_ml.features.build_features import FEATURE_RATIONALE, build_features
from churn_ml.models.evaluate_model import save_evaluation_plots, threshold_metrics, tune_threshold
from churn_ml.models.select_model import results_to_frame, select_best_model
from churn_ml.models.train_model import build_candidate_models, fit_model, predict_proba
from churn_ml.utils.logging import configure_logging

LOGGER = logging.getLogger(__name__)


def run_training_pipeline() -> dict[str, Any]:
    """Run validation, feature engineering, training, selection, and final evaluation."""
    configure_logging()
    ensure_project_directories()

    raw_df = load_raw_data(RAW_DATA_PATH)
    validation_report = validate_raw_data(raw_df, strict=True)
    for warning in validation_report.warnings:
        LOGGER.warning(warning)
    validation_report.raise_for_errors()

    clean_df = clean_churn_data(raw_df, keep_target=True)
    feature_df = build_features(clean_df)
    train_df, validation_df, test_df = split_train_validation_test(feature_df)
    X_train, y_train = split_xy(train_df)
    X_validation, y_validation = split_xy(validation_df)
    X_test, y_test = split_xy(test_df)

    decision_threshold = float(y_train.mean())
    LOGGER.info("Notebook-compatible default threshold: %.4f", decision_threshold)

    candidates = build_candidate_models(X_train)
    validation_results: dict[str, dict[str, float]] = {}
    trained_models = {}
    run_ids: dict[str, str] = {}

    mlflow_module = _configure_mlflow()
    for model_name, model in candidates.items():
        LOGGER.info("Training candidate model: %s", model_name)
        run_id = "local"
        if mlflow_module is not None:
            with mlflow_module.start_run(run_name=model_name) as run:
                run_id = run.info.run_id
                trained_model = fit_model(model, X_train, y_train)
                y_validation_proba = predict_proba(trained_model, X_validation)
                metrics = threshold_metrics(y_validation, y_validation_proba, decision_threshold)
                mlflow_module.log_params({"model_name": model_name, "threshold_policy": "training_churn_rate"})
                mlflow_module.log_metrics(metrics)
        else:
            trained_model = fit_model(model, X_train, y_train)
            y_validation_proba = predict_proba(trained_model, X_validation)
            metrics = threshold_metrics(y_validation, y_validation_proba, decision_threshold)

        validation_results[model_name] = metrics
        trained_models[model_name] = trained_model
        run_ids[model_name] = run_id

    best_model_name = select_best_model(validation_results, primary_metric=PRIMARY_METRIC)
    best_model = trained_models[best_model_name]
    validation_proba = predict_proba(best_model, X_validation)
    tuned_threshold = tune_threshold(
        y_validation,
        validation_proba,
        min_precision=MIN_PRECISION_FOR_THRESHOLD,
    )
    test_proba = predict_proba(best_model, X_test)
    test_metrics = threshold_metrics(y_test, test_proba, tuned_threshold)

    comparison_df = results_to_frame(validation_results)
    comparison_path = METRICS_DIR / "model_comparison.csv"
    test_report_path = METRICS_DIR / "final_evaluation.json"
    comparison_df.to_csv(comparison_path, index=False)
    save_evaluation_plots(y_test, test_proba, tuned_threshold, FIGURES_DIR, best_model_name)

    final_report = {
        "selected_model": best_model_name,
        "primary_metric": PRIMARY_METRIC,
        "default_threshold_policy": "training_churn_rate",
        "default_threshold": decision_threshold,
        "selected_threshold_policy": f"validation_f1_with_precision_at_least_{MIN_PRECISION_FOR_THRESHOLD}",
        "selected_threshold": tuned_threshold,
        "validation_metrics": validation_results,
        "test_metrics": test_metrics,
        "feature_rationale": FEATURE_RATIONALE,
    }
    test_report_path.write_text(json.dumps(final_report, indent=2), encoding="utf-8")

    bundle = {
        "model": best_model,
        "model_name": best_model_name,
        "threshold": tuned_threshold,
        "run_id": run_ids.get(best_model_name, "local"),
        "feature_columns": get_feature_columns(feature_df),
        "target_column": STANDARDIZED_TARGET_COLUMN,
        "primary_metric": PRIMARY_METRIC,
    }
    joblib.dump(bundle, PRODUCTION_MODEL_PATH)
    X_train.assign(**{STANDARDIZED_TARGET_COLUMN: y_train}).to_parquet(TRAINING_BASELINE_PATH, index=False)

    LOGGER.info("Selected model: %s", best_model_name)
    LOGGER.info("Saved production model bundle to %s", PRODUCTION_MODEL_PATH)
    return final_report


def _configure_mlflow():
    """Configure MLflow if available; return None when unavailable."""
    try:
        import mlflow
    except Exception as exc:
        LOGGER.warning("MLflow is unavailable; continuing without tracking: %s", exc)
        return None

    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    except Exception as exc:
        LOGGER.warning("MLflow setup failed; continuing without tracking: %s", exc)
        return None
    return mlflow


if __name__ == "__main__":
    run_training_pipeline()
