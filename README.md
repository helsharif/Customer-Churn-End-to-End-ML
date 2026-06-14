# Customer Churn End-to-End ML

A professional end-to-end machine learning project for predicting customer churn in a fictional telecommunications company. This repository is being built as a complete ML workflow, moving from exploratory analysis and baseline modeling toward deployment, monitoring, and CI/CD-ready MLOps practices.

## Business Problem

Customer churn reduces recurring revenue and increases acquisition costs. For a subscription-based telecom business, identifying customers who are likely to leave can help retention teams prioritize outreach, understand churn drivers, and design targeted interventions before customers cancel service.

## ML Objective

The primary machine learning task is binary classification: predict whether a customer will churn during the quarter. The preferred modeling target is `Churn Value`, where `1` indicates churn and `0` indicates retention.

## Dataset

The raw dataset is the IBM Telco Customer Churn dataset available on Kaggle:

[Telco Customer Churn IBM Dataset](https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset)

The dataset contains 7,043 customer records and 33 variables describing demographics, geography, subscribed services, billing information, tenure, churn outcome, and churn-related post-outcome fields. The raw CSV is expected locally at:

```text
data/raw/Telco_customer_churn.csv
```

Raw data files are intentionally excluded from version control.

## Target and Leakage Notes

`Churn Value` is the preferred target column for modeling. `Churn Label` is directly equivalent to the target and should not be used as a feature.

The following fields are treated as non-feature or leakage columns by default:

- `CustomerID`
- `Count`
- `Lat Long`
- `Latitude`
- `Longitude`
- `Churn Label`
- `Churn Value`
- `Churn Score`
- `Churn Reason`
- `Churn Category`, if present
- Other direct churn explanation or post-outcome fields, if introduced later

`Churn Score`, `Churn Reason`, and churn-category fields may contain information generated after or directly because of the churn event. They should only be used in clearly separated leakage analysis or business interpretation sections, not in model training features.

Latitude and longitude fields are also excluded from the default feature set. They may be useful for geographic analysis, but the first modeling pass will avoid raw coordinates to reduce the risk of location memorization and overly local patterns.

## Planned Workflow

1. Exploratory data analysis
2. Data validation, ingestion, and cleaning
3. Feature engineering and preprocessing
4. Interpretable baseline modeling
5. Model selection and training
6. Hyperparameter tuning
7. Model evaluation and error analysis
8. Model explainability with SHAP
9. FastAPI deployment path
10. Monitoring and data quality checks
11. CI/CD automation

## Repository Structure

```text
Customer-Churn-End-to-End-ML/
├── README.md
├── .gitignore
├── churn_ml_env001.yml
├── requirements.txt
├── pyproject.toml
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── external/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_baseline_modeling.ipynb
│   └── 04_model_selection.ipynb
├── src/churn_ml/
│   ├── config.py
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── explainability/
│   └── api/
├── reports/
├── models/
├── tests/
└── .github/workflows/
```

## Planned Modeling Approach

Early modeling will prioritize interpretable baselines such as logistic regression with clear preprocessing. Later iterations will compare stronger models such as tree-based ensembles and GPU-enabled XGBoost, then use Optuna for tuning once the validation strategy and leakage controls are stable.

## Deployment and Monitoring Plan

The repository includes a minimal FastAPI application as the starting point for service deployment. Future phases will add a production-style inference endpoint, serialized model artifacts, input validation, data quality checks with Great Expectations, monitoring metrics, and CI/CD automation for testing and deployment readiness.

## Technologies

- Python 3.12
- pandas, NumPy, SciPy
- scikit-learn
- XGBoost
- Optuna
- SHAP
- Plotly, Matplotlib, Kaleido
- FastAPI, Uvicorn, Pydantic
- Great Expectations
- pytest
- GitHub Actions

## Current Status

Phase 1 is a repository and environment scaffold. The project currently includes starter documentation, package structure, a minimal data loading and cleaning layer, a placeholder baseline training pipeline, evaluation helpers, a FastAPI health check, and starter CI.

No final model has been trained yet.

## Roadmap

- Complete EDA notebook with target distribution and feature summaries
- Validate schema and missing-value assumptions
- Build cleaned modeling dataset
- Train and evaluate baseline logistic regression
- Add model comparison experiments
- Tune selected models with Optuna
- Add SHAP explanations and business-facing interpretation
- Package inference pipeline
- Add FastAPI prediction endpoint
- Add Great Expectations validation suite
- Add monitoring and CI/CD improvements

## Setup

Create and activate the conda environment:

```bash
conda env create -f churn_ml_env001.yml
conda activate churn_ml_env001
python -m ipykernel install --user --name churn_ml_env001 --display-name "Python (churn_ml_env001)"
```

For pip-based tooling or CI, install the package with development dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Run tests:

```bash
pytest
```

Start the FastAPI health check locally:

```bash
uvicorn churn_ml.api.main:app --reload
```

Then open:

```text
http://127.0.0.1:8000/health
```

## XGBoost and GPU Note

The default conda environment installs standard CPU-compatible `xgboost` from conda-forge so the Windows setup is reliable. GPU-enabled XGBoost can be explored later as an optional optimization after the baseline workflow is stable.

If you want to experiment with CUDA XGBoost, first confirm that your NVIDIA driver, CUDA runtime, Python version, and the package wheel all match. Avoid making GPU XGBoost a required environment dependency unless the install path is proven on the target machine.
