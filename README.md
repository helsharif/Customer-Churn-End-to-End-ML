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

1. Exploratory data analysis, cleaning, and feature engineering
2. Interpretable logistic-regression baseline
3. XGBoost classification modeling
4. Cross-validated model selection and holdout evaluation
5. Hyperparameter tuning
6. Model evaluation and error analysis
7. Model explainability with SHAP
8. FastAPI deployment path
9. Monitoring and data quality checks
10. CI/CD automation

## Repository Structure

```text
Customer-Churn-End-to-End-ML/
├── README.md
├── .gitignore
├── churn_ml_env001.yml
├── requirements.txt
├── pyproject.toml
├── app/                       # Application entry points and serving assets
├── artifacts/                 # Saved models, metrics, and generated outputs
├── config/                    # Project and experiment configuration
├── data/
│   ├── raw/                   # Source data and its original data dictionary
│   ├── interim/               # Optional transient transformation outputs
│   └── processed/             # Model-ready outputs created by Notebook 01
│       ├── prediction_df_logistic_regression.csv
│       ├── prediction_df_logistic_regression_data_dictionary.md
│       ├── prediction_df_xgboost.csv
│       └── prediction_df_xgboost_data_dictionary.md
├── docker/                    # Containerization resources
├── great_expectations/        # Data-quality configuration and expectation suites
├── mlruns/                    # Local MLflow experiment tracking data
├── notebooks/
│   ├── 01_eda_data_cleaning.ipynb
│   ├── 02_baseline_modeling_logistic_regression.ipynb
│   ├── 03_xgboost_modeling.ipynb
│   └── 04_model_selection.ipynb
├── scripts/                   # Runnable project automation scripts
├── src/churn_ml/
│   ├── config.py
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── explainability/
│   └── api/
├── tests/
└── .github/workflows/
```

## Modeling Notebook Flow

Run the notebooks in numerical order. Notebook 01 produces the model-ready CSV and its field-level data dictionary. Notebooks 02 and 03 use the same stratified 80/20 holdout split for directly comparable logistic-regression and XGBoost results. Notebook 04 uses five-fold stratified cross-validation on the training split to select a candidate by PR AUC, then evaluates that selected candidate once on the untouched holdout set. Each modeling notebook defaults its prediction threshold to the training-set churn rate; set `CHURN_THRESHOLD` to a value from 0 to 1 to override it.

Notebook 01 creates two model-ready datasets, each with 7,043 records and `Churn Value` as the final column. The logistic-regression dataset uses the multicollinearity-pruned predictors and excludes `Monthly Charges`. The XGBoost dataset retains the complete encoded predictor set because tree-based models are not sensitive to linear multicollinearity in the same way. Yes/no features are encoded as 0/1, categorical features with 3–5 levels are one-hot encoded, and the XGBoost dataset retains `Total_Charges_Missing` to record a blank source `Total Charges` value.

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

The EDA/data-cleaning notebook produces a documented processed dataset. Dedicated notebooks now cover logistic-regression baseline modeling, Optuna-tuned XGBoost modeling, and cross-validated model selection. The latest experiment results favor the tuned XGBoost model for the current retention-targeting scenario, while logistic regression remains a strong, practical alternative when transparency and implementation simplicity are the priority.

## Current Model Selection

The current recommendation is to select the Optuna-tuned XGBoost model for retention targeting. In the holdout scenario that targets the top 100 customers by expected value, XGBoost produced **$24,647.51** in expected net value, compared with **$23,555.11** for logistic regression—an improvement of **$1,092.40** (about **4.6%**). The two models selected 86 of the same 100 customers, so the incremental value is concentrated in a relatively small portion of the target list.

This is a pragmatic rather than absolute choice. Logistic regression performed well and remains the preferred option where clearer explanations, simpler implementation, and easier auditing outweigh the incremental expected-value gain. The expected-value comparison depends on the assumed retention uplift and offer-acceptance rate; validate those assumptions with a controlled campaign before production deployment.

## Roadmap

- Complete EDA notebook with target distribution and feature summaries
- Validate schema and missing-value assumptions
- Build cleaned modeling dataset
- Train and evaluate baseline logistic regression
- Add model comparison experiments
- Validate the selected model and retention assumptions with a controlled campaign
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
