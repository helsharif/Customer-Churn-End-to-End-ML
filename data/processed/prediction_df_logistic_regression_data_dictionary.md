# Logistic Regression Processed Prediction Dataset Data Dictionary

- **File:** `prediction_df_logistic_regression.csv`
- **Rows:** 7,043
- **Columns:** 27 (26 predictors and 1 target)
- **Target (final column):** `Churn Value`

## Purpose and feature preparation

This dataset is the model-ready input for the logistic-regression churn classifier. It uses a multicollinearity-pruned feature set to make linear-model coefficients more stable and interpretable.

- Yes/no source features are binary encoded as `0`/`1`.
- Remaining categorical features are one-hot encoded.
- `Streaming Service` is `1` when a customer has streaming TV, streaming movies, or both.
- `Monthly Charges`, `Total Charges`, the total-charges missingness indicator, redundant streaming fields, and redundant no-service fields are excluded.
- `Churn Value` is the final column and is the binary target (`1` = churned).

| Processed column | Data type | Encoding | Description | Valid values | Missing values |
|---|---|---|---|---|---:|
| Partner | Int64 | Binary encoding | Whether the customer has a partner. | 0 = No; 1 = Yes | 0 |
| Dependents | Int64 | Binary encoding | Whether the customer has dependents. | 0 = No; 1 = Yes | 0 |
| Tenure Months | int64 | Numeric | Number of months the customer has been with the company. | Continuous numeric value | 0 |
| Phone Service | Int64 | Binary encoding | Whether the customer has phone service. | 0 = No; 1 = Yes | 0 |
| Paperless Billing | Int64 | Binary encoding | Whether the customer uses paperless billing. | 0 = No; 1 = Yes | 0 |
| Multiple Lines_No | int64 | One-hot encoding | Whether Multiple Lines is 'No'. | 0 = category not present; 1 = category present | 0 |
| Multiple Lines_Yes | int64 | One-hot encoding | Whether Multiple Lines is 'Yes'. | 0 = category not present; 1 = category present | 0 |
| Internet Service_DSL | int64 | One-hot encoding | Whether Internet Service is 'DSL'. | 0 = category not present; 1 = category present | 0 |
| Internet Service_Fiber optic | int64 | One-hot encoding | Whether Internet Service is 'Fiber optic'. | 0 = category not present; 1 = category present | 0 |
| Internet Service_No | int64 | One-hot encoding | Whether Internet Service is 'No'. | 0 = category not present; 1 = category present | 0 |
| Online Security_No | int64 | One-hot encoding | Whether Online Security is 'No'. | 0 = category not present; 1 = category present | 0 |
| Online Security_Yes | int64 | One-hot encoding | Whether Online Security is 'Yes'. | 0 = category not present; 1 = category present | 0 |
| Online Backup_No | int64 | One-hot encoding | Whether Online Backup is 'No'. | 0 = category not present; 1 = category present | 0 |
| Online Backup_Yes | int64 | One-hot encoding | Whether Online Backup is 'Yes'. | 0 = category not present; 1 = category present | 0 |
| Device Protection_No | int64 | One-hot encoding | Whether Device Protection is 'No'. | 0 = category not present; 1 = category present | 0 |
| Device Protection_Yes | int64 | One-hot encoding | Whether Device Protection is 'Yes'. | 0 = category not present; 1 = category present | 0 |
| Tech Support_No | int64 | One-hot encoding | Whether Tech Support is 'No'. | 0 = category not present; 1 = category present | 0 |
| Tech Support_Yes | int64 | One-hot encoding | Whether Tech Support is 'Yes'. | 0 = category not present; 1 = category present | 0 |
| Contract_Month-to-month | int64 | One-hot encoding | Whether Contract is 'Month-to-month'. | 0 = category not present; 1 = category present | 0 |
| Contract_One year | int64 | One-hot encoding | Whether Contract is 'One year'. | 0 = category not present; 1 = category present | 0 |
| Contract_Two year | int64 | One-hot encoding | Whether Contract is 'Two year'. | 0 = category not present; 1 = category present | 0 |
| Payment Method_Bank transfer (automatic) | int64 | One-hot encoding | Whether Payment Method is 'Bank transfer (automatic)'. | 0 = category not present; 1 = category present | 0 |
| Payment Method_Credit card (automatic) | int64 | One-hot encoding | Whether Payment Method is 'Credit card (automatic)'. | 0 = category not present; 1 = category present | 0 |
| Payment Method_Electronic check | int64 | One-hot encoding | Whether Payment Method is 'Electronic check'. | 0 = category not present; 1 = category present | 0 |
| Payment Method_Mailed check | int64 | One-hot encoding | Whether Payment Method is 'Mailed check'. | 0 = category not present; 1 = category present | 0 |
| Streaming Service | int64 | Binary encoding | Whether the customer subscribes to streaming TV, streaming movies, or both. | 0 = No; 1 = Yes | 0 |
| Churn Value | int64 | Target | Customer churn outcome. | 0 = did not churn; 1 = churned | 0 |
