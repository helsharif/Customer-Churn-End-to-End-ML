# XGBoost Processed Prediction Dataset Data Dictionary

- **File:** `prediction_df_xgboost.csv`
- **Rows:** 7,043
- **Columns:** 40 (39 predictors and 1 target)
- **Target (final column):** `Churn Value`

## Purpose and feature preparation

This dataset is the model-ready input for the XGBoost churn classifier. It retains the complete encoded predictor set, including correlated service and pricing fields, because tree-based XGBoost is not sensitive to linear multicollinearity in the way logistic regression is.

- Yes/no source features are binary encoded as `0`/`1`.
- Categorical source features with 3–5 levels are one-hot encoded.
- `Total Charges` is numeric; its 11 blank source values are stored as missing (`NaN`).
- `Total_Charges_Missing` is `1` when the original total-charge value was blank and `0` otherwise.
- `Churn Value` is the final column and is the binary target (`1` = churned).

| Processed column | Data type | Encoding | Description | Valid values | Missing values |
|---|---|---|---|---|---:|
| Partner | Int64 | Binary encoding | Whether the customer has a partner. | 0 = No; 1 = Yes | 0 |
| Dependents | Int64 | Binary encoding | Whether the customer has dependents. | 0 = No; 1 = Yes | 0 |
| Tenure Months | int64 | Numeric | Number of months the customer has been with the company. | Continuous numeric value | 0 |
| Phone Service | Int64 | Binary encoding | Whether the customer has phone service. | 0 = No; 1 = Yes | 0 |
| Paperless Billing | Int64 | Binary encoding | Whether the customer uses paperless billing. | 0 = No; 1 = Yes | 0 |
| Monthly Charges | float64 | Numeric | Customer's current monthly service charge. | Continuous numeric value | 0 |
| Total Charges | float64 | Numeric | Customer's cumulative service charges; blank source values are stored as missing. | Continuous numeric value | 11 |
| Total_Charges_Missing | int64 | Binary encoding | Whether Total Charges was blank or unavailable in the source data. | 0 = No; 1 = Yes | 0 |
| Multiple Lines_No | int64 | One-hot encoding | Whether Multiple Lines is 'No'. | 0 = category not present; 1 = category present | 0 |
| Multiple Lines_No phone service | int64 | One-hot encoding | Whether Multiple Lines is 'No phone service'. | 0 = category not present; 1 = category present | 0 |
| Multiple Lines_Yes | int64 | One-hot encoding | Whether Multiple Lines is 'Yes'. | 0 = category not present; 1 = category present | 0 |
| Internet Service_DSL | int64 | One-hot encoding | Whether Internet Service is 'DSL'. | 0 = category not present; 1 = category present | 0 |
| Internet Service_Fiber optic | int64 | One-hot encoding | Whether Internet Service is 'Fiber optic'. | 0 = category not present; 1 = category present | 0 |
| Internet Service_No | int64 | One-hot encoding | Whether Internet Service is 'No'. | 0 = category not present; 1 = category present | 0 |
| Online Security_No | int64 | One-hot encoding | Whether Online Security is 'No'. | 0 = category not present; 1 = category present | 0 |
| Online Security_No internet service | int64 | One-hot encoding | Whether Online Security is 'No internet service'. | 0 = category not present; 1 = category present | 0 |
| Online Security_Yes | int64 | One-hot encoding | Whether Online Security is 'Yes'. | 0 = category not present; 1 = category present | 0 |
| Online Backup_No | int64 | One-hot encoding | Whether Online Backup is 'No'. | 0 = category not present; 1 = category present | 0 |
| Online Backup_No internet service | int64 | One-hot encoding | Whether Online Backup is 'No internet service'. | 0 = category not present; 1 = category present | 0 |
| Online Backup_Yes | int64 | One-hot encoding | Whether Online Backup is 'Yes'. | 0 = category not present; 1 = category present | 0 |
| Device Protection_No | int64 | One-hot encoding | Whether Device Protection is 'No'. | 0 = category not present; 1 = category present | 0 |
| Device Protection_No internet service | int64 | One-hot encoding | Whether Device Protection is 'No internet service'. | 0 = category not present; 1 = category present | 0 |
| Device Protection_Yes | int64 | One-hot encoding | Whether Device Protection is 'Yes'. | 0 = category not present; 1 = category present | 0 |
| Tech Support_No | int64 | One-hot encoding | Whether Tech Support is 'No'. | 0 = category not present; 1 = category present | 0 |
| Tech Support_No internet service | int64 | One-hot encoding | Whether Tech Support is 'No internet service'. | 0 = category not present; 1 = category present | 0 |
| Tech Support_Yes | int64 | One-hot encoding | Whether Tech Support is 'Yes'. | 0 = category not present; 1 = category present | 0 |
| Streaming TV_No | int64 | One-hot encoding | Whether Streaming TV is 'No'. | 0 = category not present; 1 = category present | 0 |
| Streaming TV_No internet service | int64 | One-hot encoding | Whether Streaming TV is 'No internet service'. | 0 = category not present; 1 = category present | 0 |
| Streaming TV_Yes | int64 | One-hot encoding | Whether Streaming TV is 'Yes'. | 0 = category not present; 1 = category present | 0 |
| Streaming Movies_No | int64 | One-hot encoding | Whether Streaming Movies is 'No'. | 0 = category not present; 1 = category present | 0 |
| Streaming Movies_No internet service | int64 | One-hot encoding | Whether Streaming Movies is 'No internet service'. | 0 = category not present; 1 = category present | 0 |
| Streaming Movies_Yes | int64 | One-hot encoding | Whether Streaming Movies is 'Yes'. | 0 = category not present; 1 = category present | 0 |
| Contract_Month-to-month | int64 | One-hot encoding | Whether Contract is 'Month-to-month'. | 0 = category not present; 1 = category present | 0 |
| Contract_One year | int64 | One-hot encoding | Whether Contract is 'One year'. | 0 = category not present; 1 = category present | 0 |
| Contract_Two year | int64 | One-hot encoding | Whether Contract is 'Two year'. | 0 = category not present; 1 = category present | 0 |
| Payment Method_Bank transfer (automatic) | int64 | One-hot encoding | Whether Payment Method is 'Bank transfer (automatic)'. | 0 = category not present; 1 = category present | 0 |
| Payment Method_Credit card (automatic) | int64 | One-hot encoding | Whether Payment Method is 'Credit card (automatic)'. | 0 = category not present; 1 = category present | 0 |
| Payment Method_Electronic check | int64 | One-hot encoding | Whether Payment Method is 'Electronic check'. | 0 = category not present; 1 = category present | 0 |
| Payment Method_Mailed check | int64 | One-hot encoding | Whether Payment Method is 'Mailed check'. | 0 = category not present; 1 = category present | 0 |
| Churn Value | int64 | Target | Customer churn outcome. | 0 = did not churn; 1 = churned | 0 |
