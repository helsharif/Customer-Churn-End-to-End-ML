# Telco Customer Churn — Raw Data Dictionary

**Dataset:** `Telco_customer_churn.csv`  
**Grain:** One customer record (7,043 customers)  
**Scope:** A fictional telco company's California home-phone and internet customers in Q3.  
**Source:** [IBM Telco Customer Churn data description](https://community.ibm.com/community/user/businessanalytics/blogs/steven-macko/2019/07/11/telco-customer-churn-1113) (distributed in this project from the [Kaggle dataset](https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset)).

The definitions below follow the published data description. “Raw representation” describes how the field is stored when the CSV is read without transformations; it is not necessarily the appropriate analytical type. The documented source calls the monthly-billing field **Monthly Charge**, while this CSV names it `Monthly Charges`.

## Customer identity and geography

| Column | Raw representation | Definition | Notes / expected values |
|---|---|---|---|
| `CustomerID` | text | Unique identifier for each customer. | Identifier; one value per customer. |
| `Count` | integer | Reporting/dashboarding count used to total customers in a filtered set. | Constant value of `1` in this file. |
| `Country` | text | Country of the customer's primary residence. | `United States` in this file. |
| `State` | text | State of the customer's primary residence. | `California` in this file. |
| `City` | text | City of the customer's primary residence. | Geographic categorical field. |
| `Zip Code` | integer | ZIP code of the customer's primary residence. | Geographic code, not a continuous measurement. |
| `Lat Long` | text | Combined latitude and longitude of the customer's primary residence. | Redundant with `Latitude` and `Longitude`. |
| `Latitude` | decimal | Latitude of the customer's primary residence. | Geographic coordinate. |
| `Longitude` | decimal | Longitude of the customer's primary residence. | Geographic coordinate. |

## Customer profile and services

| Column | Raw representation | Definition | Notes / expected values |
|---|---|---|---|
| `Gender` | text | Customer gender. | `Male`, `Female`. |
| `Senior Citizen` | text | Whether the customer is age 65 or older. | `Yes`, `No`. |
| `Partner` | text | Whether the customer has a partner. | `Yes`, `No`. |
| `Dependents` | text | Whether the customer lives with dependents, such as children, parents, or grandparents. | `Yes`, `No`. |
| `Tenure Months` | integer | Total months the customer has been with the company as of the end of the quarter. | Non-negative duration. |
| `Phone Service` | text | Whether the customer subscribes to the company's home-phone service. | `Yes`, `No`. |
| `Multiple Lines` | text | Whether the customer subscribes to multiple telephone lines. | `Yes`, `No`; the raw file also uses `No phone service`. |
| `Internet Service` | text | Whether and how the customer subscribes to internet service. | Published description: `No`, `DSL`, `Fiber Optic`, `Cable`; raw file contains `No`, `DSL`, and `Fiber optic`. |
| `Online Security` | text | Whether the customer subscribes to the company's additional online-security service. | `Yes`, `No`; raw file also uses `No internet service`. |
| `Online Backup` | text | Whether the customer subscribes to the company's additional online-backup service. | `Yes`, `No`; raw file also uses `No internet service`. |
| `Device Protection` | text | Whether the customer subscribes to an additional protection plan for internet equipment. | `Yes`, `No`; raw file also uses `No internet service`. |
| `Tech Support` | text | Whether the customer subscribes to an additional technical-support plan with reduced wait times. | `Yes`, `No`; raw file also uses `No internet service`. |
| `Streaming TV` | text | Whether the customer uses internet service to stream television programming from a third-party provider. | `Yes`, `No`; raw file also uses `No internet service`. The company does not charge an additional fee for the service. |
| `Streaming Movies` | text | Whether the customer uses internet service to stream movies from a third-party provider. | `Yes`, `No`; raw file also uses `No internet service`. The company does not charge an additional fee for the service. |

## Contract and billing

| Column | Raw representation | Definition | Notes / expected values |
|---|---|---|---|
| `Contract` | text | Customer's current contract type. | `Month-to-month`, `One year`, `Two year`. |
| `Paperless Billing` | text | Whether the customer selected paperless billing. | `Yes`, `No`. |
| `Payment Method` | text | How the customer pays their bill. | Published examples include bank withdrawal, credit card, and mailed check. The raw file contains bank transfer (automatic), credit card (automatic), electronic check, and mailed check. |
| `Monthly Charges` | decimal | Customer's current total monthly charge for all company services. | Named `Monthly Charge` in the published description. |
| `Total Charges` | text (numeric amount) | Customer's total charges through the end of the specified quarter. | Convert to a numeric type during cleaning after validating all values. |

## Churn and value fields

| Column | Raw representation | Definition | Notes / expected values |
|---|---|---|---|
| `Churn Label` | text | Whether the customer left the company during the quarter. | `Yes` = left; `No` = remained. Directly corresponds to `Churn Value`. |
| `Churn Value` | integer | Numeric form of the churn outcome for the quarter. | `1` = left; `0` = remained. Directly corresponds to `Churn Label`. |
| `Churn Score` | integer | IBM SPSS Modeler predictive score based on multiple known churn factors. | Range 0–100; larger values indicate greater likelihood of churn. This is model-generated and outcome-related. |
| `CLTV` | integer | Predicted customer lifetime value, calculated using corporate formulas and existing data. | Larger values indicate a more valuable customer; high-value customers should be monitored for churn. |
| `Churn Reason` | text; nullable | Customer's specific reason for leaving the company. | Populated for churned customers; null for customers who remained. Directly related to the published `Churn Category` field, which is not present in this CSV. |

## Modeling cautions

- `CustomerID` is an identifier and `Count` is constant, so neither is a useful predictor.
- `Churn Label` duplicates the target encoded in `Churn Value`; retain only one target representation for a modeling workflow.
- `Churn Reason` is known only after churn and must be excluded from predictive features. `Churn Score` is also an existing predictive output and should be treated as potential leakage unless the use case explicitly permits it.
- `Lat Long`, `Latitude`, and `Longitude` encode the same location information, while `Zip Code` is a geographic category despite being stored numerically.
