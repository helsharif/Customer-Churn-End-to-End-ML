"""Streamlit user interface for churn prediction."""

import os

import requests
import streamlit as st

API_BASE_URL = os.getenv("CHURN_API_BASE_URL", "http://127.0.0.1:8000")


def main() -> None:
    """Render the churn prediction form."""
    st.set_page_config(page_title="Telco Churn Predictor", layout="centered")
    st.title("Telco Churn Predictor")

    profile = st.selectbox("Example profile", ["High-risk month-to-month", "Stable long-tenure"])
    defaults = _defaults(profile)

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Gender", ["Female", "Male"], index=["Female", "Male"].index(defaults["gender"]))
            senior_citizen = st.selectbox("Senior citizen", ["No", "Yes"], index=["No", "Yes"].index(defaults["senior_citizen"]))
            partner = st.selectbox("Partner", ["No", "Yes"], index=["No", "Yes"].index(defaults["partner"]))
            dependents = st.selectbox("Dependents", ["No", "Yes"], index=["No", "Yes"].index(defaults["dependents"]))
            tenure_months = st.number_input("Tenure months", min_value=0, max_value=100, value=defaults["tenure_months"])
            contract = st.selectbox(
                "Contract",
                ["Month-to-month", "One year", "Two year"],
                index=["Month-to-month", "One year", "Two year"].index(defaults["contract"]),
            )
            payment_method = st.selectbox(
                "Payment method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                index=[
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ].index(defaults["payment_method"]),
            )
        with col2:
            phone_service = st.selectbox("Phone service", ["No", "Yes"], index=["No", "Yes"].index(defaults["phone_service"]))
            multiple_lines = st.selectbox("Multiple lines", ["No", "No phone service", "Yes"], index=["No", "No phone service", "Yes"].index(defaults["multiple_lines"]))
            internet_service = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"], index=["DSL", "Fiber optic", "No"].index(defaults["internet_service"]))
            online_security = _addon_select("Online security", defaults["online_security"])
            online_backup = _addon_select("Online backup", defaults["online_backup"])
            device_protection = _addon_select("Device protection", defaults["device_protection"])
            tech_support = _addon_select("Tech support", defaults["tech_support"])
            streaming_tv = _addon_select("Streaming TV", defaults["streaming_tv"])
            streaming_movies = _addon_select("Streaming movies", defaults["streaming_movies"])

        paperless_billing = st.selectbox("Paperless billing", ["No", "Yes"], index=["No", "Yes"].index(defaults["paperless_billing"]))
        monthly_charges = st.number_input("Monthly charges", min_value=0.0, value=defaults["monthly_charges"], step=5.0)
        total_charges = st.number_input("Total charges", min_value=0.0, value=defaults["total_charges"], step=50.0)
        cltv = st.number_input("Customer lifetime value", min_value=0.0, value=defaults["cltv"], step=100.0)
        submitted = st.form_submit_button("Predict churn")

    if submitted:
        payload = {
            "gender": gender,
            "senior_citizen": senior_citizen,
            "partner": partner,
            "dependents": dependents,
            "tenure_months": tenure_months,
            "phone_service": phone_service,
            "multiple_lines": multiple_lines,
            "internet_service": internet_service,
            "online_security": online_security,
            "online_backup": online_backup,
            "device_protection": device_protection,
            "tech_support": tech_support,
            "streaming_tv": streaming_tv,
            "streaming_movies": streaming_movies,
            "contract": contract,
            "paperless_billing": paperless_billing,
            "payment_method": payment_method,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "cltv": cltv,
        }
        _submit_prediction(payload)


def _addon_select(label: str, default: str) -> str:
    options = ["No", "No internet service", "Yes"]
    return st.selectbox(label, options, index=options.index(default))


def _submit_prediction(payload: dict[str, object]) -> None:
    try:
        response = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:
        st.error(f"Could not reach the churn API at {API_BASE_URL}: {exc}")
        return

    result = response.json()
    probability = float(result["churn_probability"])
    st.metric("Prediction", result["prediction"], f"{probability:.1%} churn probability")
    st.progress(min(max(probability, 0.0), 1.0))
    st.caption(f"Decision threshold: {float(result['threshold']):.1%}. Model: {result['model_name']}.")
    if probability >= 0.7:
        st.warning("High churn risk. Prioritize retention outreach.")
    elif probability >= float(result["threshold"]):
        st.info("Elevated churn risk. Review for targeted retention action.")
    else:
        st.success("Lower churn risk under the current decision threshold.")


def _defaults(profile: str) -> dict[str, object]:
    if profile == "Stable long-tenure":
        return {
            "gender": "Female",
            "senior_citizen": "No",
            "partner": "Yes",
            "dependents": "Yes",
            "tenure_months": 48,
            "phone_service": "Yes",
            "multiple_lines": "Yes",
            "internet_service": "DSL",
            "online_security": "Yes",
            "online_backup": "Yes",
            "device_protection": "Yes",
            "tech_support": "Yes",
            "streaming_tv": "Yes",
            "streaming_movies": "Yes",
            "contract": "Two year",
            "paperless_billing": "No",
            "payment_method": "Credit card (automatic)",
            "monthly_charges": 85.0,
            "total_charges": 4080.0,
            "cltv": 4800.0,
        }
    return {
        "gender": "Female",
        "senior_citizen": "No",
        "partner": "No",
        "dependents": "No",
        "tenure_months": 6,
        "phone_service": "Yes",
        "multiple_lines": "No",
        "internet_service": "Fiber optic",
        "online_security": "No",
        "online_backup": "No",
        "device_protection": "No",
        "tech_support": "No",
        "streaming_tv": "No",
        "streaming_movies": "No",
        "contract": "Month-to-month",
        "paperless_billing": "Yes",
        "payment_method": "Electronic check",
        "monthly_charges": 75.0,
        "total_charges": 450.0,
        "cltv": 2700.0,
    }


if __name__ == "__main__":
    main()
