"""Streamlit user interface for churn prediction."""

import os
import random
from typing import Any

import requests
import streamlit as st

API_BASE_URL = os.getenv("CHURN_API_BASE_URL", "http://127.0.0.1:8000")


def main() -> None:
    """Render the churn prediction form."""
    st.set_page_config(page_title="Telco Churn Predictor", layout="centered")
    st.title("Telco Churn Predictor")

    prediction_tab, about_tab = st.tabs(["Prediction", "About"])

    with prediction_tab:
        _render_prediction_tab()

    with about_tab:
        _render_about_tab()


def _render_prediction_tab() -> None:
    profile_options = ["High-risk month-to-month", "Stable long-tenure", "Random Customer"]
    profile = st.selectbox("Example profile", profile_options)
    defaults = _profile_defaults(profile)
    widget_suffix = _widget_suffix(profile)

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            gender = _select("Gender", ["Female", "Male"], defaults["gender"], widget_suffix)
            senior_citizen = _select("Senior citizen", ["No", "Yes"], defaults["senior_citizen"], widget_suffix)
            partner = _select("Partner", ["No", "Yes"], defaults["partner"], widget_suffix)
            dependents = _select("Dependents", ["No", "Yes"], defaults["dependents"], widget_suffix)
            tenure_months = st.number_input(
                "Tenure months",
                min_value=0,
                max_value=100,
                value=int(defaults["tenure_months"]),
                key=f"tenure_months_{widget_suffix}",
            )
            contract = st.selectbox(
                "Contract",
                ["Month-to-month", "One year", "Two year"],
                index=["Month-to-month", "One year", "Two year"].index(defaults["contract"]),
                key=f"contract_{widget_suffix}",
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
                key=f"payment_method_{widget_suffix}",
            )
        with col2:
            phone_service = _select("Phone service", ["No", "Yes"], defaults["phone_service"], widget_suffix)
            multiple_lines = _select(
                "Multiple lines",
                ["No", "No phone service", "Yes"],
                defaults["multiple_lines"],
                widget_suffix,
            )
            internet_service = _select(
                "Internet service",
                ["DSL", "Fiber optic", "No"],
                defaults["internet_service"],
                widget_suffix,
            )
            online_security = _addon_select("Online security", defaults["online_security"], widget_suffix)
            online_backup = _addon_select("Online backup", defaults["online_backup"], widget_suffix)
            device_protection = _addon_select("Device protection", defaults["device_protection"], widget_suffix)
            tech_support = _addon_select("Tech support", defaults["tech_support"], widget_suffix)
            streaming_tv = _addon_select("Streaming TV", defaults["streaming_tv"], widget_suffix)
            streaming_movies = _addon_select("Streaming movies", defaults["streaming_movies"], widget_suffix)

        paperless_billing = _select(
            "Paperless billing",
            ["No", "Yes"],
            defaults["paperless_billing"],
            widget_suffix,
        )
        monthly_charges = st.number_input(
            "Monthly charges",
            min_value=0.0,
            value=float(defaults["monthly_charges"]),
            step=5.0,
            key=f"monthly_charges_{widget_suffix}",
        )
        total_charges = st.number_input(
            "Total charges",
            min_value=0.0,
            value=float(defaults["total_charges"]),
            step=50.0,
            key=f"total_charges_{widget_suffix}",
        )
        cltv = st.number_input(
            "Customer lifetime value",
            min_value=0.0,
            value=float(defaults["cltv"]),
            step=100.0,
            key=f"cltv_{widget_suffix}",
        )
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


def _render_about_tab() -> None:
    st.subheader("About this app")
    st.write(
        "This app demonstrates an end-to-end Telco Customer Churn machine learning workflow. "
        "It predicts whether a telecom customer is likely to churn and returns the churn "
        "probability from the selected production-style model."
    )
    st.write(
        "The project uses a reproducible pipeline for data validation, leakage-safe feature "
        "engineering, Logistic Regression versus XGBoost model comparison, MLflow tracking, "
        "FastAPI serving, Streamlit interaction, and local monitoring."
    )
    st.write(
        "To use it, choose an example profile or generate a random valid customer, adjust any "
        "fields you want to test, then submit the form. The UI sends the profile to the FastAPI "
        "prediction endpoint and displays the churn decision, probability, threshold, and risk "
        "interpretation."
    )


def _select(label: str, options: list[str], default: Any, widget_suffix: str) -> str:
    return st.selectbox(
        label,
        options,
        index=options.index(str(default)),
        key=f"{_key_label(label)}_{widget_suffix}",
    )


def _addon_select(label: str, default: str, widget_suffix: str) -> str:
    options = ["No", "No internet service", "Yes"]
    return _select(label, options, default, widget_suffix)


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


def _profile_defaults(profile: str) -> dict[str, object]:
    if profile == "Random Customer":
        if st.session_state.get("last_profile") != profile or "random_profile" not in st.session_state:
            st.session_state.random_profile = _random_customer()
        st.session_state.last_profile = profile
        return st.session_state.random_profile
    st.session_state.last_profile = profile
    return _defaults(profile)


def _widget_suffix(profile: str) -> str:
    if profile == "Random Customer":
        random_profile = st.session_state.get("random_profile", {})
        return f"random_{hash(tuple(sorted(random_profile.items())))}"
    return _key_label(profile)


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


def _random_customer() -> dict[str, object]:
    tenure_months = random.randint(0, 72)
    monthly_charges = round(random.uniform(20.0, 120.0), 2)
    total_charges = round(monthly_charges * max(tenure_months, 1) * random.uniform(0.75, 1.15), 2)
    internet_service = random.choice(["DSL", "Fiber optic", "No"])
    phone_service = random.choice(["No", "Yes"])

    if internet_service == "No":
        addon_default = "No internet service"
        online_security = online_backup = device_protection = tech_support = addon_default
        streaming_tv = streaming_movies = addon_default
    else:
        online_security = random.choice(["No", "Yes"])
        online_backup = random.choice(["No", "Yes"])
        device_protection = random.choice(["No", "Yes"])
        tech_support = random.choice(["No", "Yes"])
        streaming_tv = random.choice(["No", "Yes"])
        streaming_movies = random.choice(["No", "Yes"])

    multiple_lines = "No phone service" if phone_service == "No" else random.choice(["No", "Yes"])

    return {
        "gender": random.choice(["Female", "Male"]),
        "senior_citizen": random.choice(["No", "Yes"]),
        "partner": random.choice(["No", "Yes"]),
        "dependents": random.choice(["No", "Yes"]),
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
        "contract": random.choice(["Month-to-month", "One year", "Two year"]),
        "paperless_billing": random.choice(["No", "Yes"]),
        "payment_method": random.choice(
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ]
        ),
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "cltv": round(random.uniform(2000.0, 6500.0), 2),
    }


def _key_label(label: str) -> str:
    return label.lower().replace(" ", "_").replace("-", "_")


if __name__ == "__main__":
    main()
