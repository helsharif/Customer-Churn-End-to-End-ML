"""Streamlit user interface for churn prediction."""

import os
import random
from typing import Any

import requests
import streamlit as st

API_BASE_URL = os.getenv("CHURN_API_BASE_URL", "http://127.0.0.1:8000")
FIELD_LABELS = {
    "Gender": "♂️♀️ Gender",
    "Phone service": "📞 Phone service",
    "Senior citizen": "👴 Senior citizen",
    "Multiple lines": "📞📞 Multiple lines",
    "Partner": "🤝 Partner",
    "Internet service": "🌐 Internet service",
    "Dependents": "👨‍👩‍👧‍👦 Dependents",
    "Online security": "🔒 Online security",
    "Tenure months": "📅 Tenure months",
    "Online backup": "☁︎ Online backup",
    "Contract": "📝 Contract",
    "Device protection": "🛡️ Device protection",
    "Payment method": "💳 Payment method",
    "Tech support": "🛠️ Tech support",
    "Streaming TV": "📺 Streaming TV",
    "Streaming movies": "🎬 Streaming movies",
    "Paperless billing": "🧾 Paperless billing",
    "Monthly charges": "💵 Monthly charges",
    "Total charges": "💰 Total charges",
    "Customer lifetime value": "📈 Customer lifetime value",
}


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
    if "active_profile" not in st.session_state:
        st.session_state.active_profile = "High-risk month-to-month"

    st.caption("Choose a starting customer profile")
    profile_col1, profile_col2, profile_col3 = st.columns(3)
    with profile_col1:
        if st.button("High-risk month-to-month", use_container_width=True):
            st.session_state.active_profile = "High-risk month-to-month"
    with profile_col2:
        if st.button("Stable long-tenure", use_container_width=True):
            st.session_state.active_profile = "Stable long-tenure"
    with profile_col3:
        if st.button("Random Customer", use_container_width=True):
            st.session_state.active_profile = "Random Customer"
            st.session_state.random_profile = _random_customer()
            st.session_state.random_profile_id = st.session_state.get("random_profile_id", 0) + 1

    profile = st.session_state.active_profile
    defaults = _profile_defaults(profile)
    widget_suffix = _widget_suffix(profile)
    st.caption(f"Current profile: {profile}")

    with st.form("prediction_form"):
        row1_col1, row1_col2 = st.columns(2)
        with row1_col1:
            gender = _select("Gender", ["Female", "Male"], defaults["gender"], widget_suffix)
        with row1_col2:
            phone_service = _select("Phone service", ["No", "Yes"], defaults["phone_service"], widget_suffix)

        row2_col1, row2_col2 = st.columns(2)
        with row2_col1:
            senior_citizen = _select("Senior citizen", ["No", "Yes"], defaults["senior_citizen"], widget_suffix)
        with row2_col2:
            multiple_lines = _select(
                "Multiple lines",
                ["No", "No phone service", "Yes"],
                defaults["multiple_lines"],
                widget_suffix,
            )

        row3_col1, row3_col2 = st.columns(2)
        with row3_col1:
            partner = _select("Partner", ["No", "Yes"], defaults["partner"], widget_suffix)
        with row3_col2:
            internet_service = _select(
                "Internet service",
                ["DSL", "Fiber optic", "No"],
                defaults["internet_service"],
                widget_suffix,
            )

        row4_col1, row4_col2 = st.columns(2)
        with row4_col1:
            dependents = _select("Dependents", ["No", "Yes"], defaults["dependents"], widget_suffix)
        with row4_col2:
            online_security = _addon_select("Online security", defaults["online_security"], widget_suffix)

        row5_col1, row5_col2 = st.columns(2)
        with row5_col1:
            tenure_months = st.number_input(
                FIELD_LABELS["Tenure months"],
                min_value=0,
                max_value=100,
                value=int(defaults["tenure_months"]),
                key=f"tenure_months_{widget_suffix}",
            )
        with row5_col2:
            online_backup = _addon_select("Online backup", defaults["online_backup"], widget_suffix)

        row6_col1, row6_col2 = st.columns(2)
        with row6_col1:
            contract = st.selectbox(
                FIELD_LABELS["Contract"],
                ["Month-to-month", "One year", "Two year"],
                index=["Month-to-month", "One year", "Two year"].index(defaults["contract"]),
                key=f"contract_{widget_suffix}",
            )
        with row6_col2:
            device_protection = _addon_select("Device protection", defaults["device_protection"], widget_suffix)

        row7_col1, row7_col2 = st.columns(2)
        with row7_col1:
            payment_method = st.selectbox(
                FIELD_LABELS["Payment method"],
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
        with row7_col2:
            tech_support = _addon_select("Tech support", defaults["tech_support"], widget_suffix)

        row8_col1, row8_col2 = st.columns(2)
        with row8_col1:
            streaming_tv = _addon_select("Streaming TV", defaults["streaming_tv"], widget_suffix)
        with row8_col2:
            streaming_movies = _addon_select("Streaming movies", defaults["streaming_movies"], widget_suffix)

        row9_col1, row9_col2 = st.columns(2)
        with row9_col1:
            paperless_billing = _select(
                "Paperless billing",
                ["No", "Yes"],
                defaults["paperless_billing"],
                widget_suffix,
            )
        with row9_col2:
            monthly_charges = st.number_input(
                FIELD_LABELS["Monthly charges"],
                min_value=0.0,
                value=float(defaults["monthly_charges"]),
                step=5.0,
                key=f"monthly_charges_{widget_suffix}",
            )

        row10_col1, row10_col2 = st.columns(2)
        with row10_col1:
            total_charges = st.number_input(
                FIELD_LABELS["Total charges"],
                min_value=0.0,
                value=float(defaults["total_charges"]),
                step=50.0,
                key=f"total_charges_{widget_suffix}",
            )
        with row10_col2:
            cltv = st.number_input(
                FIELD_LABELS["Customer lifetime value"],
                min_value=0.0,
                value=float(defaults["cltv"]),
                step=100.0,
                key=f"cltv_{widget_suffix}",
            )
        submit_left, submit_center, submit_right = st.columns([1, 1, 1])
        with submit_center:
            submitted = st.form_submit_button("Predict churn", use_container_width=True)

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
    st.subheader("Dataset source")
    st.write(
        "The project uses the IBM Telco Customer Churn dataset, distributed for this project from "
        "the Kaggle Telco Customer Churn IBM dataset. The raw data contains 7,043 customer records "
        "for a fictional telecommunications company's California home-phone and internet customers "
        "during one quarter."
    )
    st.write(
        "Each row represents one customer. The prediction target is whether the customer churned "
        "during the quarter. Outcome-only fields such as churn reason and churn score are excluded "
        "from the prediction form to avoid target leakage."
    )

    st.subheader("Input guide")
    st.write("Use these definitions to interpret the fields in the prediction form.")
    input_groups = {
        "Customer profile": {
            "Gender": "Customer gender recorded in the source data.",
            "Senior citizen": "Whether the customer is age 65 or older.",
            "Partner": "Whether the customer has a partner.",
            "Dependents": "Whether the customer lives with dependents.",
            "Tenure months": "Total months the customer has been with the company.",
        },
        "Services": {
            "Phone service": "Whether the customer subscribes to home-phone service.",
            "Multiple lines": "Whether the customer has multiple phone lines; customers without phone service use 'No phone service'.",
            "Internet service": "Whether the customer has DSL, fiber optic, or no internet service.",
            "Online security": "Whether the customer subscribes to the online-security add-on.",
            "Online backup": "Whether the customer subscribes to the online-backup add-on.",
            "Device protection": "Whether the customer has an internet-equipment protection plan.",
            "Tech support": "Whether the customer subscribes to the technical-support plan.",
            "Streaming TV": "Whether the customer uses internet service for streaming TV.",
            "Streaming movies": "Whether the customer uses internet service for streaming movies.",
        },
        "Contract and billing": {
            "Contract": "The customer's current contract type: month-to-month, one year, or two year.",
            "Paperless billing": "Whether the customer uses paperless billing.",
            "Payment method": "How the customer pays: electronic check, mailed check, bank transfer, or credit card.",
            "Monthly charges": "Current monthly charge for the customer's subscribed services.",
            "Total charges": "Total charges accumulated through the end of the quarter.",
            "Customer lifetime value": "Predicted customer lifetime value from the source dataset's corporate formulas.",
        },
    }
    for group_name, fields in input_groups.items():
        with st.expander(group_name, expanded=group_name == "Customer profile"):
            for field, description in fields.items():
                st.markdown(f"**{FIELD_LABELS.get(field, field)}:** {description}")


def _select(label: str, options: list[str], default: Any, widget_suffix: str) -> str:
    return st.selectbox(
        FIELD_LABELS.get(label, label),
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
    threshold = float(result["threshold"])
    _render_prediction_result(
        prediction=str(result["prediction"]),
        probability=probability,
        threshold=threshold,
        model_name=str(result["model_name"]),
    )


def _render_prediction_result(
    prediction: str,
    probability: float,
    threshold: float,
    model_name: str,
) -> None:
    severity = _risk_severity(probability, threshold)
    palette = {
        "low": {
            "label": "No Churn",
            "message": "Lower churn risk under the current decision threshold.",
            "background": "#e6f6ec",
            "border": "#2e9d5b",
            "text": "#075e2b",
            "bar": "#2e9d5b",
        },
        "moderate": {
            "label": "Churn Risk",
            "message": "Elevated churn risk. Review for targeted retention action.",
            "background": "#fff4cf",
            "border": "#d79b00",
            "text": "#7a5300",
            "bar": "#d79b00",
        },
        "high": {
            "label": "High Churn Risk",
            "message": "High churn risk. Prioritize retention outreach.",
            "background": "#fde7e7",
            "border": "#d83b3b",
            "text": "#8f1d1d",
            "bar": "#d83b3b",
        },
    }[severity]
    bar_width = min(max(probability, 0.0), 1.0) * 100

    st.markdown(
        f"""
        <div style="
            border-left: 8px solid {palette["border"]};
            background: {palette["background"]};
            color: {palette["text"]};
            padding: 1rem 1.1rem;
            border-radius: 0.5rem;
            margin-top: 1rem;
            margin-bottom: 0.75rem;
        ">
            <div style="font-size: 0.85rem; font-weight: 700; text-transform: uppercase;">
                Prediction
            </div>
            <div style="font-size: 2rem; font-weight: 700; line-height: 1.2;">
                {palette["label"]}
            </div>
            <div style="font-size: 1rem; margin-top: 0.25rem;">
                {probability:.1%} churn probability
            </div>
            <div style="
                height: 0.7rem;
                background: rgba(0, 0, 0, 0.08);
                border-radius: 999px;
                margin-top: 0.9rem;
                overflow: hidden;
            ">
                <div style="
                    width: {bar_width:.1f}%;
                    height: 100%;
                    background: {palette["bar"]};
                "></div>
            </div>
            <div style="font-size: 0.85rem; margin-top: 0.75rem;">
                Decision threshold: {threshold:.1%}. Model: {model_name}.
            </div>
            <div style="font-weight: 600; margin-top: 0.75rem;">
                {palette["message"]}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"API prediction label: {prediction}")


def _risk_severity(probability: float, threshold: float) -> str:
    if probability >= 0.7:
        return "high"
    if probability >= threshold:
        return "moderate"
    return "low"


def _profile_defaults(profile: str) -> dict[str, object]:
    if profile == "Random Customer":
        if "random_profile" not in st.session_state:
            st.session_state.random_profile = _random_customer()
            st.session_state.random_profile_id = st.session_state.get("random_profile_id", 0) + 1
        return st.session_state.random_profile
    return _defaults(profile)


def _widget_suffix(profile: str) -> str:
    if profile == "Random Customer":
        return f"random_{st.session_state.get('random_profile_id', 0)}"
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
