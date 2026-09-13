import streamlit as st
import pandas as pd
import joblib

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    """
    Loads the complete trained ML pipeline.
    The pipeline should include preprocessing + model.
    """
    return joblib.load("churn_pipeline.pkl")


try:
    model = load_model()
except FileNotFoundError:
    st.error(
        "❌ churn_pipeline.pkl was not found. "
        "Please place churn_pipeline.pkl in the same folder as app.py."
    )
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()


# ============================================================
# HEADER
# ============================================================

st.title("📊 Customer Churn Prediction")

st.markdown(
    """
    Enter the customer's information below to predict whether
    the customer is likely to churn.
    """
)

st.divider()


# ============================================================
# CUSTOMER INFORMATION
# ============================================================

st.header("👤 Customer Information")

col1, col2, col3 = st.columns(3)

with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=35,
        step=1
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    region = st.selectbox(
        "Region",
        [
            "North",
            "South",
            "East",
            "West"
        ]
    )

with col2:

    tenure_months = st.number_input(
        "Tenure (Months)",
        min_value=0,
        max_value=120,
        value=12,
        step=1
    )

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        max_value=10000.0,
        value=70.0,
        step=1.0
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        max_value=1000000.0,
        value=840.0,
        step=10.0
    )

with col3:

    contract_type = st.selectbox(
        "Contract Type",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

    internet_service = st.selectbox(
        "Internet Service",
        [
            "DSL",
            "Fiber optic",
            "No"
        ]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer",
            "Credit card"
        ]
    )


# ============================================================
# SERVICES
# ============================================================

st.header("🛠️ Customer Services")

col1, col2, col3 = st.columns(3)

with col1:

    tech_support = st.selectbox(
        "Tech Support",
        [
            "Yes",
            "No"
        ]
    )

    online_security = st.selectbox(
        "Online Security",
        [
            "Yes",
            "No"
        ]
    )

with col2:

    paperless_billing = st.selectbox(
        "Paperless Billing",
        [
            "Yes",
            "No"
        ]
    )

    avg_monthly_usage_gb = st.number_input(
        "Average Monthly Usage (GB)",
        min_value=0.0,
        max_value=5000.0,
        value=100.0,
        step=1.0
    )

with col3:

    num_support_calls = st.number_input(
        "Number of Support Calls",
        min_value=0,
        max_value=100,
        value=2,
        step=1
    )

    late_payments_last_year = st.number_input(
        "Late Payments Last Year",
        min_value=0,
        max_value=50,
        value=1,
        step=1
    )


# ============================================================
# PREDICTION
# ============================================================

st.divider()

st.header("🔮 Churn Prediction")

if st.button(
    "Predict Customer Churn",
    type="primary",
    use_container_width=True
):

    # --------------------------------------------------------
    # Create dataframe with EXACT training column names
    # --------------------------------------------------------

    customer_data = pd.DataFrame({
        "age": [age],
        "gender": [gender],
        "region": [region],
        "tenure_months": [tenure_months],
        "monthly_charges": [monthly_charges],
        "total_charges": [total_charges],
        "contract_type": [contract_type],
        "internet_service": [internet_service],
        "tech_support": [tech_support],
        "online_security": [online_security],
        "paperless_billing": [paperless_billing],
        "payment_method": [payment_method],
        "num_support_calls": [num_support_calls],
        "late_payments_last_year": [late_payments_last_year],
        "avg_monthly_usage_gb": [avg_monthly_usage_gb]
    })

    # --------------------------------------------------------
    # Make prediction
    # --------------------------------------------------------

    try:

        prediction = model.predict(customer_data)[0]

        # Convert prediction to Yes/No
        if prediction in [1, "Yes", "yes", True]:
            churn_prediction = "Yes"
        else:
            churn_prediction = "No"

        # ----------------------------------------------------
        # Display prediction
        # ----------------------------------------------------

        st.subheader("Prediction Result")

        if churn_prediction == "Yes":

            st.error(
                "⚠️ HIGH RISK: This customer is likely to churn."
            )

        else:

            st.success(
                "✅ LOW RISK: This customer is likely to stay."
            )

        # ----------------------------------------------------
        # Churn probability
        # ----------------------------------------------------

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(customer_data)

            # Find probability of positive/churn class
            if hasattr(model, "classes_"):

                classes = list(model.classes_)

                if 1 in classes:
                    churn_index = classes.index(1)
                elif "Yes" in classes:
                    churn_index = classes.index("Yes")
                elif "yes" in classes:
                    churn_index = classes.index("yes")
                else:
                    churn_index = 1

                churn_probability = probabilities[0][churn_index]

            else:

                churn_probability = probabilities[0][1]

            probability_percent = churn_probability * 100

            # ------------------------------------------------
            # Metrics
            # ------------------------------------------------

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Churn Probability",
                    f"{probability_percent:.1f}%"
                )

            with col2:

                if probability_percent >= 70:
                    risk_level = "High"
                elif probability_percent >= 40:
                    risk_level = "Medium"
                else:
                    risk_level = "Low"

                st.metric(
                    "Risk Level",
                    risk_level
                )

            # ------------------------------------------------
            # Progress bar
            # ------------------------------------------------

            st.progress(
                float(churn_probability),
                text=f"Churn probability: {probability_percent:.1f}%"
            )

            # ------------------------------------------------
            # Recommendation
            # ------------------------------------------------

            st.subheader("💡 Recommended Action")

            if probability_percent >= 70:

                st.warning(
                    """
                    This customer has a high probability of churn.

                    Recommended actions:
                    - Contact the customer proactively.
                    - Offer a retention discount.
                    - Investigate recent support issues.
                    - Consider a longer-term contract.
                    """
                )

            elif probability_percent >= 40:

                st.info(
                    """
                    This customer has a moderate churn risk.

                    Recommended actions:
                    - Monitor customer activity.
                    - Improve customer engagement.
                    - Consider a personalized offer.
                    """
                )

            else:

                st.success(
                    """
                    This customer has a low churn risk.

                    Continue normal customer engagement
                    and service.
                    """
                )

        # ----------------------------------------------------
        # Display customer information
        # ----------------------------------------------------

        with st.expander("View Customer Information"):

            st.dataframe(
                customer_data,
                use_container_width=True
            )

    except Exception as e:

        st.error(
            f"❌ Prediction failed: {e}"
        )

        st.info(
            """
            Make sure that model.pkl was trained using the same
            column names and data types used by this application.
            """
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Customer Churn Prediction Dashboard | Built with Streamlit"
)
