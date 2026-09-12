"""
app_with_genai.py — example of how to extend your existing app.py.

This mirrors the standard input form your app.py already has, then adds:
  1. SHAP explanation of the prediction
  2. Retrieved retention policies
  3. A Gemini-generated retention strategy

Don't just drop this in blind — merge the "NEW:" marked sections into your
actual app.py so you keep whatever input handling / styling you already
built. Run with: streamlit run app.py
"""

import os

import streamlit as st
from dotenv import load_dotenv

from explainability import load_artifacts, explain_prediction  # NEW
from genai_retention import retrieve_relevant_policies, generate_retention_strategy  # NEW

load_dotenv()

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊", layout="centered")
st.title("Customer Churn Prediction App")

# NEW: cache the model + encoders load
@st.cache_resource
def get_artifacts():
    return load_artifacts()

model, le_gender, ohe_geo, scaler = get_artifacts()

with st.sidebar:
    st.subheader("Gemini API Key")  # NEW
    api_key = st.text_input("GEMINI_API_KEY", value=os.environ.get("GEMINI_API_KEY", ""), type="password")
    st.caption("Get a free key at https://aistudio.google.com/app/apikey")

st.divider()

with st.form("customer_form"):
    st.subheader("Customer Information")
    credit_score = st.number_input("Credit Score", 300, 900, 650)
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
    gender = st.selectbox("Gender", ["Female", "Male"])
    age = st.number_input("Age", 18, 100, 45)
    tenure = st.number_input("Tenure", 0, 10, 3)
    balance = st.number_input("Balance", 0.0, value=120000.0)
    num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
    has_cr_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
    is_active = st.selectbox("Is Active Member?", ["Yes", "No"])
    salary = st.number_input("Estimated Salary", 0.0, value=100000.0)
    submitted = st.form_submit_button("Predict Churn")

if submitted:
    raw = {
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": 1 if has_cr_card == "Yes" else 0,
        "IsActiveMember": 1 if is_active == "Yes" else 0,
        "EstimatedSalary": salary,
    }

    # --- your existing prediction logic likely already does this part ---
    proba, shap_df = explain_prediction(raw, model, le_gender, ohe_geo, scaler)  # NEW
    prediction = "Churn" if proba >= 0.5 else "No Churn"

    st.divider()
    st.subheader("Prediction Result")
    col1, col2 = st.columns(2)
    col1.metric("Churn Probability", f"{proba:.1%}")
    col2.metric("Prediction", prediction)

    # NEW: SHAP explanation
    st.markdown("### 🔍 Top Reasons")
    top_drivers = shap_df[shap_df["SHAP"] > 0].head(3)
    if top_drivers.empty:
        st.write("No strong churn-pushing factors identified for this customer.")
    else:
        for row in top_drivers.itertuples():
            st.write(f"- **{row.Label}** = {row.Value} (pushes churn risk up by {row.SHAP:.3f})")

    # NEW: retrieved policies
    st.markdown("### 📋 Relevant Retention Policies")
    for p in retrieve_relevant_policies(raw):
        with st.expander(p["title"]):
            st.write(f"**Eligibility:** {p['eligibility']}")
            st.write(f"**Benefits:** {', '.join(p['benefits'])}")
            st.write(f"**Recommended Action:** {p['action']}")

    # NEW: Gemini-generated strategy
    st.markdown("### 🤖 AI-Generated Retention Strategy")
    if prediction == "No Churn":
        st.success("This customer is not currently flagged as high risk.")
    elif not api_key:
        st.warning("Enter your Gemini API key in the sidebar to generate a strategy.")
    else:
        with st.spinner("Asking Gemini..."):
            try:
                strategy = generate_retention_strategy(api_key, raw, shap_df, proba)
                st.info(strategy)
            except Exception as e:
                st.error(f"Gemini request failed: {e}")
