import streamlit as st
import pandas as pd
import joblib

# 1. Load pipeline
pipeline = joblib.load("final_model.joblib")

# 2. Page config
st.set_page_config(page_title="Stroke Risk Predictor", layout="centered")

# 3. Header
st.title("🧠 Stroke Risk Predictor")
st.markdown("""
Fill in the patient’s clinical data below and click **Predict** to see
the estimated stroke risk.
""")

# 4. Input form
with st.form("patient_form"):
    st.subheader("Patient Information")

    # Categorical / binary
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    ever_married = st.selectbox("Ever Married?", ["No", "Yes"])
    hypertension = st.radio("Hypertension?", ["No", "Yes"])
    heart_disease = st.radio("History of Heart Disease?", ["No", "Yes"])

    work_type = st.selectbox("Work Type", [
        "Children", 
        "Government employee", 
        "Never worked", 
        "Private sector", 
        "Self-employed"
    ])
    # map display name → original code
    work_map = {
        "Children":"children",
        "Government employee":"Govt_job",
        "Never worked":"Never_worked",
        "Private sector":"Private",
        "Self-employed":"Self-employed"
    }
    work_type_code = work_map[work_type]

    residence_type = st.selectbox("Residence Type", ["Rural", "Urban"])
    smoking_status = st.selectbox("Smoking Status", [
        "never smoked", 
        "formerly smoked", 
        "smokes", 
        "Unknown"
    ])

    # Numeric sliders grouped together
    st.markdown("### Vital Signs & Labs")
    bmi               = st.slider("• Body Mass Index (BMI)", 10.0, 60.0, 25.0, 0.1)
    avg_glucose_level = st.slider("• Average Glucose Level (mg/dL)", 50.0, 300.0, 100.0, 0.1)
    age               = st.slider("• Age (years)", 0, 100, 50, 1)

    submit = st.form_submit_button("Predict")

# 5. On submit: build DataFrame, map Yes/No → 1/0
if submit:
    df = pd.DataFrame([{
        "gender": gender,
        "age": age,
        "hypertension": 1 if hypertension=="Yes" else 0,
        "heart_disease":1 if heart_disease=="Yes" else 0,
        "ever_married": ever_married,
        "work_type": work_type_code,
        "Residence_type": residence_type,
        "avg_glucose_level": avg_glucose_level,
        "bmi": bmi,
        "smoking_status": smoking_status
    }])

    # 6. Predict
    proba = pipeline.predict_proba(df)[0,1]
    thresh = 0.284
    result = "🩺 Stroke Likely" if proba >= thresh else "✅ Low Risk"

    # 7. Display
    st.markdown("## Prediction")
    st.write(f"**Probability of Stroke:** {proba:.2%}")
    st.write(f"**Result:** {result}")

    # 8. Risk gauge
    st.progress(min(int(proba*100), 100))
