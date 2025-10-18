import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Smart Healthcare Assistant", layout="wide")
st.title("Smart Healthcare Assistant for Disease Prediction")

# -----------------------------
# Load trained Random Forest model
# -----------------------------
model = joblib.load("models/health_rf.joblib")  # Must be only the RF model

# -----------------------------
# Streamlit Input Form
# -----------------------------
st.header("Enter Patient Details")
age = st.slider("Age", 0, 100, 45)
sex = st.selectbox("Sex", [0, 1])  # 0=Female, 1=Male
bmi = st.slider("BMI", 10.0, 50.0, 27.5)
systolic_bp = st.slider("Systolic BP", 80, 200, 130)
diastolic_bp = st.slider("Diastolic BP", 50, 150, 85)
cholesterol = st.slider("Cholesterol", 100, 400, 210)
glucose = st.slider("Glucose", 50, 300, 110)
smoking = st.selectbox("Smoking", [0, 1])
physical_activity = st.slider("Physical Activity (0-10)", 0, 10, 5)
family_history = st.selectbox("Family History", [0, 1])

# Create DataFrame from inputs
input_df = pd.DataFrame([{
    'age': age,
    'sex': sex,
    'bmi': bmi,
    'systolic_bp': systolic_bp,
    'diastolic_bp': diastolic_bp,
    'cholesterol': cholesterol,
    'glucose': glucose,
    'smoking': smoking,
    'physical_activity': physical_activity,
    'family_history': family_history
}])

# -----------------------------
# Predict Disease Risk
# -----------------------------
prediction = model.predict(input_df)[0]
prediction_prob = model.predict_proba(input_df)[0]

st.subheader("Prediction Result")
st.write(f"Predicted Disease Risk: {'High' if prediction==1 else 'Low'}")
st.write(f"Probability of Low Risk: {prediction_prob[0]:.2f}")
st.write(f"Probability of High Risk: {prediction_prob[1]:.2f}")

# -----------------------------
# SHAP Explainability
# -----------------------------
st.subheader("Feature Impact (SHAP Values)")
explainer = shap.TreeExplainer(model)
shap_values = explainer(input_df)

# Plot SHAP values as a bar chart
fig, ax = plt.subplots(figsize=(8, 4))
shap.bar_plot(shap_values[0].values, feature_names=input_df.columns)
st.pyplot(fig)