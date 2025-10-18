import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.set_page_config(page_title="Iris Species Predictor", layout="centered")

st.title("Iris Flower Species Predictor")
st.write("Enter the flower measurements and get model predictions.")

# Load model
model_bundle = joblib.load("models/iris_best.joblib")
model = model_bundle["model"]
target_map = model_bundle["target_map"]
feature_names = model_bundle["feature_names"]

# Input sliders
st.sidebar.header("Flower Measurements")
sepal_length = st.sidebar.slider("sepal length (cm)", 4.0, 8.0, 5.8, step=0.1)
sepal_width  = st.sidebar.slider("sepal width (cm)", 2.0, 4.5, 3.0, step=0.1)
petal_length = st.sidebar.slider("petal length (cm)", 1.0, 7.0, 4.35, step=0.1)
petal_width  = st.sidebar.slider("petal width (cm)", 0.1, 2.5, 1.3, step=0.1)

input_arr = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
prediction = model.predict(input_arr)[0]
prob = model.predict_proba(input_arr)[0] if hasattr(model, "predict_proba") else None

st.subheader("Prediction")
st.write(f"Predicted species: **{target_map[prediction]}**")

if prob is not None:
    prob_display = {target_map[i]: float(prob[i]) for i in range(len(prob))}
    st.subheader("Prediction probabilities")
    st.write(pd.DataFrame([prob_display]).T.rename(columns={0:"probability"}))

st.markdown("---")
st.write("This demo is for educational purposes only.")