import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Calories Burnt Prediction",
    layout="centered"
)

st.title("🔥 Calories Burnt Prediction App")
st.markdown(
    "Predict calories burned using a trained Random Forest Regression model"
)

# ---------------- LOAD ARTIFACT ----------------
@st.cache_resource
def load_artifact():
    artifact = joblib.load("calories_regression.pkl")
    return artifact["model"]

model, features = load_artifact()

# ---------------- USER INPUT ----------------
st.subheader("🧾 Enter Exercise Details")

weight = st.number_input(
    "Weight (kg)", min_value=30, max_value=200, value=70
)

body_temp = st.number_input(
    "Body Temperature (°C)", min_value=36.0, max_value=42.0, value=38.5
)

heart_rate = st.number_input(
    "Heart Rate (bpm)", min_value=60, max_value=200, value=120
)

duration = st.number_input(
    "Exercise Duration (minutes)", min_value=1, max_value=300, value=30
)

# ---------------- CREATE INPUT DATAFRAME ----------------
# ⚠️ ORDER MUST MATCH TRAINING FEATURES
input_data = pd.DataFrame([[
    weight,
    body_temp,
    heart_rate,
    duration
]], columns=features)

# ---------------- PREDICTION ----------------
st.markdown("---")

if st.button("🔍 Predict Calories Burned"):
    prediction = model.predict(input_data)[0]

    st.subheader("📊 Prediction Result")
    st.metric(
        label="Estimated Calories Burned",
        value=f"{prediction:.2f} kcal"
    )

    st.success("Prediction generated successfully!")

# ---------------- DEBUG (OPTIONAL) ----------------
with st.expander("🔎 Model Info"):
    st.write("Expected Features:", features)
    st.write("Model Type:", type(model).__name__)

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Model: RandomForestRegressor | Feature-selected & Deployed")
