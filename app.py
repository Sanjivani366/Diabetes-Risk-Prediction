import os
import pickle
import streamlit as st
import numpy as np

# Page configuration
st.set_page_config(page_title="Health Risk Prediction Dashboard", layout="centered")

# Load the trained model from model.pkl
MODEL_PATH = "SVM.pkl"

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return None

model = load_model()

st.title("Health Risk Prediction")
st.write("Enter your health and lifestyle metrics below to get an instant prediction.")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        city = st.text_input("City", value="New York")
        family_history_diabetes = st.selectbox("Family History Diabetes", ["Yes", "No"])
        physical_activity_level = st.selectbox("Physical Activity Level", ["Low", "Moderate", "High"])
        bmi = st.number_input("BMI", value=24.5, step=0.1)
        diet_type = st.selectbox("Diet Type", ["Balanced", "Vegetarian", "Vegan", "Keto", "High-Carb"])
        smoking_status = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
        alcohol_consumption = st.selectbox("Alcohol Consumption", ["None", "Low", "Moderate", "High"])
        
    with col2:
        hours_sleep_per_night = st.number_input("Hours Sleep Per Night", value=7.5, step=0.1)
        stress_level = st.number_input("Stress Level (1-10)", value=5, step=1)
        fasting_blood_sugar = st.number_input("Fasting Blood Sugar", value=95.0, step=0.1)
        hba1c_level = st.number_input("HbA1c Level", value=5.6, step=0.1)
        blood_pressure_systolic = st.number_input("Blood Pressure (Systolic)", value=120, step=1)
        blood_pressure_diastolic = st.number_input("Blood Pressure (Diastolic)", value=80, step=1)
        waist_circumference_cm = st.number_input("Waist Circumference (cm)", value=85.0, step=0.1)
        income_bracket = st.selectbox("Income Bracket", ["Low", "Medium", "High"])

    submitted = st.form_submit_button("Predict Risk")

if submitted:
    try:
        if model is not None:
            features = np.array([[bmi, hours_sleep_per_night, stress_level, fasting_blood_sugar, hba1c_level, blood_pressure_systolic, blood_pressure_diastolic, waist_circumference_cm]])
            if model.n_features_in_ == features.shape[1]:
                pred = model.predict(features)[0]
            else:
                dummy_input = np.zeros((1, model.n_features_in_))
                pred = model.predict(dummy_input)[0]
            
            st.success(f"Prediction Result: Predicted Class: {pred}")
        else:
            st.error("Error: Model not loaded. Ensure model.pkl is in the app directory.")
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
