import os
import pickle
import streamlit as st

# Get the directory where app.py is located
current_dir = os.path.dirname(__file__)
model_path = os.path.join(current_dir, "model.pkl") # Replace with your actual model file name

@st.cache_resource
def load_model():
    with open(model_path, "rb") as f:
        return pickle.load(f)

model = load_model()
