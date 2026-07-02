import streamlit as st
import numpy as np
import pickle

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Energy Efficiency Prediction",
    page_icon="🏠",
    layout="centered"
)

# -----------------------------
# Load Model and Scaler
# -----------------------------
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

except FileNotFoundError as e:
    st.error(f"Missing file: {e.filename}")
    st.stop()

# -----------------------------
# Title
# -----------------------------
st.title("🏠 Building Energy Efficiency Prediction")
st.write("Enter the building parameters below.")

# -----------------------------
# Inputs
# -----------------------------
relative_compactness = st.number_input(
    "Relative Compactness",
    min_value=0.50,
    max_value=1.00,
    value=0.98,
    step=0.01
)

surface_area = st.number_input(
    "Surface Area",
    value=514.5
)

wall_area = st.number_input(
    "Wall Area",
    value=294.0
)

roof_area = st.number_input(
    "Roof Area",
    value=110.25
)

overall_height = st.selectbox(
    "Overall Height",
    [3.5, 7.0]
)

orientation = st.selectbox(
    "Orientation",
    [2, 3, 4, 5]
)

glazing_area = st.selectbox(
    "Glazing Area",
    [0.0, 0.10, 0.25, 0.40]
)

glazing_distribution = st.selectbox(
    "Glazing Area Distribution",
    [0, 1, 2, 3, 4, 5]
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):

    sample = np.array([[
        relative_compactness,
        surface_area,
        wall_area,
        roof_area,
        overall_height,
        orientation,
        glazing_area,
        glazing_distribution
    ]])

    sample_scaled = scaler.transform(sample)

    prediction = model.predict(sample_scaled)

    heating_load = prediction[0][0]
    cooling_load = prediction[0][1]

    st.success("Prediction Completed Successfully!")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Heating Load (Y1)", f"{heating_load:.2f}")

    with col2:
        st.metric("Cooling Load (Y2)", f"{cooling_load:.2f}")