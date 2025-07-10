import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("power_consumption_model.pkl")

# App title and description
st.title("⚡ Power Consumption Predictor")
st.markdown("""
This app predicts **power consumption in Zone A** (Wellington, NZ) based on environmental and meteorological inputs.
""")

# Sidebar for user input
st.sidebar.header("Input Environmental Data")

def get_user_input():
    temperature = st.sidebar.slider("Temperature (°C)", 0.0, 35.0, 15.0)
    humidity = st.sidebar.slider("Humidity (%)", 0.0, 100.0, 75.0)
    wind_speed = st.sidebar.slider("Wind Speed (m/s)", 0.0, 5.0, 0.5)
    general_diffuse = st.sidebar.slider("General Diffuse Radiation", 0.0, 0.2, 0.05)
    diffuse = st.sidebar.slider("Diffuse Radiation", 0.0, 0.2, 0.1)
    aqi = st.sidebar.slider("Air Quality Index (PM)", 0, 200, 150)
    cloudiness = st.sidebar.radio("Cloudiness", [0, 1])
    
    return {
        "Temperature": temperature,
        "Humidity": humidity,
        "Wind Speed": wind_speed,
        "general diffuse flows": general_diffuse,
        "diffuse flows": diffuse,
        "Air Quality Index (PM)": aqi,
        "Cloudiness": cloudiness
    }

# Get user input
input_data = get_user_input()

# Preprocess input
def preprocess_input(data):
    temp = data['Temperature']
    humidity = data['Humidity']
    wind = data['Wind Speed']
    gdif = data['general diffuse flows']
    dif = data['diffuse flows']
    aqi = data['Air Quality Index (PM)']
    cloud = data['Cloudiness']

    # Feature transformations (same as training)
    temp_hum = temp * humidity
    temp_sq = temp ** 2
    log_diffuse = np.log1p(dif)
    sqrt_wind = np.sqrt(wind)

    # Final feature array (must match training order)
    features = np.array([[
        temp, humidity, wind, gdif, dif, aqi, cloud,
        temp_hum, temp_sq, log_diffuse, sqrt_wind
    ]])

    return features

# Predict
processed_input = preprocess_input(input_data)

if st.button("Predict Power Consumption"):
    prediction = model.predict(processed_input)
    st.success(f"🔋 Estimated Power Consumption: **{prediction[0]:,.2f} kWh**")
