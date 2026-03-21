import streamlit as st
import numpy as np
import joblib
import pandas as pd

# ===== ตั้งค่าหน้า =====
st.set_page_config(page_title="Temperature Prediction", layout="centered")

st.title("🌡️ ระบบทำนายอุณหภูมิ (RandomForest)")
st.write("กรอกข้อมูลสภาพอากาศเพื่อทำนายอุณหภูมิ")

# ===== โหลดโมเดล =====
@st.cache_resource
def load_model():
    return joblib.load("random_forest_model.pkl")  # ไฟล์ที่คุณเซฟ

model = load_model()

# ===== รับค่า input =====
st.subheader("📋 กรอกข้อมูล")

col1, col2 = st.columns(2)

with col1:
    humidity = st.number_input("relative_humidity", 0.0, 100.0, 60.0)
    dew_point = st.number_input("dew_point", -50.0, 50.0, 20.0)
    pressure = st.number_input("pressure_msl (hPa)", 900.0, 1100.0, 1013.0)
    cloud = st.number_input("cloud_cover (%)", 0.0, 100.0, 50.0)
    wind_speed = st.number_input("wind_speed_10m (km/h)", 0.0, 150.0, 10.0)

with col2:
    rain = st.number_input("rain (mm)", 0.0, 200.0, 0.0)
    snow = st.number_input("snowfall (cm)", 0.0, 50.0, 0.0)
    surface_pressure = st.number_input("surface_pressure (hPa)", 900.0, 1100.0, 1010.0)
    wind_dir = st.number_input("wind_direction", 0.0, 360.0, 180.0)
    is_day = st.selectbox("is_Day", [0, 1])

# ===== lag features (สำคัญมาก เพราะคุณ train แบบนี้) =====
st.subheader("⏱️ อุณหภูมิย้อนหลัง 12 ชั่วโมง")

lags = []
for i in range(1, 13):
    val = st.number_input(f"lag_{i}", value=25.0)
    lags.append(val)

# ===== Predict =====
if st.button("🔍 ทำนายอุณหภูมิ"):

    # ⚠️ ต้องเรียง feature ให้ตรงตอน train
    input_data = np.array([[
        *lags  # lag_1 ถึง lag_12
    ]])

    prediction = model.predict(input_data)[0]

    st.success(f"🌡️ อุณหภูมิที่คาดการณ์: {prediction:.2f} °C")
