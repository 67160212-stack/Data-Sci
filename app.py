import streamlit as st
import numpy as np
import joblib
import pandas as pd

# ===== ตั้งค่าหน้า =====
st.set_page_config(page_title="Temperature Prediction", layout="wide")

st.title("🌡️ ระบบทำนายอุณหภูมิ (XGBoost)")
st.write("กรอกข้อมูลสภาพอากาศและข้อมูลย้อนหลังเพื่อทำนายอุณหภูมิ")

# ===== โหลดโมเดล =====
@st.cache_resource
def load_model():
    # เปลี่ยนชื่อไฟล์ให้ตรงกับที่คุณอัปโหลดมาล่าสุด
    return joblib.load("xgb_model.pkl") 

model = load_model()

# ===== รับค่า input สภาพอากาศปัจจุบัน (10 Features) =====
st.subheader("📋 ข้อมูลสภาพอากาศปัจจุบัน")
col1, col2 = st.columns(2)

with col1:
    humidity = st.number_input("Relative Humidity (%)", 0.0, 100.0, 60.0)
    dew_point = st.number_input("Dew Point (°C)", -50.0, 50.0, 20.0)
    pressure = st.number_input("Pressure MSL (hPa)", 900.0, 1100.0, 1013.0)
    cloud = st.number_input("Cloud Cover (%)", 0.0, 100.0, 50.0)
    wind_speed = st.number_input("Wind Speed 10m (km/h)", 0.0, 150.0, 10.0)

with col2:
    rain = st.number_input("Rain (mm)", 0.0, 200.0, 0.0)
    snow = st.number_input("Snowfall (cm)", 0.0, 50.0, 0.0)
    surface_pressure = st.number_input("Surface Pressure (hPa)", 900.0, 1100.0, 1010.0)
    wind_dir = st.number_input("Wind Direction (degrees)", 0.0, 360.0, 180.0)
    is_day = st.selectbox("Is Day? (0=No, 1=Yes)", [0, 1])

# ===== ข้อมูลอุณหภูมิย้อนหลัง (12 Features) =====
st.subheader("⏱️ อุณหภูมิย้อนหลัง 12 ชั่วโมง")
lags = []
# สร้างแถวละ 4 ช่องเพื่อให้ดูง่ายขึ้น
for row in range(3):
    cols = st.columns(4)
    for i in range(4):
        idx = row * 4 + i + 1
        with cols[i]:
            val = st.number_input(f"Lag {idx} (ชั่วโมงที่แล้ว)", value=25.0)
            lags.append(val)

# ===== Predict =====
if st.button("🔍 ทำนายอุณหภูมิ", type="primary", use_container_width=True):
    try:
        # เตรียม Features ให้ครบ 22 ตัว (10 ปัจจุบัน + 12 Lags)
        current_weather = [humidity, dew_point, pressure, cloud, wind_speed, rain, snow, surface_pressure, wind_dir, is_day]
        all_features = current_weather + lags
        
        # แปลงเป็น DataFrame หรือ Numpy Array ตามที่ XGBoost ต้องการ
        # ส่วนใหญ่ XGBoost ที่ save ผ่าน joblib จะรับ numpy array (1, 22)
        final_features = np.array([all_features])

        prediction = model.predict(final_features)[0]

        st.divider()
        st.success(f"### 🌡️ อุณหภูมิที่คาดการณ์: {prediction:.2f} °C")
        
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาด: {e}")
        st.info(f"จำนวน Features ที่ส่งไป: {len(all_features)} ตัว")
