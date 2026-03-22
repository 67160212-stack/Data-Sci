import streamlit as st
import numpy as np
import joblib

# ===== ตั้งค่าหน้า =====
st.set_page_config(page_title="Temperature Prediction AI", layout="wide")

st.title("🌡️ AI ทำนายอุณหภูมิ (XGBoost)")
st.markdown("ระบบนี้ใช้ Machine Learning วิเคราะห์ข้อมูลสภาพอากาศ + อุณหภูมิย้อนหลัง เพื่อทำนายอุณหภูมิในอนาคต")

# ===== โหลดโมเดล =====
@st.cache_resource
def load_model():
    return joblib.load("xgb_model.pkl")

model = load_model()

# ===== STEP 1 =====
st.header("🧭 STEP 1: กรอกข้อมูลสภาพอากาศปัจจุบัน")
st.caption("ข้อมูลเหล่านี้ช่วยให้ AI เข้าใจสภาพแวดล้อม ณ ปัจจุบัน")

col1, col2 = st.columns(2)

with col1:
    humidity = st.slider("💧 ความชื้น (%)", 0, 100, 60)
    dew_point = st.number_input("🌫️ Dew Point (°C)", -50.0, 50.0, 20.0)
    pressure = st.number_input("📊 Pressure MSL (hPa)", 900.0, 1100.0, 1013.0)
    cloud = st.slider("☁️ Cloud Cover (%)", 0, 100, 50)
    wind_speed = st.number_input("🌬️ Wind Speed (km/h)", 0.0, 150.0, 10.0)

with col2:
    rain = st.number_input("🌧️ Rain (mm)", 0.0, 200.0, 0.0)
    snow = st.number_input("❄️ Snowfall (cm)", 0.0, 50.0, 0.0)
    surface_pressure = st.number_input("📉 Surface Pressure (hPa)", 900.0, 1100.0, 1010.0)
    wind_dir = st.slider("🧭 Wind Direction (°)", 0, 360, 180)
    is_day = st.radio("🌞 กลางวัน / กลางคืน", ["กลางคืน", "กลางวัน"])

is_day = 1 if is_day == "กลางวัน" else 0

# ===== STEP 2 =====
st.header("⏱️ STEP 2: อุณหภูมิย้อนหลัง 12 ชั่วโมง")
st.caption("ข้อมูลนี้สำคัญมาก เพราะอุณหภูมิในอดีตช่วยให้ AI เห็นแนวโน้ม (Trend)")

lags = []
cols = st.columns(4)

for i in range(12):
    with cols[i % 4]:
        val = st.number_input(f"Lag {i+1} ชั่วโมงก่อนหน้า", value=25.0, key=f"lag_{i}")
        lags.append(val)

# ===== STEP 3 =====
st.header("🔍 STEP 3: ทำนายผล")

if st.button("🚀 ทำนายอุณหภูมิ", use_container_width=True):

    # รวม Feature
    current_weather = [
        humidity, dew_point, pressure, cloud, wind_speed,
        rain, snow, surface_pressure, wind_dir, is_day
    ]

    base_features = current_weather + lags

    # ===== ตรวจสอบ Feature =====
    if len(base_features) > 43:
        st.error("❌ จำนวน Feature เกิน 43 ตัว")
    else:
        # เติมค่าให้ครบ
        padding = [0.0] * (43 - len(base_features))
        final_features_list = base_features + padding

        final_features = np.array([final_features_list])

        try:
            prediction = model.predict(final_features)[0]

            st.divider()
            st.success(f"🌡️ อุณหภูมิที่คาดการณ์: {prediction:.2f} °C")

            # ===== อธิบายผล =====
            st.info(f"""
            💡 การทำนายนี้ใช้ข้อมูลทั้งหมด **{len(final_features_list)} Features**

            🔎 หลักการ:
            - ใช้ข้อมูลปัจจุบัน → บอกสภาพอากาศตอนนี้  
            - ใช้ Lag 12 ชั่วโมง → บอกแนวโน้มอุณหภูมิ  
            - AI จะรวมสองส่วนนี้เพื่อทำนายค่าในอนาคต
            """)

        except Exception as e:
            st.error(f"❌ เกิดข้อผิดพลาด: {e}")
            st.warning("⚠️ ตรวจสอบว่า Feature ที่ใช้ตรงกับตอน Train หรือไม่")
