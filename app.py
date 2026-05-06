import streamlit as st
import pandas as pd
import requests
import joblib
import time
import plotly.express as px

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(page_title="Heart Attack Monitor", layout="wide")

# ==========================
# LOAD MODELS
# ==========================
rf_model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# ==========================
# BLYNK CONFIG
# ==========================
BLYNK_TOKEN = "JOXNOAXdalPOa5tQ3MCmEBRGz9KIn6G6"

URLS = {
    "HeartRate": f"https://blynk.cloud/external/api/get?token={BLYNK_TOKEN}&V0",
    "SpO2": f"https://blynk.cloud/external/api/get?token={BLYNK_TOKEN}&V1",
    "ECG": f"https://blynk.cloud/external/api/get?token={BLYNK_TOKEN}&V2",
}

# ==========================
# FUNCTIONS
# ==========================
def fetch(url):
    return float(requests.get(url, timeout=5).text)

# Send STRING → V1
def update_blynk_string(value):
    try:
        url = f"https://blynk.cloud/external/api/update?token={BLYNK_TOKEN}&V3={value}"
        requests.get(url)
        print("📡 V3 (String):", value)
    except Exception as e:
        print("❌ Blynk V1 Error:", e)

# Send INTEGER → V0
def update_blynk_int(value):
    try:
        url = f"https://blynk.cloud/external/api/update?token={BLYNK_TOKEN}&V6={value}"
        requests.get(url)
        print("📡 V6 (Integer):", value)
    except Exception as e:
        print("❌ Blynk V0 Error:", e)

def predict():
    data = {
        "HeartRate": fetch(URLS["HeartRate"]),
        "SpO2": fetch(URLS["SpO2"]),
        "ECG": fetch(URLS["ECG"]),
    }
    df = pd.DataFrame([data])
    scaled = scaler.transform(df)
    pred = rf_model.predict(scaled)
    result = label_encoder.inverse_transform(pred)[0]
    update_blynk_string(result)
    return result, data

# ==========================
# CSV STORAGE
# ==========================
CSV_FILE = "history.csv"

def save_data(data, result):
    df = pd.DataFrame([{**data, "Result": result, "Time": pd.Timestamp.now()}])
    try:
        old = pd.read_csv(CSV_FILE)
        df = pd.concat([old, df], ignore_index=True)
    except:
        pass
    df.to_csv(CSV_FILE, index=False)

# ==========================
# UI DESIGN
# ==========================
st.markdown("""
    <style>
    body {
        background: linear-gradient(to right, #1f4037, #99f2c8);
    }
    .metric-box {
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 20px;
        font-weight: bold;
        color: white;
    }
    .safe { background-color: #28a745; }
    .danger { background-color: #dc3545; }
    </style>
""", unsafe_allow_html=True)

st.title("❤️ Heart Attack Prediction Dashboard")

# ==========================
# FETCH BUTTON
# ==========================
if st.button("🔄 Fetch & Predict"):
    result, data = predict()
    save_data(data, result)

    col1, col2, col3 = st.columns(3)

    col1.metric("Heart Rate", data["HeartRate"])
    col2.metric("SpO2", data["SpO2"])
    col3.metric("ECG", data["ECG"])

    # Highlight Result
    if result.lower() == "normal":
        st.markdown(f'<div class="metric-box safe">Result: {result}</div>', unsafe_allow_html=True)
        update_blynk_int(0)
    else:
        st.markdown(f'<div class="metric-box danger">⚠️ Result: {result}</div>', unsafe_allow_html=True)
        update_blynk_int(1)

# ==========================
# HISTORY SECTION
# ==========================
st.subheader("📜 History Data")

try:
    history = pd.read_csv(CSV_FILE)
    st.dataframe(history, use_container_width=True)

    st.download_button("⬇️ Download CSV", history.to_csv(index=False), "history.csv")

    # ==========================
    # EDA CHARTS
    # ==========================
    st.subheader("📊 Data Analysis")

    fig1 = px.line(history, x="Time", y="HeartRate", title="Heart Rate Trend")
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.line(history, x="Time", y="SpO2", title="SpO2 Trend")
    st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.histogram(history, x="Result", title="Prediction Distribution")
    st.plotly_chart(fig3, use_container_width=True)

except:
    st.info("No history available yet. Click fetch to generate data.")
