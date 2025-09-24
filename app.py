%%writefile app.py
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import datetime
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model

MODEL_PATH = "stock_model.h5"
SCALER_PATH = "scaler.pkl"

# ✅ Fix 1: Use st.cache_resource instead of st.cache
@st.cache_resource
def load_resources():
    model = load_model(MODEL_PATH, custom_objects={"mse": tf.keras.losses.MeanSquaredError()})
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

model, scaler = load_resources()

st.title("📈 Stock Price Prediction App")

ticker = st.text_input("Enter Stock Symbol (e.g., AAPL, TSLA):", "AAPL")

if st.button("Predict"):
    # Download latest stock data
    data = yf.download(ticker, start="2015-01-01", end=datetime.date.today().strftime("%Y-%m-%d"))
    data = data[['Close']]

    last_60 = data[-60:].values
    scaled_last_60 = scaler.transform(last_60)

    X_test = np.reshape(scaled_last_60, (1, 60, 1))
    pred_price = model.predict(X_test)
    pred_price = scaler.inverse_transform(pred_price)

    st.success(f"Predicted Next Day Closing Price: ${pred_price[0][0]:.2f}")


from pyngrok import ngrok, conf

NGROK_AUTH_TOKEN = "YOUR_NGROK_AUTH_TOKEN"
conf.get_default().auth_token = NGROK_AUTH_TOKEN

ngrok.kill()  # close old tunnels
public_url = ngrok.connect(8501)
print("🌍 Streamlit public URL:", public_url)

!streamlit run app.py --server.port 8501 &>/dev/null&
