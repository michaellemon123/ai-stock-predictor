
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="Stock Trend Predictor", layout="centered")
st.title("📈 Stock Trend Predictor (Lite Version)")

ticker = st.text_input("Enter Stock Ticker (e.g., TSLA)", "TSLA").upper()

if st.button("Predict"):
    with st.spinner("Fetching data and analyzing..."):
        data = yf.download(ticker, period="6mo", interval="1d")
        if data.empty:
            st.error("Invalid ticker or no data available.")
        else:
            data["SMA_5"] = data["Close"].rolling(window=5).mean()
            data["SMA_20"] = data["Close"].rolling(window=20).mean()
            data["Momentum"] = data["Close"] - data["Close"].shift(5)

            latest_close = data["Close"].iloc[-1]
            sma_5 = data["SMA_5"].iloc[-1]
            sma_20 = data["SMA_20"].iloc[-1]
            momentum = data["Momentum"].iloc[-1]

            # Basic prediction logic
            trend = "🔼 Uptrend Expected" if sma_5 > sma_20 and momentum > 0 else "🔽 Downtrend Expected"
            signal = "✅ BUY" if sma_5 > sma_20 and momentum > 0 else "❌ SELL"

            st.subheader("🔍 Analysis")
            st.metric("Latest Close", f"${latest_close:.2f}")
            st.metric("5-Day SMA", f"${sma_5:.2f}")
            st.metric("20-Day SMA", f"${sma_20:.2f}")
            st.metric("5-Day Momentum", f"${momentum:.2f}")

            st.subheader("📌 Prediction:")
            st.success(trend)
            st.info(f"Signal: {signal}")

            st.subheader("📊 Price Chart")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data["Close"], name="Close"))
            fig.add_trace(go.Scatter(x=data.index, y=data["SMA_5"], name="5-Day SMA"))
            fig.add_trace(go.Scatter(x=data.index, y=data["SMA_20"], name="20-Day SMA"))
            st.plotly_chart(fig)
