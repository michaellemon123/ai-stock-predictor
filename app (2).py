
import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="AI Stock Predictor", layout="centered")
st.title("📈 AI Stock Predictor")

ticker = st.text_input("Enter a stock ticker (e.g. TSLA, AAPL):", value="TSLA")

@st.cache_data
def load_data(ticker):
    df = yf.download(ticker, period="6mo", interval="1d")
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

    # Ensure 1D input for all indicators
    close_series = df['Close']

    df['RSI'] = RSIIndicator(close=close_series).rsi()
    macd = MACD(close=close_series)
    df['MACD'] = macd.macd_diff()
    bb = BollingerBands(close=close_series)
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()
    df['SMA_10'] = close_series.rolling(window=10).mean()
    df['SMA_20'] = close_series.rolling(window=20).mean()
    df['Target_1d'] = np.where(close_series.shift(-1) > close_series, 1, 0)
    df['Target_5d'] = np.where(close_series.shift(-5) > close_series, 1, 0)

    df.dropna(inplace=True)
    return df

try:
    df = load_data(ticker)

    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD',
                'BB_High', 'BB_Low', 'SMA_10', 'SMA_20']
    X = df[features]
    y_1d = df['Target_1d']
    y_5d = df['Target_5d']

    X_train_1d, X_test_1d, y_train_1d, y_test_1d = train_test_split(X, y_1d, test_size=0.2, shuffle=False)
    X_train_5d, X_test_5d, y_train_5d, y_test_5d = train_test_split(X, y_5d, test_size=0.2, shuffle=False)

    model_1d = RandomForestClassifier(n_estimators=100, random_state=42)
    model_1d.fit(X_train_1d, y_train_1d)

    model_5d = RandomForestClassifier(n_estimators=100, random_state=42)
    model_5d.fit(X_train_5d, y_train_5d)

    latest = X.iloc[[-1]]
    pred_1d = model_1d.predict(latest)[0]
    pred_5d = model_5d.predict(latest)[0]
    proba_1d = model_1d.predict_proba(latest)[0][1]
    proba_5d = model_5d.predict_proba(latest)[0][1]

    st.subheader("📊 AI Prediction")
    st.markdown(f"**Next-Day Trend:** {'📈 Bullish' if pred_1d else '📉 Bearish'} ({proba_1d * 100:.2f}% confidence)")
    st.markdown(f"**Next-Week Trend:** {'📈 Bullish' if pred_5d else '📉 Bearish'} ({proba_5d * 100:.2f}% confidence)")

    st.subheader("💵 Historical Price Chart")
    st.line_chart(df['Close'][-90:])

except Exception as e:
    st.error(f"❌ Something went wrong: {e}")
