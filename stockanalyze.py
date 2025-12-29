import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------
# SETTINGS
# ---------------------------------------------------
TICKER = "^NSEI"
START = "2025-11-05"
END = "2025-11-26"

# ---------------------------------------------------
# 1. DOWNLOAD DATA
# ---------------------------------------------------
print(f"Downloading data for {TICKER}...")
df = yf.download(TICKER, start=START, end=END)

if df.empty:
    print("Error: No data found.")
    exit()

# ---------------------------------------------------
# 2. INDICATORS
# ---------------------------------------------------

# SMA - Simple Moving Average
df["SMA20"] = df["Close"].rolling(window=20).mean()
df["SMA50"] = df["Close"].rolling(window=50).mean()

# EMA - Exponential Moving Average
df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()

# RSI - Relative Strength Index
def compute_rsi(series, period=14):
    delta = series.diff()

    gain = (delta.where(delta > 0, 0))
    loss = (-delta.where(delta < 0, 0))

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


df["RSI"] = compute_rsi(df["Close"])

# MACD
df["EMA12"] = df["Close"].ewm(span=12, adjust=False).mean()
df["EMA26"] = df["Close"].ewm(span=26, adjust=False).mean()
df["MACD"] = df["EMA12"] - df["EMA26"]
df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

# ---------------------------------------------------
# 3. FEATURE ENGINEERING
# ---------------------------------------------------

df["Return_1d"] = df["Close"].pct_change()
df["Return_3d"] = df["Close"].pct_change(3)

df["Volatility_5d"] = df["Return_1d"].rolling(5).std()
df["Volatility_10d"] = df["Return_1d"].rolling(10).std()

df["MACD_hist"] = df["MACD"] - df["Signal"]

df["RSI_overbought"] = (df["RSI"] > 70).astype(int)
df["RSI_oversold"] = (df["RSI"] < 30).astype(int)
features = [
    "SMA20", "SMA50", "EMA20",
    "RSI", "MACD", "Signal", "MACD_hist",
    "Return_1d", "Return_3d",
    "Volatility_5d", "Volatility_10d",
    "RSI_overbought", "RSI_oversold"
]

df_ml = df[features + ["Target"]].dropna()


# ---------------------------------------------------
# 4. PLOT THE RESULTS
# ---------------------------------------------------

plt.style.use("seaborn-v0_8")

# --------------- PRICE + SMA/EMA -------------------
plt.figure(figsize=(14, 7))
plt.plot(df["Close"], label="Close", linewidth=2)
plt.plot(df["SMA20"], label="SMA20")
plt.plot(df["SMA50"], label="SMA50")
plt.plot(df["EMA20"], label="EMA20")
plt.title(f"{TICKER} Price with SMA & EMA")
plt.legend()
plt.grid(True)
plt.show()

# --------------- RSI -------------------
plt.figure(figsize=(14, 4))
plt.plot(df["RSI"], color="purple", label="RSI")
plt.axhline(70, color="red", linestyle="--")
plt.axhline(30, color="green", linestyle="--")
plt.title("RSI Indicator")
plt.legend()
plt.grid(True)
plt.show()

# --------------- MACD -------------------
plt.figure(figsize=(14, 4))
plt.plot(df["MACD"], label="MACD")
plt.plot(df["Signal"], label="Signal")
plt.title("MACD Indicator")
plt.legend()
plt.grid(True)
plt.show()

# ---------------------------------------------------
# 4. Save results
# ---------------------------------------------------
df.to_csv(f"{TICKER}_analysis.csv")
print(f"Analysis saved to {TICKER}_analysis.csv!")
