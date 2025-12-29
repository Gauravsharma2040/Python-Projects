import pandas as pd

def add_sma(df, windows):
    for w in windows:
        df[f"SMA{w}"] = df["Close"].rolling(w).mean()
    return df

def add_ema(df, windows):
    for w in windows:
        df[f"EMA{w}"] = df["Close"].ewm(span=w, adjust=False).mean()
    return df

def add_rsi(df, period):
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df

def add_macd(df, fast, slow, signal):
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()

    df["MACD"] = ema_fast - ema_slow
    df["Signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["Signal"]
    return df
def max_drawdown(equity):
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    return drawdown.min()