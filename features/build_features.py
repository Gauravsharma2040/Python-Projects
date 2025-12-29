def build_features(df, horizon):
    # ---------- Returns ----------
    df["Return_1d"] = df["Close"].pct_change()
    df["Return_3d"] = df["Close"].pct_change(3)

    # ---------- RSI Signals ----------
    df["RSI_oversold"] = (df["RSI"] < 30).astype(int)
    df["RSI_overbought"] = (df["RSI"] > 70).astype(int)
    df["RSI_mid"] = ((df["RSI"] >= 40) & (df["RSI"] <= 60)).astype(int)

    # ---------- MACD Signals ----------
    df["MACD_cross_up"] = (
        (df["MACD"] > df["Signal"]) &
        (df["MACD"].shift(1) <= df["Signal"].shift(1))
    ).astype(int)

    df["MACD_cross_down"] = (
        (df["MACD"] < df["Signal"]) &
        (df["MACD"].shift(1) >= df["Signal"].shift(1))
    ).astype(int)

    df["MACD_positive"] = (df["MACD"] > 0).astype(int)

    # ---------- Trend Signals ----------
    df["Price_above_EMA20"] = (df["Close"] > df["EMA20"]).astype(int)
    df["EMA20_above_SMA50"] = (df["EMA20"] > df["SMA50"]).astype(int)

    # ---------- Volatility Signals ----------
    df["Volatility_10d"] = df["Return_1d"].rolling(10).std()
    df["High_volatility"] = (
        df["Volatility_10d"] >
        df["Volatility_10d"].rolling(50).mean()
    ).astype(int)

    # ---------- Target ----------
    threshold = 0.002 
    df["Target"] = (
    (df["Close"].shift(-horizon) / df["Close"] - 1) > threshold
    ).astype(int)

    df["Trend_conflict"] = (
    (df["Price_above_EMA20"] == 0) & (df["MACD_positive"] == 1)
    ).astype(int)
    signal_cols = [
    "RSI_oversold", "RSI_overbought",
    "MACD_cross_up", "MACD_cross_down",
    "MACD_positive",
    "Price_above_EMA20", "EMA20_above_SMA50"
    ]
    df["Signal_strength"] = df[signal_cols].sum(axis=1)
    return df
