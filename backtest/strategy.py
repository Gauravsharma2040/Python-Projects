def backtest(df, predictions):
    df = df.copy()

    df["Prediction"] = predictions

    # Strategy returns (signal-based)
    df["Strategy_Return"] = (
        df["Prediction"].shift(1) * df["Return_1d"]
    )

    # Buy & Hold returns
    df["BuyHold_Return"] = df["Return_1d"]

    # Cumulative curves
    df["Strategy_Equity"] = (1 + df["Strategy_Return"]).cumprod()
    df["BuyHold_Equity"] = (1 + df["BuyHold_Return"]).cumprod()

    return df
