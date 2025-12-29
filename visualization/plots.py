import matplotlib.pyplot as plt

def plot_price_indicators(df):
    plt.figure(figsize=(14,7))
    plt.plot(df["Close"], label="Close")
    plt.plot(df["SMA20"], label="SMA20")
    plt.plot(df["EMA20"], label="EMA20")
    plt.legend()
    plt.grid()
    plt.show()

def plot_rsi(df):
    plt.figure(figsize=(14,4))
    plt.plot(df["RSI"])
    plt.axhline(70, linestyle="--")
    plt.axhline(30, linestyle="--")
    plt.show()

def plot_equity_curves(df):
    plt.figure(figsize=(12, 5))

    plt.plot(df.index, df["Strategy_Equity"], label="ML Strategy")
    plt.plot(df.index, df["BuyHold_Equity"], label="Buy & Hold", linestyle="--")

    plt.title("Equity Curve Comparison")
    plt.legend()
    plt.grid()
    plt.show()
