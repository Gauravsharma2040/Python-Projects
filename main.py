from config import *
from data.loader import fetch_data
from indicators.technicals import*
from features.build_features import build_features
from models.random_forest import train_random_forest
from backtest.strategy import backtest
from visualization.plots import *
from sklearn.model_selection import train_test_split


df = fetch_data(TICKER, START, END)
df = add_sma(df, SMA_WINDOWS)
df = add_ema(df, EMA_WINDOWS)
df = add_rsi(df, RSI_PERIOD)
df = add_macd(df, MACD_FAST, MACD_SLOW, MACD_SIGNAL)

df = build_features(df, PREDICTION_HORIZON)

df = df.dropna()

FEATURES = [
    "RSI_oversold",
    "RSI_overbought",
    "RSI_mid",
    "MACD_cross_up",
    "MACD_cross_down",
    "MACD_positive",
    "Price_above_EMA20",
    "EMA20_above_SMA50",
    "High_volatility",
    "Trend_conflict"
]

df = df.dropna()

X = df[FEATURES]

y = df["Target"]
if len(df) < 50:
    raise ValueError(
        f"Not enough data after preprocessing. Rows left: {len(df)}"
    )


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, shuffle=False
)

model = train_random_forest(X_train, y_train, RANDOM_STATE)

preds = model.predict(X)

df_bt = backtest(df, preds)
print("ML DD:", max_drawdown(df_bt["Strategy_Equity"]))
print("BH DD:", max_drawdown(df_bt["BuyHold_Equity"]))
print(df_bt[["Prediction", "Strategy_Return"]].head(10))
print(df_bt["Prediction"].mean())
print(y.mean())
plot_equity_curves(df_bt)
