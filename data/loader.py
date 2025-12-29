import yfinance as yf
import pandas as pd

def fetch_data(ticker, start, end):
    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False
    )

    if df.empty:
        raise ValueError("No data downloaded")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if "Close" not in df.columns:
        raise ValueError(f"'Close' column missing. Found: {df.columns}")

    return df   
