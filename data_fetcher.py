"""Fetches 20 years price data for baseline model 
   and 6 months data for sentiment model for any given stock."""

import yfinance as yf
import pandas as pd

def format_stock_name(stock_name):
    return stock_name.upper() + ".NS"

def clean_yfinance_df(df):
    # If MultiIndex, keep only the first level (like 'Close' from ('Close', 'RELIANCE.NS'))
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def get_price_data(stock_name, period_for_baseline="20y", period_for_sentiment="6mo"):
    ticker = format_stock_name(stock_name)

    try:
        print(f"📥 Downloading the previous 20 years data for {ticker}...")
        df_20y = yf.download(ticker, period=period_for_baseline, interval="1d", progress=False)
        df_20y = clean_yfinance_df(df_20y)
        df_20y.reset_index(inplace=True)
        df_20y['Date'] = pd.to_datetime(df_20y['Date']).dt.date
        df_20y['Company'] = stock_name.upper()

        print(f"📥 Downloading recent 6 months data for {ticker}...")
        df_6m = yf.download(ticker, period=period_for_sentiment, interval="1d", progress=False)
        df_6m = clean_yfinance_df(df_6m)
        df_6m.reset_index(inplace=True)
        df_6m['Date'] = pd.to_datetime(df_6m['Date']).dt.date
        df_6m['Company'] = stock_name.upper()

        return df_20y, df_6m

    except Exception as e:
        print(f"Error fetching data: {e}")
        return None, None

print("data_fetcher loaded successfully (latest version)")
