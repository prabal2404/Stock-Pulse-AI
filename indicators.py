# import pandas as pd
# import ta # 'ta' is the technical analysis library

# In this module we'll create the technical indicators on the basis of the data downloaded from the "yfinnance" 

# indicators.py

import pandas as pd
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volume import OnBalanceVolumeIndicator, VolumePriceTrendIndicator

def add_technical_indicators(df):
    df = df.copy()

    # 💡 Fix if accidentally 2D
    df['Close'] = df['Close'].squeeze()
    df['Volume'] = df['Volume'].squeeze()

    # EMA
    df['EMA_9'] = EMAIndicator(close=df['Close'], window=9).ema_indicator()
    df['EMA_15'] = EMAIndicator(close=df['Close'], window=15).ema_indicator()
    df['EMA_50'] = EMAIndicator(close=df['Close'], window=50).ema_indicator()

    # MACD
    macd = MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['Signal_Line'] = macd.macd_signal()

    # RSI
    df['RSI_14'] = RSIIndicator(close=df['Close'], window=14).rsi()

    # OBV
    obv = OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume'])
    df['OBV'] = obv.on_balance_volume()

    # VPT
    vpt = VolumePriceTrendIndicator(close=df['Close'], volume=df['Volume'])
    df['VPT'] = vpt.volume_price_trend()

    # Drop rows with NaNs
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df
