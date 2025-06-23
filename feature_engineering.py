import pandas as pd

def add_custom_features(df):
    
    """
    Add custom engineered features to the dataframe.
    
    """
    df = df.copy()
    df['Price_Change'] = df['Close'] - df['Open']
    df['Volatility'] = df['High'] - df['Low']
    df['Price_Range_Percent'] = ((df['High'] - df['Low']) / df['Open']) * 100
    df['Is_Green_Candle'] = (df['Close'] > df['Open']).astype(int)
    return df

def get_feature_list():
    
    """
    Returns the list of features to be used for modeling.
    Update this list if you add/remove features.
    
    """
    # return [
    #     'EMA_9', 'EMA_15', 'EMA_50', 'MACD', 'Signal_Line',
    #     'RSI_14', 'OBV', 'VPT', 
    #     'Price_Change', 'Volatility', 'Price_Range_Percent', 'Is_Green_Candle'
    # ]

    return [
        'Close', 'MACD', 'RSI_14',
        'Price_Change', 'Volatility', 'Price_Range_Percent', 'Is_Green_Candle'
    ]

def get_final_features(df):
    """
    Returns the dataframe with selected features and drops rows with NaNs.
    """
    features = get_feature_list()
    df = df.copy()
    df = df.dropna(subset=features)  # Drop rows where any feature is NaN
    X = df[features]
    return X
