# def create_labels(df,n_days):

#     """Creates binary classification labels based on price movement  after n_days
    
#     Args: 
#         df (pd.DataFrame):  Data Frame with "Close" column
#         n_days (int): number of days ahead to compare price

#     Returns:
#         pd.DataFrame : Data Frame with new 'label' column ahead 
        
#     """
#     df = df.copy()
#     df['Future_Close'] = df['Close'].shift(-n_days)
#     df["Label"] = (df['Future_Close'] > df['Close'].astype(int))
#     df.dropna(inplace=True) # drop last few days column cause they are NaN according to 'n_days'

#     return df

def create_labels(df, n_days=2, threshold=1.0):
    """
    Creates binary labels:
    1 => If stock price increases by more than `threshold` percent in `n_days`
    0 => Otherwise
    """
    df = df.copy()
    df['Future_Close'] = df['Close'].shift(-n_days)
    df['Future_Return'] = ((df['Future_Close'] - df['Close']) / df['Close']) * 100

    df['Label'] = (df['Future_Return'] > threshold).astype(int)

    return df
