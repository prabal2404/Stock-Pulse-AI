import pandas as pd
def load_nse_stocks():
    df = pd.read_csv("stock_list.csv")
    return df['SYMBOL'].tolist()
    