import pandas as pd
from data_fetcher import get_price_data
from news_scraper import fetch_news_google
from news_cleaner import clean_news_df
from vader_sentiment import apply_sentiment

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import numpy as np



def prepare_stock_sentiment_data(stock_name, max_articles = 200, sentiment_ffill_limit= 3, target_shift = 2):
    """
    Fetches stock price data and news sentiment data for a given symbol,
    merges them, applies forward fill on sentiment, and creates labels for classification.

    Args:
        symbol (str): Stock symbol e.g. 'reliance'.
        max_articles (int): Max number of news articles to fetch.
        sentiment_ffill_limit (int): Limit of forward fill for missing sentiment values.
        target_shift (int): Number of days to shift target for future return prediction.

    Returns:
        pd.DataFrame: Merged dataframe with price, sentiment, and label columns.
        
    """

    # get the data
    df_20y, df_6m = get_price_data(stock_name)

    # create some labels
    df_6m['pct_change'] = df_6m['Close'].pct_change()
    df_6m['target'] = df_6m['Close'].pct_change().shift(-target_shift)
    df_6m['label'] = df_6m['target'].apply(lambda x: 1 if x > 0 else 0)
    df_6m['Date'] = pd.to_datetime(df_6m['Date'])

    # fetch data,clean it, sentiment the data by vader
    df_news = fetch_news_google(stock_name, max_articles=max_articles)
    df_clean = clean_news_df(df_news)
    df_sent = apply_sentiment(df_clean)

    # sentiment dataframe
    df_sent = df_sent[['published', 'final_sentiment']].rename(columns={'published': 'Date'})
    df_sent['Date'] = pd.to_datetime(df_sent['Date']).dt.normalize()
    df_sent_agg = df_sent.groupby('Date')['final_sentiment'].mean().reset_index()

    # Merge the sentiment and 6 months stock data
    df_merge = pd.merge(df_6m, df_sent_agg, on='Date', how='inner')

    # fill miising values (limit)
    df_merge['final_sentiment'] = df_merge['final_sentiment'].ffill(limit=sentiment_ffill_limit)

    # create label 
    df_merge['label'] = df_merge['target'].apply(lambda x: 1 if x > 0 else 0)

    return df_merge

def train_sentiment_model(stock_name):
    df = prepare_stock_sentiment_data(stock_name)
    # print(f"\ncolumns in df_merge{df.columns}\n")
    features = ['pct_change','Volume', 'final_sentiment']
    X = df[features].fillna(0)
    y = df['label']
    
    # split data (because time series data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # logistic regression model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # prediction & evaluation
    y_pred = model.predict(X_test_scaled)
    print("Logistic Regression Results:")
    print(classification_report(y_test, y_pred))
    
    return model, X_test_scaled, y_test
