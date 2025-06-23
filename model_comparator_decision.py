
from baseline_model import train_baseline_model
from data_merge_sentiment_model_train import train_sentiment_model
from news_scraper import fetch_news_google
from news_cleaner import clean_news_df
from vader_sentiment import apply_sentiment, get_5_days_sentiment_signal
import pandas as pd

def compare_models_and_decide(stock_name='RELIANCE'):
    print(f"\n Training baseline model (20y price only) for {stock_name}......\n")
    base_model, base_X_test, base_y_test = train_baseline_model(stock_name)

    print(f"\nTraining sentiment-based model (6m price + news) for {stock_name}......\n")
    senti_model, senti_X_test, senti_y_test = train_sentiment_model(stock_name)

    base_pred = base_model.predict(base_X_test[-1:])[0]
    senti_pred = senti_model.predict(senti_X_test[-1:])[0]

    if base_pred == senti_pred:
        final_decision = 'Buy' if base_pred == 1 else 'Sell'
    else:
        final_decision = 'Hold'

    print(f"\n Baseline Model: {'Up' if base_pred == 1 else 'Down'}")
    print(f"\nSentiment Model: {'Up' if senti_pred == 1 else 'Down'}")
    print(f"\nFinal Combined Decision: {final_decision}")

    df_news = fetch_news_google(stock_name, max_articles=150)
    df_clean = clean_news_df(df_news)
    df_sent = apply_sentiment(df_clean)

    signal_decision, signal_df = get_5_days_sentiment_signal(df_sent)

    return {
        'baseline_prediction': int(base_pred),
        'sentiment_prediction': int(senti_pred),
        'final_decision': final_decision,
        'signal_decision': signal_decision,
        'signal_df': signal_df
    }

    
