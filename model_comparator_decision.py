from baseline_model import train_baseline_model
from data_merge_sentiment_model_train import train_sentiment_model
from news_scraper import fetch_news_google
from news_cleaner import clean_news_df
from vader_sentiment import apply_sentiment, get_5_days_sentiment_signal

def compare_models_and_decide(stock_name='RELIANCE'):

    print("\nTraining baseline model (20y price only).........\n")
    base_model, base_X_test, base_y_test = train_baseline_model(stock_name)

    print("\nTraining sentiment-based model (news + 6m price).......\n")
    senti_model, senti_X_test, senti_y_test = train_sentiment_model(stock_name)

    # latest prediction
    base_pred = base_model.predict(base_X_test[-1:])[0] # means last raw all columns of dataframe X_test and [0] is the single element but it is in array
    senti_pred = senti_model.predict(senti_X_test[-1:])[0]

    if base_pred == senti_pred:
        final_decision = 'Buy' if base_pred == 1 else 'Sell'
    else:
        final_decision = 'Hold'

    print(f"\nBaseline Model Prediction: {'Up' if base_pred == 1 else 'Down'}")
    print(f" Sentiment Model Prediction: {'Up' if senti_pred == 1 else 'Down'}")
    print(f"Final Decision: {final_decision}")
    

    symbol = stock_name
    df_news = fetch_news_google(stock_name, max_articles=150)
    df_clean = clean_news_df(df_news)
    df_sent = apply_sentiment(df_clean)
    
    signal = get_5_days_sentiment_signal(df_sent)


    print(f"Also see the last five days ")
    print("\n\nFinal 5-day sentiment based trading signal:\n\n", signal)


    
    return {
        'baseline_prediction': int(base_pred),
        'sentiment_prediction': int(senti_pred),
        'final_decision': final_decision
    }

    
    
