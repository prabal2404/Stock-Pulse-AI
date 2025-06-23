import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

try:
    # Check if vader_lexicon already exists
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    # If not found, download it
    nltk.download('vader_lexicon')

# initiazlize vader

vader = SentimentIntensityAnalyzer()

def get_sentiment_scores(text):
    if not isinstance(text, str):
        return pd.Series([0, 0, 0, 0])  # if NaN values
    
    score = vader.polarity_scores(text)
    return pd.Series([
        score['neg'],
        score['neu'],
        score['pos'],
        score['compound']
    ])

def apply_sentiment(df):
    """
    Apply sentiment analysis to "clean_title + clean_summary"
    Adds 4 new columns for each: neg, neu, pos, compound

    """
    # title sentiments on basis of 'clean_title'
    df[['title_neg', 'title_neu', 'title_pos', 'title_compound']] = df['clean_title'].apply(get_sentiment_scores)

    # summary sentiments on basis of 'clean_summary'
    df[['summary_neg', 'summary_neu', 'summary_pos', 'summary_compound']] = df['clean_summary'].apply(get_sentiment_scores)

    # average sentiment 
    df['final_sentiment'] = (df['title_compound'] + df['summary_compound']) / 2

    return df

def get_5_days_sentiment_signal(df):
    """
    Takes in a df with 'published' and 'final_sentiment' columns
    Returns 'BUY', 'SELL', or 'HOLD' based on last 5 days average sentiment trend

    """
    # Ensure datetime format
    df['published'] = pd.to_datetime(df['published']).dt.normalize()

    # Group by Date and take mean sentiment
    df_daily = df.groupby('published')['final_sentiment'].mean().reset_index()

    # Last 5 days (sorted)
    last_5 = df_daily.sort_values(by='published', ascending=False).head(5).sort_values(by='published')

    print("Last 5 Days Sentiment:")
    print(last_5)

    # Decision logic
    pos_days = (last_5['final_sentiment'] > 0).sum()
    neg_days = (last_5['final_sentiment'] < 0).sum()

    if pos_days >= 3:
        decision = "BUY"
    elif neg_days >= 3:
        decision = "SELL"
    else:
        decision = "HOLD"

    return decision, last_5