import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords

# One-time download if not already
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()  # lowercase
    text = re.sub(r'<.*?>', '', text)  # remove HTML tags
    text = re.sub(r'http\S+|www\S+', '', text)  # remove URLs
    text = re.sub(r'\d+', '', text)  # remove digits
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]  # remove stopwords
    return ' '.join(tokens)

def clean_news_df(df):
    """
    Takes a news dataframe and adds 'clean_title' and 'clean_summary'
    
    """
    df['clean_title'] = df['title'].apply(clean_text)
    df['clean_summary'] = df['summary'].apply(clean_text)
    return df
