# Stock-Pulse-AI

## Stock Market Trend Prediction using News Sentiment and Technical Indicators

This project combines Machine Learning with Natural Language Processing (NLP) to predict stock market trends more accurately. It integrates 20 years of historical stock price data with recent news sentiment to give a final Buy/Sell/Hold recommendation. The model is deployed through an interactive Streamlit web application.

---

## 🔍 Overview

Financial markets are influenced by both technical patterns and market sentiment. This project builds two separate models:
1. A **price-based model** that uses technical indicators.
2. A **news-based sentiment model** that uses VADER sentiment scores from scraped news headlines.

Both models are compared using a custom decision engine, and a final recommendation is given to the user.

---

## 🚀 Features

- 📈 Fetches and processes 20 years of historical stock data from NSE.
- 🧠 Applies technical indicators like SMA, EMA, RSI, MACD.
- 📰 Scrapes recent financial news for the selected stock.
- 🧾 Cleans and analyzes text using VADER sentiment analysis.
- 🔄 Merges sentiment with price data to improve prediction accuracy.
- 🧮 Compares predictions from both models and outputs a final decision: **Buy / Sell / Hold**.
- 📊 Displays a 5-day sentiment trend (positive/neutral/negative).
- 🌐 Fully interactive **Streamlit dashboard** for real-time exploration.

---

## 📁 Project Structure

⚙️ How the Project Works (Module-wise Summary):

* `data_fetcher.py`
Fetches 20 years of stock data (df_20y) and 6 months of recent data (df_6m) for analysis and sentiment merging.

* `indicators.py`
Adds key technical indicators (SMA, EMA, RSI, MACD) to historical price data.

* `feature_engineering.py`
Creates additional custom features and final feature set for training ML models.

* `label_generator.py`
Generates target labels like Buy, Sell, or Hold based on future stock performance.

* `auto_eda.py`
Automatically creates visual EDA reports (trends, distributions, etc.) from stock data.

* `baseline_model.py`
Trains multiple price-based ML models and selects the best based on accuracy or other metrics.

* `news_scraper.py`
Scrapes latest financial news headlines for the selected stock using Google News RSS.

* `news_cleaner.py`
Cleans the scraped headlines — removes noise, stopwords, symbols, etc.

* `vader_sentiment.py`
Applies VADER sentiment analysis to cleaned news text and aggregates scores for the last 5 days.

* `data_merge_sentiment_model_train.py`
Merges recent stock data with sentiment scores and trains a sentiment-based logistic regression model.

* `model_comparator.py`
Compares the predictions of both models (price-based & sentiment-based) and gives a final decision: Buy / Sell / Hold.

* `streamlit_app.py`
Frontend dashboard where user inputs stock name → entire pipeline runs → final results, EDA, and sentiment visualization are shown.

-------------------------------
## 🧠 ML Techniques Used

Supervised learning (Logistic Regression, Random Forest, etc.)

Feature engineering from price data and sentiment

Model evaluation and comparison

NLP using VADER for sentiment analysis

-------------------------------
## 📚 Libraries & Tools
Python (pandas, sklearn, numpy)

Streamlit (for web app interface)

BeautifulSoup & Feedparser (for news scraping)

VADER Sentiment (for analyzing news text)

Matplotlib & Seaborn (for visualizations)

-------------------------------

## 📊 Streamlit App

The app provides:
- A text input for stock name (with NSE autocomplete).
- Visual EDA summary of the stock’s historical data.
- Output from both price and sentiment models.
- Final prediction with **Buy / Sell / Hold** signal.
- Recent 5-day sentiment analysis visual.

Run it using:
streamlit run app/streamlit_app.py

## 📌 Sample Output

Buy/Sell/Hold prediction and EDA summary
