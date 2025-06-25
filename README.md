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

stock-sentiment-predictor/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/ # Sample or test data
│ └── example_stock_data.csv
│
├── src/ # Main backend logic
│ ├── data_fetcher.py
│ ├── indicators.py
│ ├── feature_engineering.py
│ ├── label_generator.py
│ ├── auto_eda.py
│ ├── baseline_model.py
│ ├── news_scraper.py
│ ├── news_cleaner.py
│ ├── vader_sentiment.py
│ ├── data_merge_sentiment_model_train.py
│ └── model_comparator.py
│
├── app/ # Streamlit app
│ └── streamlit_app.py
│
├── outputs/ # Charts and screenshots
│ └── sample_predictions.png
│ └── sentiment_trend_plot.png


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
