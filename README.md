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

