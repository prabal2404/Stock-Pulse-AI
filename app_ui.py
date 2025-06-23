import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from stock_names import load_nse_stocks
from baseline_model import train_all_models
from model_comparator_decision import compare_models_and_decide
from news_scraper import fetch_news_google
from news_cleaner import clean_news_df
from vader_sentiment import apply_sentiment, get_5_days_sentiment_signal
from auto_eda import smart_eda
from data_fetcher import get_price_data
from label_generator import create_labels
import sys
import io

def background_color(val):
    if val > 0:
        color = '#004d00'  # dark green background
    else:
        color = '#8b0000'  # dark red background
    text_color = '#000000'  # black text for both cases
    return f'background-color: {color}; color: {text_color}; font-weight: bold;'



st.set_page_config(page_title="Stock Pulse AI", layout="centered")


nse_stocks = load_nse_stocks()
stock_name = st.selectbox("🔍 Select or Type NSE Stock Name", nse_stocks, index=nse_stocks.index("RELIANCE") if "RELIANCE" in nse_stocks else 0,key="stock_input")

st.markdown("""
    <style>
        .centered {
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
            justify-content: flex-start !important;
            padding-top: 60px !important;
            min-height: 80vh !important;
            width: 100% !important;
            margin: 0 auto !important;
        }
        .block-container {
            padding-top: 60px !important;
            max-width: 900px !important;
            margin-left: auto !important;
            margin-right: auto !important;
        }
       .stButton>button {
            width: 250px;
            display: block;
            margin: 0 auto;
            background-color: #4CAF50;
            color: white;
            border-radius: 10px;
            border: 2px solid #4CAF50;
            transition: background-color 0.4s ease, color 0.4s ease;
            position: relative;
            overflow: hidden;
            padding: 10px 0;
            font-size: 18px;
            letter-spacing: normal;
            text-align: center;
            font-weight: 400;        
            padding-left: 1px;         
        }
        .stButton>button:hover {
            color: black;              
            background-color: #4CAF50;  
            border-color: #4CAF50;      
            z-index: 1;
        }
        
        label[data-for="stock_input"] {
            display: block;
            text-align: center;
            font-weight: bold;
            font-size: 20px;
            margin-bottom: 8px;
        }
        .buy-box {
            background-color: #d4edda;
            color: #155724;
            padding: 15px;
            border-radius: 12px;
            text-align: center;
            font-size: 18px;
        }
        .sell-box {
            background-color: #f8d7da;
            color: #721c24;
            padding: 15px;
            border-radius: 12px;
            text-align: center;
            font-size: 18px;
        }
        .hold-box {
            background-color: #fff3cd;
            color: #856404;
            padding: 15px;
            border-radius: 12px;
            text-align: center;
            font-size: 18px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <h1 style='
        text-align: center;
        font-size: 60px;
        font-family: "Segoe UI", "Helvetica Neue", sans-serif;
        font-weight: 800;
        color: #ffffff;    /* White color for dark bg */
        margin-top: 20px;
        margin-bottom: 20px;
    '>
        Stock Pulse AI
    </h1>
""", unsafe_allow_html=True)



st.write("Get trading decisions based on **20-year price data**, **news sentiment**, and **5-day sentiment signal**")



df_20y, _ = get_price_data(stock_name)
if df_20y is None or df_20y.empty:
    st.error("🚫 Unable to fetch stock data. Please check the stock name.")
    st.stop()


train_all_models_option = st.checkbox("Train different classification models on differernt parameters for 20 years data (may take 5-10 mins)")
run_eda = st.checkbox("📊 Run Auto EDA on 20-Year Price Data")


if st.button("🔍 Predict"):
    if not stock_name:
        st.warning("Please enter a NSE stock name and click Predict !!!!!!")

    else:
        if train_all_models_option:
            with st.spinner("Training all models....."):
                df_results = train_all_models(stock_name, run_eda=False)
                st.success("✅ Training completed!")
                with st.expander("📊 Model Training Results"):
                    st.dataframe(df_results)
            
        if run_eda:
            st.markdown("⏳ Running `smart_eda()` on df_20y...")
    
            # 1. Redirect stdout
            buffer = io.StringIO()
            sys.stdout = buffer
        
            # 2. run EDA
            df_with_label = create_labels(df_20y)
            smart_eda(df_with_label)
        
            # 3. reset
            sys.stdout = sys.__stdout__
        
            # 4. ret output
            output = buffer.getvalue()
        
            # 5. show in streamlit
            with st.expander("📂 View Auto EDA Report"):
                st.code(output, language='bash')
        
        with st.spinner(f"Running models and fetching news sentiment for {stock_name}......"):
            results = compare_models_and_decide(stock_name)
            
        
        st.subheader("🧾 Model Predictions")
    
        col1, col2, col3 = st.columns(3)
        col1.metric("🧠 Baseline Model", "Up" if results['baseline_prediction'] == 1 else "Down")
        col2.metric("💬 Sentiment Model", "Up" if results['sentiment_prediction'] == 1 else "Down")
        col3.metric("✅ Final Decision", results['final_decision'])
    
        st.markdown("---")
    
        st.subheader("📌 Final Trading Signal")
        if results['final_decision'] == 'Buy':
            st.markdown("<div class='buy-box'>💹 You should consider: <b>BUY</b></div>", unsafe_allow_html=True)
        elif results['final_decision'] == 'Sell':
            st.markdown("<div class='sell-box'>🔻 You should consider: <b>SELL</b></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='hold-box'>⏸️ You should consider: <b>HOLD</b></div>", unsafe_allow_html=True)
    
    
        # 5days sentiment signal
        st.subheader("🗓️ Last 5-Day Sentiment Signal")
        st.info(f"Signal based on past 5 days: **{results['signal_decision']}**")
    
        df_signal = results['signal_df']
        df_signal['Sentiment'] = df_signal['final_sentiment'].apply(lambda x: 'Positive' if x > 0 else 'Negative')
    
    
        st.bar_chart(df_signal['Sentiment'].value_counts())
    
        styled_df = df_signal.style.applymap(background_color, subset=['final_sentiment'])
    
        st.write("📊 Detailed Daily Sentiments:")
        st.dataframe(styled_df)
    

else:
    st.warning("Please enter a NSE stock name and click Predict !!!!!!")


st.markdown("</div>", unsafe_allow_html=True)
