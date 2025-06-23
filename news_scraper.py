import feedparser
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime

def clean_html(raw_html):
    return BeautifulSoup(raw_html, "html.parser").get_text()

def fetch_news_google(stock_name, max_articles=100):
    """
    Fetch latest news about given stock_name using Google News RSS
    """
    stock_query = stock_name + " stock"  
    url = f"https://news.google.com/rss/search?q={stock_query.replace(' ', '+')}&hl=en-IN&gl=IN&ceid=IN:en"
    
    print("RSS URL:", url)  # Debug
    feed = feedparser.parse(url)
    print("Articles found:", len(feed.entries))  

    news_list = []
    for entry in feed.entries[:max_articles]:
        title = entry.title
        summary = clean_html(entry.summary)
        link = entry.link
        published = entry.get('published', None)

        news_list.append({
            "title": title,
            "summary": summary,
            "link": link,
            "published": published,
            "source": "Google News",
            "company": stock_name.upper()
        })

    df = pd.DataFrame(news_list)
    if 'published' in df.columns:
        df['published'] = pd.to_datetime(df['published'], errors='coerce')
    return df

# reddit backup
# def fetch_news_reddit(stock_name, max_articles=50):
#     url = f"https://www.reddit.com/r/stocks/.rss"
#     feed = feedparser.parse(url)

#     news_list = []
#     for entry in feed.entries[:max_articles]:
#         if stock_name.lower() in entry.title.lower():
#             summary = clean_html(entry.summary)
#             news_list.append({
#                 "title": entry.title,
#                 "summary": summary,
#                 "link": entry.link,
#                 "published": entry.get('published', None),
#                 "source": "Reddit",
#                 "company": stock_name.upper()
#             })

#     df = pd.DataFrame(news_list)
#     if 'published' in df.columns:
#         df['published'] = pd.to_datetime(df['published'], errors='coerce')
#     return df

