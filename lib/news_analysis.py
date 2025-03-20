import re
from newspaper import Article
from textblob import TextBlob
import yfinance as yf
import logging
from datetime import datetime
import streamlit as st


def fetch_news(ticker_symbol):
    """Fetch news data for a given ticker symbol with error handling."""
    try:
        logging.debug(
            f"[news_analysis.py][fetch_news] Fetching news for ticker: {ticker_symbol}"
        )
        dnews = yf.Ticker(ticker_symbol).news
        if not dnews:
            logging.warning(
                f"[news_analysis.py][fetch_news] No news found for ticker '{ticker_symbol}'."
            )
            return []
        return dnews
    except Exception as e:
        logging.error(
            f"[news_analysis.py][fetch_news] Failed to fetch news for ticker '{ticker_symbol}': {e}"
        )
        return []


def analyze_news_article(article_info):
    """Analyze a single news article and return its processed information."""
    try:
        logging.debug(
            f"[news_analysis.py][analyze_news_article] Analyzing article: {article_info['link']}"
        )
        article = Article(article_info["link"])
        article.download()
        article.parse()
    except Exception as e:
        logging.error(
            f"[news_analysis.py][analyze_news_article] Error processing article at {article_info['link']}: {e}"
        )
        return None
    blob = TextBlob(article.text)
    polarity = blob.sentiment.polarity
    days_ago = (
        datetime.now() - datetime.fromtimestamp(article_info["providerPublishTime"])
    ).days
    return {
        "Title": article_info["title"],
        "Link": article_info["link"],
        "Publisher": article_info["publisher"],
        "Sentiment": polarity,
        "Days Ago": days_ago,
    }


def get_news_data(ticker_symbol):
    """Fetch and analyze news data and calculate total polarity for a given ticker symbol."""
    logging.debug(
        f"[news_analysis.py][get_news_data] Fetching and analyzing news data for ticker: {ticker_symbol}"
    )
    dnews = fetch_news(ticker_symbol)
    total_polarity = 0
    news_data = []
    for article_info in dnews:
        if all(
            k in article_info
            for k in ["link", "providerPublishTime", "title", "publisher"]
        ):
            article_data = analyze_news_article(article_info)
            if article_data:
                total_polarity += article_data["Sentiment"]
                news_data.append(article_data)
        else:
            logging.error(
                f"[news_analysis.py][get_news_data] Missing required keys in article info: {article_info}"
            )
    return news_data, total_polarity


def display_news_articles(news_data):
    """Display formatted news articles data"""
    logging.debug(f"[news_analysis.py][display_news_articles] Displaying news articles")
    if not news_data:
        st.write("No news data available.")
        return
    for news_item in news_data:
        title = news_item["Title"]
        if len(title) > 70:
            title = title[:67] + "..."
        rounded_sentiment = round(news_item["Sentiment"], 2)
        days_ago = int(news_item["Days Ago"])
        st.markdown(
            f"{rounded_sentiment} - [{re.sub(':', '', title)}]({news_item['Link']}) - ({days_ago} days ago)"
        )
