from newspaper import Article
from textblob import TextBlob
from datetime import datetime
import logging
import re
import time


class SentimentAnalyzer:
    """Class for analyzing news sentiment and content"""

    def analyze_news_article(self, article_info):
        """Analyze a single news article and return its processed information."""
        try:
            article = Article(article_info["link"])
            article.download()
            article.parse()
        except Exception as e:
            logging.error(f"Error processing article at {article_info['link']}: {e}")
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

    def analyze_news_batch(self, news_data):
        """Analyze a batch of news articles and calculate total sentiment"""
        total_polarity = 0
        processed_news = []

        for article_info in news_data:
            if all(
                k in article_info
                for k in ["link", "providerPublishTime", "title", "publisher"]
            ):
                article_data = self.analyze_news_article(article_info)
                if article_data:
                    total_polarity += article_data["Sentiment"]
                    processed_news.append(article_data)
            else:
                logging.error(f"Missing required keys in article info: {article_info}")

        return processed_news, total_polarity

    def fetch_and_analyze_news(
        self, ticker_symbol, news_fetcher, max_retries=1, delay=2
    ):
        """Fetch and analyze news for a ticker symbol with retry functionality"""
        news_data = []
        total_polarity = 0
        attempts = 0

        while attempts < max_retries:
            try:
                dnews = news_fetcher(ticker_symbol)
                if dnews:
                    news_data, total_polarity = self.analyze_news_batch(dnews)
                    break
            except Exception as e:
                logging.error(
                    f"Attempt {attempts + 1}: Failed to fetch or analyze news for '{ticker_symbol}': {e}"
                )
                time.sleep(delay)
                attempts += 1

        if not news_data:
            logging.error(
                f"All {max_retries} attempts failed for ticker symbol '{ticker_symbol}'."
            )

        return news_data, total_polarity
