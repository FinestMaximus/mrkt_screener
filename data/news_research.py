from datetime import datetime
import yfinance as yf
import streamlit as st
from utils.logger import info, debug, warning, error  # Updated import for logging


class SentimentAnalyzer:
    """Class for analyzing news sentiment and content"""

    @st.cache_data(show_spinner="Fetching news from API...", ttl="1h", persist=True)
    def _fetch_news(_self, ticker_symbol):
        """Fetch news data for a given ticker symbol."""
        try:
            info(f"Attempting to fetch news for ticker: {ticker_symbol}")
            ticker = yf.Ticker(ticker_symbol)
            dnews = ticker.get_news()
            processed_news = []

            # Log the number of news items received
            if dnews:
                info(
                    f"Successfully fetched {len(dnews)} news items for {ticker_symbol}"
                )

                # Log structure of first news item to understand the data format
                if len(dnews) > 0:
                    debug(f"First news item structure: {dnews[0].keys()}")

                    # Process each news item
                    for item in dnews:
                        try:
                            # Check that the item has required fields
                            if "id" not in item or "content" not in item:
                                debug(
                                    f"News item missing required fields: {item.keys()}"
                                )
                                continue

                            content = item["content"]

                            # Extract title from content
                            title = content.get("title", "No title")

                            # Extract summary from content
                            summary = content.get("summary", "")

                            # Skip items without summary
                            if not summary:
                                debug(f"Missing summary: {title}")
                                continue

                            # Extract preview URL (direct link to the article)
                            news_url = content.get("clickThroughUrl", {}).get(
                                "url"
                            ) or content.get("canonicalUrl", {}).get("url")

                            # Skip items with invalid URLs
                            if not news_url:
                                debug(f"Missing valid URL for: {title}")
                                continue

                            # Extract publisher name
                            publisher_name = content.get("provider", {}).get(
                                "displayName", "Unknown source"
                            )

                            # Get publish time
                            publish_time = content.get("pubDate") or content.get(
                                "displayTime"
                            )

                            # Get thumbnail URL if available
                            thumbnail = None
                            if (
                                "thumbnail" in content
                                and "resolutions" in content["thumbnail"]
                            ):
                                resolutions = content["thumbnail"]["resolutions"]
                                if (
                                    resolutions
                                    and isinstance(resolutions, list)
                                    and len(resolutions) > 0
                                ):
                                    thumbnail = resolutions[0].get("url")

                            processed_item = {
                                "title": title,
                                "publisher": publisher_name,
                                "providerPublishTime": publish_time,
                                "link": news_url,
                                "thumbnail": thumbnail,
                                "summary": summary,
                            }

                            debug(
                                f"Processed item publisher: {processed_item['publisher']}"
                            )
                            processed_news.append(processed_item)

                        except Exception as e:
                            warning(f"Could not process news item: {e}")
                            continue

                if not processed_news:
                    warning(f"No usable news found for ticker '{ticker_symbol}'.")
                else:
                    info(
                        f"Successfully processed {len(processed_news)} news items for {ticker_symbol}"
                    )

                return processed_news
            else:
                warning(f"No news found for ticker '{ticker_symbol}'.")
                return []
        except Exception as e:
            error(f"Failed to fetch ticker data or news for '{ticker_symbol}': {e}")
            return []

    def display_news_without_sentiment(self, ticker_symbol):
        """Display news articles in a user-friendly format"""
        try:
            info(f"Starting to display news for {ticker_symbol}")

            # Use the cached news fetching method
            news_data = self._fetch_news(ticker_symbol)

            info(f"Processing {len(news_data)} news items for display")

            # Custom CSS for compact, dense news layout
            st.markdown(
                """
            <style>
            .news-container {
                max-height: 500px;
                overflow-y: auto;
                padding-right: 10px;
                border-radius: 5px;
            }
            .news-card {
                display: flex;
                flex-direction: column;
                padding: 6px 0;
                margin-bottom: 4px;
                background-color: transparent;
            }
            .news-header {
                display: flex;
                flex-direction: row;
            }
            .news-thumbnail {
                margin-right: 8px;
                width: 60px;
                min-width: 60px;
                height: 60px;
                object-fit: cover;
            }
            .news-text {
                display: flex;
                flex-direction: column;
                flex: 1;
            }
            .news-title {
                font-size: 14px;
                font-weight: 500;
                color: #fff;
                margin-bottom: 2px;
                text-decoration: none;
                line-height: 1.2;
            }
            .news-title:hover {
                text-decoration: underline;
            }
            .news-meta {
                font-size: 10px;
                color: #aaa;
                line-height: 1.2;
                margin-bottom: 2px;
            }
            .news-desc {
                font-size: 12px;
                color: #fff;
                line-height: 1.2;
                padding-top: 4px;
            }
            </style>
            """,
                unsafe_allow_html=True,
            )

            if not news_data:
                st.info(
                    "No news items available at this time. Check back later for updates."
                )
                return

            # Create a container for the news items
            with st.container():
                # Render each news item individually instead of building one large HTML string
                for i, news_item in enumerate(news_data):
                    debug(f"Processing news item {i+1}/{len(news_data)}")

                    title = news_item.get("title", "No title")
                    link = news_item.get("link", "#")
                    if link is None or link == "/None" or not link:
                        link = "#"
                        debug(f"Invalid URL found for article: {title}")

                    summary = news_item.get("summary", "")
                    publisher = news_item.get("publisher", "")
                    publish_time = news_item.get("providerPublishTime", "")
                    formatted_date = self._format_date(publish_time)
                    thumbnail = news_item.get("thumbnail", "")

                    # Create news card HTML for this specific item
                    item_html = f"""
                    <div class="news-card" style="margin-bottom: 8px; padding: 6px;">
                        <div class="news-header" style="display: flex; margin-bottom: 2px;">
                            {"<img src='" + thumbnail + "' class='news-thumbnail'>" if thumbnail else "<div style='width:60px;min-width:60px;'></div>"}
                            <div class="news-text">
                                <a href="{link}" target="_blank" class="news-title">{title}</a>
                                <div class="news-meta">{publisher + " · " if publisher else ""}{formatted_date if formatted_date else ""}</div>
                                <div class="news-desc" style="margin-top: 2px;">{summary}</div>
                            </div>
                        </div>
                    </div>
                    """

                    # Render this item
                    st.markdown(item_html, unsafe_allow_html=True)

            info(f"Completed displaying news for {ticker_symbol}")

        except Exception as e:
            error(f"Error displaying news for {ticker_symbol}: {e}")
            st.error("Error loading news data. Please try again later.")

    def _format_date(self, timestamp):
        """Format timestamp into a readable date/time."""
        try:
            if not timestamp:
                return ""

            # Parse the timestamp (assuming ISO format like '2025-04-04T22:36:22Z')
            try:
                dt = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ")
            except ValueError:
                # Try alternate format if first one fails
                try:
                    dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    return timestamp  # Return original if parsing fails

            # Get current time for relative time calculation
            now = datetime.now()
            diff = now - dt

            # Format as relative time
            if diff.days == 0:
                hours = diff.seconds // 3600
                if hours == 0:
                    minutes = diff.seconds // 60
                    if minutes == 0:
                        return "Just now"
                    return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
                return f"{hours} hour{'s' if hours > 1 else ''} ago"
            elif diff.days == 1:
                return "Yesterday"
            elif diff.days < 7:
                return f"{diff.days} days ago"
            else:
                # Format as date
                return dt.strftime("%b %d, %Y")
        except Exception as e:
            debug(f"Error formatting date: {e}")
            return ""
