import tempfile
import os
import base64
import io
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import mplfinance as mpf
import traceback
from utils.logger import info, debug, warning, error, critical


class ReportGenerator:
    """Class to handle report generation for stock analysis"""

    def __init__(self):
        """Initialize the ReportGenerator"""
        info("Initializing ReportGenerator")
        # Initialize session data containers
        self.charts_data = {}
        self.news_data = {}
        self.sentiment_data = {}

    def capture_chart(self, ticker_symbol, fig):
        """Capture a matplotlib figure as base64 string for the report

        Args:
            ticker_symbol (str): The ticker symbol for the chart
            fig (matplotlib.figure.Figure): The matplotlib figure to capture

        Returns:
            str: Base64 encoded string of the chart image
        """
        info(f"Capturing chart for {ticker_symbol} for the report")

        try:
            # Save the figure to a bytes buffer
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            buf.seek(0)

            # Convert to base64 string
            img_str = base64.b64encode(buf.getvalue()).decode()

            # Store in the instance
            self.charts_data[ticker_symbol] = img_str

            info(f"Chart for {ticker_symbol} captured successfully")
            return img_str
        except Exception as e:
            error(f"Error capturing chart for {ticker_symbol}: {str(e)}")
            error(traceback.format_exc())
            return None

    def capture_news(self, ticker_symbol, news_items):
        """Capture news articles for the report

        Args:
            ticker_symbol (str): The ticker symbol for the news
            news_items (list): List of news article dictionaries

        Returns:
            list: The captured news items
        """
        info(f"Capturing news for {ticker_symbol} for the report")

        try:
            # Store in the instance - handle empty or None news_items
            if not news_items:
                debug(f"No news items to capture for {ticker_symbol}")
                self.news_data[ticker_symbol] = []
                return []

            # Process news items to ensure we have all required fields
            processed_items = []
            for item in news_items:
                # Ensure all required fields are present with defaults
                processed_item = {
                    "title": item.get("title", f"News about {ticker_symbol}"),
                    "published": item.get("published", "Recent"),
                    "url": item.get("url", "#"),
                    "publisher": item.get("publisher", "Unknown source"),
                    "summary": item.get("summary", "No summary available"),
                }

                # Add sentiment scores if available
                if "sentiment" in item:
                    processed_item["sentiment"] = item["sentiment"]

                processed_items.append(processed_item)

            self.news_data[ticker_symbol] = processed_items

            info(
                f"News for {ticker_symbol} captured successfully ({len(processed_items)} articles)"
            )
            return processed_items
        except Exception as e:
            error(f"Error capturing news for {ticker_symbol}: {str(e)}")
            error(traceback.format_exc())
            self.news_data[ticker_symbol] = []
            return []

    def capture_sentiment(self, ticker_symbol, sentiment_data):
        """Capture sentiment analysis data for the report

        Args:
            ticker_symbol (str): The ticker symbol
            sentiment_data (dict): Sentiment analysis results

        Returns:
            dict: The captured sentiment data
        """
        info(f"Capturing sentiment data for {ticker_symbol}")

        try:
            self.sentiment_data[ticker_symbol] = sentiment_data
            info(f"Sentiment data for {ticker_symbol} captured successfully")
            return sentiment_data
        except Exception as e:
            error(f"Error capturing sentiment data for {ticker_symbol}: {str(e)}")
            return {}

    def _create_html_header(self, timestamp):
        """Create the HTML header section for the report

        Args:
            timestamp (str): Timestamp for the report

        Returns:
            str: HTML header section
        """
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Stock Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ text-align: center; margin-bottom: 30px; }}
                .metrics-table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
                .metrics-table th, .metrics-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                .metrics-table th {{ background-color: #f2f2f2; }}
                .chart-section {{ page-break-before: always; }}
                .chart-container {{ margin-bottom: 30px; border-bottom: 1px solid #eee; padding-bottom: 20px; }}
                .footer {{ text-align: center; font-size: 0.8em; margin-top: 30px; }}
                h2 {{ color: #333; border-bottom: 2px solid #ddd; padding-bottom: 5px; margin-top: 30px; }}
                h3 {{ color: #444; margin-top: 25px; }}
                .market-data {{ margin: 15px 0; padding: 10px; background-color: #f9f9f9; border-radius: 5px; }}
                .market-data table {{ width: 100%; border-collapse: collapse; }}
                .market-data th {{ background-color: #f2f2f2; padding: 8px; text-align: left; }}
                .market-data td {{ padding: 8px; border-bottom: 1px solid #eee; }}
                .news-section {{ margin-top: 20px; }}
                .news-container {{ background-color: #f9f9f9; border-left: 3px solid #2196F3; margin-bottom: 15px; padding: 10px; }}
                .news-container.positive {{ border-left-color: #4CAF50; }}
                .news-container.negative {{ border-left-color: #F44336; }}
                .news-container.neutral {{ border-left-color: #9E9E9E; }}
                .news-date {{ color: #666; font-size: 0.8em; margin-bottom: 5px; }}
                .news-title {{ font-weight: bold; margin-bottom: 5px; }}
                .news-source {{ color: #888; font-size: 0.9em; }}
                .news-summary {{ margin-top: 8px; }}
                .sentiment-badge {{ 
                    display: inline-block; 
                    border-radius: 4px; 
                    padding: 2px 6px; 
                    font-size: 0.8em; 
                    margin-left: 8px;
                }}
                .sentiment-positive {{ background-color: rgba(76, 175, 80, 0.2); color: #2E7D32; }}
                .sentiment-negative {{ background-color: rgba(244, 67, 54, 0.2); color: #C62828; }}
                .sentiment-neutral {{ background-color: rgba(158, 158, 158, 0.2); color: #616161; }}
                @media print {{
                    .chart-section {{ page-break-before: always; }}
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Stock Analysis Report</h1>
                <p>Generated on: {timestamp}</p>
            </div>
            
            <h2>Metrics Overview</h2>
        """

    def _create_metrics_section(self, metrics):
        """Create the metrics table section of the report

        Args:
            metrics (dict): Dictionary of metrics data

        Returns:
            str: HTML for the metrics section
        """
        html_content = ""

        if metrics and "company_labels" in metrics:
            companies = metrics["company_labels"]
            html_content += '<table class="metrics-table"><tr><th>Company</th>'

            # Create a DataFrame to make it easier to work with metrics
            metrics_df = pd.DataFrame()
            metrics_df["Company"] = companies

            # Add metrics columns that we want to include in the report
            columns_to_include = [
                "currentPrice",
                "marketCap",
                "recommendationMean",
                "dividendYield",
                "targetMeanPrice",
                "beta",
                "trailingPE",
                "forwardPE",
            ]

            column_display_names = {
                "currentPrice": "Current Price",
                "marketCap": "Market Cap",
                "recommendationMean": "Recommendation",
                "dividendYield": "Dividend Yield",
                "targetMeanPrice": "Target Price",
                "beta": "Beta",
                "trailingPE": "Trailing P/E",
                "forwardPE": "Forward P/E",
            }

            # Add columns to the table header
            for col in columns_to_include:
                if col in metrics:
                    display_name = column_display_names.get(col, col)
                    html_content += f"<th>{display_name}</th>"
                    # Add data to DataFrame
                    if len(metrics[col]) == len(companies):
                        metrics_df[col] = metrics[col]

            html_content += "</tr>"

            # Add data rows
            for idx, row in metrics_df.iterrows():
                html_content += "<tr>"
                html_content += f'<td>{row["Company"]}</td>'

                for col in columns_to_include:
                    if col in metrics_df.columns:
                        value = row[col]
                        # Format values appropriately
                        if col == "marketCap" and isinstance(value, (int, float)):
                            if value >= 1e9:
                                formatted_value = f"${value/1e9:.2f}B"
                            elif value >= 1e6:
                                formatted_value = f"${value/1e6:.2f}M"
                            else:
                                formatted_value = f"${value:,.0f}"
                        elif col == "dividendYield" and isinstance(value, (int, float)):
                            formatted_value = f"{value:.2%}" if value else "N/A"
                        elif col == "recommendationMean" and isinstance(
                            value, (int, float)
                        ):
                            # Add a color-coded recommendation value
                            if value < 2.0:
                                formatted_value = f"<span style='color:#4CAF50;font-weight:bold;'>{value:.2f} (Buy)</span>"
                            elif value < 3.0:
                                formatted_value = f"<span style='color:#2196F3;font-weight:bold;'>{value:.2f} (Overweight)</span>"
                            elif value < 4.0:
                                formatted_value = f"<span style='color:#9E9E9E;'>{value:.2f} (Hold)</span>"
                            else:
                                formatted_value = f"<span style='color:#F44336;'>{value:.2f} (Sell)</span>"
                        elif isinstance(value, (int, float)):
                            formatted_value = f"{value:.2f}"
                        else:
                            formatted_value = str(value) if value is not None else "N/A"

                        html_content += f"<td>{formatted_value}</td>"

                html_content += "</tr>"

            html_content += "</table>"

            # Add market data overview section
            html_content += """
            <h2>Market Analysis</h2>
            <div class="market-data">
                <p>This section provides key market indicators and financial metrics for the analyzed companies.</p>
                <table>
                    <tr>
                        <th>Indicator</th>
                        <th>Description</th>
                    </tr>
                    <tr>
                        <td>POC (Point of Control)</td>
                        <td>The price level with the highest traded volume, representing a significant support/resistance level.</td>
                    </tr>
                    <tr>
                        <td>Value Area</td>
                        <td>The range between VA High and VA Low where 70% of trading occurred, representing fair value range.</td>
                    </tr>
                    <tr>
                        <td>Cash Flow</td>
                        <td>Indicates the company's ability to generate cash and fund operations. Positive trends suggest financial health.</td>
                    </tr>
                    <tr>
                        <td>Debt to Equity</td>
                        <td>Measures financial leverage. Lower ratios typically indicate stronger financial positions.</td>
                    </tr>
                </table>
            </div>
            """

        return html_content

    def _create_charts_section(self, charts_data):
        """Create the charts section of the report

        Args:
            charts_data (dict): Dictionary of chart data by ticker symbol

        Returns:
            str: HTML for the charts section
        """
        html_content = ""

        if charts_data and len(charts_data) > 0:
            html_content += (
                '<div class="chart-section"><h2>Price Charts and Volume Analysis</h2>'
            )
            for symbol, chart_img in charts_data.items():
                html_content += f"""
                <div class="chart-container">
                    <h3>{symbol} - Price and Volume Analysis</h3>
                    <p>The chart below shows price history with point of control (POC) and value area analysis.</p>
                    <img src="data:image/png;base64,{chart_img}" style="width: 100%; max-width: 800px; margin: 0 auto; display: block;">
                    <p><small>POC line (red) indicates the price with highest traded volume. Value Area (blue lines) shows the range where 70% of trading occurred.</small></p>
                </div>
                """
            html_content += "</div>"

        return html_content

    def _create_cashflow_section(self, metrics):
        """Create the cashflow analysis section of the report

        Args:
            metrics (dict): Dictionary of metrics data

        Returns:
            str: HTML for the cashflow section
        """
        html_content = ""

        if (
            metrics
            and "opCashflow" in metrics
            and any(isinstance(item, list) for item in metrics["opCashflow"])
        ):
            html_content += """
            <div class="chart-section">
                <h2>Financial Analysis</h2>
                <p>These charts show key financial metrics over time, including:</p>
                <ul>
                    <li><strong>Operating Cash Flow</strong>: Cash generated from core business operations</li>
                    <li><strong>Free Cash Flow</strong>: Cash that can be distributed to shareholders or reinvested</li>
                    <li><strong>Stock Repurchases</strong>: Company buying back its own shares, potentially increasing shareholder value</li>
                </ul>
                <p>Positive trends in these metrics generally indicate financial strength and effective capital allocation.</p>
            """

            # Add specific financial highlights if data is available
            if "company_labels" in metrics and len(metrics["company_labels"]) > 0:
                companies = metrics["company_labels"]

                # Check for financial metrics to highlight
                if "opCashflow" in metrics and "freeCashflow" in metrics:
                    html_content += "<h3>Financial Highlights</h3><ul>"

                    for i, company in enumerate(companies):
                        if i < len(metrics.get("opCashflow", [])) and i < len(
                            metrics.get("freeCashflow", [])
                        ):
                            opcf = metrics["opCashflow"][i]
                            fcf = metrics["freeCashflow"][i]

                            # Only add if we have valid data
                            if (
                                isinstance(opcf, list)
                                and isinstance(fcf, list)
                                and len(opcf) > 0
                                and len(fcf) > 0
                            ):
                                # Get the trend (positive/negative) for the last year
                                opcf_trend = "positive" if opcf[-1] > 0 else "negative"
                                fcf_trend = "positive" if fcf[-1] > 0 else "negative"

                                html_content += f"""
                                <li><strong>{company}</strong>: 
                                    Operating cash flow is <span style="color: {'green' if opcf_trend == 'positive' else 'red'};">
                                    {opcf_trend}</span> at ${abs(opcf[-1]):.2f}M. 
                                    Free cash flow is <span style="color: {'green' if fcf_trend == 'positive' else 'red'};">
                                    {fcf_trend}</span> at ${abs(fcf[-1]):.2f}M.
                                </li>
                                """

                    html_content += "</ul>"

            html_content += "</div>"

        return html_content

    def _create_news_section(self, news_data):
        """Create the news section of the report

        Args:
            news_data (dict): Dictionary of news data by ticker symbol

        Returns:
            str: HTML for the news section
        """
        html_content = ""

        if news_data and len(news_data) > 0:
            html_content += (
                '<div class="chart-section"><h2>Market News & Sentiment</h2>'
            )

            for symbol, articles in news_data.items():
                if articles and len(articles) > 0:
                    html_content += f"""
                    <h3>{symbol} - Recent News Articles</h3>
                    <div class="news-section">
                    """

                    # Add up to 5 news articles for each symbol
                    for idx, article in enumerate(articles[:5]):
                        # Get properties safely with defaults
                        title = article.get("title", "No title available")
                        published = article.get("published", "Unknown date")
                        url = article.get("url", "#")
                        publisher = article.get("publisher", "Unknown source")
                        summary = article.get("summary", "No summary available")

                        # Get sentiment if available
                        sentiment_class = "neutral"
                        sentiment_badge = ""
                        if "sentiment" in article:
                            sentiment = article["sentiment"]
                            if isinstance(sentiment, dict):
                                score = sentiment.get("score", 0)
                                if score > 0.2:
                                    sentiment_class = "positive"
                                    sentiment_badge = f"""<span class="sentiment-badge sentiment-positive">Positive ({score:.2f})</span>"""
                                elif score < -0.2:
                                    sentiment_class = "negative"
                                    sentiment_badge = f"""<span class="sentiment-badge sentiment-negative">Negative ({score:.2f})</span>"""
                                else:
                                    sentiment_badge = f"""<span class="sentiment-badge sentiment-neutral">Neutral ({score:.2f})</span>"""

                        html_content += f"""
                        <div class="news-container {sentiment_class}">
                            <div class="news-date">{published}</div>
                            <div class="news-title"><a href="{url}" target="_blank">{title}</a>{sentiment_badge}</div>
                            <div class="news-source">Source: {publisher}</div>
                            <div class="news-summary">{summary}</div>
                        </div>
                        """

                    html_content += "</div>"

            html_content += "</div>"

        return html_content

    def _create_footer(self):
        """Create the footer section of the report

        Returns:
            str: HTML for the footer section
        """
        return """
            <div class="footer">
                <p>This report is generated for informational purposes only and does not constitute investment advice.</p>
                <p>Data may be delayed or subject to errors. Always perform your own research before making investment decisions.</p>
                <p>© Stock Analysis Dashboard | Generated with Streamlit</p>
            </div>
        </body>
        </html>
        """

    def generate_report(
        self, metrics, charts_data=None, news_data=None, sentiment_data=None
    ):
        """Generate an HTML report with metrics, charts and news data

        Args:
            metrics (dict): Dictionary of metrics data
            charts_data (dict, optional): Dictionary of chart data by ticker symbol. Defaults to None.
            news_data (dict, optional): Dictionary of news data by ticker symbol. Defaults to None.
            sentiment_data (dict, optional): Dictionary of sentiment data by ticker symbol. Defaults to None.

        Returns:
            bytes: HTML report as bytes
        """
        info("Generating HTML report")

        # Use instance data if not explicitly provided
        if charts_data is None:
            charts_data = self.charts_data

        if news_data is None:
            news_data = self.news_data

        if sentiment_data is None:
            sentiment_data = self.sentiment_data

        try:
            # Create a temporary HTML file
            with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
                html_file = f.name

            # Format timestamp for the report
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Start building HTML content
            html_content = self._create_html_header(timestamp)

            # Add metrics table
            html_content += self._create_metrics_section(metrics)

            # Add charts section
            html_content += self._create_charts_section(charts_data)

            # Add cashflow analysis section
            html_content += self._create_cashflow_section(metrics)

            # Add news section
            html_content += self._create_news_section(news_data)

            # Add footer
            html_content += self._create_footer()

            # Write the HTML content to the file
            with open(html_file, "w", encoding="utf-8") as f:
                f.write(html_content)

            # Read the HTML file and return its content
            with open(html_file, "rb") as f:
                html_data = f.read()

            # Clean up the temporary file
            os.unlink(html_file)

            info("HTML report generated successfully")
            return html_data

        except Exception as e:
            error(f"Error generating HTML report: {str(e)}")
            error(traceback.format_exc())
            return None

    def create_download_link(self, html_data, filename="stock_analysis_report.html"):
        """Create a download link for the HTML file

        Args:
            html_data (bytes): HTML report as bytes
            filename (str, optional): Filename for download. Defaults to "stock_analysis_report.html".

        Returns:
            str: HTML anchor tag for download
        """
        b64 = base64.b64encode(html_data).decode()
        href = f'<a href="data:text/html;base64,{b64}" download="{filename}">Download HTML Report</a>'
        return href

    def create_monkey_patches(self, chart_generator, sentiment_analyzer):
        """Create monkey patches for chart and news capture

        Args:
            chart_generator: The chart generator instance
            sentiment_analyzer: The sentiment analyzer instance

        Returns:
            tuple: Original methods that were patched
        """
        info("Creating monkey patches for data capture")

        # Patch chart display method
        original_display_market_profile_chart = (
            chart_generator.candlestick_charts._display_market_profile_chart
        )

        def patched_display_market_profile_chart(
            self, ticker_symbol, data, va_high, va_low, poc_price
        ):
            """Patch to capture chart data before displaying it"""
            try:
                info(f"Capturing chart for {ticker_symbol} before display")

                # Create a figure for capturing that matches the actual chart style
                fig = plt.figure(figsize=(12, 9))
                gs = GridSpec(
                    5,
                    5,
                    figure=fig,
                    height_ratios=[3, 3, 3, 1.5, 1.5],
                    width_ratios=[0.12, 0.88, 0.88, 0.88, 1],
                )

                # Set up the layout similar to the actual display
                ax1 = fig.add_subplot(gs[0:3, 1:4])  # Main price chart
                ax_volume = fig.add_subplot(gs[3:5, 1:4], sharex=ax1)  # Volume chart
                ax2 = fig.add_subplot(gs[0:3, 4], sharey=ax1)  # Volume profile

                plt.style.use("dark_background")

                # Plot candlestick chart with a cleaner style
                mc = mpf.make_marketcolors(
                    up="#54ff54",  # Green for up days
                    down="#ff5454",  # Red for down days
                    edge="inherit",
                    wick="inherit",
                    volume={"up": "#54ff54", "down": "#ff5454"},
                )
                custom_style = mpf.make_mpf_style(
                    marketcolors=mc, gridstyle=":", y_on_right=False
                )

                # If we have data, plot it
                if not data.empty:
                    mpf.plot(
                        data,
                        type="candle",
                        style=custom_style,
                        ax=ax1,
                        volume=ax_volume,
                        show_nontrading=False,
                    )

                    # Add POC and Value Area lines
                    ax1.axhline(
                        y=poc_price,
                        color="#ff5050",
                        linestyle="dashed",
                        linewidth=2,
                        label="POC",
                    )
                    ax1.axhline(
                        y=va_high,
                        color="#5050ff",
                        linestyle="dashed",
                        linewidth=1.5,
                        label="VA High",
                    )
                    ax1.axhline(
                        y=va_low,
                        color="#5050ff",
                        linestyle="dashed",
                        linewidth=1.5,
                        label="VA Low",
                    )

                    # Add a legend
                    ax1.legend(["POC", "VA High", "VA Low"], loc="upper left")

                # Add title
                fig.suptitle(
                    f"{ticker_symbol} - Price Analysis",
                    fontsize=16,
                    color="white",
                )

                # Capture the chart for the report
                self.capture_chart(ticker_symbol, fig)
                plt.close(fig)

                # Now call the original method to display in Streamlit
                return original_display_market_profile_chart(
                    ticker_symbol, data, va_high, va_low, poc_price
                )

            except Exception as e:
                error(f"Error in chart capture: {str(e)}")
                error(traceback.format_exc())
                # Still try to display the original chart if capture fails
                return original_display_market_profile_chart(
                    ticker_symbol, data, va_high, va_low, poc_price
                )

        # Bind the method to this instance
        patched_display_market_profile_chart = (
            patched_display_market_profile_chart.__get__(self)
        )

        # Apply the patch
        chart_generator.candlestick_charts._display_market_profile_chart = (
            patched_display_market_profile_chart
        )

        # Patch news display method
        original_display_news_without_sentiment = (
            sentiment_analyzer.display_news_without_sentiment
        )

        def patched_display_news_without_sentiment(self, ticker_symbol):
            """Patch to capture news data before displaying it"""
            try:
                info(f"Getting news for {ticker_symbol} for capture")

                # Get news data
                news_items = sentiment_analyzer.get_news_articles(ticker_symbol)

                # Capture the news for the report
                if news_items:
                    # If sentiment analyzer has a sentiment analysis method, use it
                    enhanced_items = []
                    for item in news_items:
                        enhanced_item = item.copy()

                        # Try to get sentiment if it's not already included
                        if "sentiment" not in enhanced_item and hasattr(
                            sentiment_analyzer, "analyze_sentiment"
                        ):
                            try:
                                sentiment = sentiment_analyzer.analyze_sentiment(
                                    item.get("title", "")
                                    + " "
                                    + item.get("summary", "")
                                )
                                enhanced_item["sentiment"] = sentiment
                            except:
                                # If sentiment analysis fails, continue without it
                                pass

                        enhanced_items.append(enhanced_item)

                    self.capture_news(ticker_symbol, enhanced_items)

                # Call the original method to display the news in Streamlit
                return original_display_news_without_sentiment(ticker_symbol)

            except Exception as e:
                error(f"Error in news capture for {ticker_symbol}: {str(e)}")
                error(traceback.format_exc())
                # Still try to display the original news if capture fails
                return original_display_news_without_sentiment(ticker_symbol)

        # Bind the method to this instance
        patched_display_news_without_sentiment = (
            patched_display_news_without_sentiment.__get__(self)
        )

        # Apply the patch
        sentiment_analyzer.display_news_without_sentiment = (
            patched_display_news_without_sentiment
        )

        return (
            original_display_market_profile_chart,
            original_display_news_without_sentiment,
        )
