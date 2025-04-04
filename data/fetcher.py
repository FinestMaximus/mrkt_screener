import yfinance as yf
import pandas as pd
import concurrent.futures
import random
import requests
import logging
from datetime import datetime, timedelta
import streamlit as st


class DataFetcher:
    """Class responsible for fetching financial data from external APIs"""

    def __init__(self):
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:124.0) Gecko/20100101 Firefox/124.0",
            "Mozilla/5.0 (X11; Linux x86_64; rv:124.0) Gecko/20100101 Firefox/124.0",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 Edg/123.0.0.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3 Safari/605.1.15",
            "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
            "Mozilla/5.0 (Linux; Android 14; SM-S918B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Mobile Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 OPR/109.0.0.0",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 Brave/1.62.153",
            "Mozilla/5.0 (Linux; Android 14; SAMSUNG SM-G998B) AppleWebKit/537.36 (KHTML, like Gecko) SamsungBrowser/23.0 Chrome/115.0.0.0 Mobile Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 Vivaldi/6.5.3232.45",
            "Mozilla/5.0 (Linux; U; Android 13; en-US; Pixel 7) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/78.0.3904.108 UCBrowser/13.4.0.1306 Mobile Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 YaBrowser/24.2.0 Yowser/2.5 Safari/537.36",
            "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 DuckDuckGo/7 Safari/605.1.15",
        ]

    def get_date_range(self, days_back):
        """Helper function to compute start and end date strings."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")

    @st.cache_data(show_spinner="Fetching data from API...", persist=True)
    def fetch_ticker_data(self, symbol):
        """Fetch basic ticker data for a single symbol"""
        try:
            user_agent = random.choice(self.user_agents)
            session = requests.Session()
            session.headers["User-Agent"] = user_agent

            ticker = yf.Ticker(symbol)
            return ticker
        except Exception as e:
            logging.error(f"Error fetching data for {symbol}: {e}")
            return None

    @st.cache_data(show_spinner="Fetching data from API...", persist=True)
    def fetch_multiple_tickers(self, symbols, max_workers=3):
        """Fetch ticker data for multiple symbols concurrently"""
        results = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(self._worker, symbols))

        return results

    def _worker(self, symbol):
        """Worker function for concurrent ticker fetching"""
        try:
            user_agent = random.choice(self.user_agents)
            session = requests.Session()
            session.headers["User-Agent"] = user_agent

            ticker = yf.Ticker(symbol)
            metrics = self._populate_metrics(ticker)

            return metrics
        except Exception as e:
            logging.error(f"Worker error for {symbol}: {e}")
            return None

    def _populate_metrics(self, ticker):
        """Extract relevant metrics from ticker object"""
        metrics = {}
        if ticker and hasattr(ticker, "info"):
            try:
                stock_info = ticker.info

                # Initialize metrics dictionary with relevant fields
                important_fields = [
                    "symbol",
                    "longName",
                    "sector",
                    "industry",
                    "marketCap",
                    "currentPrice",
                    "trailingPE",
                    "forwardPE",
                    "priceToBook",
                    "dividendYield",
                    "beta",
                    "fiftyTwoWeekLow",
                    "fiftyTwoWeekHigh",
                    "fiftyDayAverage",
                    "twoHundredDayAverage",
                    "shortRatio",
                    "profitMargins",
                    "operatingMargins",
                    "returnOnAssets",
                    "returnOnEquity",
                    "revenueGrowth",
                    "earningsGrowth",
                    "totalCash",
                    "totalDebt",
                    "debtToEquity",
                    "currentRatio",
                    "bookValue",
                    "priceToSalesTrailing12Months",
                    "targetMeanPrice",
                    "recommendationKey",
                    "averageAnalystRating",
                    "trailingAnnualDividendYield",
                ]

                # Populate metrics with available data
                for field in important_fields:
                    if field in stock_info:
                        metrics[field] = stock_info[field]
                    else:
                        metrics[field] = None

                return metrics

            except Exception as e:
                logging.error(f"Failed to process ticker {ticker.ticker}: {e}")
                return {
                    "error": str(e),
                    "symbol": ticker.ticker if hasattr(ticker, "ticker") else "unknown",
                }
        else:
            logging.warning(f"Skipped ticker due to missing info or invalid object")
            return {
                "error": "Missing info or invalid ticker object",
                "symbol": ticker.ticker if hasattr(ticker, "ticker") else "unknown",
            }

    @st.cache_data(show_spinner="Fetching historical data from API...", persist=True)
    def fetch_historical_data(
        self, symbol, start_date, end_date, period=None, interval="3mo"
    ):
        """Fetch historical price data for a ticker"""
        ticker = yf.Ticker(symbol)
        try:
            if period:
                data = ticker.history(period=period, interval=interval)
            else:
                data = ticker.history(start=start_date, end=end_date, interval=interval)
            return data
        except Exception as e:
            logging.error(f"Error fetching historical data for {symbol}: {e}")
            return pd.DataFrame()

    def classify_by_industry(self, tickers):
        """Classify tickers by industry/sector"""
        industries = {}
        for ticker in tickers.tickers.values():
            sector = ticker.info.get("sector")
            if sector:
                industries.setdefault(sector, []).append(ticker.ticker)
        return industries

    def fetch_industries(self, companies):
        """Fetch and classify industries for a list of companies"""
        tickers = yf.Tickers(" ".join(companies))
        industries = self.classify_by_industry(tickers)
        return industries

    @st.cache_data(show_spinner="Fetching recommendations from API...", persist=True)
    def fetch_recommendations_summary(self, symbol):
        """Fetch analyst recommendations summary for a ticker"""
        ticker = yf.Ticker(symbol)
        try:
            rec_data = ticker.get_recommendations_summary()

            if not rec_data.empty:
                return {
                    row["period"]: {
                        "strongBuy": row["strongBuy"],
                        "buy": row["buy"],
                        "hold": row["hold"],
                        "sell": row["sell"],
                        "strongSell": row["strongSell"],
                    }
                    for index, row in rec_data.iterrows()
                }
            else:
                return {"message": "No recommendation data available."}
        except Exception as e:
            return {"error": f"Error: {str(e)}"}

    @st.cache_data(show_spinner="Fetching additional data from API...", persist=True)
    def fetch_additional_metrics_data(self, companies):
        """Fetch additional metrics data for multiple companies"""
        tickers = yf.Tickers(" ".join(companies))
        metrics = {
            metric: []
            for metric in [
                "recommendations_summary",
            ]
        }

        for company in companies:
            try:
                self.populate_additional_metrics(
                    tickers.tickers[company].ticker, metrics
                )
            except KeyError:
                logging.warning(f"Ticker {company} not found. Skipping.")

        return metrics

    def populate_additional_metrics(self, ticker_symbol, metrics):
        """Populate additional metrics for a specific ticker"""
        ticker = yf.Ticker(ticker_symbol)
        if not hasattr(ticker, "info") or not hasattr(ticker, "cashflow"):
            raise AttributeError(
                "The ticker object must have 'info' and 'cashflow' attributes"
            )

        try:
            recommendations_summary = self.fetch_recommendations_summary(ticker_symbol)
            metrics["recommendations_summary"].append(recommendations_summary)
        except Exception as e:
            metrics["recommendations_summary"].append(None)

        fields_to_add = {
            "freeCashflow": None,
            "opCashflow": None,
            "repurchaseCapStock": None,
        }

        self.get_cash_flows(ticker_symbol, fields_to_add, metrics)

    @st.cache_data(show_spinner="Fetching cashflow from API...", persist=True)
    def get_cash_flows(self, ticker_symbol, fields_to_add, metrics):
        """Get cash flow data for a ticker"""
        ticker = yf.Ticker(ticker_symbol)
        try:
            df = ticker.cashflow
        except Exception as e:
            df = None

        for key, value in fields_to_add.items():
            if key not in metrics:
                metrics[key] = []

            if df is not None and key in [
                "freeCashflow",
                "opCashflow",
                "repurchaseCapStock",
            ]:
                try:
                    if key == "freeCashflow":
                        free_cash_flow = df.iloc[0, :].tolist()
                        metrics[key].append(free_cash_flow)
                    elif key == "opCashflow":
                        op_cash_flow = df.iloc[33, :].tolist()
                        metrics[key].append(op_cash_flow)
                    elif key == "repurchaseCapStock":
                        repurchase_capital_stock = df.iloc[1, :].tolist()
                        metrics[key].append(repurchase_capital_stock)
                except Exception as e:
                    logging.error(f"{ticker.ticker} Failed to process {key}: {e}")
                    metrics[key].append(None)
            else:
                metrics[key].append(None)

    @st.cache_data(show_spinner="Fetching news from API...", persist=True)
    def fetch_news(self, ticker_symbol):
        """Fetch news data for a given ticker symbol."""
        try:
            dnews = yf.Ticker(ticker_symbol).news
            if not dnews:
                logging.warning(f"No news found for ticker '{ticker_symbol}'.")
                return []
            else:
                return dnews
        except Exception as e:
            logging.error(
                f"Failed to fetch ticker data or news for '{ticker_symbol}': {e}"
            )
            return []
