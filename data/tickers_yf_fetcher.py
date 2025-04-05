import yfinance as yf
import pandas as pd
import concurrent.futures
import random
import requests
from datetime import datetime, timedelta
import streamlit as st
from bs4 import BeautifulSoup
import re
from utils.logger import info, debug, success, warning, error, critical


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

        self.METRICS = [
            "short_name",
            "eps_values",
            "pe_values",
            "sector",
            "industry",
            "peg_values",
            "fullTimeEmployees",
            "gross_margins",
            "company_labels",
            "boardRisk",
            "compensationRisk",
            "shareHolderRightsRisk",
            "overallRisk",
            "exDividendDate",
            "dividendYield",
            "dividendRate",
            "priceHint",
            "fiftyTwoWeekLow",
            "forwardPE",
            "marketCap",
            "beta",
            "fiveYearAvgDividendYield",
            "payoutRatio",
            "ebitdaMargins",
            "website",
            "operatingMargins",
            "financialCurrency",
            "trailingPegRatio",
            "fiftyTwoWeekHigh",
            "priceToSalesTrailing12Months",
            "fiftyDayAverage",
            "twoHundredDayAverage",
            "trailingAnnualDividendRate",
            "trailingAnnualDividendYield",
            "currency",
            "enterpriseValue",
            "profitMargins",
            "floatShares",
            "sharesOutstanding",
            "sharesShort",
            "sharesShortPriorMonth",
            "sharesShortPreviousMonthDate",
            "dateShortInterest",
            "sharesPercentSharesOut",
            "heldPercentInsiders",
            "heldPercentInstitutions",
            "shortRatio",
            "shortPercentOfFloat",
            "bookValue",
            "priceToBook",
            "lastFiscalYearEnd",
            "nextFiscalYearEnd",
            "mostRecentQuarter",
            "earningsQuarterlyGrowth",
            "netIncomeToCommon",
            "forwardEps",
            "lastSplitFactor",
            "lastSplitDate",
            "enterpriseToRevenue",
            "enterpriseToEbitda",
            "exchange",
            "quoteType",
            "symbol",
            "underlyingSymbol",
            "longName",
            "firstTradeDateEpochUtc",
            "timeZoneFullName",
            "timeZoneShortName",
            "uuid",
            "gmtOffSetMilliseconds",
            "currentPrice",
            "targetHighPrice",
            "targetLowPrice",
            "targetMeanPrice",
            "targetMedianPrice",
            "recommendationMean",
            "recommendationKey",
            "numberOfAnalystOpinions",
            "totalCash",
            "totalCashPerShare",
            "ebitda",
            "totalDebt",
            "quickRatio",
            "currentRatio",
            "totalRevenue",
            "debtToEquity",
            "revenuePerShare",
            "returnOnAssets",
            "returnOnEquity",
            "freeCashflow",
            "operatingCashflow",
            "earningsGrowth",
            "revenueGrowth",
        ]

    def fetch_ticker_data(_self, symbol):
        """Fetch basic ticker data for a single symbol"""
        try:
            user_agent = random.choice(_self.user_agents)
            session = requests.Session()
            session.headers["User-Agent"] = user_agent

            ticker = yf.Ticker(symbol)
            return ticker
        except Exception as e:
            error(f"Error fetching data for {symbol}: {e}")
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
            error(f"Worker error for {symbol}: {e}")
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
                error(f"Failed to process ticker {ticker.ticker}: {e}")
                return {
                    "error": str(e),
                    "symbol": ticker.ticker if hasattr(ticker, "ticker") else "unknown",
                }
        else:
            warning(f"Skipped ticker due to missing info or invalid object")
            return {
                "error": "Missing info or invalid ticker object",
                "symbol": ticker.ticker if hasattr(ticker, "ticker") else "unknown",
            }

    @st.cache_data(show_spinner="Fetching historical data from API...", persist=True)
    def fetch_historical_data(
        _self, symbol, start_date, end_date, period=None, interval="3mo"
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
            error(f"Error fetching historical data for {symbol}: {e}")
            return pd.DataFrame()

    @st.cache_data(show_spinner="Fetching cashflow from API...", persist=True)
    def get_cash_flows(_self, ticker_symbol, fields_to_add, metrics):
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
                    error(f"{ticker.ticker} Failed to process {key}: {e}")
                    metrics[key].append(None)
            else:
                metrics[key].append(None)

    @st.cache_data(show_spinner="Fetching ticker info...", persist=True)
    def fetch_ticker_info(_self, companies, fetch_func=None):
        """
        Fetch initial metrics data for filtering companies

        Args:
            companies (list): List of company tickers
            fetch_func (callable, optional): Custom function for fetching data with retries
        """
        # Change the return structure to a list of dictionaries (one per company)
        metrics_list = []

        important_metrics_keys = [
            "symbol",
            "longName",
            "sector",
            "industry",
            "marketCap",
            "currentPrice",
            "targetHighPrice",
            "targetLowPrice",
            "targetMeanPrice",
            "targetMedianPrice",
            "recommendationMean",
            "dividendYield",
            "trailingPE",
            "forwardPE",
            "priceToBook",
            "freeCashflow",
            "operatingCashflow",
            "revenueGrowth",
            "grossMargins",
            "returnOnEquity",
        ]

        for company in companies:
            try:
                if fetch_func:
                    info = fetch_func(company)
                else:
                    stock = yf.Ticker(company)
                    info = stock.info

                if info:
                    # Filter to only include important metrics
                    important_metrics = {
                        key: info[key] for key in important_metrics_keys if key in info
                    }
                    important_metrics["company"] = company
                    metrics_list.append(important_metrics)
                else:
                    warning(f"No info returned for {company}")
                    metrics_list.append(
                        {"company": company, "error": "No data returned"}
                    )

            except Exception as e:
                error(f"Error fetching data for {company}: {str(e)}")
                # Add company with error info to maintain list integrity
                metrics_list.append({"company": company, "error": str(e)})
                continue

        return metrics_list
