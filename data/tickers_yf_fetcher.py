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
import time


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
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 Opera/90.0.0.0",
            "Mozilla/5.0 (Linux; Android 14; SM-G998B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Mobile Safari/537.36 Opera/90.0.0.0",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 Firefox/90.0.0.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 Firefox/90.0.0.0",
            "Mozilla/5.0 (Linux; Android 14; SM-G998B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Mobile Safari/537.36 Firefox/90.0.0.0",
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
            "symbols",
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
            "forwardPegRatio",
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
    def fetch_multiple_tickers(self, symbols, max_workers=5, batch_size=100):
        """Fetch ticker data for multiple symbols concurrently with batching"""
        all_results = []

        # Process in batches to avoid overwhelming the API
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i : i + batch_size]
            info(
                f"Processing batch {i//batch_size + 1}/{(len(symbols) + batch_size - 1)//batch_size} ({len(batch)} symbols)"
            )

            results = []
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers
            ) as executor:
                results = list(executor.map(self._worker, batch))

            # Filter out None results and retry failed ones with delay
            failed_symbols = [
                symbol for symbol, result in zip(batch, results) if result is None
            ]
            if failed_symbols:
                info(f"Retrying {len(failed_symbols)} failed symbols with delay...")
                time.sleep(5)  # Wait before retrying
                retry_results = []
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=2
                ) as retry_executor:
                    retry_results = list(
                        retry_executor.map(self._worker, failed_symbols)
                    )

            # Replace None values with retry results
            for j, symbol in enumerate(batch):
                if results[j] is None and symbol in failed_symbols:
                    idx = failed_symbols.index(symbol)
                    results[j] = retry_results[idx]

            all_results.extend(results)

            # Add delay between batches to avoid rate limiting
            if i + batch_size < len(symbols):
                info(f"Pausing between batches to avoid rate limits...")
                time.sleep(random.uniform(3, 5))

        return all_results

    def _worker(self, symbol):
        """Worker function for concurrent ticker fetching"""
        try:
            user_agent = random.choice(self.user_agents)
            session = requests.Session()
            session.headers["User-Agent"] = user_agent

            # Add random delay to avoid rate limiting
            time.sleep(random.uniform(0.5, 2))

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
                    "dividendYield",  # Make sure this is included
                    "trailingAnnualDividendYield",  # Also include this
                    "fiveYearAvgDividendYield",  # And this
                    # ... other fields ...
                ]

                # Populate metrics with available data
                for field in important_fields:
                    if field in stock_info:
                        # For dividend fields, ensure they're numeric
                        if field in [
                            "dividendYield",
                            "trailingAnnualDividendYield",
                            "fiveYearAvgDividendYield",
                        ]:
                            try:
                                value = (
                                    float(stock_info[field])
                                    if stock_info[field] is not None
                                    else None
                                )
                                metrics[field] = value
                                if (
                                    field == "dividendYield"
                                    and value is not None
                                    and value > 0
                                ):
                                    debug(
                                        f"Found dividend paying stock: {ticker.ticker} with yield {value}"
                                    )
                            except (ValueError, TypeError):
                                metrics[field] = None
                                warning(
                                    f"Could not convert {field} to number for {ticker.ticker}"
                                )
                        else:
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

    @st.cache_data(show_spinner=False, persist=True)
    def _cached_fetch_ticker_info(_self, company):
        """Cached part of fetching ticker info"""
        try:
            stock = yf.Ticker(company)
            return stock.info
        except Exception as e:
            error(f"Error fetching data for {company}: {str(e)}")
            return None

    def fetch_ticker_info(
        self,
        tickers,
        _progress_callback=None,
        max_workers=20,
        batch_size=100,
        use_cache=True,
    ):
        """Fetch ticker information in parallel batches with progress tracking

        Args:
            tickers: List of ticker symbols to fetch
            _progress_callback: Optional callback function to report progress
            max_workers: Number of concurrent workers for parallel processing
            batch_size: Size of each batch of tickers
            use_cache: Whether to assume cache is present and use larger batches
        """
        # Use larger batch sizes when cache is expected to be present
        if use_cache:
            # When cache is present, we can process more tickers at once
            batch_size = max(batch_size, 5000)  # Increase to at least 200
            max_workers = max(max_workers, 30)  # Increase worker count too

        metrics_list = []
        total_companies = len(tickers)
        processed_count = 0

        # Process in batches to avoid overwhelming the API
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i : i + batch_size]
            info(
                f"Processing batch {i//batch_size + 1}/{(len(tickers) + batch_size - 1)//batch_size} ({len(batch)} symbols)"
            )

            # Process batch in parallel
            batch_results = []
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers
            ) as executor:
                # Create a map of future to company for tracking
                future_to_company = {
                    executor.submit(self._cached_fetch_ticker_info, company): company
                    for company in batch
                }

                for future in concurrent.futures.as_completed(future_to_company):
                    company = future_to_company[future]
                    processed_count += 1

                    try:
                        # Get the complete ticker info dictionary
                        ticker_info = future.result()
                        if ticker_info:
                            # Add the company symbol to the info dict
                            ticker_info["symbol"] = company
                            batch_results.append(ticker_info)
                        else:
                            warning(f"No info returned for {company}")
                            batch_results.append(
                                {"symbol": company, "error": "No data returned"}
                            )
                    except Exception as e:
                        error(f"Error processing {company}: {str(e)}")
                        batch_results.append({"symbol": company, "error": str(e)})

                    # Update progress after each ticker is processed
                    if _progress_callback and callable(_progress_callback):
                        _progress_callback(processed_count, total_companies, company)

            # Add batch results to the full metrics list
            metrics_list.extend(batch_results)

            # Add delay between batches to avoid rate limiting
            if i + batch_size < len(tickers):
                info(f"Pausing between batches to avoid rate limits...")
                time.sleep(random.uniform(2, 4))

        return metrics_list
