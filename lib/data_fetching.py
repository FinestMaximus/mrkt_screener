import concurrent.futures
from bs4 import BeautifulSoup
import requests
import yfinance as yf
import streamlit as st
import pandas as pd
from lib.metrics_handling import populate_metrics
import logging
import time
from tqdm import tqdm
import re

logging.getLogger("yfinance").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger().setLevel(logging.INFO)

METRICS = [
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


def fetch_metrics_data_for_initial_filtering(companies, fetch_func=None):
    """
    Fetch initial metrics data for filtering companies

    Args:
        companies (list): List of company tickers
        fetch_func (callable, optional): Custom function for fetching data with retries
    """
    metrics = {
        "company_labels": [],
        "eps": [],
        "peg_ratio": [],
        "gross_margins": [],
        # ... other metrics ...
    }

    for company in companies:
        try:
            if fetch_func:
                info = fetch_func(company)
            else:
                stock = yf.Ticker(company)
                info = stock.info

            if info:
                # Process the data
                metrics["company_labels"].append(company)
                # ... rest of the metrics processing ...

        except Exception as e:
            logging.error(f"Error fetching data for {company}: {str(e)}")
            continue

    return metrics


@st.cache_data(ttl=3600)
def fetch_company_data(company):
    """Fetch data for a single company with caching."""
    try:
        ticker = yf.Ticker(company)
        info = ticker.info
        if not info:
            return {}, False

        company_data = {
            "company_labels": company,
            "short_name": info.get("shortName", company),
            "eps_values": info.get("trailingEps", 0),
            "pe_values": info.get("trailingPE", 0),
            "sector": info.get("sector", "Unknown"),
            "industry": info.get("industry", "Unknown"),
            "peg_values": info.get("pegRatio", 0),
            "gross_margins": info.get("grossMargins", 0),
        }

        for metric in METRICS:
            if metric not in company_data and metric in info:
                company_data[metric] = info[metric]

        return company_data, True
    except Exception:
        return {}, False


@st.cache_data(ttl=3600)
def fetch_historical_data(
    ticker_symbol, start_date=None, end_date=None, period=None, interval="3mo"
):
    """Fetch historical price data with caching."""
    ticker = yf.Ticker(ticker_symbol)
    try:
        if period:
            data = ticker.history(period=period, interval=interval)
        else:
            data = ticker.history(start=start_date, end=end_date, interval=interval)
        return data
    except Exception:
        st.error(
            f"Failed to fetch historical data for {ticker_symbol}. Please try again later."
        )
        return pd.DataFrame()


def classify_by_industry(tickers):
    try:
        industries = {}
        for ticker in tickers.tickers.values():
            sector = ticker.info.get("sector")
            if sector:
                industries.setdefault(sector, []).append(ticker.ticker)
        return industries
    except Exception:
        return {}


def fetch_industries(companies):
    try:
        tickers = yf.Tickers(" ".join(companies))
        industries = classify_by_industry(tickers)
        return industries
    except Exception:
        return {}


def fetch_recommendations_summary(ticker_symbol):
    try:
        ticker = yf.Ticker(ticker_symbol)
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
    except Exception:
        return {"error": "Error occurred."}


def get_ticker_object(symbol):
    if not isinstance(symbol, str):
        raise ValueError("Symbol must be a string.")
    ticker = yf.Ticker(symbol)
    return ticker


def fetch_additional_metrics_data(companies):
    tickers = yf.Tickers(" ".join(companies))
    metrics = {
        metric: []
        for metric in [
            "recommendations_summary",
        ]
    }
    for company in companies:
        try:
            from lib.metrics_handling import populate_additional_metrics

            populate_additional_metrics(tickers.tickers[company].ticker, metrics)
        except KeyError:
            pass
    return metrics


def fetch_market_sentiment(url):
    try:
        logging.debug(
            "[main.py][fetch_market_sentiment] Initiating request to fetch market sentiment from URL: %s",
            url,
        )
        response = requests.get(url)
        logging.debug(
            "[main.py][fetch_market_sentiment] Received response with status code: %s",
            response.status_code,
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        logging.debug(
            "[main.py][fetch_market_sentiment] Parsed HTML content successfully."
        )
        selector = (
            "div > div > div:nth-of-type(2) > div:nth-of-type(1) > p:nth-of-type(2)"
        )
        extracted_text = soup.select_one(selector).text
        logging.debug(
            "[main.py][fetch_market_sentiment] Extracted text: %s", extracted_text
        )
        match = re.search(r"\d+%", extracted_text)
        if match:
            percentage_value = float(match.group().strip("%"))
            logging.debug(
                "[main.py][fetch_market_sentiment] Extracted percentage value: %s",
                percentage_value,
            )

            if percentage_value >= 75:
                sentiment = "Extreme Greed"
                color_code = "red"
            elif 60 <= percentage_value < 75:
                sentiment = "Greed 😨"
                color_code = "orange"
            elif 40 <= percentage_value < 60:
                sentiment = "Neutral 😐"
                color_code = "white"
            elif 25 <= percentage_value < 40:
                sentiment = "Fear 😏"
                color_code = "yellow"
            else:
                sentiment = "Extreme Fear"
                color_code = "green"

            logging.debug(
                "[main.py][fetch_market_sentiment] Determined sentiment: %s, with color code: %s",
                sentiment,
                color_code,
            )
            return match.group(), sentiment, color_code
        else:
            logging.info(
                "[main.py][fetch_market_sentiment] No percentage found in the extracted text."
            )
            return None, None, None
    except requests.RequestException as e:
        logging.error(
            "[main.py][fetch_market_sentiment] RequestException occurred - Error: %s",
            e,
        )
        return None, None, None
