import concurrent.futures
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from market_profile import MarketProfile
from newspaper import Article
from textblob import TextBlob
import mplfinance as mpf
import textwrap
import logging
import re
import time
import random
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_date_range(days_back):
    """Helper function to compute start and end date strings."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")


def populate_metrics(ticker):
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
            # Avoid using st.error in threads
            logging.error(f"Failed to process ticker {ticker.ticker}: {e}")
            return {
                "error": str(e),
                "symbol": ticker.ticker if hasattr(ticker, "ticker") else "unknown",
            }
    else:
        # Avoid using st.write in threads
        logging.warning(f"Skipped ticker due to missing info or invalid object")
        return {
            "error": "Missing info or invalid ticker object",
            "symbol": ticker.ticker if hasattr(ticker, "ticker") else "unknown",
        }


def worker(company):
    """Worker function to fetch ticker data"""
    user_agents = [
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

    try:
        user_agent = random.choice(user_agents)

        session = requests.Session()
        session.headers["User-Agent"] = user_agent

        ticker = yf.Ticker(company)
        metrics = populate_metrics(ticker)

        return metrics
    except Exception as e:
        return None


@st.cache_data(show_spinner="Fetching data from API...", persist=True)
def fetch_metrics_data(companies):
    max_workers = 3
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(worker, companies))

    return results


def classify_by_industry(tickers):
    industries = {}
    for ticker in tickers.tickers.values():
        sector = ticker.info.get("sector")
        if sector:
            industries.setdefault(sector, []).append(ticker.ticker)
    return industries


def fetch_industries(companies):
    tickers = yf.Tickers(" ".join(companies))
    industries = classify_by_industry(tickers)
    return industries


def fetch_recommendations_summary(ticker_symbol):
    ticker = yf.Ticker(ticker_symbol)
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


def populate_additional_metrics(ticker_symbol, metrics):
    ticker = yf.Ticker(ticker_symbol)
    if not hasattr(ticker, "info") or not hasattr(ticker, "cashflow"):
        raise AttributeError(
            "The ticker object must have 'info' and 'cashflow' attributes"
        )

    try:
        recommendations_summary = fetch_recommendations_summary(ticker_symbol)
        metrics["recommendations_summary"].append(recommendations_summary)
    except Exception as e:
        metrics["recommendations_summary"].append(None)

    fields_to_add = {
        "freeCashflow": None,
        "opCashflow": None,
        "repurchaseCapStock": None,
    }

    get_cash_flows(ticker_symbol, fields_to_add, metrics)


@st.cache_data(show_spinner="Fetching cashflow from API...", persist=True)
def get_cash_flows(ticker_symbol, fields_to_add, metrics):
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
                print(f"{ticker.ticker} Failed to process {key}: {e}")
                metrics[key].append(None)


def get_ticker_object(symbol):
    if not isinstance(symbol, str):
        raise ValueError("Symbol must be a string.")

    ticker = yf.Ticker(symbol)
    return ticker


@st.cache_data(show_spinner="Fetching additional data from API...", persist=True)
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
            populate_additional_metrics(tickers.tickers[company].ticker, metrics)

        except KeyError:
            print(f"Warning: Ticker {company} not found. Skipping.")

    return metrics


def build_combined_metrics(company_symbols, metrics, metrics_filtered):
    combined_metrics = {}

    # Add company symbols
    combined_metrics["company_labels"] = company_symbols

    # Add other metrics from the metrics dictionary
    for key, value in metrics.items():
        if key != "company_labels":  # Skip this as we've already added it
            combined_metrics[key] = value

    # Add metrics from metrics_filtered
    for key, value in metrics_filtered.items():
        if key in combined_metrics:
            # Check if lengths match before combining
            if len(value) != len(company_symbols):
                # Handle length mismatch by padding with None values
                print(
                    f"Warning: Length mismatch in combined metrics for key: {key}. Padding with None values."
                )
                # If the filtered data is shorter, extend it
                if len(value) < len(company_symbols):
                    value = value + [None] * (len(company_symbols) - len(value))
                # If the filtered data is longer, truncate it
                else:
                    value = value[: len(company_symbols)]
        combined_metrics[key] = value

    return combined_metrics


@st.cache_data(show_spinner="Fetching historical data from API...", persist=True)
def fetch_historical_data(
    ticker_symbol, start_date, end_date, period=None, interval="3mo"
):
    ticker = yf.Ticker(ticker_symbol)
    try:
        if period:
            data = ticker.history(period=period, interval=interval)
        else:
            data = ticker.history(start=start_date, end=end_date, interval=interval)
        return data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()


def plot_sector_distribution_interactive(industries, title):
    sector_counts = {sector: len(tickers) for sector, tickers in industries.items()}

    labels = list(sector_counts.keys())
    sizes = list(sector_counts.values())

    fig = go.Figure(data=[go.Pie(labels=labels, values=sizes, hole=0.3)])

    fig.update_layout(
        title_text=title,
        annotations=[
            dict(text="Sectors", x=0.50, y=0.5, font_size=20, showarrow=False)
        ],
    )

    return fig


def plot_combined_interactive(combined_metrics):
    if not combined_metrics or not isinstance(combined_metrics, dict):
        raise ValueError("combined_metrics must be a non-empty dictionary.")

    company_labels = combined_metrics.get("company_labels", [])
    eps_values = combined_metrics.get("eps_values", [])

    if not all(len(lst) == len(company_labels) for lst in [eps_values]):
        raise ValueError("Inconsistent data lengths found in combined_metrics.")

    high_diffs = [
        combined_metrics["price_diff"].get(company, {}).get("high_diff", 0)
        for company in company_labels
    ]
    low_diffs = [
        combined_metrics["price_diff"].get(company, {}).get("low_diff", 0)
        for company in company_labels
    ]
    market_caps = combined_metrics.get("market_caps", [])
    priceToBook = combined_metrics.get("priceToBook", [])
    pe_values = combined_metrics.get("pe_values", [])
    peg_values = combined_metrics.get("peg_values", [])
    priceToSalesTrailing12Months = combined_metrics.get(
        "priceToSalesTrailing12Months", []
    )
    gross_margins = combined_metrics.get("gross_margins", [])
    recommendations_summary = combined_metrics.get("recommendations_summary", [])
    earningsGrowth = combined_metrics.get("earningsGrowth", [])
    revenueGrowth = combined_metrics.get("revenueGrowth", [])
    freeCashflow = combined_metrics.get("freeCashflow", [])
    opCashflow = combined_metrics.get("opCashflow", [])
    repurchaseCapStock = combined_metrics.get("repurchaseCapStock", [])

    peg_min, peg_max = min(peg_values, default=0), max(peg_values, default=1)

    fig = make_subplots(
        rows=4,
        cols=3,
        subplot_titles=(
            "Price Difference % Over the Last Year",
            "EPS vs P/E Ratio",
            "Gross Margin (%)",
            "EPS vs P/B Ratio",
            "EPS vs PEG Ratio",
            "EPS vs P/S Ratio",
            "Upgrades & Downgrades Timeline",
            "Earnings Growth vs Revenue Growth",
            "Free Cash Flow",
            "Operational Cashflow",
            "Repurchase of Capital Stock",
        ),
        specs=[[{}, {}, {}], [{}, {}, {}], [{"colspan": 2}, None, {}], [{}, {}, {}]],
        vertical_spacing=0.1,
    )

    colors = {
        company: f"hsl({(i / len(company_labels) * 360)},100%,50%)"
        for i, company in enumerate(company_labels)
    }

    for i, company in enumerate(company_labels):
        try:
            legendgroup = f"group_{company}"
            marker_size = max(market_caps[i] / max(market_caps, default=1) * 50, 5)

            fig.add_trace(
                go.Scatter(
                    x=[high_diffs[i]],
                    y=[low_diffs[i]],
                    marker=dict(size=10, color=colors[company]),
                    legendgroup=legendgroup,
                    name=company,
                    hoverinfo="none",
                    hovertemplate=f"Company: {company}<br>High Diff: %{{x}}<br>Low Diff: %{{y}}<extra></extra>",
                ),
                row=1,
                col=1,
            )

            fig.add_trace(
                go.Scatter(
                    x=[eps_values[i]],
                    y=[pe_values[i]],
                    marker=dict(size=marker_size, color=colors[company]),
                    legendgroup=legendgroup,
                    showlegend=False,
                    hoverinfo="none",
                    hovertemplate=f"Company: {company}<br>EPS: ${eps_values[i]}<br>P/E Ratio: {pe_values[i]}<extra></extra>",
                ),
                row=1,
                col=2,
            )

            fig.add_trace(
                go.Bar(
                    x=[company_labels[i]],
                    y=[gross_margins[i] * 100],
                    marker=dict(color=colors[company]),
                    legendgroup=legendgroup,
                    showlegend=False,
                    width=0.8,
                ),
                row=1,
                col=3,
            )

            fig.add_trace(
                go.Scatter(
                    x=[eps_values[i]],
                    y=[priceToBook[i]],
                    marker=dict(size=marker_size, color=colors[company]),
                    legendgroup=legendgroup,
                    showlegend=False,
                    hoverinfo="none",
                    hovertemplate=f"Company: {company}<br>EPS: ${eps_values[i]}<br>P/B Ratio: {priceToBook[i]}<extra></extra>",
                ),
                row=2,
                col=1,
            )

            fig.add_trace(
                go.Scatter(
                    x=[eps_values[i]],
                    y=[peg_values[i]],
                    marker=dict(size=marker_size, color=colors[company]),
                    legendgroup=legendgroup,
                    showlegend=False,
                    hoverinfo="none",
                    hovertemplate=f"Company: {company}<br>EPS: ${eps_values[i]}<br>PEG Ratio: {peg_values[i]}<extra></extra>",
                ),
                row=2,
                col=2,
            )

            fig.add_trace(
                go.Scatter(
                    x=[eps_values[i]],
                    y=[priceToSalesTrailing12Months[i]],
                    marker=dict(size=marker_size, color=colors[company]),
                    legendgroup=legendgroup,
                    showlegend=False,
                    hoverinfo="none",
                    hovertemplate=f"Company: {company}<br>EPS: ${eps_values[i]}<br>P/S Ratio: {priceToSalesTrailing12Months[i]}<extra></extra>",
                ),
                row=2,
                col=3,
            )

            current_recommendations = recommendations_summary[i]

            if (
                isinstance(current_recommendations, dict)
                and "0m" in current_recommendations
            ):
                ratings = current_recommendations["0m"]
                rating_categories = ["strongBuy", "buy", "hold", "sell", "strongSell"]
                rating_values = [
                    ratings.get(category, 0) for category in rating_categories
                ]

                bar_heights = rating_values

                fig.add_trace(
                    go.Bar(
                        x=rating_categories,
                        y=bar_heights,
                        marker=dict(color=colors[company]),
                        name=company,
                        legendgroup=legendgroup,
                        showlegend=False,
                        text=company,
                        hoverinfo="y+text",
                    ),
                    row=3,
                    col=1,
                )

                fig.update_yaxes(range=[0, max(peg_values)], row=2, col=2)

                for row in range(1, 3):
                    for col in range(1, 3):
                        fig.update_yaxes(range=[0, "auto"], row=row, col=col)

            else:
                continue

            now = datetime.now()
            year = now.year

            if now.month < 4:
                year -= 1

            years = [str(year - i) for i in range(3, -1, -1)]

            if isinstance(freeCashflow[i], list):
                fig.add_trace(
                    go.Scatter(
                        x=years[: len(freeCashflow[i])],
                        y=[cf for cf in freeCashflow[i]],
                        name=company_labels[i],
                        hoverinfo="none",
                        legendgroup=legendgroup,
                        showlegend=False,
                        hovertemplate=f"Company: {company}<br>Year: %{{x}}<br> Free Cashflow: %{{y}}<extra></extra>",
                    ),
                    row=4,
                    col=3,
                )
            else:
                continue

            fig.add_trace(
                go.Scatter(
                    x=[revenueGrowth[i]],
                    y=[earningsGrowth[i]],
                    marker=dict(size=10, color=colors[company]),
                    legendgroup=legendgroup,
                    showlegend=False,
                    hoverinfo="none",
                    hovertemplate=f"Company: {company}<br>Revenue Growth: {revenueGrowth[i]}<br>Earnings Growth: {earningsGrowth[i]}<extra></extra>",
                ),
                row=3,
                col=3,
            )

            if isinstance(opCashflow[i], list):
                fig.add_trace(
                    go.Scatter(
                        x=years[: len(opCashflow[i])],
                        y=[cf for cf in opCashflow[i]],
                        mode="lines",
                        name=company_labels[i],
                        hoverinfo="none",
                        legendgroup=legendgroup,
                        showlegend=False,
                        hovertemplate=f"Company: {company}<br>Year: %{{x}}<br> Operational Cashflow: %{{y}}<extra></extra>",
                    ),
                    row=4,
                    col=1,
                )
            else:
                continue

            if isinstance(repurchaseCapStock[i], list):
                fig.add_trace(
                    go.Scatter(
                        x=years[: len(repurchaseCapStock[i])],
                        y=[-cf for cf in repurchaseCapStock[i]],
                        mode="lines",
                        name=company_labels[i],
                        hoverinfo="none",
                        legendgroup=legendgroup,
                        showlegend=False,
                        hovertemplate=f"Company: {company}<br>Year: %{{x}}<br> Repurchase of Capital Stock: %{{y}}<extra></extra>",
                    ),
                    row=4,
                    col=2,
                )
            else:
                continue

        except (ValueError, TypeError, IndexError) as error:
            print(f"Error plotting data for {company}: {error}")
            continue

    titles = [
        ("High Diff (%)", "Low Diff (%)"),
        ("EPS", "P/E Ratio"),
        ("Company", "Gross Margin (%)"),
        ("Price to Books", "EPS"),
        ("PEG", "EPS"),
        ("P/S", "EPS"),
        ("Earnings Growth", "Revenue Growth"),
        ("Years", "Free Cash Flow"),
        ("Years", "Operational Cashflow"),
        ("Years", "Repurchase of Capital Stock"),
    ]

    for col, (x_title, y_title) in enumerate(titles, start=1):
        fig.update_xaxes(title_text=x_title, row=1, col=col)
        fig.update_yaxes(title_text=y_title, row=1, col=col)

    fig.update_xaxes(title_text="Recommendation Type", row=1, col=4)
    fig.update_yaxes(title_text="Number of Recommendations", row=1, col=4)

    fig.update_layout(height=1500)

    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=list(
                    [
                        dict(
                            args=[{"visible": "legendonly"}],
                            label="Hide All",
                            method="restyle",
                        ),
                        dict(
                            args=[{"visible": True}],
                            label="Show All",
                            method="restyle",
                        ),
                    ]
                ),
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0,
                xanchor="left",
                y=-0.15,
                yanchor="top",
            ),
        ]
    )

    st.plotly_chart(fig)


def get_dash_metrics(ticker_symbol, combined_metrics):
    # Default return values
    default_return = (None,) * 14  # Returns 14 None values

    try:
        # First check if all required keys exist
        required_keys = [
            "company_labels",
            "eps_values",
            "pe_values",
            "priceToSalesTrailing12Months",
            "priceToBook",
            "peg_values",
            "gross_margins",
            "fiftyTwoWeekHigh",
            "fiftyTwoWeekLow",
            "currentPrice",
            "targetMedianPrice",
            "targetLowPrice",
            "targetMeanPrice",
            "targetHighPrice",
            "recommendationMean",
        ]

        # Check if all required keys exist
        for key in required_keys:
            if key not in combined_metrics:
                print(f"Missing key in combined_metrics: '{key}'")
                return default_return

        if ticker_symbol in combined_metrics["company_labels"]:
            index = combined_metrics["company_labels"].index(ticker_symbol)

            # Check if index is valid for all lists
            for key in required_keys[1:]:  # Skip company_labels
                if len(combined_metrics[key]) <= index:
                    print(
                        f"Index {index} out of range for key '{key}' with length {len(combined_metrics[key])}"
                    )
                    return default_return

            eps = combined_metrics["eps_values"][index]
            pe = combined_metrics["pe_values"][index]
            ps = combined_metrics["priceToSalesTrailing12Months"][index]
            pb = combined_metrics["priceToBook"][index]
            peg = combined_metrics["peg_values"][index]
            gm = combined_metrics["gross_margins"][index]
            wh52 = combined_metrics["fiftyTwoWeekHigh"][index]
            wl52 = combined_metrics["fiftyTwoWeekLow"][index]
            currentPrice = combined_metrics["currentPrice"][index]
            targetMedianPrice = combined_metrics["targetMedianPrice"][index]
            targetLowPrice = combined_metrics["targetLowPrice"][index]
            targetMeanPrice = combined_metrics["targetMeanPrice"][index]
            targetHighPrice = combined_metrics["targetHighPrice"][index]
            recommendationMean = combined_metrics["recommendationMean"][index]

            return (
                eps,
                pe,
                ps,
                pb,
                peg,
                gm,
                wh52,
                wl52,
                currentPrice,
                targetMedianPrice,
                targetLowPrice,
                targetMeanPrice,
                targetHighPrice,
                recommendationMean,
            )
        else:
            print(f"Ticker '{ticker_symbol}' not found in the labels list.")
            return default_return
    except Exception as e:
        print(f"An error occurred in get_dash_metrics: {e}")
        return default_return


def format_business_summary(summary):
    summary_no_colons = summary.replace(":", "\:")
    wrapped_summary = textwrap.fill(summary_no_colons)
    return wrapped_summary


def calculate_market_profile(data):
    mp = MarketProfile(data)
    mp_slice = mp[data.index.min() : data.index.max()]

    va_low, va_high = mp_slice.value_area
    poc_price = mp_slice.poc_price
    profile_range = mp_slice.profile_range

    return va_high, va_low, poc_price, profile_range


def fetch_ticker_news_with_retry(ticker_symbol, max_retries=1, delay=2):
    news_data = []
    total_polarity = 0
    attempts = 0

    while attempts < max_retries:
        try:
            dnews = yf.Ticker(ticker_symbol).news
            if dnews:
                news_data = dnews
                break
        except Exception as e:
            logging.error(
                f"Attempt {attempts + 1}: Failed to fetch ticker data or news for '{ticker_symbol}': {e}"
            )
            time.sleep(delay)
            attempts += 1

    if not news_data:
        logging.error(
            f"All {max_retries} attempts failed for ticker symbol '{ticker_symbol}'."
        )

    return news_data, total_polarity


def plot_with_volume_profile(
    ticker_symbol,
    start_date,
    end_date,
    combined_metrics,
    final_shortlist_labels,
    option,
):
    ticker = yf.Ticker(ticker_symbol)
    data = fetch_historical_data(ticker_symbol, start_date, end_date)

    (
        eps,
        pe,
        ps,
        pb,
        peg,
        gm,
        wh52,
        wl52,
        currentPrice,
        targetMedianPrice,
        targetLowPrice,
        targetMeanPrice,
        targetHighPrice,
        recommendationMean,
    ) = get_dash_metrics(ticker_symbol, combined_metrics)

    if not data.empty:
        va_high, va_low, poc_price, _ = calculate_market_profile(data)
        price = ticker.info["currentPrice"]

        if option[0] == "va_high":
            if price > va_high:
                logging.info(
                    f"{ticker_symbol} - current price is above value area: {price} {va_high} {poc_price}"
                )
                return 0
        elif option[0] == "poc_price":
            if price > poc_price:
                logging.info(
                    f"{ticker_symbol} - price is above price of control: {price} {va_high} {poc_price}"
                )
                return 0
        else:
            pass

        website = ticker.info["website"]
        shortName = ticker.info["shortName"]

        header_with_link = f"[🔗]({website}){shortName} - {ticker_symbol}"

        st.markdown(f"### {header_with_link}", unsafe_allow_html=True)

        final_shortlist_labels.append(ticker_symbol)

        col1, col2, col3, col4, col5, col6, col7 = st.columns(7)

        with col1:
            if peg:
                st.metric(label="PEG", value=f"{round(peg,2)}")
            else:
                st.metric(label="PEG", value="-")

        with col2:
            if eps:
                st.metric(label="EPS", value=f"{round(eps,2)}")
            else:
                st.metric(label="EPS", value="-")

        with col3:
            if pe:
                st.metric(label="P/E", value=f"{round(pe,2)}")
            else:
                st.metric(label="P/E", value="-")

        with col4:
            if ps:
                st.metric(label="P/S", value=f"{round(ps,2)}")
            else:
                st.metric(label="P/S", value="-")

        with col5:
            if pb:
                st.metric(label="P/B", value=f"{round(pb,2)}")
            else:
                st.metric(label="P/B", value="-")

        with col6:
            if "marketCap" in ticker.info:
                market_cap = ticker.info["marketCap"]
                if market_cap >= 1e9:
                    market_cap_display = f"{market_cap / 1e9:.2f} B"
                elif market_cap >= 1e6:
                    market_cap_display = f"{market_cap / 1e6:.2f} M"
                else:
                    market_cap_display = f"{market_cap:.2f}"
                st.metric(
                    label=f"Market Cap ({ticker.info['financialCurrency']})",
                    value=market_cap_display,
                )
            else:
                st.metric(label="Market Cap", value="-")

        with col7:
            if pb:
                st.metric(label="Gross Margin", value=f"{round(gm*100,1)}%")
            else:
                st.metric(label="Gross Margin", value="-")

        summary_text = ticker.info["longBusinessSummary"]

        formatted_summary = format_business_summary(summary_text)

        with st.container():
            st.markdown(formatted_summary)

        news_data, total_polarity = get_news_data(ticker_symbol)

        col1_weight, col2_weight, col3_weight = 1, 2, 1
        col1, col2, col3 = st.columns([col1_weight, col2_weight, col3_weight])

        with col1:
            if len(news_data) > 0:
                average_sentiment = total_polarity / len(news_data)

                if average_sentiment >= 0.5:
                    color = "green"
                elif average_sentiment >= 0:
                    color = "orange"
                else:
                    color = "red"

                fig = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=average_sentiment,
                        domain={"x": [0, 1], "y": [0, 1]},
                        gauge={"axis": {"range": [-1, 1]}, "bar": {"color": color}},
                    )
                )

                fig.update_layout(width=300, height=300)

                st.plotly_chart(fig)

            else:
                logging.info(f"No news data available for {ticker_symbol}.")
                st.write("No sentiment or news data available.")

        with col2:
            display_news_articles(news_data)
        with col3:
            poc_line = pd.Series(poc_price, index=data.index)
            va_high_line = pd.Series(va_high, index=data.index)
            va_low_line = pd.Series(va_low, index=data.index)

            apds = [
                mpf.make_addplot(
                    poc_line, type="line", color="red", linestyle="dashed", width=3
                ),
                mpf.make_addplot(
                    va_high_line,
                    type="line",
                    color="blue",
                    linestyle="dashed",
                    width=0.7,
                ),
                mpf.make_addplot(
                    va_low_line,
                    type="line",
                    color="blue",
                    linestyle="dashed",
                    width=0.7,
                ),
            ]

            fig, ax = mpf.plot(
                data,
                type="candle",
                addplot=apds,
                style="yahoo",
                volume=True,
                show_nontrading=False,
                returnfig=True,
            )

            st.pyplot(fig)

    else:
        print(f"No data found for {ticker_symbol} in the given date range.")


def plot_candle_charts_per_symbol(
    industries, start_date, end_date, combined_metrics, final_shortlist_labels, option
):
    logging.info("Started plotting candle charts for each symbol")

    for sector, symbol_list in industries.items():
        logging.info(f"Processing sector: {sector} with symbols: {len(symbol_list)}")

        container = st.container()

        sector_commands = []
        all_skipped = True
        for ticker_symbol in symbol_list:
            logging.debug(
                f"Attempting to plot candle chart for symbol: {ticker_symbol}"
            )

            response = plot_with_volume_profile(
                ticker_symbol,
                start_date,
                end_date,
                combined_metrics,
                final_shortlist_labels,
                option,
            )

            if response == 0:
                logging.warning(
                    f"Skipped plotting for {ticker_symbol} due to no response"
                )
                continue

            all_skipped = False

        if not all_skipped:
            with container.expander(f"Sector: {sector}", expanded=False):
                for cmd in sector_commands:
                    st.write(cmd)

        logging.info("Finished plotting candle charts for all symbols")


@st.cache_data(show_spinner="Fetching news data from API...", persist=True)
def fetch_news(ticker_symbol):
    """Fetch news data for a given ticker symbol."""
    try:
        dnews = yf.Ticker(ticker_symbol).news
        if not dnews:
            logging.warning(f"No news found for ticker '{ticker_symbol}'.")
            return []
        else:
            return dnews
    except Exception as e:
        logging.error(f"Failed to fetch ticker data or news for '{ticker_symbol}': {e}")
        return []


def analyze_news_article(article_info):
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


@st.cache_data(show_spinner="Fetching news data from API...", persist=True)
def get_news_data(ticker_symbol):
    """Fetch and analyze news data and calculate total polarity for a given ticker symbol."""
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
            logging.error(f"Missing required keys in article info: {article_info}")

    return news_data, total_polarity


def display_news_articles(news_data):
    """Display formatted news articles data"""
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


def show_calendar_data(data):
    if data:

        st.write("### Key Dates")
        for key, value in data.items():
            if "Date" in key:
                if isinstance(value, list):
                    dates = ", ".join([date.strftime("%Y-%m-%d") for date in value])
                    st.write(f"**{key}**: {dates}")
                else:
                    st.write(f"**{key}**: {value.strftime('%Y-%m-%d')}")

        st.write("\n### Financial Metrics")
        st.metric(label="Earnings High", value=f"${data['Earnings High']:.2f}")
        st.metric(label="Earnings Low", value=f"${data['Earnings Low']:.2f}")
        st.metric(label="Earnings Average", value=f"${data['Earnings Average']:.2f}")
        revenue_fmt = lambda x: f"${x:,}"
        st.metric(label="Revenue High", value=revenue_fmt(data["Revenue High"]))
        st.metric(label="Revenue Low", value=revenue_fmt(data["Revenue Low"]))
        st.metric(label="Revenue Average", value=revenue_fmt(data["Revenue Average"]))

    else:
        st.write("No calendar data available.")


def filter_companies(
    list_metrics_all_tickers,
    eps_threshold,
    peg_threshold_low,
    peg_threshold_high,
    gross_margin_threshold,
):
    try:
        if not isinstance(list_metrics_all_tickers, list):
            raise ValueError("list_metrics_all_tickers must be a list")

        # Convert list of dictionaries to DataFrame directly
        df = pd.DataFrame(list_metrics_all_tickers)

        # Apply filters using the original column names
        # Note: These criteria might need adjustment based on the actual data
        if not df.empty:
            # Calculate EPS if not directly available
            if "currentPrice" in df.columns and "trailingPE" in df.columns:
                df["calculatedEPS"] = df["currentPrice"] / df["trailingPE"]
                eps_column = "calculatedEPS"
            else:
                eps_column = None

            # Keep track of valid criteria
            valid_criteria = []

            # EPS filter
            if eps_column:
                valid_criteria.append(df[eps_column] > eps_threshold)

            # Profit margins filter (instead of gross_margin)
            if "profitMargins" in df.columns:
                margin_criteria = df["profitMargins"] * 100 > gross_margin_threshold
                valid_criteria.append(margin_criteria)

            # PEG filter - use trailingPE and earningsGrowth
            if "trailingPE" in df.columns and "earningsGrowth" in df.columns:
                df["calculatedPEG"] = df["trailingPE"] / df["earningsGrowth"]
                peg_criteria = (df["calculatedPEG"] > peg_threshold_low) & (
                    df["calculatedPEG"] <= peg_threshold_high
                )
                valid_criteria.append(peg_criteria)

            # Only apply filtering if we have criteria
            if valid_criteria:
                combined_criteria = valid_criteria[0]
                for criterion in valid_criteria[1:]:
                    combined_criteria = combined_criteria & criterion

                filtered_df = df[combined_criteria]

                # Sort by trailingPE if available
                if "trailingPE" in filtered_df.columns:
                    filtered_df_sorted = filtered_df.sort_values(
                        by="trailingPE", ascending=True
                    )
                else:
                    filtered_df_sorted = filtered_df

                print(
                    f"Filtered down to {len(filtered_df_sorted)} companies based on criteria."
                )
                return filtered_df_sorted
            else:
                print("No valid criteria could be applied.")
                return df
        else:
            print("DataFrame is empty.")
            return df

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback

        traceback.print_exc()
        return pd.DataFrame()
