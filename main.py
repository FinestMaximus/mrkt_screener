import streamlit as st

from PIL import Image
import json
import concurrent.futures
import time

import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
from newspaper import Article
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from IPython.display import display, Markdown
from tabulate import tabulate
from textblob import TextBlob
import yfinance as yf

from tqdm import tqdm
from market_profile import MarketProfile
from datetime import datetime, timedelta

st.title("Stock Data Analyzer")

with st.sidebar:
    st.header("Configuration")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        companies = data["ticker"].tolist()
    else:
        companies = []

    days_history = st.number_input(
        "Days History", min_value=365, max_value=3650, value=3650, step=100
    )
    eps_threshold = st.number_input("EPS Threshold", value=1.0)
    gross_margin_threshold = st.number_input("Gross Margin Threshold", value=0.8)
    peg_threshold_low = st.number_input("PEG Lower Threshold", value=0.0)
    peg_threshold_high = st.number_input("PEG Upper Threshold", value=1.5)


def populate_metrics(ticker, metrics):
    if ticker and hasattr(ticker, "info"):
        stock_info = ticker.info
        metrics["eps_values"].append(stock_info.get("trailingEps", 0))
        metrics["pe_values"].append(stock_info.get("trailingPE", 0))
        metrics["peg_values"].append(stock_info.get("pegRatio", 0))
        metrics["gross_margins"].append(stock_info.get("grossMargins", 0))
        metrics["company_labels"].append(ticker.ticker)
    else:
        print(f"Skipped a company ticker due to missing info or an invalid object.")


def worker(company, metrics):
    try:
        ticker = yf.Ticker(company)
        populate_metrics(ticker, metrics)
    except Exception as e:
        print(f"Failed to fetch data for {company}. Error: {e}")


def fetch_metrics_data(companies):
    metrics = {
        metric: []
        for metric in [
            "eps_values",
            "pe_values",
            "peg_values",
            "gross_margins",
            "company_labels",
        ]
    }

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(worker, company, metrics) for company in companies]

        for i, future in enumerate(
            tqdm(
                concurrent.futures.as_completed(futures),
                total=len(companies),
                desc="Fetching metrics",
            )
        ):
            # This loop is primarily to keep tqdm updated, handling of results (if any) would go here

            # Pause for 2 seconds after every 500 requests
            if i != 0 and i % 1000 == 0:
                time.sleep(30)

    return metrics


metrics = fetch_metrics_data(companies)

####################################
######### STEP 1 COMPLETE ##########
####################################


def get_date_range(days_back):
    """Helper function to compute start and end date strings."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")


start_date_str, end_date_str = get_date_range(days_history)

####################################
######### STEP 2 COMPLETE ##########
####################################


def filter_companies(
    metrics,
    eps_threshold,
    peg_threshold_low,
    peg_threshold_high,
    gross_margin_threshold,
):

    try:

        if not isinstance(metrics, dict):
            raise ValueError("metrics must be a dictionary")
        required_keys = [
            "company_labels",
            "eps_values",
            "pe_values",
            "peg_values",
            "gross_margins",
        ]
        for key in required_keys:
            if key not in metrics:
                raise KeyError(f"Key '{key}' not found in metrics")
            if not isinstance(metrics[key], list):
                raise TypeError(f"Value of '{key}' must be a list")
            if not metrics[key]:
                raise ValueError(f"List for '{key}' is empty")

        if not (
            isinstance(eps_threshold, (int, float))
            and isinstance(peg_threshold_low, (int, float))
            and isinstance(peg_threshold_high, (int, float))
            and isinstance(gross_margin_threshold, (int, float))
        ):
            raise TypeError("Thresholds must be numeric")

        if peg_threshold_low >= peg_threshold_high:
            raise ValueError("peg_threshold_low must be less than peg_threshold_high")

        df = pd.DataFrame(
            {
                "company": metrics["company_labels"],
                "eps": metrics["eps_values"],
                "pe": metrics["pe_values"],
                "peg": metrics["peg_values"],
                "gross_margin": metrics["gross_margins"],
            }
        )

        if df["gross_margin"].max() <= 1:
            df["gross_margin"] *= 100

        criteria = (
            (df["eps"] > eps_threshold)
            & (df["gross_margin"] > gross_margin_threshold)
            & (df["peg"] > peg_threshold_low)
            & (df["peg"] <= peg_threshold_high)
        )

        filtered_df = df[criteria]

        filtered_df_sorted = filtered_df.sort_values(by="pe", ascending=True)

        print(
            f"Filtered down to {len(filtered_df_sorted)} companies based on criteria."
        )

        return filtered_df_sorted

    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame()


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


def fetch_recommendations_summary(ticker):
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


def populate_additional_metrics(ticker, metrics):
    if not hasattr(ticker, "info") or not hasattr(ticker, "cashflow"):
        raise AttributeError(
            "The ticker object must have 'info' and 'cashflow' attributes"
        )

    try:
        stock_info = ticker.info
    except Exception as e:
        print(f"Failed to fetch stock info: {e}")
        return metrics

    try:
        recommendations_summary = fetch_recommendations_summary(ticker)
        metrics["recommendations_summary"].append(recommendations_summary)
    except Exception as e:
        print(f"Failed to fetch recommendations summary: {e}")
        metrics["recommendations_summary"].append(None)

    metrics["ps_values"].append(stock_info.get("priceToSalesTrailing12Months", 0))
    metrics["pb_values"].append(stock_info.get("priceToBook", 0))
    metrics["market_caps"].append(stock_info.get("marketCap", 0))

    fields_to_add = {
        "forwardPE": "forwardPE",
        "profitMargins": "profitMargins",
        "heldPercentInsiders": "heldPercentInsiders",
        "heldPercentInstitutions": "heldPercentInstitutions",
        "forwardEps": "forwardEps",
        "recommendationMean": "recommendationMean",
        "recommendationKey": "recommendationKey",
        "numberOfAnalystOpinions": "numberOfAnalystOpinions",
        "totalCashPerShare": "totalCashPerShare",
        "debtToEquity": "debtToEquity",
        "earningsGrowth": "earningsGrowth",
        "revenueGrowth": "revenueGrowth",
        "freeCashflow": None,
        "opCashflow": None,
        "repurchaseCapStock": None,
    }

    try:
        df = ticker.cashflow
    except Exception as e:
        print(f"Failed to fetch cashflow data: {e}")
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
        else:
            metrics[key].append(stock_info.get(value, 0))

    return metrics


def augment_metrics_with_live_data(companies, original_metrics):
    if not isinstance(companies, list) or not all(
        isinstance(item, str) for item in companies
    ):
        raise ValueError("companies must be a list of strings")

    if not isinstance(original_metrics, dict):
        raise ValueError("original_metrics must be a dictionary")

    augmented_data = {metric: [] for metric in original_metrics}

    augmented_data.update(
        {
            "recommendations_summary": [],
            "news": [],
            "ps_values": [],
            "pb_values": [],
            "market_caps": [],
        }
    )

    for company_symbol in companies:
        try:
            ticker = get_ticker_object(company_symbol)

            company_metrics = {metric: [] for metric in augmented_data}

            populate_additional_metrics(ticker, company_metrics)

            for key, values in company_metrics.items():
                if key in augmented_data:
                    augmented_data[key].extend(
                        values if isinstance(values, list) else [values]
                    )
                else:
                    print(
                        f"Unexpected metric {key} found in company_metrics, skipping..."
                    )
        except Exception as e:
            print(f"Error processing company {company_symbol}: {e}")

    return augmented_data


def get_ticker_object(symbol):
    if not isinstance(symbol, str):
        raise ValueError("Symbol must be a string.")

    ticker = yf.Ticker(symbol)
    return ticker


def fetch_additional_metrics_data(companies):
    """Fetches and structures various financial metrics for the given list of company tickers."""
    tickers = yf.Tickers(" ".join(companies))
    metrics = {
        metric: []
        for metric in [
            "ps_values",
            "pb_values",
            "market_caps",
            "recommendations_summary",
        ]
    }
    metrics["price_diff"] = {}

    for company in companies:
        try:
            ticker = tickers.tickers[company]
            populate_additional_metrics(ticker, metrics)
        except KeyError:
            print(f"Warning: Ticker {company} not found. Skipping.")

    return metrics


def build_combined_metrics(filtered_company_symbols, metrics, metrics_filtered):
    if not isinstance(filtered_company_symbols, list):
        raise ValueError("filtered_company_symbols must be a list")
    if not isinstance(metrics, dict):
        raise ValueError("metrics must be a dictionary")
    if not isinstance(metrics_filtered, dict):
        raise ValueError("metrics_filtered must be a dictionary")

    metrics.pop("companies_fetched", None)
    metrics_filtered.pop("companies_fetched", None)

    combined_keys = set(metrics.keys()).union(metrics_filtered.keys()) - {
        "company_labels",
        "companies_fetched",
    }
    combined_metrics = {key: [] for key in combined_keys}

    combined_metrics["company_labels"] = filtered_company_symbols

    for symbol in filtered_company_symbols:
        if "company_labels" in metrics and not isinstance(
            metrics["company_labels"], list
        ):
            raise ValueError("'company_labels' in metrics must be a list")

        metrics_index = (
            metrics["company_labels"].index(symbol)
            if "company_labels" in metrics and symbol in metrics["company_labels"]
            else -1
        )

        for key in combined_metrics:
            if key == "company_labels":
                continue

            if key in metrics and metrics_index >= 0:
                value = (
                    metrics[key][metrics_index]
                    if isinstance(metrics[key], list)
                    and len(metrics[key]) > metrics_index
                    else None
                )
                combined_metrics[key].append(value)
            elif key in metrics_filtered:
                filtered_index = filtered_company_symbols.index(symbol)
                value = (
                    metrics_filtered[key][filtered_index]
                    if isinstance(metrics_filtered[key], list)
                    and len(metrics_filtered[key]) > filtered_index
                    else None
                )
                combined_metrics[key].append(value)
            else:
                combined_metrics[key].append(None)

    expected_length = len(filtered_company_symbols)
    for key, values_list in combined_metrics.items():
        if len(values_list) != expected_length:
            raise ValueError(f"Length mismatch in combined metrics for key: {key}")

    return combined_metrics


####################################
######### STEP 3 COMPLETE ##########
####################################

filtered_companies_df = filter_companies(
    metrics,
    eps_threshold,
    peg_threshold_low,
    peg_threshold_high,
    gross_margin_threshold,
)
print(filtered_companies_df)
filtered_company_symbols = filtered_companies_df["company"].tolist()

metrics_filtered = fetch_additional_metrics_data(filtered_company_symbols)
combined_metrics = build_combined_metrics(
    filtered_company_symbols, metrics, metrics_filtered
)

filtered_industries = fetch_industries(filtered_company_symbols)

####################################
######### STEP 4 COMPLETE ##########
####################################


def fetch_historical_data(ticker, start_date, end_date, period=None, interval="3mo"):
    try:
        if period:
            data = ticker.history(period=period, interval=interval)
        else:
            data = ticker.history(start=start_date, end=end_date, interval=interval)
        return data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()  # Return an empty DataFrame


def calculate_price_diff(companies):
    tickers = yf.Tickers(" ".join(companies))
    price_diff = {}  # Store price difference info here

    for company in companies:
        ticker = tickers.tickers[company]
        try:
            hist = fetch_historical_data(ticker, None, None, period="1y")
            if not hist.empty:
                today_price = hist["Close"].iloc[-1]
                high_52week = max(hist["High"])
                low_52week = min(hist["Low"])
                high_percent_diff = ((today_price - high_52week) / high_52week) * 100
                low_percent_diff = ((today_price - low_52week) / low_52week) * 100
                price_diff[company] = {
                    "high_diff": -1 * high_percent_diff,
                    "low_diff": low_percent_diff,
                }
        except Exception as e:
            print(f"Error processing data for {company}: {e}")

    return price_diff


def fetch_price_diff(companies, combined_metrics):
    try:
        price_diff = calculate_price_diff(companies)

        if price_diff is not None and not price_diff:
            print("No price difference data was found for the provided companies.")
        else:
            combined_metrics["price_diff"] = price_diff
    except Exception as e:
        print(f"An error occurred while fetching price differences: {e}")
    finally:
        return combined_metrics


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
    pb_values = combined_metrics.get("pb_values", [])
    pe_values = combined_metrics.get("pe_values", [])
    peg_values = combined_metrics.get("peg_values", [])
    ps_values = combined_metrics.get("ps_values", [])
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

            # Plot 1: Price Difference
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

            # Plot 2: EPS vs P/E Ratio
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

            # Plot 3: Gross Margin Bar Chart
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

            # Plot 4:  EPS vs P/B Ratio
            fig.add_trace(
                go.Scatter(
                    x=[eps_values[i]],
                    y=[pb_values[i]],
                    marker=dict(size=marker_size, color=colors[company]),
                    legendgroup=legendgroup,
                    showlegend=False,
                    hoverinfo="none",
                    hovertemplate=f"Company: {company}<br>EPS: ${eps_values[i]}<br>P/B Ratio: {pb_values[i]}<extra></extra>",
                ),
                row=2,
                col=1,
            )

            # Plot 5: EPS vs PEG Ratio
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

            # Plot 6: EPS vs P/S Ratio
            fig.add_trace(
                go.Scatter(
                    x=[eps_values[i]],
                    y=[ps_values[i]],
                    marker=dict(size=marker_size, color=colors[company]),
                    legendgroup=legendgroup,
                    showlegend=False,
                    hoverinfo="none",
                    hovertemplate=f"Company: {company}<br>EPS: ${eps_values[i]}<br>P/S Ratio: {ps_values[i]}<extra></extra>",
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

    # Update axes titles
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

    # Layout adjustments for readability and aesthetics
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=list(
                    [
                        dict(
                            args=[
                                {"visible": "legendonly"}
                            ],  # This sets non-selected traces to be hidden.
                            label="Hide All",
                            method="restyle",
                        ),
                        dict(
                            args=[{"visible": True}],  # This makes all traces visible.
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
    # Show the combined plot
    fig.show()


####################################
######### STEP 5 COMPLETE ##########
####################################

combined_metrics = fetch_price_diff(filtered_company_symbols, combined_metrics)
company_labels = list(combined_metrics["company_labels"])
combined_metrics["company_labels"] = company_labels

####################################
######### STEP 6 COMPLETE ##########
####################################


def get_eps_pe_pb_ps_peg(ticker_symbol):
    try:
        if ticker_symbol in combined_metrics["company_labels"]:
            index = combined_metrics["company_labels"].index(ticker_symbol)
            eps = combined_metrics["eps_values"][index]
            pe = combined_metrics["pe_values"][index]
            ps = combined_metrics["ps_values"][index]
            pb = combined_metrics["pb_values"][index]
            peg = combined_metrics["peg_values"][index]

            return eps, pe, ps, pb, peg
        else:
            print(f"Ticker '{ticker_symbol}' not found in the labels list.")
            return None, None, None, None, None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None, None, None


final_shortlist_labels = []


def calculate_market_profile(data):
    mp = MarketProfile(data)
    mp_slice = mp[data.index.min() : data.index.max()]

    va_high, va_low = mp_slice.value_area
    poc_price = mp_slice.poc_price
    profile_range = mp_slice.profile_range

    return va_high, va_low, poc_price, profile_range


def plot_with_volume_profile(ticker_symbol, start_date, end_date):

    ticker = yf.Ticker(ticker_symbol)
    data = fetch_historical_data(ticker, start_date, end_date)

    eps, pe, ps, pb, peg = get_eps_pe_pb_ps_peg(ticker_symbol)

    if not data.empty:
        va_high, va_low, poc_price, _ = calculate_market_profile(data)
        price = ticker.info["currentPrice"]
        if price > va_low:
            return 0

        final_shortlist_labels.append(ticker_symbol)

        display(
            Markdown(
                f"## {ticker_symbol} - {yf.Ticker(ticker_symbol).info['shortName']}"
            )
        )

        # Website
        print(yf.Ticker(ticker_symbol).info["website"])

        # Business Desc
        display(Markdown(f"{yf.Ticker(ticker_symbol).info['longBusinessSummary']}"))

        # Market Cap
        display(
            Markdown(
                f"Market Cap: {yf.Ticker(ticker_symbol).info['marketCap']/1000000} Millions USD"
            )
        )

        poc_line = pd.Series(poc_price, index=data.index)
        va_high_line = pd.Series(va_high, index=data.index)
        va_low_line = pd.Series(va_low, index=data.index)

        apds = [
            mpf.make_addplot(
                poc_line, type="line", color="red", linestyle="dashed", width=3
            ),
            mpf.make_addplot(
                va_high_line, type="line", color="blue", linestyle="dashed", width=0.7
            ),
            mpf.make_addplot(
                va_low_line, type="line", color="blue", linestyle="dashed", width=0.7
            ),
        ]

        title = (
            f"{ticker.info['shortName']}\n\n"
            f" EPS={eps}, P/E={pe}, P/S={ps}, \n P/B={pb}, PEG ratio={peg}\n\n\n"
        )

        mpf.plot(
            data,
            type="candle",
            addplot=apds,
            title=title,
            style="yahoo",
            volume=True,
            show_nontrading=False,
        )
    else:
        print(f"No data found for {ticker_symbol} in the given date range.")


def plot_candle_charts_per_sector(industries, start_date, end_date):
    for sector, symbol_list in industries.items():
        display(Markdown(f"# Sector: {sector}"))
        for symbol in symbol_list:

            response = plot_with_volume_profile(symbol, start_date, end_date)

            if response == 0:
                continue

            symbol_sentiments = []

            # News
            total_polarity = 0  # Initialize total polarity
            try:
                ndata = yf.Ticker(symbol).news
                news_data = []

                for item in ndata:
                    publish_datetime = datetime.fromtimestamp(
                        item["providerPublishTime"]
                    )
                    now = datetime.now()
                    days_ago = (now - publish_datetime).days

                    # Get full article content, not just the title
                    article = Article(item["link"])
                    article.download()
                    article.parse()
                    article_text = article.text

                    # Run sentiment analysis using TextBlob on the article text
                    blob = TextBlob(article_text)
                    polarity = blob.sentiment.polarity
                    total_polarity += polarity  # Add this article's polarity to total

                    news_data.append(
                        {
                            "Title": f"[{item['title']}]({item['link']})",
                            "Publisher": item["publisher"],
                            "Sentiment": polarity,
                            "Days Ago": days_ago,
                        }
                    )

                # Compute average polarity (sentiment) for this symbol
                average_polarity = (
                    total_polarity / len(news_data) if news_data else None
                )
                symbol_sentiments.append(average_polarity)

                # Display the table with news
                table_str = tabulate(
                    news_data, headers="keys", tablefmt="pipe", showindex="always"
                )
                display(Markdown(table_str))

            except Exception as e:
                print(f"An error occurred when trying to fetch data for {symbol}: {e}")

            if symbol_sentiments:
                average_symbol_sentiment = sum(
                    x for x in symbol_sentiments if x is not None
                ) / len(symbol_sentiments)
                display(
                    Markdown(
                        f"Average new sentiment for <span style='color:yellow'>{symbol}</span>: <span style='color:red'>**{average_symbol_sentiment}**</span>"
                    )
                )
            else:
                display(Markdown(f"No sentiment data available for sector {sector}."))

            # Calendar
            data = yf.Ticker(symbol).calendar
            data["Earnings Date"] = [
                date.strftime("%B %d, %Y") for date in data["Earnings Date"]
            ]
            for key, value in data.items():
                if key.startswith("Revenue") and isinstance(value, int):
                    data[key] = "{:,.2f}".format(value)
            table = [[key] + [val] for key, val in data.items()]
            table_str = tabulate(table, headers=["Metric", "Value"], tablefmt="pipe")
            display(Markdown(table_str))

            plt.tight_layout()

    plt.show()


####################################
######### STEP 7 COMPLETE ##########
####################################

plot_candle_charts_per_sector(filtered_industries, start_date_str, end_date_str)

####################################
######### STEP 8 COMPLETE ##########
####################################

indices_to_keep = [
    combined_metrics["company_labels"].index(label)
    for label in final_shortlist_labels
    if label in combined_metrics["company_labels"]
]

st.header("Analysis Results")

filtered_data = {}
for key, values in combined_metrics.items():
    if isinstance(values, list) and len(values) == len(
        combined_metrics["company_labels"]
    ):  # Ensure the list corresponds to 'company_labels' in size.
        # Keep only the elements at the calculated indices.
        filtered_data[key] = [values[i] for i in indices_to_keep]
    else:
        # Copy the value as-is if it doesn't match the list condition.
        filtered_data[key] = values

plot_combined_interactive(filtered_data)

final_industries = fetch_industries(final_shortlist_labels)
plot_sector_distribution_interactive(
    final_industries, "Interactive Ticker Distribution by Sector for Filtered Tickers"
)
