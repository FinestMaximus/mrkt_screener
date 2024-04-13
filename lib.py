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

def populate_metrics(ticker, metrics):
    if ticker and hasattr(ticker, "info"):
        try:
            stock_info = ticker.info
            metrics["eps_values"].append(stock_info.get("trailingEps", 0))
            metrics["pe_values"].append(stock_info.get("trailingPE", 0))
            metrics["peg_values"].append(stock_info.get("pegRatio", 0))
            metrics["gross_margins"].append(stock_info.get("grossMargins", 0))
            metrics["sector"].append(stock_info.get("sector", ""))
            metrics["short_name"].append(stock_info.get("shortName", ""))
            metrics["fullTimeEmployees"].append(stock_info.get("fullTimeEmployees", ""))
            metrics["boardRisk"].append(stock_info.get("boardRisk", ""))
            metrics["industry"].append(stock_info.get("industry", ""))
            metrics["compensationRisk"].append(stock_info.get("compensationRisk", ""))
            metrics["shareHolderRightsRisk"].append(
                stock_info.get("shareHolderRightsRisk", "")
            )
            metrics["overallRisk"].append(stock_info.get("overallRisk", ""))
            metrics["exDividendDate"].append(stock_info.get("exDividendDate", ""))
            metrics["dividendYield"].append(stock_info.get("dividendYield", ""))
            metrics["dividendRate"].append(stock_info.get("dividendRate", ""))
            metrics["priceHint"].append(stock_info.get("priceHint", ""))
            metrics["fiftyTwoWeekLow"].append(stock_info.get("fiftyTwoWeekLow", ""))
            metrics["forwardPE"].append(stock_info.get("forwardPE", ""))
            metrics["marketCap"].append(stock_info.get("marketCap", ""))
            metrics["beta"].append(stock_info.get("beta", ""))
            metrics["fiveYearAvgDividendYield"].append(
                stock_info.get("fiveYearAvgDividendYield", "")
            )
            metrics["payoutRatio"].append(stock_info.get("payoutRatio", ""))
            metrics["ebitdaMargins"].append(stock_info.get("ebitdaMargins", ""))
            metrics["website"].append(stock_info.get("website", ""))
            metrics["operatingMargins"].append(stock_info.get("operatingMargins", ""))
            metrics["financialCurrency"].append(stock_info.get("financialCurrency", ""))
            metrics["trailingPegRatio"].append(stock_info.get("trailingPegRatio", ""))
            metrics["fiftyTwoWeekHigh"].append(stock_info.get("fiftyTwoWeekHigh", ""))
            metrics["priceToSalesTrailing12Months"].append(
                stock_info.get("priceToSalesTrailing12Months", "")
            )
            metrics["fiftyDayAverage"].append(stock_info.get("fiftyDayAverage", ""))
            metrics["twoHundredDayAverage"].append(
                stock_info.get("twoHundredDayAverage", "")
            )
            metrics["trailingAnnualDividendRate"].append(
                stock_info.get("trailingAnnualDividendRate", "")
            )
            metrics["trailingAnnualDividendYield"].append(
                stock_info.get("trailingAnnualDividendYield", "")
            )
            metrics["currency"].append(stock_info.get("currency", ""))
            metrics["enterpriseValue"].append(stock_info.get("enterpriseValue", ""))
            metrics["profitMargins"].append(stock_info.get("profitMargins", ""))
            metrics["floatShares"].append(stock_info.get("floatShares", ""))
            metrics["sharesOutstanding"].append(stock_info.get("sharesOutstanding", ""))
            metrics["sharesShort"].append(stock_info.get("sharesShort", ""))
            metrics["sharesShortPriorMonth"].append(
                stock_info.get("sharesShortPriorMonth", "")
            )
            metrics["sharesShortPreviousMonthDate"].append(
                stock_info.get("sharesShortPreviousMonthDate", "")
            )
            metrics["dateShortInterest"].append(stock_info.get("dateShortInterest", ""))
            metrics["sharesPercentSharesOut"].append(
                stock_info.get("sharesPercentSharesOut", "")
            )
            metrics["heldPercentInsiders"].append(
                stock_info.get("heldPercentInsiders", "")
            )
            metrics["heldPercentInstitutions"].append(
                stock_info.get("heldPercentInstitutions", "")
            )
            metrics["shortRatio"].append(stock_info.get("shortRatio", ""))
            metrics["shortPercentOfFloat"].append(
                stock_info.get("shortPercentOfFloat", "")
            )
            metrics["bookValue"].append(stock_info.get("bookValue", ""))
            metrics["priceToBook"].append(stock_info.get("priceToBook", ""))
            metrics["lastFiscalYearEnd"].append(stock_info.get("lastFiscalYearEnd", ""))
            metrics["nextFiscalYearEnd"].append(stock_info.get("nextFiscalYearEnd", ""))
            metrics["mostRecentQuarter"].append(stock_info.get("mostRecentQuarter", ""))
            metrics["earningsQuarterlyGrowth"].append(
                stock_info.get("earningsQuarterlyGrowth", "")
            )
            metrics["netIncomeToCommon"].append(stock_info.get("netIncomeToCommon", ""))
            metrics["forwardEps"].append(stock_info.get("forwardEps", ""))
            metrics["lastSplitFactor"].append(stock_info.get("lastSplitFactor", ""))
            metrics["lastSplitDate"].append(stock_info.get("lastSplitDate", ""))
            metrics["enterpriseToRevenue"].append(
                stock_info.get("enterpriseToRevenue", "")
            )
            metrics["enterpriseToEbitda"].append(
                stock_info.get("enterpriseToEbitda", "")
            )
            metrics["exchange"].append(stock_info.get("exchange", ""))
            metrics["quoteType"].append(stock_info.get("quoteType", ""))
            metrics["symbol"].append(stock_info.get("symbol", ""))
            metrics["underlyingSymbol"].append(stock_info.get("underlyingSymbol", ""))
            metrics["shortName"].append(stock_info.get("shortName", ""))
            metrics["longName"].append(stock_info.get("longName", ""))
            metrics["firstTradeDateEpochUtc"].append(
                stock_info.get("firstTradeDateEpochUtc", "")
            )
            metrics["timeZoneFullName"].append(stock_info.get("timeZoneFullName", ""))
            metrics["timeZoneShortName"].append(stock_info.get("timeZoneShortName", ""))
            metrics["uuid"].append(stock_info.get("uuid", ""))
            metrics["gmtOffSetMilliseconds"].append(
                stock_info.get("gmtOffSetMilliseconds", "")
            )
            metrics["currentPrice"].append(stock_info.get("currentPrice", ""))
            metrics["targetHighPrice"].append(stock_info.get("targetHighPrice", ""))
            metrics["targetLowPrice"].append(stock_info.get("targetLowPrice", ""))
            metrics["targetMeanPrice"].append(stock_info.get("targetMeanPrice", ""))
            metrics["targetMedianPrice"].append(stock_info.get("targetMedianPrice", ""))
            metrics["recommendationMean"].append(
                stock_info.get("recommendationMean", "")
            )
            metrics["recommendationKey"].append(stock_info.get("recommendationKey", ""))
            metrics["numberOfAnalystOpinions"].append(
                stock_info.get("numberOfAnalystOpinions", "")
            )
            metrics["totalCash"].append(stock_info.get("totalCash", ""))
            metrics["totalCashPerShare"].append(stock_info.get("totalCashPerShare", ""))
            metrics["ebitda"].append(stock_info.get("ebitda", ""))
            metrics["totalDebt"].append(stock_info.get("totalDebt", ""))
            metrics["quickRatio"].append(stock_info.get("quickRatio", ""))
            metrics["currentRatio"].append(stock_info.get("currentRatio", ""))
            metrics["totalRevenue"].append(stock_info.get("totalRevenue", ""))
            metrics["debtToEquity"].append(stock_info.get("debtToEquity", ""))
            metrics["revenuePerShare"].append(stock_info.get("revenuePerShare", ""))
            metrics["returnOnAssets"].append(stock_info.get("returnOnAssets", ""))
            metrics["returnOnEquity"].append(stock_info.get("returnOnEquity", ""))
            metrics["freeCashflow"].append(stock_info.get("freeCashflow", ""))
            metrics["operatingCashflow"].append(stock_info.get("operatingCashflow", ""))
            metrics["earningsGrowth"].append(stock_info.get("earningsGrowth", ""))
            metrics["revenueGrowth"].append(stock_info.get("revenueGrowth", ""))
            metrics["company_labels"].append(ticker.ticker)
        except Exception as e:
            st.error(f"Failed to process ticker {ticker.ticker}: {e}")
    else:
        st.write(f"Skipped a company ticker due to missing info or an invalid object.")

def worker(company, metrics):
    try:
        ticker = yf.Ticker(company)
        populate_metrics(ticker, metrics)
    except Exception as e:
        print(f"Failed to fetch data for {company}. Error: {e}")

@st.cache_data(show_spinner="Fetching data from API...", persist=True)
def fetch_metrics_data(companies):
    metrics = {
        metric: []
        for metric in [
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
            "shortName",
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
    }

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(worker, company, metrics) for company in companies]
        concurrent.futures.wait(futures)
    return metrics

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

@st.cache_data(show_spinner="Fetching recommendations from API...", persist=True)
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
        print(f"Failed to fetch recommendations summary: {e}")
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
                if isinstance(metrics[key][metrics_index], list):
                    value = metrics[key][metrics_index]
                else:
                    value = metrics[key][metrics_index] if len(metrics[key]) > metrics_index else None
            elif key in metrics_filtered:
                filtered_index = filtered_company_symbols.index(symbol)
                value = metrics_filtered[key][filtered_index] if len(metrics_filtered[key]) > filtered_index else None
            else:
                value = None

            # Append or extend based on the type of value
            if isinstance(value, list) and not isinstance(value, str) and key == 'freeCashflow':
                # Assuming we want to extend to flatten the list of lists where key is 'freeCashflow'
                combined_metrics[key].extend(value) if isinstance(combined_metrics[key], list) else combined_metrics[key].append([value])
            else:
                combined_metrics[key].append(value)
                
    expected_length = len(filtered_company_symbols)
    for key, values_list in combined_metrics.items():
        if len(values_list) != expected_length:
            raise ValueError(f"Length mismatch in combined metrics for key: {key}")

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

    try:
        if ticker_symbol in combined_metrics["company_labels"]:
            index = combined_metrics["company_labels"].index(ticker_symbol)
            eps = combined_metrics["eps_values"][index]
            pe = combined_metrics["pe_values"][index]
            ps = combined_metrics["priceToSalesTrailing12Months"][index]
            pb = combined_metrics["priceToBook"][index]
            peg = combined_metrics["peg_values"][index]
            gm = combined_metrics["gross_margins"][index]
            wh52 = combined_metrics["fiftyTwoWeekHigh"][index]
            wl52 = combined_metrics["fiftyTwoWeekLow"][index]
            currentPrice = combined_metrics["currentPrice"][index]
            targetMedianPrice= combined_metrics["targetMedianPrice"][index]
            targetLowPrice= combined_metrics["targetLowPrice"][index]
            targetMeanPrice= combined_metrics["targetMeanPrice"][index]
            targetHighPrice= combined_metrics["targetHighPrice"][index]
            recommendationMean= combined_metrics["recommendationMean"][index]

            return eps, pe, ps, pb, peg, gm, wh52, wl52, currentPrice, targetMedianPrice, targetLowPrice, targetMeanPrice, targetHighPrice, recommendationMean
        
        else:
            print(f"Ticker '{ticker_symbol}' not found in the labels list.")
            return None, None, None, None, None, None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None, None, None, None

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

    eps, pe, ps, pb, peg, gm, wh52, wl52, currentPrice, targetMedianPrice, targetLowPrice, targetMeanPrice, targetHighPrice, recommendationMean = get_dash_metrics(ticker_symbol, combined_metrics)

    if not data.empty:
        va_high, va_low, poc_price, _ = calculate_market_profile(data)
        price = ticker.info["currentPrice"]

        if option[0] == "va_high":
            if price > va_high:
                logging.info(f"{ticker_symbol} - current price is above value area: {price} {va_high} {poc_price}")
                return 0
        elif option[0] == "poc_price":
            if price > poc_price:
                logging.info(f"{ticker_symbol} - price is above price of control: {price} {va_high} {poc_price}")
                return 0
        else :
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
                    market_cap_display = (f"{market_cap / 1e6:.2f} M"
                    )
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

        # col1, col2 = st.columns(2)

        # with col1:
            # if peg:
            #     st.metric(label="PEG", value=f"{round(peg,2)}")
            # else:
                # st.metric(label="PEG", value="-")

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

            # try:
            #     calendar_data = yf.Ticker(ticker_symbol).calendar
            #     logging.debug(f"Displaying calendar data for {ticker_symbol}")
            #     show_calendar_data(calendar_data)
            # except Exception as e:
            #     logging.error(f"Failed to fetch or display calendar data for {ticker_symbol}: {e}")
            
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

def fetch_news(ticker_symbol):
    """Fetch news data for a given ticker symbol with error handling."""
    try:
        dnews = yf.Ticker(ticker_symbol).news
        if not dnews:
            logging.warning(f"No news found for ticker '{ticker_symbol}'.")
            return []
        return dnews
    except Exception as e:
        logging.error(f"Failed to fetch news for ticker '{ticker_symbol}': {e}")
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

    days_ago = (datetime.now() - datetime.fromtimestamp(article_info["providerPublishTime"])).days

    return {
        "Title": article_info["title"],
        "Link": article_info["link"],
        "Publisher": article_info["publisher"],
        "Sentiment": polarity,
        "Days Ago": days_ago,
    }

def get_news_data(ticker_symbol):
    """Fetch and analyze news data and calculate total polarity for a given ticker symbol."""
    dnews = fetch_news(ticker_symbol)
    total_polarity = 0
    news_data = []
    print(dnews)
    for article_info in dnews:
        if all(k in article_info for k in ["link", "providerPublishTime", "title", "publisher"]):
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
        st.markdown(f"{rounded_sentiment} - [{re.sub(':', '', title)}]({news_item['Link']}) - ({days_ago} days ago)")

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
                "short_name": metrics["short_name"],
                "fullTimeEmployees": metrics["fullTimeEmployees"],
                "boardRisk": metrics["boardRisk"],
                "industry": metrics["industry"],
                "sector": metrics["sector"],
                "compensationRisk": metrics["compensationRisk"],
                "shareHolderRightsRisk": metrics["shareHolderRightsRisk"],
                "overallRisk": metrics["overallRisk"],
                "exDividendDate": metrics["exDividendDate"],
                "dividendYield": metrics["dividendYield"],
                "dividendRate": metrics["dividendRate"],
                "priceHint": metrics["priceHint"],
                "fiftyTwoWeekLow": metrics["fiftyTwoWeekLow"],
                "forwardPE": metrics["forwardPE"],
                "marketCap": metrics["marketCap"],
                "beta": metrics["beta"],
                "fiveYearAvgDividendYield": metrics["fiveYearAvgDividendYield"],
                "payoutRatio": metrics["payoutRatio"],
                "ebitdaMargins": metrics["ebitdaMargins"],
                "website": metrics["website"],
                "operatingMargins": metrics["operatingMargins"],
                "financialCurrency": metrics["financialCurrency"],
                "trailingPegRatio": metrics["trailingPegRatio"],
                "fiftyTwoWeekHigh": metrics["fiftyTwoWeekHigh"],
                "priceToSalesTrailing12Months": metrics["priceToSalesTrailing12Months"],
                "fiftyDayAverage": metrics["fiftyDayAverage"],
                "twoHundredDayAverage": metrics["twoHundredDayAverage"],
                "trailingAnnualDividendRate": metrics["trailingAnnualDividendRate"],
                "trailingAnnualDividendYield": metrics["trailingAnnualDividendYield"],
                "currency": metrics["currency"],
                "fullTimeEmployees": metrics["fullTimeEmployees"],
                "enterpriseValue": metrics["enterpriseValue"],
                "profitMargins": metrics["profitMargins"],
                "floatShares": metrics["floatShares"],
                "sharesOutstanding": metrics["sharesOutstanding"],
                "sharesShort": metrics["sharesShort"],
                "sharesShortPriorMonth": metrics["sharesShortPriorMonth"],
                "sharesShortPreviousMonthDate": metrics["sharesShortPreviousMonthDate"],
                "dateShortInterest": metrics["dateShortInterest"],
                "sharesPercentSharesOut": metrics["sharesPercentSharesOut"],
                "heldPercentInsiders": metrics["heldPercentInsiders"],
                "heldPercentInstitutions": metrics["heldPercentInstitutions"],
                "shortRatio": metrics["shortRatio"],
                "shortPercentOfFloat": metrics["shortPercentOfFloat"],
                "bookValue": metrics["bookValue"],
                "priceToBook": metrics["priceToBook"],
                "lastFiscalYearEnd": metrics["lastFiscalYearEnd"],
                "nextFiscalYearEnd": metrics["nextFiscalYearEnd"],
                "mostRecentQuarter": metrics["mostRecentQuarter"],
                "earningsQuarterlyGrowth": metrics["earningsQuarterlyGrowth"],
                "netIncomeToCommon": metrics["netIncomeToCommon"],
                "forwardEps": metrics["forwardEps"],
                "lastSplitFactor": metrics["lastSplitFactor"],
                "lastSplitDate": metrics["lastSplitDate"],
                "enterpriseToRevenue": metrics["enterpriseToRevenue"],
                "enterpriseToEbitda": metrics["enterpriseToEbitda"],
                "exchange": metrics["exchange"],
                "quoteType": metrics["quoteType"],
                "symbol": metrics["symbol"],
                "underlyingSymbol": metrics["underlyingSymbol"],
                "shortName": metrics["shortName"],
                "longName": metrics["longName"],
                "firstTradeDateEpochUtc": metrics["firstTradeDateEpochUtc"],
                "timeZoneFullName": metrics["timeZoneFullName"],
                "timeZoneShortName": metrics["timeZoneShortName"],
                "uuid": metrics["uuid"],
                "gmtOffSetMilliseconds": metrics["gmtOffSetMilliseconds"],
                "currentPrice": metrics["currentPrice"],
                "targetHighPrice": metrics["targetHighPrice"],
                "targetLowPrice": metrics["targetLowPrice"],
                "targetMeanPrice": metrics["targetMeanPrice"],
                "targetMedianPrice": metrics["targetMedianPrice"],
                "recommendationMean": metrics["recommendationMean"],
                "recommendationKey": metrics["recommendationKey"],
                "numberOfAnalystOpinions": metrics["numberOfAnalystOpinions"],
                "totalCash": metrics["totalCash"],
                "totalCashPerShare": metrics["totalCashPerShare"],
                "ebitda": metrics["ebitda"],
                "totalDebt": metrics["totalDebt"],
                "quickRatio": metrics["quickRatio"],
                "currentRatio": metrics["currentRatio"],
                "totalRevenue": metrics["totalRevenue"],
                "debtToEquity": metrics["debtToEquity"],
                "revenuePerShare": metrics["revenuePerShare"],
                "returnOnAssets": metrics["returnOnAssets"],
                "returnOnEquity": metrics["returnOnEquity"],
                "freeCashflow": metrics["freeCashflow"],
                "operatingCashflow": metrics["operatingCashflow"],
                "earningsGrowth": metrics["earningsGrowth"],
                "revenueGrowth": metrics["revenueGrowth"],
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
