import streamlit as st
import logging
import pandas as pd
from lib import data_fetching, market_analysis, metrics_handling
import time
import yfinance as yf

logging.basicConfig(level=logging.DEBUG)
st.set_page_config(layout="wide")

RATE_LIMIT_SECONDS = 5
MAX_RETRIES = 5


with st.sidebar:
    url = "https://pyinvesting.com/fear-and-greed/"
    logging.debug("[main.py][sidebar] Fetching market sentiment for URL: %s", url)
    percentage, sentiment, color_code = data_fetching.fetch_market_sentiment(url)

    if percentage and sentiment and color_code:
        logging.debug("[main.py][sidebar] Successfully fetched market sentiment.")
        info_text = "% Stocks in the market that are in an uptrend trading above their 6 month exponential moving average (EMA)."
        col1, col2 = st.columns(2)

        with col1:
            st.metric(label="Sentiment:", value=percentage, help=info_text)

        with col2:
            st.markdown(
                f"<h1 style='color: {color_code};'>{sentiment}</h1>",
                unsafe_allow_html=True,
            )
    else:
        logging.info(
            "[main.py][sidebar] Market sentiment data is incomplete or missing."
        )

    # Converting days input to a dropdown for years
    years_options = [1, 2, 3, 4, 5, 10]  # You can adjust the range as needed
    years_selected = st.selectbox(
        "Years History", years_options, index=4
    )  # Default index can be adjusted
    days_history = years_selected * 365
    st.write(f"Days History: {days_history} days")

    eps_threshold = st.number_input("EPS Threshold", value=2.0)
    gross_margin_threshold = st.number_input("Gross Margin Threshold", value=0.7)
    peg_threshold_low = st.number_input("PEG Lower Threshold", value=-0.1)
    peg_threshold_high = st.number_input("PEG Upper Threshold", value=1.1)

    st.subheader("Price Type Selection")
    st.write(
        "Select the type of price you want to analyze. Hover over each option for more details to help you decide."
    )
    eps_threshold = st.number_input("EPS Threshold", value=5.0)
    gross_margin_threshold = st.number_input("Gross Margin Threshold", value=0.8)
    peg_threshold_low = st.number_input("PEG Lower Threshold", value=0.0)
    peg_threshold_high = st.number_input("PEG Upper Threshold", value=1.0)

with st.sidebar:
    st.sidebar.subheader("Price Type Selection")
    option = st.radio(
        "Select the price threshold:",
        options=[
            ("va_high", "Current Price inside VA"),
            ("poc_price", "Current Price below POC"),
            ("disabled", "Disable Price Area Filter"),
        ],
        format_func=lambda x: x[1],
    )
    logging.debug("[main.py][sidebar] Price Type Selection set to: %s", option)

    # Add a ticker search input field in the sidebar
    search_ticker_1 = st.text_input(
        "Search for a Ticker (use ';' to separate multiple tickers):",
        key="search_ticker_1",
    )
    search_button_1 = st.button("Search", key="search_button_1")


def init_session_state():
    if "companies" not in st.session_state:
        st.session_state.companies = []
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False
    if "metrics" not in st.session_state:
        st.session_state.metrics = {}
    if "combined_metrics" not in st.session_state:
        st.session_state.combined_metrics = {}


def display_metrics(metrics_dict):
    if not metrics_dict:
        st.write("No metrics available.")
        return

    for key, value in metrics_dict.items():
        st.subheader(f"Metric: {key}")
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                st.write(f"{sub_key}: {sub_value}")
        else:
            st.write(f"Value: {value}")


def replace_with_zero(lst):
    if lst is None:
        return [0, 0, 0, 0]
    return [0.0 if (pd.isna(x) or str(x).lower() == "nan") else x for x in lst]


def fetch_with_retry(ticker):
    for attempt in range(MAX_RETRIES):
        time.sleep(RATE_LIMIT_SECONDS)
        stock = yf.Ticker(ticker)
        info = stock.info
        if info:
            return info
    return None


def main():
    init_session_state()
    file_path = "data/tickers.csv"
    logging.debug(f"[main.py][main] Loading tickers from: {file_path}")

    try:
        df = pd.read_csv(file_path)
        if "ticker" not in df.columns:
            st.error("CSV file must contain a 'ticker' column")
            return

        companies = df["ticker"].tolist()
        if not companies:
            st.error("No tickers found in CSV file")
            return

    except (FileNotFoundError, pd.errors.EmptyDataError) as e:
        st.error(f"Error loading tickers file: {str(e)}")
        return

    with st.spinner("Fetching company data... This may take a few minutes."):
        progress_bar = st.progress(0)
        progress_text = st.empty()

        for i, company in enumerate(companies):
            progress = (i + 1) / len(companies)
            progress_bar.progress(progress)
            progress_text.text(f"Processing {company} ({i+1}/{len(companies)})")

        metrics = data_fetching.fetch_metrics_data_for_initial_filtering(
            companies, fetch_with_retry
        )

        if (
            not metrics
            or "company_labels" not in metrics
            or not metrics["company_labels"]
        ):
            st.error("Failed to fetch metrics data or no company labels found.")
            return

        st.session_state.companies = companies
        st.session_state.metrics = metrics
        st.session_state.data_loaded = True

        logging.info(
            f"Fetched metrics for {len(metrics.get('company_labels', []))} companies"
        )

        with st.spinner("Filtering companies..."):
            filtered_companies_df = market_analysis.filter_companies(
                metrics,
                eps_threshold,
                peg_threshold_low,
                peg_threshold_high,
                gross_margin_threshold,
            )

            if filtered_companies_df.empty:
                st.warning("No companies matched the filtering criteria")
                return

            filtered_company_symbols = filtered_companies_df["company"].tolist()

            with st.spinner("Fetching additional metrics..."):
                metrics_filtered = data_fetching.fetch_additional_metrics_data(
                    filtered_company_symbols
                )

        st.session_state.combined_metrics = metrics_handling.build_combined_metrics(
            filtered_company_symbols, metrics, metrics_filtered
        )

        display_filtered_results(filtered_companies_df, metrics_filtered, days_history)


def display_filtered_results(filtered_df, metrics_filtered, days_history):
    st.markdown("# Analysis Results - Short List")

    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Total Finds", value=f"{len(filtered_df)} Companies")
    with col2:
        st.metric(
            label="Volume Profile Range", value=f"{round(days_history/365)} Years"
        )

    display_df = filtered_df.copy()
    if "freeCashflow" in metrics_filtered:
        display_df["freeCashflow"] = metrics_filtered["freeCashflow"]

    st.dataframe(
        display_df,
        width=10000,
        column_config={
            "company_labels": st.column_config.TextColumn("Company Labels"),
            "shortName": st.column_config.TextColumn("Short Name"),
            "sector": st.column_config.TextColumn("Sector"),
            "industry": st.column_config.TextColumn("Industry"),
            "fullTimeEmployees": st.column_config.TextColumn("Full Time Employees"),
            "overallRisk": st.column_config.TextColumn("Overall Risk"),
            "freeCashflow": st.column_config.LineChartColumn(
                "Free Cashflow (4y)", y_min=-200, y_max=200
            ),
        },
        hide_index=True,
    )


if __name__ == "__main__":
    main()
