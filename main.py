import re
import streamlit as st
import logging
import pandas as pd
import requests
from bs4 import BeautifulSoup
import plotly.express as px
import plotly.graph_objects as go
from lib import *
import random


logging.basicConfig(level=logging.INFO)
st.set_page_config(layout="wide")


def fetch_market_sentiment(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        selector = (
            "div > div > div:nth-of-type(2) > div:nth-of-type(1) > p:nth-of-type(2)"
        )
        extracted_text = soup.select_one(selector).text
        match = re.search(r"\d+%", extracted_text)
        if match:
            percentage_value = float(match.group().strip("%"))

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

            return match.group(), sentiment, color_code
        else:
            logging.info("Failed to find the percentage in the extractor text.")
            return None, None, None
    else:
        logging.error(
            "Failed to retrieve the webpage - Status code: %s", response.status_code
        )
        return None, None, None


with st.sidebar:
    url = "https://pyinvesting.com/fear-and-greed/"
    percentage, sentiment, color_code = fetch_market_sentiment(url)

    if percentage and sentiment and color_code:
        info_text = "Percentage of stocks in the market that are in an uptrend trading above their 6-month exponential moving average (EMA)."
        col1, col2 = st.columns(2)

        with col1:
            st.metric(label="Sentiment:", value=percentage, help=info_text)

        with col2:
            st.markdown(
                f"<h1 style='color: {color_code};'>{sentiment}</h1>",
                unsafe_allow_html=True,
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

    options = [
        ("poc_price", "Current Price below POC"),
        ("disabled", "Disable Price Area Filter"),
        ("va_high", "Current Price inside VA"),
    ]
    option = st.radio(
        "Select the price threshold:",
        options=options,
        format_func=lambda x: x[1],
        help="Value Area High (va_high) refers to the highest price level within the Value Area where the majority of trading activity took place. \n\nPoint of Control Price (poc_price) is the price level for the time period with the highest traded volume.",
    )

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


def search_ticker_and_display(ticker, start_date_str, end_date_str, option):
    try:
        # Get the ticker object from Yahoo Finance
        ticker_object = get_ticker_object(ticker)

        # Fetch historical data for the given ticker
        historical_data = fetch_historical_data(ticker, start_date_str, end_date_str)

        if historical_data.empty:
            st.error(f"No historical data found for ticker '{ticker}'.")
            return

        # Calculate market profile
        va_high, va_low, poc_price, _ = calculate_market_profile(historical_data)

        # Decision based on the selected price threshold option
        current_price = ticker_object.info.get("currentPrice", None)
        if current_price is None:
            st.error(f"Current price data missing for ticker '{ticker}'.")
            return

        if option[0] == "va_high" and current_price > va_high:
            logging.info(
                f"{ticker} - Current price is above value area: {current_price} > {va_high}"
            )
            st.warning(
                f"{ticker} - Current price is above the value area high ({va_high})."
            )
            return
        elif option[0] == "poc_price" and current_price > poc_price:
            logging.info(
                f"{ticker} - Current price is above price of control: {current_price} > {poc_price}"
            )
            st.warning(
                f"{ticker} - Current price is above the point of control ({poc_price})."
            )
            return

        # Displaying the ticker details
        website = ticker_object.info.get("website", "#")
        short_name = ticker_object.info.get("shortName", ticker)
        header_with_link = f"[🔗]({website}){short_name} - {ticker}"
        st.markdown(f"### {header_with_link}", unsafe_allow_html=True)

        # Display ticker financial metrics
        display_ticker_metrics(ticker, ticker_object.info)

        # Display news and sentiment
        news_data, total_polarity = get_news_data(ticker)
        col1_weight, col2_weight, col3_weight = 1, 2, 1
        col1, col2, col3 = st.columns([col1_weight, col2_weight, col3_weight])

        with col1:
            display_sentiment_gauge(news_data, total_polarity)

        with col2:
            display_news_articles(news_data)

        with col3:
            plot_candle_chart_with_volume_profile(
                historical_data, va_high, va_low, poc_price
            )

    except Exception as e:
        st.error(
            f"An error occurred while fetching and displaying ticker data for '{ticker}': {e}"
        )
        logging.error(f"Error in search_ticker_and_display: {e}")


def display_ticker_metrics(ticker, info):
    try:
        metrics_labels = {
            "pegRatio": "PEG Ratio",
            "trailingEps": "EPS",
            "trailingPE": "P/E Ratio",
            "priceToSalesTrailing12Months": "P/S Ratio",
            "priceToBook": "P/B Ratio",
            "grossMargins": "Gross Margin (%)",
            "marketCap": "Market Cap",
        }

        cols = st.columns(len(metrics_labels))

        for col, (key, label) in zip(cols, metrics_labels.items()):
            value = info.get(key, None)
            if value is not None:
                try:
                    if isinstance(value, str):
                        value = float(value.replace("%", "").strip())

                    if "Gross Margin" in label:
                        value = f"{value * 100:.1f}%"
                    elif "Market Cap" in label:
                        value = (
                            f"{value/1e9:.2f} B"
                            if value >= 1e9
                            else (f"{value/1e6:.2f} M" if value >= 1e6 else value)
                        )
                    else:
                        value = round(value, 2)
                except ValueError:
                    value = "N/A"  # If the value can't be converted to float

                col.metric(label=label, value=value)
            else:
                col.metric(label=label, value="N/A")

    except Exception as e:
        st.error(f"Failed to display ticker metrics for '{ticker}': {e}")
        logging.error(f"Error in display_ticker_metrics: {e}")


def plot_candle_chart_with_volume_profile(data, va_high, va_low, poc_price):
    try:
        # Create lines for Value Area High, Value Area Low, and Point of Control
        poc_line = pd.Series(poc_price, index=data.index)
        va_high_line = pd.Series(va_high, index=data.index)
        va_low_line = pd.Series(va_low, index=data.index)

        # Add these lines to the candlestick plot
        # Placeholder for mplfinance solution
        # Please adjust according to your implementation logic
        raise NotImplementedError("Candlestick plot logic to be implemented.")
    except Exception as e:
        st.error(f"Failed to plot candlestick chart for the ticker: {e}")
        logging.error(f"Error in plot_candle_chart_with_volume_profile: {e}")


def display_sentiment_gauge(news_data, total_polarity):
    try:
        if len(news_data) > 0:
            average_sentiment = total_polarity / len(news_data)
            color = (
                "green"
                if average_sentiment >= 0.5
                else ("orange" if average_sentiment >= 0 else "red")
            )

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
            st.write("No sentiment or news data available.")
    except Exception as e:
        st.error(f"Failed to display sentiment gauge: {e}")
        logging.error(f"Error in display_sentiment_gauge: {e}")


def display_scatter_plot_eps_pe(df) -> None:
    try:
        # Ensure columns are of the correct type
        df["forwardPE"] = pd.to_numeric(df["forwardPE"], errors="coerce")
        df["marketCap"] = pd.to_numeric(df["marketCap"], errors="coerce")
        df["forwardEps"] = pd.to_numeric(df["forwardEps"], errors="coerce")

        # Logging data types for debugging
        logging.info(f"Data types before plotting: {df.dtypes}")

        # Drop rows with NaN values in these columns to avoid type comparison issues
        df = df.dropna(subset=["forwardPE", "marketCap", "forwardEps"])

        # Classify market cap into small, mid, and large cap
        size_bins = [0, 2e9, 10e9, float("inf")]
        size_labels = ["Small-Cap", "Mid-Cap", "Large-Cap"]
        df["capSize"] = pd.cut(df["marketCap"], bins=size_bins, labels=size_labels)

        # Define marker sizes for each category
        size_mapping = {"Small-Cap": 5, "Mid-Cap": 10, "Large-Cap": 20}
        df["markerSize"] = df["capSize"].map(size_mapping)

        fig = px.scatter(
            df,
            x="forwardPE",
            y="forwardEps",
            size="markerSize",
            color="shortName",  # Use capSize for color distinction
            hover_name="shortName",
            title="Stocks Comparison: PEG vs Forward EPS",
            labels={
                "forwardPE": "Forward PE",
                "forwardEps": "Forward EPS",
                "marketCap": "Market Cap",
            },
            size_max=20,  # Ensure the largest size for large-cap
        )
        # Update layout and marker properties
        fig.update_layout(clickmode="event+select")
        fig.update_traces(marker=dict(sizemode="area", sizemin=4))

        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"An error occurred while displaying the scatter plot: {e}")
        logging.error(f"Error in display_scatter_plot: {e}")


def display_scatter_plot_roe_roa(df) -> None:
    try:
        # Ensure these columns exist
        if "returnOnEquity" not in df.columns or "returnOnAssets" not in df.columns:
            st.error(
                "The required 'returnOnEquity' or 'returnOnAssets' columns are missing from the DataFrame."
            )
            logging.error(
                "Missing columns: 'returnOnEquity' or 'returnOnAssets'. Available columns: %s",
                df.columns.tolist(),
            )
            return

        # Ensure columns are of the correct type
        df["returnOnEquity"] = pd.to_numeric(df["returnOnEquity"], errors="coerce")
        df["marketCap"] = pd.to_numeric(df["marketCap"], errors="coerce")
        df["returnOnAssets"] = pd.to_numeric(df["returnOnAssets"], errors="coerce")

        # Logging data types for debugging
        logging.info(f"Data types before plotting: {df.dtypes}")

        # Drop rows with NaN values in these columns to avoid type comparison issues
        df = df.dropna(subset=["returnOnEquity", "marketCap", "returnOnAssets"])

        # Classify market cap into small, mid, and large cap
        size_bins = [0, 2e9, 10e9, float("inf")]
        size_labels = ["Small-Cap", "Mid-Cap", "Large-Cap"]
        df["capSize"] = pd.cut(df["marketCap"], bins=size_bins, labels=size_labels)

        # Define marker sizes for each category
        size_mapping = {"Small-Cap": 5, "Mid-Cap": 10, "Large-Cap": 20}
        df["markerSize"] = df["capSize"].map(size_mapping)

        fig = px.scatter(
            df,
            x="returnOnEquity",
            y="returnOnAssets",
            size="markerSize",
            color="shortName",  # Use capSize for color distinction
            hover_name="shortName",
            title="Stocks Comparison: returnOnEquity vs returnOnAssets",
            labels={
                "returnOnEquity": "Return On Equity",
                "returnOnAssets": "Return On Assets",
                "marketCap": "Market Cap",
            },
            size_max=20,  # Ensure the largest size for large-cap
        )
        # Update layout and marker properties
        fig.update_layout(clickmode="event+select")
        fig.update_traces(marker=dict(sizemode="area", sizemin=4))

        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"An error occurred while displaying the scatter plot: {e}")
        logging.error(f"Error in display_scatter_plot_roe_roa: {e}")


def main():
    init_session_state()

    if search_button_1:
        st.session_state.search_clicked = True
        st.session_state.search_ticker = search_ticker_1
    else:
        st.session_state.search_clicked = False

    if not st.session_state.data_loaded:
        file_path = "tickers.csv"
        df = pd.read_csv(file_path)
        st.session_state.companies = df["ticker"].tolist()
        st.session_state.data_loaded = True
        st.session_state.metrics = fetch_metrics_data(st.session_state.companies)

    if st.session_state.search_clicked and st.session_state.search_ticker:
        tickers = [
            ticker.strip() for ticker in st.session_state.search_ticker.split(";")
        ]
        start_date_str, end_date_str = get_date_range(days_history)

        for ticker in tickers:
            search_ticker_and_display(
                ticker,
                start_date_str,
                end_date_str,
                option,
            )
        return

    if not st.session_state.companies:
        st.info("No companies found in the uploaded file.")
        return

    start_date_str, end_date_str = get_date_range(days_history)

    filtered_companies_df = filter_companies(
        st.session_state.metrics,
        eps_threshold,
        peg_threshold_low,
        peg_threshold_high,
        gross_margin_threshold,
    )

    if "company" in filtered_companies_df.columns:
        filtered_company_symbols = filtered_companies_df["company"].tolist()
    else:
        st.error("The expected 'company' column was not found.")
        return

    metrics_filtered = fetch_additional_metrics_data(filtered_company_symbols)

    st.session_state.combined_metrics = build_combined_metrics(
        filtered_company_symbols, st.session_state.metrics, metrics_filtered
    )

    filtered_industries = fetch_industries(filtered_company_symbols)
    final_shortlist_labels = filtered_company_symbols  # Ensure correct filling of shortlist labels as required

    st.markdown("# Analysis Results - Short List")

    col_total_finds, col_volume_profile_range = st.columns(2)

    with col_total_finds:
        st.metric(label="Total Finds", value=f"{len(filtered_companies_df)} Companies")

    with col_volume_profile_range:
        st.metric(
            label="Volume Profile Range", value=f"{round(days_history/365)} Years"
        )

    df = pd.DataFrame(st.session_state.combined_metrics)

    logging.info(f"Available columns in DataFrame: {df.columns.tolist()}")

    columns_to_display = [
        "company_labels",
        "shortName",
        "sector",
        "industry",
        "fullTimeEmployees",
        "overallRisk",
        "opCashflow",
        "repurchaseCapStock",
        "forwardEps",
        "forwardPE",
        "marketCap",
        "returnOnEquity",
        "returnOnAssets",  # Make sure to include these columns here
    ]

    # Check if the required columns exist
    missing_columns = [col for col in columns_to_display if col not in df.columns]
    if missing_columns:
        st.error(f"Missing columns in DataFrame: {missing_columns}")
        logging.error(f"Missing columns: {missing_columns}")
        return
    else:
        filtered_df = df[columns_to_display].copy()

    filtered_df["opCashflow"] = filtered_df["opCashflow"].apply(
        lambda x: replace_with_zero(x)
    )
    filtered_df["repurchaseCapStock"] = filtered_df["repurchaseCapStock"].apply(
        lambda x: replace_with_zero(x)
    )

    filtered_df["fullTimeEmployees"] = (
        pd.to_numeric(filtered_df["fullTimeEmployees"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    filtered_df["overallRisk"] = (
        pd.to_numeric(filtered_df["overallRisk"], errors="coerce")
        .fillna(0)
        .astype(float)
    )

    sectors = filtered_df["sector"].unique().tolist()
    industries = filtered_df["industry"].unique().tolist()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        selected_sector = st.selectbox("Filter by Sector:", ["All"] + sectors)

    with col2:
        selected_industry = st.selectbox("Filter by Industry:", ["All"] + industries)

    with col3:
        min_employees, max_employees = st.slider(
            "Filter by Full Time Employees:",
            0,
            int(filtered_df["fullTimeEmployees"].max()),
            (0, int(filtered_df["fullTimeEmployees"].max())),
        )

    with col4:
        min_risk, max_risk = st.slider(
            "Filter by Overall Risk:",
            float(filtered_df["overallRisk"].min()),
            float(filtered_df["overallRisk"].max()),
            (
                float(filtered_df["overallRisk"].min()),
                float(filtered_df["overallRisk"].max()),
            ),
        )

    if selected_sector != "All":
        filtered_df = filtered_df[filtered_df["sector"] == selected_sector]

    if selected_industry != "All":
        filtered_df = filtered_df[filtered_df["industry"] == selected_industry]

    filtered_df = filtered_df[
        (filtered_df["fullTimeEmployees"] >= min_employees)
        & (filtered_df["fullTimeEmployees"] <= max_employees)
    ]
    filtered_df = filtered_df[
        (filtered_df["overallRisk"] >= min_risk)
        & (filtered_df["overallRisk"] <= max_risk)
    ]

    # Add scatter plots
    if not filtered_df.empty:

        col1, col2 = st.columns(2)

        with col1:
            display_scatter_plot_eps_pe(filtered_df)

        with col2:
            display_scatter_plot_roe_roa(filtered_df)

    st.dataframe(
        filtered_df,
        width=10000,
        column_config={
            "company_labels": st.column_config.TextColumn("Company Labels"),
            "shortName": st.column_config.TextColumn("Short Name"),
            "sector": st.column_config.TextColumn("Sector"),
            "industry": st.column_config.TextColumn("Industry"),
            "fullTimeEmployees": st.column_config.TextColumn("Full Time Employees"),
            "overallRisk": st.column_config.TextColumn("Overall Risk"),
            # "freeCashflow": st.column_config.LineChartColumn(
            #     "Free Cashflow (4y)", y_min=-200, y_max=200
            # ),
            "opCashflow": st.column_config.BarChartColumn(
                "Operating Cashflow (4y)", y_min=-100, y_max=100
            ),
            "repurchaseCapStock": st.column_config.BarChartColumn(
                "Stock Repurchase Value (4y)", y_min=-50, y_max=50
            ),
        },
        hide_index=True,
    )

    st.markdown("# Detailed Analysis")

    # Call the updated plot_candle_charts_per_symbol function
    plot_candle_charts_per_symbol(
        filtered_industries,
        start_date_str,
        end_date_str,
        st.session_state.combined_metrics,
        final_shortlist_labels,
        option,
    )

    indices_to_keep = [
        st.session_state.combined_metrics["company_labels"].index(label)
        for label in final_shortlist_labels
        if label in st.session_state.combined_metrics["company_labels"]
    ]


if __name__ == "__main__":
    main()
