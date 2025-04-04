import streamlit as st
import logging
import pandas as pd
from lib import *
from market_sentiment import fetch_market_sentiment

logging.basicConfig(level=logging.INFO)
st.set_page_config(layout="wide")


with st.sidebar:
    url = "https://pyinvesting.com/fear-and-greed/"
    percentage, sentiment, color_code = fetch_market_sentiment(url)

    if percentage and sentiment and color_code:
        info_text = "% Stocks in the market that are in an uptrend trading above their 6 month exponential moving average (EMA)."
        col1, col2 = st.columns(2)

        with col1:
            st.metric(label="Sentiment:", value=percentage, help=info_text)

        with col2:
            st.markdown(
                f"<h1 style='color: {color_code};'>{sentiment}</h1>",
                unsafe_allow_html=True,
            )

    days_history = st.number_input(
        "Days History", min_value=365, max_value=36500, value=1825, step=365
    )
    eps_threshold = st.number_input(
        "EPS Threshold", value=0.0, min_value=0.0, max_value=100.0
    )
    gross_margin_threshold = st.number_input(
        "Gross Margin Threshold (%)", value=20.0, min_value=0.0, max_value=100.0
    )
    peg_threshold_low = st.number_input(
        "PEG Lower Threshold", value=0.0, min_value=-100.0, max_value=100.0
    )
    peg_threshold_high = st.number_input(
        "PEG Upper Threshold", value=30.0, min_value=0.0, max_value=100.0
    )

with st.sidebar:
    st.sidebar.subheader("Price Type Selection")
    st.sidebar.write(
        "Select the type of price you want to analyze. Hover over each option for more details to help you decide."
    )

    option = st.radio(
        "Select the price threshold:",
        options=[
            ("va_high", "Current Price inside VA"),
            ("poc_price", "Current Price below POC"),
            ("disabled", "Disable Price Area Filter"),
        ],
        format_func=lambda x: x[1],
        help="Value Area High (va_high) refers to the highest price level within the Value Area where the majority of trading activity took place. \n\nPoint of Control Price (poc_price) is the price level for the time period with the highest traded volume.",
    )


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


def main():
    global companies, combined_metrics

    file_path = "tickers.csv"
    df = pd.read_csv(file_path)
    companies = df["ticker"].tolist()

    if not companies:
        st.info("No companies found in the uploaded file.")
        return

    print("companies: ", companies)
    start_date_str, end_date_str = get_date_range(days_history)

    print("start_date_str: ", start_date_str)
    print("end_date_str: ", end_date_str)

    list_metrics_all_tickers = fetch_metrics_data(companies)

    # Convert the list to a DataFrame
    filtered_companies_df = pd.DataFrame(list_metrics_all_tickers)

    print(filtered_companies_df)

    # Extract company symbols from the dataframe
    if "company" in filtered_companies_df.columns:
        filtered_company_symbols = filtered_companies_df["company"].tolist()
    else:
        if "symbol" in filtered_companies_df.columns:
            filtered_company_symbols = filtered_companies_df["symbol"].tolist()
        else:
            st.error("Neither 'company' nor 'symbol' column was found.")
            return

    # Initialize metrics before fetching additional data
    metrics = {"company_labels": filtered_company_symbols}

    # Now fetch additional metrics
    metrics_filtered = fetch_additional_metrics_data(filtered_company_symbols)

    combined_metrics = build_combined_metrics(
        filtered_company_symbols, metrics, metrics_filtered
    )

    filtered_industries = fetch_industries(filtered_company_symbols)
    final_shortlist_labels = []

    st.markdown("# Analysis Results - Full List")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            label="Total Companies", value=f"{len(filtered_companies_df)} Companies"
        )

    with col2:
        st.metric(
            label="Volume Profile Range", value=f"{round(days_history/365)} Years"
        )

    df = pd.DataFrame(combined_metrics)

    # Check what columns are actually available
    print("Available columns:", df.columns.tolist())

    # Define columns to display based on what's available
    # Original desired columns
    desired_columns = [
        "company_labels",
        "shortName",
        "sector",
        "industry",
        "fullTimeEmployees",
        "overallRisk",
        "opCashflow",
        "repurchaseCapStock",
    ]

    # Filter to only include columns that actually exist
    columns_to_display = [col for col in desired_columns if col in df.columns]

    # If no columns match, use all available columns
    if not columns_to_display:
        st.warning(
            "None of the specified columns were found in the data. Displaying available columns instead."
        )
        columns_to_display = df.columns.tolist()

    # Now use the filtered list
    filtered_df = df[columns_to_display].copy()

    # Only try to process columns that exist
    if "opCashflow" in filtered_df.columns:
        filtered_df["opCashflow"] = filtered_df["opCashflow"].apply(
            lambda x: replace_with_zero(x)
        )

    if "repurchaseCapStock" in filtered_df.columns:
        filtered_df["repurchaseCapStock"] = filtered_df["repurchaseCapStock"].apply(
            lambda x: replace_with_zero(x)
        )
        # Only transform if the column exists
        filtered_df["repurchaseCapStock"] = filtered_df["repurchaseCapStock"].apply(
            lambda x: [-y for y in x] if isinstance(x, list) else -x
        )

    st.dataframe(
        filtered_df,
        width=10000,
        column_config={
            "company_labels": st.column_config.TextColumn("Company Labels"),
            "shortName": st.column_config.TextColumn("Short Name"),
            "sector": st.column_config.TextColumn("Sector"),
            "industry": st.column_config.TextColumn("Industry"),
            "fullTimeEmployees": st.column_config.TextColumn("fullTimeEmployees"),
            "overallRisk": st.column_config.TextColumn("Overall Risk"),
            # "freeCashflow": st.column_config.LineChartColumn(
            #     "Free Cashflow (4y)", y_min=-200, y_max=200
            # ),
            "opCashflow": st.column_config.LineChartColumn(
                "Operating Cashflow (4y)", y_min=-100, y_max=100
            ),
            "repurchaseCapStock": st.column_config.LineChartColumn(
                "Stock Repurchase Value (4y)", y_min=-50, y_max=50
            ),
        },
        hide_index=True,
    )

    st.markdown("# Detailed Analysis")

    plot_candle_charts_per_symbol(
        filtered_industries,
        start_date_str,
        end_date_str,
        combined_metrics,
        final_shortlist_labels,
        option,
    )

    indices_to_keep = [
        combined_metrics["company_labels"].index(label)
        for label in final_shortlist_labels
        if label in combined_metrics["company_labels"]
    ]


if __name__ == "__main__":
    main()
