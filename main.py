import re
import streamlit as st
import logging
import pandas as pd
import requests
from bs4 import BeautifulSoup
from lib import *

logging.basicConfig(level=logging.INFO)
st.set_page_config(layout="wide")

def fetch_market_sentiment(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        selector = 'div > div > div:nth-of-type(2) > div:nth-of-type(1) > p:nth-of-type(2)'
        extracted_text = soup.select_one(selector).text
        match = re.search(r'\d+%', extracted_text)
        if match:
            percentage_value = float(match.group().strip('%'))

            if percentage_value >= 75:
                sentiment = "Extreme Greed" 
                color_code = 'red'
            elif 50 <= percentage_value < 75:
                sentiment = "Greed 😨" 
                color_code = 'orange'
            elif 25 <= percentage_value < 50:
                sentiment = "Fear 😏" 
                color_code = 'yellow'
            else:
                sentiment = "Extreme Fear" 
                color_code = 'green'

            return match.group(), sentiment, color_code
        else:
            logging.info("Failed to find the percentage in the extractor text.")
            return None, None, None
    else:
        logging.error("Failed to retrieve the webpage - Status code: %s", response.status_code)
        return None, None, None

with st.sidebar:
    url = 'https://pyinvesting.com/fear-and-greed/'
    percentage, sentiment, color_code = fetch_market_sentiment(url)

    if percentage and sentiment and color_code:
        info_text = "% Stocks in the market that are in an uptrend trading above their 6 month exponential moving average (EMA)."
        col1, col2 = st.columns(2)

        with col1:
            st.metric(label="Sentiment:", value=percentage, help=info_text)

        with col2:
            st.markdown(f"<h1 style='color: {color_code};'>{sentiment}</h1>", unsafe_allow_html=True)

    days_history = st.number_input(
        "Days History", min_value=365, max_value=36500, value=1825, step=365
    )
    eps_threshold = st.number_input("EPS Threshold", value=5.0)
    gross_margin_threshold = st.number_input("Gross Margin Threshold", value=0.8)
    peg_threshold_low = st.number_input("PEG Lower Threshold", value=0.0)
    peg_threshold_high = st.number_input("PEG Upper Threshold", value=1.0)

with st.sidebar:
    st.sidebar.subheader("Price Type Selection")
    st.sidebar.write("Select the type of price you want to analyze. Hover over each option for more details to help you decide.")

    option = st.radio(
        "Select the price threshold:",
        options=[
            ('va_high', 'Current Price inside VA'),
            ('poc_price', 'Current Price below POC'), 
            ('disabled', "Disable Price Area Filter")
        ],
        format_func=lambda x: x[1],  
        help="Value Area High (va_high) refers to the highest price level within the Value Area where the majority of trading activity took place. \n\nPoint of Control Price (poc_price) is the price level for the time period with the highest traded volume."
    )

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
    return [0.0 if (pd.isna(x) or x == 'nan') else x for x in lst]

def main():
    init_session_state()


    if not st.session_state.data_loaded:
        file_path = "tickers.csv"
        df = pd.read_csv(file_path)
        st.session_state.companies = df["ticker"].tolist()
        st.session_state.data_loaded = True

        st.session_state.metrics = fetch_metrics_data(st.session_state.companies)

    # Prevent further execution if no companies found
    if not st.session_state.companies:
        st.info("No companies found in the uploaded file.")
        return

    # Assuming get_date_range() doesn't rely on Streamlit commands to function
    start_date_str, end_date_str = get_date_range(days_history)

    # Assuming filtering logic doesn't depend on the Streamlit visual commands
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
    final_shortlist_labels = []

    st.header("Analysis Results - Short List")
    
    df = pd.DataFrame(st.session_state.combined_metrics)
    
    columns_to_display = ['company_labels', 'shortName', 'overallRisk', 'freeCashflow', 'opCashflow', 'repurchaseCapStock']
    filtered_df: pd.DataFrame = df[columns_to_display].copy()
    filtered_df['opCashflow'] = filtered_df['opCashflow'].apply(lambda x: replace_with_zero(x))
    filtered_df['repurchaseCapStock'] = filtered_df['repurchaseCapStock'].apply(lambda x: replace_with_zero(x))

    assert len(metrics_filtered['freeCashflow']) == len(filtered_df), "Mismatch in number of records"

    # THIS IS A QUICK FIX FOR MALFOMED freeCashflow inside  build_combined_metrics() - TBF later
    filtered_df['freeCashflow'] = metrics_filtered['freeCashflow']
    filtered_df['repurchaseCapStock'] = filtered_df['repurchaseCapStock'].apply(lambda x: [-y for y in x] if isinstance(x, list) else -x)

    st.dataframe(
        filtered_df,
        width=10000,
        column_config={
            "company_labels": st.column_config.TextColumn("Company Labels"),
            "shortName": st.column_config.TextColumn("Short Name"),
            "overallRisk": st.column_config.TextColumn("Overall Risk"),
            "freeCashflow": st.column_config.LineChartColumn(
                "Free Cashflow (4y)", y_min=-200, y_max=200
            ),
            "opCashflow": st.column_config.LineChartColumn(
                "Operating Cashflow (4y)", y_min=-100, y_max=100
            ),
            "repurchaseCapStock": st.column_config.LineChartColumn(
                "Stock Repurchase Value (4y)", y_min=-50, y_max=50
            ),
        },
        hide_index=True,
    )

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
