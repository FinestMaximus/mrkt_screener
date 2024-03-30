import streamlit as st
import logging
import pandas as pd


from lib import *

logging.basicConfig(level=logging.INFO)
st.set_page_config(layout="wide")

with st.sidebar:
    st.header("Configuration")

    st.markdown(
        """
        This app helps in screening the market based on various metrics. 
        The source code can be found at [GitHub](https://github.com/FinestMaximus/mrkt_screener).

        """
    )

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
            ('va_high', 'Value Area High'),
            ('poc_price', 'Point of Control Price')
        ],
        format_func=lambda x: x[1],  # Displaying more descriptive text for the options
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


def main():

    st.title("Interesting Stocks")

    init_session_state()

    if not st.session_state.data_loaded:
        file_path = "tickers.csv"
        df = pd.read_csv(file_path)
        st.session_state.companies = df["ticker"].tolist()
        st.session_state.data_loaded = True

        st.session_state.metrics = fetch_metrics_data(st.session_state.companies)

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

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(label="Total Finds", value=f"{len(filtered_companies_df)} Companies")

    with col2:
        st.metric(
            label="Volume Profile Range", value=f"{round(days_history/365)} Years"
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

    plot_candle_charts_per_symbol(
        filtered_industries,
        start_date_str,
        end_date_str,
        st.session_state.combined_metrics,
        final_shortlist_labels,
        option,
    )

    # indices_to_keep = [
    #     st.session_state.combined_metrics["company_labels"].index(label)
    #     for label in final_shortlist_labels
    #     if label in st.session_state.combined_metrics["company_labels"]
    # ]

    # st.header("Analysis Results")

    # filtered_data = {}
    # for key, values in st.session_state.combined_metrics.items():
    #     if isinstance(values, list) and len(values) == len(
    #         st.session_state.combined_metrics["company_labels"]
    #     ):
    #         filtered_data[key] = [values[i] for i in indices_to_keep]
    #     else:
    #         filtered_data[key] = values
    # st.table(filtered_data)

    # plot_combined_interactive(filtered_data)

    # final_industries = fetch_industries(company_labels)
    # plot_title = "Sector Distribution"
    # fig = plot_sector_distribution_interactive(final_industries, plot_title)
    # st.plotly_chart(fig)


if __name__ == "__main__":
    main()
