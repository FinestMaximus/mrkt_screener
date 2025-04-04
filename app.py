import streamlit as st
import logging
import pandas as pd
from analysis.lib import (
    build_combined_metrics,
    fetch_additional_metrics_data,
    fetch_metrics_data,
    plot_candle_charts_per_symbol,
)
from data.fetcher import DataFetcher
from analysis.metrics import FinancialMetrics
from analysis.news_sentiment import SentimentAnalyzer
from analysis.market_profile import MarketProfileAnalyzer
from analysis.market_sentiment import fetch_market_sentiment
from visualization.charts import ChartGenerator
from utils.helpers import get_date_range, replace_with_zero
import traceback
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("stock_analysis.log"),
        logging.StreamHandler(sys.stdout),
    ],
)

# Create a logger for this module
logger = logging.getLogger(__name__)

# Configure page
st.set_page_config(layout="wide", page_title="Stock Analysis Dashboard")
logger.info("Application started")


def display_market_sentiment():
    """Display market sentiment in the sidebar"""
    logger.info("Fetching market sentiment data")
    url = "https://pyinvesting.com/fear-and-greed/"
    percentage, sentiment, color_code = fetch_market_sentiment(url)

    if percentage and sentiment and color_code:
        logger.info(f"Market sentiment data retrieved: {sentiment} ({percentage})")
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
        logger.warning("Failed to retrieve market sentiment data")


def get_config_inputs():
    """Get configuration inputs from sidebar"""
    logger.info("Getting configuration inputs from sidebar")
    st.header("Configuration")

    # Load tickers directly from CSV file
    file_path = "tickers.csv"
    try:
        logger.info(f"Loading tickers from {file_path}")
        df = pd.read_csv(file_path)
        company_symbols = df["ticker"].tolist() if "ticker" in df.columns else []
        logger.info(f"Loaded {len(company_symbols)} tickers from tickers.csv")
        st.info(f"Loaded {len(company_symbols)} tickers from tickers.csv")
    except Exception as e:
        logger.error(f"Error loading tickers.csv: {str(e)}")
        st.error(f"Error loading tickers.csv: {str(e)}")
        company_symbols = []

    # Time range configuration
    days_back = st.slider(
        "Days of historical data",
        365,
        3650,
        1825,
        365,
        help="Number of days to look back for historical data analysis",
    )
    logger.debug(f"Selected days_back: {days_back}")

    # Filtering options
    st.subheader("Filtering Options")
    eps_threshold = st.number_input("Minimum EPS", 0.0, 100.0, 0.5, 0.1)
    gross_margin_threshold = st.number_input(
        "Minimum Gross Margin (%)", 0.0, 100.0, 20.0, 1.0
    )
    peg_low = st.number_input("Minimum PEG Ratio", -100.0, 100.0, 0.5, 0.1)
    peg_high = st.number_input("Maximum PEG Ratio", 0.0, 100.0, 2.5, 0.1)

    logger.debug(
        f"Filtering options - EPS: {eps_threshold}, Gross Margin: {gross_margin_threshold}, PEG: {peg_low} to {peg_high}"
    )

    # Price type selection
    st.subheader("Price Area Analysis")
    price_option = st.radio(
        "Select price threshold:",
        options=[
            ("va_high", "Current Price inside VA"),
            ("poc_price", "Current Price below POC"),
            ("disabled", "Disable Price Area Filter"),
        ],
        format_func=lambda x: x[1],
        help="Value Area High (va_high): The highest price level within the Value Area.\n\nPoint of Control (poc_price): The price level with the highest traded volume.",
    )
    logger.debug(f"Selected price option: {price_option}")

    logger.info(f"Configuration complete.")

    return {
        "company_symbols": company_symbols,
        "days_back": days_back,
        "eps_threshold": eps_threshold,
        "gross_margin_threshold": gross_margin_threshold,
        "peg_low": peg_low,
        "peg_high": peg_high,
        "price_option": price_option,
    }


def display_metrics_dashboard(combined_metrics):
    """Display metrics dashboard with dataframe view"""
    logger.info("Displaying metrics dashboard")
    st.markdown("## Analysis Results - Full List")

    col1, col2 = st.columns(2)
    with col1:
        total_companies = len(combined_metrics.get("company_labels", []))
        logger.info(f"Total companies in analysis: {total_companies}")
        st.metric(
            label="Total Companies",
            value=f"{total_companies} Companies",
        )
    with col2:
        days = combined_metrics.get("days_history", 1825)
        logger.info(f"Analysis period: {round(days/365)} years")
        st.metric(label="Analysis Period", value=f"{round(days/365)} Years")

    df = pd.DataFrame(combined_metrics)
    logger.debug(f"Dashboard dataframe shape: {df.shape}")

    # Define columns to display
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
    logger.debug(f"Columns available for display: {columns_to_display}")

    # If no columns match, use all available columns
    if not columns_to_display:
        logger.warning(
            "None of the specified columns were found in the data. Using all available columns."
        )
        st.warning(
            "None of the specified columns were found in the data. Displaying available columns instead."
        )
        columns_to_display = df.columns.tolist()

    # Prepare dataframe
    filtered_df = df[columns_to_display].copy()

    # Transform cashflow data for better visualization
    if "opCashflow" in filtered_df.columns:
        logger.debug("Transforming operating cashflow data")
        filtered_df["opCashflow"] = filtered_df["opCashflow"].apply(
            lambda x: replace_with_zero(x)
        )

    if "repurchaseCapStock" in filtered_df.columns:
        logger.debug("Transforming stock repurchase data")
        filtered_df["repurchaseCapStock"] = filtered_df["repurchaseCapStock"].apply(
            lambda x: replace_with_zero(x)
        )
        filtered_df["repurchaseCapStock"] = filtered_df["repurchaseCapStock"].apply(
            lambda x: [-y for y in x] if isinstance(x, list) else -x
        )

    # Display dataframe with custom column configuration matching hello.py
    logger.info("Displaying final metrics dashboard")
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
            "opCashflow": st.column_config.LineChartColumn(
                "Operating Cashflow (4y)", y_min=-100, y_max=100
            ),
            "repurchaseCapStock": st.column_config.LineChartColumn(
                "Stock Repurchase Value (4y)", y_min=-50, y_max=50
            ),
        },
        hide_index=True,
    )
    logger.info("Metrics dashboard displayed successfully")


def main():
    """Main application entry point"""
    logger.info("Main function started")

    # Create a layout with title and button side by side
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("Stock Analysis Dashboard")
    with col2:
        run_analysis = st.button("Run Analysis", use_container_width=True)

    # Sidebar for market sentiment and inputs
    logger.info("Setting up sidebar")
    with st.sidebar:
        display_market_sentiment()
        config = get_config_inputs()

    company_symbols = config["company_symbols"]
    days_back = config["days_back"]
    eps_threshold = config["eps_threshold"]
    gross_margin_threshold = config["gross_margin_threshold"]
    peg_low = config["peg_low"]
    peg_high = config["peg_high"]
    price_option = (
        config["price_option"][0]
        if isinstance(config["price_option"], tuple)
        else config["price_option"]
    )

    if not company_symbols:
        logger.error("No ticker symbols found in tickers.csv file")
        st.error("No ticker symbols found in tickers.csv file")
        return

    # Show progress
    st.subheader("Analysis Progress")
    progress_bar = st.progress(0)
    status_text = st.empty()

    if run_analysis:
        logger.info("Starting analysis with the following configuration:")
        try:
            # Get date range for historical data
            logger.info("Calculating date range")
            start_date, end_date = get_date_range(days_back)
            logger.info(f"Using date range: {start_date} to {end_date}")
            status_text.text("Calculating date ranges...")
            progress_bar.progress(20)

            # Add debug output
            st.write(f"Using company symbols: {company_symbols}")
            st.write(f"Date range: {start_date} to {end_date}")

            # Fetch metrics using hello.py approach
            logger.info("Fetching metrics data")
            status_text.text("Fetching metrics data...")
            list_metrics_all_tickers = fetch_metrics_data(company_symbols)

            # Debug output to see what we got
            logger.info(f"Fetched metrics for {len(list_metrics_all_tickers)} tickers")
            st.write(f"Fetched metrics for {len(list_metrics_all_tickers)} tickers")
            progress_bar.progress(35)

            # Convert list to DataFrame for display
            filtered_companies_df = pd.DataFrame(list_metrics_all_tickers)
            logger.debug(f"Metrics dataframe shape: {filtered_companies_df.shape}")

            # Debug output
            st.write(f"DataFrame columns: {filtered_companies_df.columns.tolist()}")
            logger.debug(f"DataFrame columns: {filtered_companies_df.columns.tolist()}")

            # Ensure we have company symbols
            if "company" in filtered_companies_df.columns:
                filtered_company_symbols = filtered_companies_df["company"].tolist()
                logger.debug("Using 'company' column for symbols")
            elif "symbol" in filtered_companies_df.columns:
                filtered_company_symbols = filtered_companies_df["symbol"].tolist()
                logger.debug("Using 'symbol' column for symbols")
            else:
                filtered_company_symbols = company_symbols
                logger.warning("No symbol column found, using original company symbols")

            logger.info(
                f"Using {len(filtered_company_symbols)} filtered company symbols"
            )
            st.write(f"Using filtered company symbols: {filtered_company_symbols}")

            # Initialize metrics
            metrics = {"company_labels": filtered_company_symbols}

            # Fetch additional metrics
            logger.info("Fetching additional metrics")
            status_text.text("Fetching additional metrics...")
            metrics_filtered = fetch_additional_metrics_data(filtered_company_symbols)

            # Debug output
            logger.debug(f"Additional metrics keys: {list(metrics_filtered.keys())}")
            st.write(f"Additional metrics keys: {list(metrics_filtered.keys())}")
            progress_bar.progress(50)

            # Build combined metrics
            logger.info("Building combined metrics")
            status_text.text("Building combined metrics...")
            combined_metrics = build_combined_metrics(
                filtered_company_symbols, metrics, metrics_filtered
            )
            combined_metrics["days_history"] = days_back

            # Debug output
            logger.debug(f"Combined metrics keys: {list(combined_metrics.keys())}")
            st.write(f"Combined metrics keys: {list(combined_metrics.keys())}")
            progress_bar.progress(65)

            # Fetch industries - from the DataFetcher class since this might be missing
            logger.info("Fetching industry data")
            status_text.text("Fetching industry data...")
            data_fetcher = DataFetcher()
            filtered_industries = data_fetcher.fetch_industries(
                filtered_company_symbols
            )

            logger.info("Industry data fetched successfully")
            st.write(f"Fetched industries data")
            final_shortlist_labels = []
            progress_bar.progress(80)

            status_text.text("Analysis complete!")
            progress_bar.progress(100)
            logger.info("Core analysis completed successfully")

            # Display the metrics dashboard
            display_metrics_dashboard(combined_metrics)

            # Display detailed candle charts
            st.markdown("## Detailed Analysis")
            st.write("About to generate candle charts...")
            logger.info("Starting candle chart generation")

            # Call the plotting function with debug output
            try:
                logger.info("Generating candle charts")
                plot_candle_charts_per_symbol(
                    filtered_industries,
                    start_date,
                    end_date,
                    combined_metrics,
                    final_shortlist_labels,
                    price_option,
                )
                logger.info("Candle charts generated successfully")
                st.write("Candle charts generated successfully")
            except Exception as chart_error:
                error_msg = f"Error generating charts: {str(chart_error)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                st.error(error_msg)

        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            st.error(error_msg)

            # Show more detailed error information
            logger.debug("Displaying traceback to user")
            st.code(traceback.format_exc())


if __name__ == "__main__":
    try:
        logger.info("Starting application")
        main()
        logger.info("Application completed successfully")
    except Exception as e:
        logger.critical(f"Unhandled exception: {str(e)}")
        logger.critical(traceback.format_exc())
        st.error(f"Critical application error: {str(e)}")
        st.code(traceback.format_exc())
