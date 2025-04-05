import streamlit as st
import pandas as pd
from data.tickers_yf_fetcher import DataFetcher
from data.fear_greed_indicator import FearGreedIndicator
from visualization.charts import ChartGenerator
from utils.helpers import get_date_range, replace_with_zero
import traceback
from utils.logger import info, debug, warning, error, critical
from analysis.metrics import FinancialMetrics
from analysis.market_profile import MarketProfileAnalyzer
from data.news_research import SentimentAnalyzer


# Configure page
st.set_page_config(
    layout="wide",
    page_title="Stock Analysis Dashboard",
    initial_sidebar_state="expanded",
)
info("Application started")


def display_market_sentiment():
    """Display market sentiment in the sidebar"""
    info("Fetching market sentiment data")
    percentage, sentiment, color_code = FearGreedIndicator().fetch_market_sentiment()

    if percentage and sentiment and color_code:
        info(f"Market sentiment data retrieved: {sentiment} ({percentage})")
        info_text = "% Stocks in the market that are in an uptrend trading above their 6 month exponential moving average (EMA)."

        st.markdown("### Market Sentiment")
        st.markdown(
            f"<div style='background-color: rgba(0,0,0,0.1); padding: 15px; border-radius: 5px;'>",
            unsafe_allow_html=True,
        )
        col1, col2 = st.columns(2)

        with col1:
            st.metric(label="Sentiment:", value=percentage, help=info_text)

        with col2:
            st.markdown(
                f"<h3 style='color: {color_code}; margin-top: 10px;'>{sentiment}</h3>",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        warning("Failed to retrieve market sentiment data")
        st.warning("Market sentiment data currently unavailable")


def get_config_inputs():
    """Get configuration inputs from sidebar"""
    info("Getting configuration inputs from sidebar")
    st.markdown("### Configuration")
    st.markdown("<hr style='margin: 5px 0px 15px 0px'>", unsafe_allow_html=True)

    # Load tickers directly from CSV file
    file_path = "tickers.csv"
    try:
        info(f"Loading tickers from {file_path}")
        df = pd.read_csv(file_path)
        company_symbols = df["ticker"].tolist() if "ticker" in df.columns else []
        info(f"Loaded {len(company_symbols)} tickers from tickers.csv")
        st.success(f"Loaded {len(company_symbols)} tickers from tickers.csv")
    except Exception as e:
        error(f"Error loading tickers.csv: {str(e)}")
        st.error(f"Error loading tickers.csv: {str(e)}")
        company_symbols = []

    # Time range configuration
    st.markdown("#### Analysis Period")
    days_back = st.slider(
        "Days of historical data",
        365,
        3650,
        1825,
        365,
        help="Number of days to look back for historical data analysis",
    )
    debug(f"Selected days_back: {days_back}")

    # Price type selection
    st.markdown("#### Price Area Analysis")
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
    debug(f"Selected price option: {price_option}")

    info(f"Configuration complete.")

    # Return configuration with default filtering values
    return {
        "company_symbols": company_symbols,
        "days_back": days_back,
        "price_option": price_option,
    }


def display_metrics_dashboard(metrics):
    """Display metrics dashboard with dataframe view"""
    info("Displaying metrics dashboard")

    # Create metrics cards at the top
    st.markdown("<div style='margin-bottom: 20px;'>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        total_companies = len(metrics.get("company_labels", []))
        info(f"Total companies in analysis: {total_companies}")
        st.metric(
            label="Total Companies",
            value=f"{total_companies}",
        )
    with col2:
        days = metrics.get("days_history", 1825)
        info(f"Analysis period: {round(days/365)} years")
        st.metric(label="Analysis Period", value=f"{round(days/365)} Years")
    st.markdown("</div>", unsafe_allow_html=True)

    df = pd.DataFrame(metrics)
    debug(f"Dashboard dataframe shape: {df.shape}")
    debug(f"Dashboard dataframe columns: {df.columns.tolist()}")

    # Display all available columns instead of trying to filter
    columns_to_display = df.columns.tolist()
    debug(f"Columns available for display: {columns_to_display}")

    # Prepare dataframe
    filtered_df = df.copy()

    # Transform cashflow data for better visualization if it exists
    if "opCashflow" in filtered_df.columns:
        debug("Transforming operating cashflow data")
        filtered_df["opCashflow"] = filtered_df["opCashflow"].apply(
            lambda x: replace_with_zero(x)
        )

    if "repurchaseCapStock" in filtered_df.columns:
        debug("Transforming stock repurchase data")
        filtered_df["repurchaseCapStock"] = filtered_df["repurchaseCapStock"].apply(
            lambda x: replace_with_zero(x)
        )
        filtered_df["repurchaseCapStock"] = filtered_df["repurchaseCapStock"].apply(
            lambda x: [-y for y in x] if isinstance(x, list) else -x
        )

    # Display dataframe with dynamic column configuration
    info("Displaying final metrics dashboard")

    # Create dynamic column config based on available columns
    column_config = {}
    for col in filtered_df.columns:
        if col == "opCashflow":
            column_config[col] = st.column_config.LineChartColumn(
                "Operating Cashflow (4y)", y_min=-100, y_max=100
            )
        elif col == "repurchaseCapStock":
            column_config[col] = st.column_config.LineChartColumn(
                "Stock Repurchase Value (4y)", y_min=-50, y_max=50
            )
        elif col == "freeCashflow":
            column_config[col] = st.column_config.LineChartColumn(
                "Free Cashflow (4y)", y_min=-100, y_max=100
            )
        elif col == "totalCash":
            column_config[col] = st.column_config.LineChartColumn(
                "Total Cash (4y)", y_min=0, y_max=1000
            )
        elif col == "totalDebt":
            column_config[col] = st.column_config.LineChartColumn(
                "Total Debt (4y)", y_min=0, y_max=1000
            )
        elif col == "debtToEquity":
            column_config[col] = st.column_config.TextColumn("Debt to Equity Ratio")
        elif col == "currentRatio":
            column_config[col] = st.column_config.TextColumn("Current Ratio")
        elif col == "trailingPE":
            column_config[col] = st.column_config.TextColumn("Trailing P/E Ratio")
        elif col == "forwardPE":
            column_config[col] = st.column_config.TextColumn("Forward P/E Ratio")
        elif col == "priceToBook":
            column_config[col] = st.column_config.TextColumn("Price to Book Ratio")
        elif col == "dividendYield":
            column_config[col] = st.column_config.TextColumn("Dividend Yield")
        elif col == "beta":
            column_config[col] = st.column_config.TextColumn("Beta")
        elif col == "marketCap":
            column_config[col] = st.column_config.TextColumn("Market Capitalization")
        elif col == "recommendationMean":
            column_config[col] = st.column_config.TextColumn("Recommendation Mean")
        elif col == "targetHighPrice":
            column_config[col] = st.column_config.TextColumn("Target High Price")
        elif col == "targetLowPrice":
            column_config[col] = st.column_config.TextColumn("Target Low Price")
        elif col == "targetMeanPrice":
            column_config[col] = st.column_config.TextColumn("Target Mean Price")
        elif col == "targetMedianPrice":
            column_config[col] = st.column_config.TextColumn("Target Median Price")
        elif col == "currentPrice":
            column_config[col] = st.column_config.TextColumn("Current Price")
        elif col == "revenueGrowth":
            column_config[col] = st.column_config.TextColumn("Revenue Growth")
        elif col == "grossMargins":
            column_config[col] = st.column_config.TextColumn("Gross Margins")
        elif col == "returnOnEquity":
            column_config[col] = st.column_config.TextColumn("Return on Equity")

    st.dataframe(
        filtered_df,
        height=400,  # Limit height to ensure it doesn't take too much space
        use_container_width=True,  # Use container width instead of fixed width
        column_config=column_config,
        hide_index=True,
    )
    info("Metrics dashboard displayed successfully")


def main():
    """Main application entry point"""
    info("Main function started")

    # Custom CSS for better spacing and styling
    st.markdown(
        """
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1rem;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(0, 0, 0, 0.05);
        border-radius: 4px 4px 0 0;
    }
    .card {
        background-color: #f8f9fa;
        border-radius: 6px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Create a layout with title and button side by side
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("Stock Analysis Dashboard")
    with col2:
        run_analysis = st.button(
            "Run Analysis", use_container_width=True, type="primary"
        )

    # Sidebar for market sentiment and inputs
    info("Setting up sidebar")
    with st.sidebar:
        display_market_sentiment()
        config = get_config_inputs()

    company_symbols = config["company_symbols"]
    days_back = config["days_back"]
    price_option = (
        config["price_option"][0]
        if isinstance(config["price_option"], tuple)
        else config["price_option"]
    )

    if not company_symbols:
        error("No ticker symbols found in tickers.csv file")
        st.error("No ticker symbols found in tickers.csv file")
        return

    # Show progress
    progress_container = st.container()

    if run_analysis:
        info("Starting analysis with the following configuration:")
        try:
            # Progress container
            with progress_container:
                st.markdown("## Analysis Progress")
                progress_bar = st.progress(0)
                status_text = st.empty()

            # Get date range for historical data
            info("Calculating date range")
            start_date, end_date = get_date_range(days_back)
            info(f"Using date range: {start_date} to {end_date}")
            status_text.text("Calculating date ranges...")
            progress_bar.progress(20)

            # Initialize DataFetcher
            data_fetcher = DataFetcher()

            # Fetch metrics using hello.py approach
            info("Fetching metrics data")
            status_text.text("Fetching metrics data...")
            try:
                list_metrics_all_tickers = data_fetcher.fetch_ticker_info(
                    company_symbols
                )

            except Exception as e:
                error(f"Error fetching metrics data: {e}")
                st.error("An error occurred while fetching metrics data.")
                return  # Exit early if there's an error

            # Debug output to see what we got
            debug(f"Fetched metrics for {len(list_metrics_all_tickers)} tickers")
            progress_bar.progress(35)

            # Convert list to DataFrame for display - handle uneven data structures
            try:
                # First, check if we have a list of dictionaries
                if list_metrics_all_tickers and all(
                    isinstance(item, dict) for item in list_metrics_all_tickers
                ):
                    # Create DataFrame with proper error handling
                    filtered_companies_df = pd.DataFrame(list_metrics_all_tickers)
                else:
                    # Handle case where we have inconsistent data structures
                    warning(
                        "Inconsistent data structures in metrics data. Attempting to normalize."
                    )

                    # Alternative approach - create an empty DataFrame and add rows carefully
                    filtered_companies_df = pd.DataFrame()
                    for ticker_data in list_metrics_all_tickers:
                        if isinstance(ticker_data, dict):
                            # Add one row at a time
                            filtered_companies_df = pd.concat(
                                [filtered_companies_df, pd.DataFrame([ticker_data])],
                                ignore_index=True,
                            )
            except ValueError as ve:
                error(f"Error creating DataFrame: {str(ve)}")
                st.error(f"Error processing metrics data: {str(ve)}")

                # Create a basic DataFrame with just the company symbols to continue
                info("Creating simplified DataFrame with just company symbols")
                filtered_companies_df = pd.DataFrame({"company": company_symbols})

            debug(f"Metrics dataframe shape: {filtered_companies_df.shape}")

            # Debug output
            debug(f"DataFrame columns: {filtered_companies_df.columns.tolist()}")

            # Ensure we have company symbols
            if "company" in filtered_companies_df.columns:
                filtered_company_symbols = filtered_companies_df["company"].tolist()
                debug("Using 'company' column for symbols")
            elif "symbol" in filtered_companies_df.columns:
                filtered_company_symbols = filtered_companies_df["symbol"].tolist()
                debug("Using 'symbol' column for symbols")
            else:
                filtered_company_symbols = company_symbols
                warning("No symbol column found, using original company symbols")

            info(f"Using {len(filtered_company_symbols)} filtered company symbols")

            # Initialize metrics with all the data from the DataFrame
            # Convert DataFrame to dictionary format that matches the expected structure
            metrics = {}
            metrics["company_labels"] = filtered_company_symbols
            metrics["days_history"] = days_back

            # Add all columns from filtered_companies_df to metrics
            for column in filtered_companies_df.columns:
                if (
                    column != "company" and column != "symbol"
                ):  # Avoid duplicating these
                    metrics[column] = filtered_companies_df[column].tolist()

            # Debug output
            debug(f"Combined metrics keys: {list(metrics.keys())}")
            progress_bar.progress(65)

            status_text.text("Analysis complete!")
            progress_bar.progress(100)
            info("Core analysis completed successfully")

            # Create tabs for different views
            tab1, tab2 = st.tabs(["📊 Metrics Overview", "📈 Detailed Analysis"])

            with tab1:
                st.markdown("## Analysis Results")
                display_metrics_dashboard(metrics)

            with tab2:
                # Display detailed candle charts
                st.markdown("## Stock Charts")
                st.markdown(
                    "*Charts are limited to 50% of screen width for better visibility*"
                )
                info("Starting candle chart generation")

                # Call the plotting function with debug output
                try:
                    info("Generating candle charts")

                    # Create required analyzer instances for ChartGenerator
                    info("Creating analyzer instances for ChartGenerator")
                    metrics_analyzer = FinancialMetrics()
                    market_profile_analyzer = MarketProfileAnalyzer()
                    sentiment_analyzer = SentimentAnalyzer()

                    # Initialize ChartGenerator with required parameters
                    chart_generator = ChartGenerator(
                        data_fetcher,
                        metrics_analyzer,
                        market_profile_analyzer,
                        sentiment_analyzer,
                    )

                    info("chart_generator.plot_candle_charts_per_symbol to run next")

                    # Add a container for the charts with limited width
                    chart_container = st.container()
                    with chart_container:
                        # Use the properly initialized chart_generator instance
                        # We'll need to modify the ChartGenerator class to respect width settings
                        # For now, we'll wrap the output in a container
                        st.markdown(
                            "<div style='max-width: 50%; margin: 0 auto;'>",
                            unsafe_allow_html=True,
                        )
                        chart_generator.plot_candle_charts_per_symbol(
                            start_date,
                            end_date,
                            metrics,
                            price_option,
                        )
                        st.markdown("</div>", unsafe_allow_html=True)

                    info("Candle charts generated successfully")
                except Exception as chart_error:
                    error_msg = f"Error generating charts: {str(chart_error)}"
                    error(error_msg)
                    error(traceback.format_exc())
                    st.error(error_msg)

        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            error(error_msg)
            error(traceback.format_exc())
            st.error(error_msg)

            # Show more detailed error information in a collapsible section
            with st.expander("Error Details"):
                st.code(traceback.format_exc())


if __name__ == "__main__":
    try:
        info("Starting application")
        main()
        info("Application completed successfully")
    except Exception as e:
        critical(f"Unhandled exception: {str(e)}")
        critical(traceback.format_exc())
        st.error(f"Critical application error: {str(e)}")
        with st.expander("Error Details", expanded=False):
            st.code(traceback.format_exc())
