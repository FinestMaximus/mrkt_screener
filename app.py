import streamlit as st
import pandas as pd
from data.tickers_yf_fetcher import DataFetcher
from visualization.charts import ChartGenerator
from utils.helpers import get_date_range, replace_with_zero
import traceback
from utils.logger import info, debug, warning, error, critical
from analysis.metrics import FinancialMetrics
from analysis.market_profile import MarketProfileAnalyzer
from data.news_research import SentimentAnalyzer
from datetime import datetime
from reporting import ReportGenerator
from presentation.sidebar import SidebarManager
from presentation.dashboard import DashboardManager


# Configure page
st.set_page_config(
    layout="wide",
    page_title="Stock Analysis Dashboard",
    initial_sidebar_state="expanded",
)
info("Application started")


def main():
    """Main application entry point"""
    info("Main function started")

    # Create the report generator and dashboard manager at the beginning
    report_generator = ReportGenerator()
    dashboard_manager = DashboardManager()

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

    # Create a layout with title and buttons side by side
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.title("")
    with col2:
        run_analysis = st.button(
            "Run Analysis", use_container_width=True, type="primary"
        )
    with col3:
        generate_report = st.button("📄 Print Report", use_container_width=True)

    # Use the sidebar manager instead of direct sidebar operations
    info("Setting up sidebar")
    sidebar_manager = SidebarManager()
    config = sidebar_manager.display()

    company_symbols = config["company_symbols"]
    days_back = config["days_back"]
    price_option = (
        config["price_option"][0]
        if isinstance(config["price_option"], tuple)
        else config["price_option"]
    )

    # Extract filter settings - only include the filters that exist in the config
    filters = {}
    filter_keys = [
        "pe_filter",
        "pb_filter",
        "gm_filter",
        "roe_filter",
        "div_yield_filter",
        "peg_filter",
    ]

    for key in filter_keys:
        if key in config:
            filters[key] = config[key]

    info(
        f"Filter settings loaded: {', '.join(f'{k}' for k, v in filters.items() if v['enabled'])}"
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

            # Progress reporting function
            def update_progress(percentage, message):
                progress_bar.progress(percentage)
                status_text.text(message)

            update_progress(20, "Calculating date ranges...")

            # Initialize DataFetcher
            data_fetcher = DataFetcher()

            # Define a progress callback function
            def update_ticker_progress(current, total, ticker):
                # Calculate overall progress: 20% for date range + up to 45% for fetching data
                fetch_progress = (current / total) * 45
                overall_progress = 20 + fetch_progress
                update_progress(
                    int(overall_progress),
                    f"Fetching data for {ticker} ({current}/{total})",
                )

            # Fetch metrics using hello.py approach
            info("Fetching metrics data")
            try:
                list_metrics_all_tickers = data_fetcher.fetch_ticker_info(
                    company_symbols, _progress_callback=update_ticker_progress
                )
            except Exception as e:
                error(f"Error fetching metrics data: {e}")
                st.error("An error occurred while fetching metrics data.")
                return  # Exit early if there's an error

            # Debug output to see what we got
            debug(f"Fetched metrics for {len(list_metrics_all_tickers)} tickers")

            # Update progress
            update_progress(65, "Processing data...")

            # Use dashboard manager to create DataFrame instead of embedding the logic here
            filtered_companies_df, filtered_company_symbols = (
                dashboard_manager.create_companies_table(
                    list_metrics_all_tickers, company_symbols, filters=filters
                )
            )

            # Create metrics dictionary from filtered_companies_df
            metrics = {
                "company_labels": filtered_company_symbols,
                "days_history": days_back,
            }

            # Add all columns from filtered_companies_df to metrics
            for column in filtered_companies_df.columns:
                if column not in ["company", "symbol"]:  # Avoid duplicating these
                    metrics[column] = filtered_companies_df[column].tolist()

            # Further filter companies based on price area analysis if needed
            if price_option != "disabled":
                status_text.text("Filtering by price area analysis...")
                filtered_company_symbols = dashboard_manager.filter_by_price_area(
                    metrics, filtered_company_symbols, price_option
                )

                # Update the filtered dataframe to match the new filtered symbols
                filtered_companies_df = filtered_companies_df[
                    filtered_companies_df["symbol"].isin(filtered_company_symbols)
                ]

                # Update metrics with the filtered data
                metrics["company_labels"] = filtered_company_symbols

                # Update all other metrics lists to match the filtered data
                for column in filtered_companies_df.columns:
                    if column not in ["company", "symbol"]:
                        metrics[column] = filtered_companies_df[column].tolist()

                status_text.text(
                    f"Found {len(filtered_company_symbols)} companies matching price area filter..."
                )

            debug(f"Metrics dataframe shape: {filtered_companies_df.shape}")
            debug(f"DataFrame columns: {filtered_companies_df.columns.tolist()}")
            info(f"Using {len(filtered_company_symbols)} filtered company symbols")

            # Debug output
            debug(f"Combined metrics keys: {list(metrics.keys())}")
            update_progress(100, "Analysis complete!")
            info("Core analysis completed successfully")

            # Store the metrics in session state for the report generation
            st.session_state["metrics"] = metrics
            st.session_state["analysis_complete"] = True

            # Create tabs for different views
            tab1, tab2 = st.tabs(["📊 Metrics Overview", "📈 Detailed Analysis"])

            with tab1:
                st.markdown("## Analysis Results")
                dashboard_manager.display_metrics_dashboard(metrics, filters=filters)

            with tab2:
                # Display detailed candle charts
                st.markdown("## Stock Charts")
                info("Starting candle chart generation")

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

                    # Create monkey patches to capture data during analysis
                    report_generator.create_monkey_patches(
                        chart_generator, sentiment_analyzer
                    )

                    # Display charts in container with consistent styling
                    chart_container = st.container()
                    with chart_container:
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
                    error(f"Error generating charts: {str(chart_error)}")
                    error(traceback.format_exc())
                    st.error(f"Error generating charts: {str(chart_error)}")

        except Exception as e:
            error(f"An error occurred: {str(e)}")
            error(traceback.format_exc())
            st.error(f"An error occurred: {str(e)}")

            # Show more detailed error information in a collapsible section
            with st.expander("Error Details"):
                st.code(traceback.format_exc())

            # Reset session state
            if "analysis_complete" in st.session_state:
                st.session_state["analysis_complete"] = False

    # Handle report generation
    if generate_report:
        if (
            "analysis_complete" in st.session_state
            and st.session_state["analysis_complete"]
        ):
            with st.spinner("Generating HTML report..."):
                metrics = st.session_state.get("metrics", {})

                # Generate the report using the report_generator
                result = report_generator.generate_report(metrics)

                if result:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"stock_analysis_report_{timestamp}.html"

                    st.download_button(
                        label="📥 Download HTML Report",
                        data=result,
                        file_name=filename,
                        mime="text/html",
                    )
                    st.success(
                        "HTML report generated successfully! Click above to download."
                    )
                else:
                    st.error(
                        "Failed to generate report. Please check logs for details."
                    )
        else:
            st.warning("Please run the analysis first before generating a report.")


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
