import streamlit as st
import pandas as pd
from data.tickers_yf_fetcher import DataFetcher
from presentation.charts import ChartGenerator
from utils.helpers import get_date_range
import traceback
from utils.logger import info, debug, warning, error, critical
from analysis.market_profile import MarketProfileAnalyzer
from data.news_research import SentimentAnalyzer
from datetime import datetime
from reporting import ReportGenerator
from presentation.sidebar import SidebarManager
from presentation.dashboard import DashboardManager
from presentation.styles import apply_custom_styling


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

    # Apply custom styling from the styles module
    apply_custom_styling()

    # Create a layout with title and report button side by side
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title("")
    with col2:
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

    # Run analysis automatically when needed
    if st.session_state.should_run_analysis:
        info("Starting analysis with the following configuration:")
        try:
            # Reset the flag
            st.session_state.should_run_analysis = False

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
                all_metrics_all_tickers = data_fetcher.fetch_ticker_info(
                    company_symbols, _progress_callback=update_ticker_progress
                )
            except Exception as e:
                error(f"Error fetching metrics data: {e}")
                st.error("An error occurred while fetching metrics data.")
                return  # Exit early if there's an error

            # Debug output to see what we got
            debug(f"Fetched metrics for {len(all_metrics_all_tickers)} tickers")

            # Update progress
            update_progress(65, "Processing data...")

            # Use dashboard manager to create DataFrame instead of embedding the logic here
            filtered_df = dashboard_manager.create_companies_table(
                all_metrics_all_tickers, filters=filters
            )

            # Extract the filtered company symbols from the dataframe

            # Add days_history column to the DataFrame for reference
            filtered_df["days_history"] = days_back

            # Further filter companies based on price area analysis if needed
            if price_option != "disabled":
                status_text.text("Filtering by price area analysis...")

                # Create required instances
                market_profile_analyzer = MarketProfileAnalyzer()

                # Add market profile data to DataFrame
                for idx, row in filtered_df.iterrows():
                    symbol = row["symbol"]
                    try:
                        # Fetch historical data for this symbol
                        data = data_fetcher.fetch_historical_data(
                            symbol, start_date, end_date
                        )

                        if not data.empty:
                            # Calculate market profile
                            va_high, va_low, poc_price, _ = (
                                market_profile_analyzer.calculate_market_profile(data)
                            )

                            # Get current price (use last closing price as approximation)
                            current_price = (
                                data["Close"].iloc[-1] if not data.empty else None
                            )

                            # Store values in DataFrame
                            filtered_df.at[idx, "poc_price"] = poc_price
                            filtered_df.at[idx, "va_high"] = va_high
                            filtered_df.at[idx, "va_low"] = va_low
                            filtered_df.at[idx, "current_price"] = current_price
                        else:
                            filtered_df.at[idx, "poc_price"] = None
                            filtered_df.at[idx, "va_high"] = None
                            filtered_df.at[idx, "va_low"] = None
                            filtered_df.at[idx, "current_price"] = None
                    except Exception as e:
                        debug(
                            f"Error calculating market profile for {symbol}: {str(e)}"
                        )
                        filtered_df.at[idx, "poc_price"] = None
                        filtered_df.at[idx, "va_high"] = None
                        filtered_df.at[idx, "va_low"] = None
                        filtered_df.at[idx, "current_price"] = None

                # Now filter with the updated DataFrame
                filtered_symbols = dashboard_manager.filter_by_price_area(
                    filtered_df, price_option
                )

                # Update the filtered dataframe to match the new filtered symbols
                filtered_df = filtered_df[filtered_df["symbol"].isin(filtered_symbols)]

                status_text.text(
                    f"Found {len(filtered_df)} companies matching price area filter..."
                )

            debug(f"Metrics dataframe shape: {filtered_df.shape}")
            debug(f"DataFrame columns: {filtered_df.columns.tolist()}")
            info(f"Using {len(filtered_df)} filtered companies")

            update_progress(100, "Analysis complete!")
            info("Core analysis completed successfully")

            # Store the filtered DataFrame in session state for the report generation
            st.session_state["filtered_df"] = filtered_df
            st.session_state["analysis_complete"] = True

            # Create tabs for different views
            tab1, tab2 = st.tabs(["📊 Metrics Overview", "📈 Detailed Analysis"])

            with tab1:
                st.markdown("## Analysis Results")
                dashboard_manager.display_metrics_dashboard(
                    filtered_df, filters=filters
                )

            with tab2:
                # Display detailed candle charts
                st.markdown("## Stock Charts")
                info("Starting candle chart generation")

                try:
                    info("Generating candle charts")

                    # Create required analyzer instances for ChartGenerator
                    info("Creating analyzer instances for ChartGenerator")
                    market_profile_analyzer = MarketProfileAnalyzer()
                    sentiment_analyzer = SentimentAnalyzer()

                    # Initialize ChartGenerator with required parameters
                    chart_generator = ChartGenerator(
                        data_fetcher,
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
                            filtered_df,
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
                filtered_df = st.session_state.get("filtered_df", pd.DataFrame())

                # Generate the report using the report_generator
                result = report_generator.generate_report(filtered_df)

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
            st.warning(
                "Please wait for the analysis to complete before generating a report."
            )


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
