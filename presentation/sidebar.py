import streamlit as st
import pandas as pd
from utils.logger import info, debug, error
from presentation.vix_chart import VIXChartManager
from presentation.pe_chart import PERatioChartManager
from presentation.fear_greed_chart import FearGreedChartManager
from presentation.styles import apply_sidebar_styling


class SidebarManager:
    """Class to handle all Streamlit sidebar operations"""

    def __init__(self):
        """Initialize the sidebar manager"""
        info("Initializing SidebarManager")
        # Define colors for different filter categories
        self.colors = {
            "financial": "#1E88E5",  # Blue
            "profitability": "#43A047",  # Green
            "debt": "#E53935",  # Red
            "cash_flow": "#FBC02D",  # Yellow
            "growth": "#8E24AA",  # Purple
        }

        # Initialize chart managers
        self.fear_greed_manager = FearGreedChartManager()
        self.vix_manager = VIXChartManager()
        self.pe_manager = PERatioChartManager()

        # Initialize session state for tracking filter changes
        if "previous_config" not in st.session_state:
            st.session_state.previous_config = {}

        if "should_run_analysis" not in st.session_state:
            st.session_state.should_run_analysis = True  # True on first load

    def display(self):
        """Display all sidebar elements and return configuration"""
        with st.sidebar:
            self.display_market_sentiment()
            config = self.get_config_inputs()

            # Check if config has changed compared to previous run
            if self._has_config_changed(config):
                st.session_state.should_run_analysis = True
                st.session_state.previous_config = config.copy()
                debug("Filter configuration changed, triggering new analysis")

            return config

    def _has_config_changed(self, current_config):
        """Check if the current configuration differs from the previous one"""
        # Compare current config with stored previous config
        previous_config = st.session_state.previous_config

        # If no previous config exists, we should run analysis
        if not previous_config:
            return True

        # Compare relevant configuration options
        key_configs = [
            "days_back",
            "price_option",
        ]

        # Check if basic settings have changed
        for key in key_configs:
            if key in current_config and key in previous_config:
                if current_config[key] != previous_config[key]:
                    debug(f"Config change detected in {key}")
                    return True

        # Check filter settings
        filter_keys = [
            "pe_filter",
            "pb_filter",
            "gm_filter",
            "roe_filter",
            "div_yield_filter",
            "peg_filter",
        ]

        for key in filter_keys:
            if key in current_config and key in previous_config:
                current_filter = current_config[key]
                previous_filter = previous_config[key]

                # Compare each filter property
                if (
                    current_filter["enabled"] != previous_filter["enabled"]
                    or current_filter["min"] != previous_filter["min"]
                    or current_filter["max"] != previous_filter["max"]
                ):
                    debug(f"Filter change detected in {key}")
                    return True

        # No changes detected
        return False

    def display_market_sentiment(self):
        """Display market sentiment indicators in the sidebar"""
        st.sidebar.markdown("### Market Sentiment")

        # Get the days_back from the config
        days_back = st.session_state.get(
            "days_back", 2555
        )  # Default to ~7 years if not set

        # Create columns for the charts
        col1 = st.sidebar.container()
        col2 = st.sidebar.container()
        col3 = st.sidebar.container()

        # Display Fear & Greed Index chart in col1
        with col1:
            self.fear_greed_manager.display_fear_greed_chart_compact(days=days_back)

        # Display VIX in col2
        with col2:
            self.vix_manager.display_vix_chart_compact(days=days_back)

        # Display PE Ratio in col3
        with col3:
            self.pe_manager.display_pe_chart_compact(days=days_back)

    def get_config_inputs(self):
        """Get configuration inputs from sidebar"""
        # Apply sidebar filter styling
        apply_sidebar_styling()

        info("Getting configuration inputs from sidebar")
        st.markdown(
            "<h3 style='margin-top: 25px;'>Configuration</h3>", unsafe_allow_html=True
        )
        st.markdown(
            "<hr style='margin: 5px 0px 15px 0px; border-color: rgba(0,0,0,0.1);'>",
            unsafe_allow_html=True,
        )

        # Load tickers directly from CSV file
        file_path = "neomerc.csv"
        try:
            info(f"Loading tickers from {file_path}")
            # Read the CSV file with comment lines handled properly
            df = pd.read_csv(file_path, comment="#", skip_blank_lines=True)
            print(df["ticker"].tolist())

            # Extract only the ticker symbols
            company_symbols = df["ticker"].tolist() if "ticker" in df.columns else []

            # Remove any potential NaN values or empty strings
            company_symbols = [
                ticker
                for ticker in company_symbols
                if isinstance(ticker, str) and ticker.strip()
            ]

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
            2555,
            365,
            help="Number of days to look back for historical data analysis",
        )
        debug(f"Selected days_back: {days_back}")

        # Store in session state for use in charts
        st.session_state["days_back"] = days_back

        # Price type selection
        st.markdown("#### Price Area Analysis")
        price_option = st.radio(
            "Select price threshold:",
            options=[
                (
                    "va_high",
                    "Current Price inside VA",
                ),  # Keep name for backward compatibility
                (
                    "poc_price",
                    "Current Price below POC",
                ),  # Keep name for backward compatibility
                ("disabled", "Disable Price Area Filter"),
            ],
            format_func=lambda x: x[1],
            index=1,  # Select "Current Price below POC" by default
            help="""
            Value Area (VA): The price range where 70% of the trading volume occurred. Stocks with current price inside this range are considered fairly valued.
            
            Point of Control (POC): The price level with the highest traded volume. Stocks with current price below POC may present buying opportunities.
            
            These filters help identify stocks that are trading in favorable price zones based on volume profile analysis.
            """,
        )
        debug(f"Selected price option: {price_option}")

        # Add a note to make it clear what these filters do
        if price_option != "disabled":
            desc = {
                "va_high": "Showing only stocks with current price INSIDE the Value Area (between VA Low and VA High)",
                "poc_price": "Showing only stocks with current price BELOW the Point of Control",
            }
            filter_desc = desc.get(
                price_option[0] if isinstance(price_option, tuple) else price_option, ""
            )
            if filter_desc:
                st.info(filter_desc)

        # Improved Filter Functions
        def create_filter_section(title, color, filters_dict):
            """Create a styled filter section with multiple filters"""
            st.markdown(
                f"""
                <div class="filter-section" style="border-left-color: {color};">
                    <div class="filter-header">{title} Filters</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            return filters_dict

        def create_enhanced_filter(
            label,
            min_default=0,
            max_default=100,
            is_percent=False,
            enabled_default=False,
            help_text="",
            min_help="",
            max_help="",
            max_value=None,
            single_value=False,
            unit="",
            step=1,
            key=None,
        ):
            """Create an enhanced filter with better UX and consistent numeric types"""
            # Create the filter container
            filter_container = st.container()

            # Convert all numeric values to the same type (float)
            min_default = float(min_default)
            if max_default is not None:
                max_default = float(max_default)
            if max_value is not None:
                max_value = float(max_value)
            step = float(step)

            # If max_value is not specified, use max_default as the max_value
            if max_value is None:
                max_value = max_default * 2 if max_default > 0 else 100.0

            with filter_container:
                # Enable/disable checkbox
                enabled = st.checkbox(label, value=enabled_default, help=help_text)

                # Filter values UI
                if enabled:
                    if single_value:
                        # For single value filters
                        min_val = st.number_input(
                            f"Minimum {label}{' (%)' if is_percent else ''}",
                            min_value=0.0,
                            value=min_default,
                            help=min_help,
                        )
                        max_val = None
                    else:
                        # For range filters
                        cols = st.columns([1, 1])
                        with cols[0]:
                            min_val = st.number_input(
                                f"Min{' (%)' if is_percent else ''}",
                                min_value=0.0,
                                max_value=max_value,
                                value=min_default,
                                help=min_help,
                            )
                        with cols[1]:
                            max_val = st.number_input(
                                f"Max{' (%)' if is_percent else ''}",
                                min_value=0.0,
                                max_value=max_value,
                                value=max_default,
                                help=max_help,
                            )

                        # Visual indicator of range
                        range_text = (
                            f"{min_val} to {max_val}{' %' if is_percent else unit}"
                        )
                        st.markdown(
                            f"<div class='filter-values'>Range: {range_text}</div>",
                            unsafe_allow_html=True,
                        )
                else:
                    min_val = None
                    max_val = None

            return {"enabled": enabled, "min": min_val, "max": max_val}

        # Create filter sections with the new styling
        st.markdown(
            "### Default filters are set for a value investing strategy",
            unsafe_allow_html=True,
        )

        # Financial Filters Section - Keep P/E, P/B, and Gross Margin
        with st.expander("Financial Filters", expanded=True):
            # P/E Ratio filter
            pe_filter = create_enhanced_filter(
                "P/E Ratio",
                0.0,
                20.0,
                is_percent=True,
                enabled_default=False,
                help_text="Price-to-Earnings (P/E) ratio - lower values may indicate undervaluation",
                min_help="Minimum P/E ratio (0 for no minimum)",
                max_help="Maximum P/E ratio (value investors typically prefer P/E < 15)",
            )

            # Price to Book filter
            pb_filter = create_enhanced_filter(
                "Price to Book",
                0.0,
                2,
                enabled_default=False,
                help_text="Price-to-Book (P/B) ratio - lower values may indicate undervaluation",
                min_help="Minimum Price to Book ratio (0 for no minimum)",
                max_help="Maximum Price to Book ratio (value investors typically prefer P/B < 1.5)",
            )

            # Gross Margin filter
            gm_filter = create_enhanced_filter(
                "Gross Margin",
                25.0,
                100.0,
                is_percent=True,
                enabled_default=True,  # Enable Gross Margin filter by default
                max_value=100.0,
                help_text="Gross Margin percentage - higher values indicate better profitability",
                min_help="Minimum Gross Margin percentage (value investors look for healthy margins)",
                max_help="Maximum Gross Margin percentage (100 for no maximum)",
            )

        # Profitability Filters Section - Add back ROE filter
        with st.expander("Profitability Filters", expanded=True):
            # Return on Equity filter
            roe_filter = create_enhanced_filter(
                "Return on Equity",
                10.0,
                100.0,
                is_percent=True,
                enabled_default=False,
                max_value=100.0,
                help_text="Return on Equity (ROE) percentage - higher values indicate better capital efficiency",
                min_help="Minimum Return on Equity percentage (value investors prefer > 10%)",
                max_help="Maximum Return on Equity percentage (100 for no maximum)",
            )

        # Cash Flow Filters Section - Add back Dividend Yield filter
        with st.expander("Cash Flow Filters", expanded=False):
            # Add a "Dividend Payers Only" checkbox
            dividend_payers_only = st.checkbox(
                "Show Only Dividend Paying Companies",
                value=False,
                help="When enabled, only companies that pay dividends will be shown (dividend yield > 0)",
            )

            # Dividend Yield filter
            div_yield_filter = create_enhanced_filter(
                "Dividend Yield",
                0.5,
                10.0,
                is_percent=True,
                enabled_default=False,
                max_value=50.0,
                help_text="Dividend Yield percentage - income from dividends",
                min_help="Minimum Dividend Yield percentage",
                max_help="Maximum Dividend Yield percentage (very high yields may be unsustainable)",
            )

        # Growth Value Filters Section - Add back PEG filter
        with st.expander("Growth Value Filters", expanded=False):
            # PEG Ratio filter
            peg_filter = create_enhanced_filter(
                "PEG Ratio",
                0.0,
                5.0,
                enabled_default=False,
                max_value=20.0,
                help_text="Price/Earnings to Growth ratio - lower values may indicate growth at reasonable price",
                min_help="Minimum PEG Ratio",
                max_help="Maximum PEG Ratio (value < 1 may indicate undervaluation)",
            )

        info(f"Configuration complete.")

        # Return configuration with the relevant filtering values
        return {
            "company_symbols": company_symbols,
            "days_back": days_back,
            "price_option": price_option,
            "pe_filter": pe_filter,
            "pb_filter": pb_filter,
            "gm_filter": gm_filter,
            "roe_filter": roe_filter,
            "div_yield_filter": div_yield_filter,
            "peg_filter": peg_filter,
            "dividend_payers_only": dividend_payers_only,
        }
