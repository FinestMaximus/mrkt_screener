import streamlit as st
import pandas as pd
from utils.logger import info, debug, warning, error
from presentation.styles import create_metrics_container, get_dataframe_config
from analysis.market_profile import MarketProfileAnalyzer


class DashboardManager:
    """Class to handle metrics dashboard display and configuration"""

    def __init__(self):
        """Initialize the dashboard manager"""
        info("Initializing DashboardManager")

    def display_metrics_dashboard(self, df, filters=None):
        """
        Display the metrics dashboard using the DataFrame

        Parameters:
        - df: DataFrame containing all metrics data
        - filters: Dictionary of filter settings
        """
        info("Displaying metrics dashboard")

        # Create metrics cards at the top using the styling helper
        def render_metrics():
            total_companies = len(df["symbol"].tolist())
            info(f"Total companies in analysis: {total_companies}")
            st.metric(
                label="Total Companies",
                value=f"{total_companies}",
            )

        # Use the container helper from styles
        create_metrics_container(render_metrics)

        # Verify array lengths before creating DataFrame
        array_lengths = {
            k: len(v)
            for k, v in df.items()
            if isinstance(v, list) and k != "days_history"
        }

        if len(set(array_lengths.values())) > 1:
            error(f"Inconsistent array lengths detected: {array_lengths}")
            st.error("Data inconsistency detected. Please try again.")
            return

        try:
            debug(f"Dashboard dataframe shape: {df.shape}")
            debug(f"Dashboard dataframe columns: {df.columns.tolist()}")

            # Apply filters if provided
            if filters:
                df = self._apply_filters(df, filters)
                debug(f"After filters, dataframe shape: {df.shape}")

            # Display dataframe with dynamic column configuration
            info("Displaying final metrics dashboard")
            column_config = self._create_column_config(df)

            # Get standard dataframe configuration from styles
            df_config = get_dataframe_config()

            st.dataframe(
                df,
                column_config=column_config,
                **df_config,  # Apply all standard config options
            )
            info("Metrics dashboard displayed successfully")

        except Exception as e:
            error(f"Error creating DataFrame: {str(e)}")
            st.error(f"Error processing metrics data: {str(e)}")

    def _apply_filters(self, df, filters):
        """Apply financial filters to the dataframe"""
        original_count = len(df)
        info(f"Applying filters to dataframe with {original_count} rows")

        # Create a copy to avoid SettingWithCopyWarning
        df = df.copy()

        # Convert numeric columns to proper numeric types
        numeric_columns = [
            "trailingPE",
            "forwardPE",
            "priceToBook",
            "grossMargins",
            "returnOnEquity",
            "currentPrice",
            "dividendYield",
            "forwardPegRatio",
        ]

        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Log column data types to help with debugging
        info(f"DataFrame columns and types: {df.dtypes}")

        # Debug info about data
        for col in numeric_columns:
            if col in df.columns:
                non_null = df[col].count()
                total = len(df)
                info(
                    f"Column {col}: {non_null}/{total} non-null values ({non_null/total*100:.1f}%)"
                )
                if non_null > 0:
                    info(f"  Range: {df[col].min()} to {df[col].max()}")

        # Apply P/E Ratio filter if enabled
        if "pe_filter" in filters and filters["pe_filter"]["enabled"]:
            pe_min = filters["pe_filter"]["min"]
            pe_max = filters["pe_filter"]["max"]

            # Try to filter by trailing P/E first, then forward P/E if necessary
            pe_columns = ["trailingPE", "forwardPE"]
            for pe_col in pe_columns:
                if pe_col in df.columns and df[pe_col].count() > 0:
                    debug(f"Filtering by {pe_col} between {pe_min} and {pe_max}")
                    df = self._apply_range_filter(df, pe_col, pe_min, pe_max)
                    info(f"After {pe_col} filter: {len(df)} rows remaining")
                    break
            else:
                warning(
                    "No usable P/E column found in dataframe. P/E filter not applied."
                )

        # Apply PEG Ratio filter if enabled
        if "peg_filter" in filters and filters["peg_filter"]["enabled"]:
            peg_min = filters["peg_filter"]["min"]
            peg_max = filters["peg_filter"]["max"]

            info(f"PEG filter enabled with min={peg_min}, max={peg_max}")

            # Check if PEG data exists at all
            peg_columns = ["forwardPegRatio", "pegRatio"]
            peg_col_exists = False
            for col in peg_columns:
                if col in df.columns:
                    peg_col_exists = True
                    info(
                        f"Found PEG column: {col} with {df[col].notnull().sum()} non-null values"
                    )
                    if df[col].notnull().sum() > 0:
                        # Show sample of values
                        sample_values = df[col].dropna().head(5).tolist()
                        info(f"Sample PEG values: {sample_values}")

            if not peg_col_exists:
                warning("No PEG ratio columns found in data. Available columns are:")
                for col in df.columns:
                    info(f"  - {col}")

            # Try different PEG Ratio column names
            for peg_col in peg_columns:
                if peg_col in df.columns and df[peg_col].count() > 0:
                    debug(f"Filtering by {peg_col} between {peg_min} and {peg_max}")

                    # Save the dataframe length before filtering
                    before_filter = len(df)

                    df = self._apply_range_filter(df, peg_col, peg_min, peg_max)

                    info(
                        f"After {peg_col} filter: {len(df)} rows remaining (removed {before_filter - len(df)})"
                    )
                    break
            else:
                warning(
                    "No usable PEG ratio column found in dataframe. PEG filter not applied."
                )

        # Apply Price to Book filter if enabled
        if "pb_filter" in filters and filters["pb_filter"]["enabled"]:
            pb_min = filters["pb_filter"]["min"]
            pb_max = filters["pb_filter"]["max"]

            # Try different Price to Book column names
            pb_columns = ["priceToBook", "pb_ratio"]
            for pb_col in pb_columns:
                if pb_col in df.columns:
                    debug(f"Filtering by {pb_col} between {pb_min} and {pb_max}")
                    df = self._apply_range_filter(df, pb_col, pb_min, pb_max)
                    break
            else:
                warning(
                    "No Price to Book column found in dataframe. P/B filter not applied."
                )

        # Apply Gross Margin filter if enabled
        if "gm_filter" in filters and filters["gm_filter"]["enabled"]:
            gm_min = (
                filters["gm_filter"]["min"] / 100.0
            )  # Convert percentage to decimal
            gm_max = (
                filters["gm_filter"]["max"] / 100.0
            )  # Convert percentage to decimal

            # Try different Gross Margin column names
            gm_columns = ["grossMargins", "gross_margins"]
            for gm_col in gm_columns:
                if gm_col in df.columns:
                    debug(f"Filtering by {gm_col} between {gm_min} and {gm_max}")
                    df = self._apply_range_filter(df, gm_col, gm_min, gm_max)
                    break

        # Apply Return on Equity filter if enabled
        if "roe_filter" in filters and filters["roe_filter"]["enabled"]:
            roe_min = (
                filters["roe_filter"]["min"] / 100.0
            )  # Convert percentage to decimal
            roe_max = (
                filters["roe_filter"]["max"] / 100.0
            )  # Convert percentage to decimal

            # Try different ROE column names
            roe_columns = ["returnOnEquity", "roe"]
            for roe_col in roe_columns:
                if roe_col in df.columns:
                    debug(f"Filtering by {roe_col} between {roe_min} and {roe_max}")
                    df = self._apply_range_filter(df, roe_col, roe_min, roe_max)
                    break
            else:
                warning(
                    "No Return on Equity column found in dataframe. ROE filter not applied."
                )

        # Apply Dividend Yield filter if enabled
        if "div_yield_filter" in filters and filters["div_yield_filter"]["enabled"]:
            div_yield_min = (
                filters["div_yield_filter"]["min"] / 100.0
            )  # Convert percentage to decimal
            div_yield_max = (
                filters["div_yield_filter"]["max"] / 100.0
            )  # Convert percentage to decimal

            # Try different Dividend Yield column names
            div_yield_columns = ["dividendYield", "fiveYearAvgDividendYield"]
            for div_yield_col in div_yield_columns:
                if div_yield_col in df.columns:
                    debug(
                        f"Filtering by {div_yield_col} between {div_yield_min} and {div_yield_max}"
                    )
                    df = self._apply_range_filter(
                        df, div_yield_col, div_yield_min, div_yield_max
                    )
                    break
            else:
                warning(
                    "No Dividend Yield column found in dataframe. Dividend filter not applied."
                )

        info(
            f"After applying all filters: {len(df)} rows (filtered out {original_count - len(df)} companies)"
        )

        # If we filtered out too many companies (>90%), log a warning
        if len(df) < original_count * 0.1 and original_count > 10:
            warning(
                f"Filters removed {original_count - len(df)} out of {original_count} companies ({(original_count - len(df))/original_count*100:.1f}%). Consider relaxing filters."
            )

        return df

    def _apply_range_filter(self, df, column, min_value, max_value):
        """Apply min/max range filter to a dataframe column with improved debugging"""
        before_count = len(df)

        # Create masks for min and max filtering
        if min_value > 0:
            # Only filter by minimum if it's greater than zero
            min_mask = df[column].notnull() & (df[column] >= min_value)
            df = df[min_mask]
            mid_count = len(df)
            debug(
                f"  Min filter ({column} >= {min_value}) removed {before_count - mid_count} rows"
            )
        else:
            mid_count = before_count

        # Apply max filter if it's a reasonable limiting value
        if max_value < 100:  # Only apply max filter if it's set to a limiting value
            max_mask = df[column].notnull() & (df[column] <= max_value)
            df = df[max_mask]
            after_count = len(df)
            debug(
                f"  Max filter ({column} <= {max_value}) removed {mid_count - after_count} rows"
            )

        return df

    def _create_column_config(self, df):
        """Create dynamic column configuration based on available columns"""
        # Define column configurations by type
        line_chart_configs = {
            "opCashflow": {
                "title": "Operating Cashflow (4y)",
                "y_min": -100,
                "y_max": 100,
            },
            "repurchaseCapStock": {
                "title": "Stock Repurchase Value (4y)",
                "y_min": -50,
                "y_max": 50,
            },
            "freeCashflow": {
                "title": "Free Cashflow (4y)",
                "y_min": -100,
                "y_max": 100,
            },
            "totalCash": {"title": "Total Cash (4y)", "y_min": 0, "y_max": 1000},
            "totalDebt": {"title": "Total Debt (4y)", "y_min": 0, "y_max": 1000},
        }

        # Build the configuration dictionary
        column_config = {}

        return column_config

    def create_companies_table(self, all_metrics_all_tickers, filters=None):
        """
        Create and return a DataFrame of filtered company metrics.

        Args:
            list_metrics_all_tickers: List of dictionaries containing ticker metrics
            filters: Dictionary containing filter settings

        Returns:
            tuple: (filtered_all_metrics_all_tickers)
        """
        # Create DataFrame from metrics data
        if not all_metrics_all_tickers:
            raise ValueError("No metrics data provided")

        if all(isinstance(item, dict) for item in all_metrics_all_tickers):
            # Create DataFrame with proper error handling
            filtered_all_metrics_all_tickers = pd.DataFrame(all_metrics_all_tickers)
        else:
            # Handle case where we have inconsistent data structures
            warning(
                "Inconsistent data structures in metrics data. Attempting to normalize."
            )
            filtered_all_metrics_all_tickers = pd.DataFrame()
            for ticker_data in all_metrics_all_tickers:
                if isinstance(ticker_data, dict):
                    filtered_all_metrics_all_tickers = pd.concat(
                        [filtered_all_metrics_all_tickers, pd.DataFrame([ticker_data])],
                        ignore_index=True,
                    )

        # Apply financial filters if provided
        if filters and not filtered_all_metrics_all_tickers.empty:
            filtered_all_metrics_all_tickers = self._apply_filters(
                filtered_all_metrics_all_tickers, filters
            )

        return filtered_all_metrics_all_tickers

    def _extract_company_symbols(self, df):
        """Extract company symbols from DataFrame or use fallback"""
        if "symbol" in df.columns:
            symbols = df["symbol"].tolist()
            debug("Using 'symbol' column for symbols")

        return symbols

    # Add a method to filter companies based on price area analysis
    def filter_by_price_area(self, df, price_option):
        """
        Filter companies based on their current price relative to value area using MarketProfileAnalyzer

        Parameters:
        - df: DataFrame containing all metrics including market profile data
        - price_option: String indicating which price filter to apply

        Returns:
        - List of filtered symbols
        """
        if price_option == "disabled":
            # If filtering is disabled, return all symbols
            return df["symbol"].tolist()

        # Extract first part of tuple if price_option is a tuple
        if isinstance(price_option, tuple) and len(price_option) > 0:
            price_option = price_option[0]

        # Create an instance of MarketProfileAnalyzer for proper comparison logic
        market_profile_analyzer = MarketProfileAnalyzer()

        # Create a list to store filtered symbols
        filtered_symbols = []

        # Process each row in the DataFrame
        for _, row in df.iterrows():
            symbol = row["symbol"]

            # Safely extract price data, handling both scalar values and pandas Series
            try:
                # Handle different possible data structures for these values
                def extract_value(field):
                    value = row.get(field)
                    if value is None:
                        return None
                    # Handle list type
                    if isinstance(value, list):
                        return value[0] if value else None
                    # Handle pandas Series type
                    if hasattr(value, "values"):
                        return value.values[0] if len(value.values) > 0 else None
                    # Handle scalar value
                    return value

                current_price = extract_value("currentPrice")
                poc_price = extract_value("poc_price")
                va_high = extract_value("va_high")
                va_low = extract_value("va_low")

                # Log the extracted values for debugging
                debug(
                    f"Symbol: {symbol}, Current: {current_price}, POC: {poc_price}, VA High: {va_high}, VA Low: {va_low}"
                )

                # Skip if any required value is None or not numeric
                if any(
                    not isinstance(val, (int, float))
                    for val in [current_price, poc_price, va_high, va_low]
                ):
                    debug(
                        f"Skipping {symbol} due to missing or non-numeric market profile data"
                    )
                    continue

                # Apply filtering based on price_option directly
                if price_option == "va_high":
                    # INSIDE VALUE AREA: Price must be BETWEEN va_low and va_high
                    if va_low <= current_price <= va_high:
                        debug(f"Symbol {symbol} INSIDE value area - KEEP")
                        filtered_symbols.append(symbol)
                    else:
                        debug(f"Symbol {symbol} OUTSIDE value area - FILTER OUT")
                elif price_option == "poc_price":
                    # BELOW POC: Price must be BELOW poc_price
                    if current_price < poc_price:
                        debug(f"Symbol {symbol} BELOW POC - KEEP")
                        filtered_symbols.append(symbol)
                    else:
                        debug(f"Symbol {symbol} NOT below POC - FILTER OUT")
                elif price_option == "above_va":
                    if current_price > va_high:
                        filtered_symbols.append(symbol)
                elif price_option == "below_va":
                    if current_price < va_low:
                        filtered_symbols.append(symbol)
                elif price_option == "inside_va":
                    if va_low <= current_price <= va_high:
                        filtered_symbols.append(symbol)
                elif price_option == "near_poc":
                    # Price is near point of control (within 2%)
                    poc_distance_percent = (
                        abs(current_price - poc_price) / poc_price * 100
                        if poc_price
                        else 0
                    )
                    if poc_distance_percent < 2:  # Within 2% of POC
                        filtered_symbols.append(symbol)

            except Exception as e:
                warning(f"Error processing {symbol} in filter_by_price_area: {str(e)}")
                # Skip this symbol on error

        info(
            f"Price area filter '{price_option}' returned {len(filtered_symbols)} symbols out of {len(df)}"
        )
        return filtered_symbols
