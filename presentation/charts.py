import streamlit as st
import textwrap
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
import matplotlib.dates as mdates
from scipy import stats

# Import custom logger functions
from utils.logger import info, debug, warning, error, critical

# Import styles module
from presentation.styles import (
    setup_chart_grid,
    enhance_chart_aesthetics,
    style_price_indicators,
    style_volume_profile,
    style_candlestick_chart,
)

# Import metrics display
from presentation.metrics_display import FinancialMetricsDisplay

# Import helper functions
from utils.helpers import (
    calculate_price_bins,
    distribute_volume_by_price,
    check_price_in_value_area,
)


class CandlestickCharts:
    """Class for handling candlestick chart functionality"""

    def __init__(self, parent):
        """Initialize with reference to parent ChartGenerator"""
        self.parent = parent

    def _identify_price_levels(self, data, window=10, prominence=0.01):
        """
        Identify resistance and support levels based on price peaks and valleys

        Parameters:
        - data: OHLC price data
        - window: lookback window for peak detection
        - prominence: minimum relative height to be considered a significant peak

        Returns:
        - resistance_levels: list of detected resistance prices
        - support_levels: list of detected support prices
        """
        highs = data["High"].values
        lows = data["Low"].values

        resistance_levels = []
        support_levels = []

        # Find peaks (resistance) and valleys (support) in a single pass
        for i in range(window, len(highs) - window):
            # Check for resistance (peaks in highs)
            if highs[i] == max(highs[i - window : i + window + 1]):
                prominence_value = (highs[i] - np.mean(highs)) / np.mean(highs)
                if prominence_value >= prominence:
                    resistance_levels.append(highs[i])

            # Check for support (valleys in lows)
            if lows[i] == min(lows[i - window : i + window + 1]):
                prominence_value = (np.mean(lows) - lows[i]) / np.mean(lows)
                if prominence_value >= prominence:
                    support_levels.append(lows[i])

        # Consolidate similar levels
        resistance_levels = self._consolidate_price_levels(resistance_levels)
        support_levels = self._consolidate_price_levels(support_levels)

        return resistance_levels, support_levels

    def _consolidate_price_levels(self, levels, threshold=0.005):
        """
        Consolidate similar price levels to avoid cluttering the chart
        """
        if not levels:
            return []

        levels = sorted(levels)
        consolidated = []

        current_group = [levels[0]]

        for i in range(1, len(levels)):
            if abs(levels[i] - current_group[0]) / current_group[0] <= threshold:
                current_group.append(levels[i])
            else:
                # Add average of current group
                consolidated.append(sum(current_group) / len(current_group))
                current_group = [levels[i]]

        # Add the last group
        if current_group:
            consolidated.append(sum(current_group) / len(current_group))

        return consolidated

    def _calculate_trend_lines(self, data, days=30):
        """
        Calculate both up and down trend lines using linear regression

        Parameters:
        - data: OHLC price data
        - days: number of days to use for trend calculation

        Returns:
        - Tuple containing parameters for both trend lines:
          (up_slope, up_intercept, down_slope, down_intercept, x_range, date_index)
        """
        # Use only the specified number of most recent days
        recent_data = data.iloc[-days:].copy()

        # Convert index to numeric for regression
        x = np.arange(len(recent_data))

        # Calculate uptrend using highs
        up_slope, up_intercept, _, _, _ = stats.linregress(
            x, recent_data["High"].values
        )

        # Calculate downtrend using lows
        down_slope, down_intercept, _, _, _ = stats.linregress(
            x, recent_data["Low"].values
        )

        return up_slope, up_intercept, down_slope, down_intercept, x, recent_data.index

    def _display_market_profile_chart(
        self, ticker_symbol, data, va_high, va_low, poc_price, option=None
    ):
        """Display market profile chart with improved layout and styling"""
        # Calculate price bins and distribute volume
        price_levels, bin_size, _ = calculate_price_bins(data)
        buy_volume_by_price, sell_volume_by_price = distribute_volume_by_price(
            data, price_levels, bin_size
        )

        # Get current price
        current_price = self.parent._get_current_price(ticker_symbol, data)

        # Check if price meets filtering criteria
        if not check_price_in_value_area(
            current_price, va_high, va_low, poc_price, option
        ):
            info(f"{ticker_symbol} - filtered out based on price criteria")
            return 0

        # Setup chart components
        fig, gs, axes = setup_chart_grid(fig_size=(12, 9), is_volume_profile=True)

        # Draw the main candlestick chart
        style_candlestick_chart(axes["price"], axes["volume"], plt.gcf(), data)

        # Properly format the date axis based on the data timespan
        days_span = (data.index[-1] - data.index[0]).days

        # Format x-axis date labels based on the number of days
        if days_span > 1095:  # More than 3 years
            date_format = mdates.DateFormatter("%b\n%Y")  # Month and Year
            locator = mdates.MonthLocator(interval=3)  # Show every 3 months
        elif days_span > 365:  # More than 1 year
            date_format = mdates.DateFormatter("%b\n%Y")  # Month and Year
            locator = mdates.MonthLocator(interval=2)  # Show every 2 months
        elif days_span > 180:  # More than 6 months
            date_format = mdates.DateFormatter("%b\n%Y")
            locator = mdates.MonthLocator(interval=1)  # Show every month
        else:  # Less than 6 months
            date_format = mdates.DateFormatter("%d\n%b")  # Day and Month
            locator = mdates.WeekdayLocator(interval=2)  # Show every 2 weeks

        # Apply the formatting to both axes
        for ax in [axes["price"], axes["volume"]]:
            ax.xaxis.set_major_formatter(date_format)
            ax.xaxis.set_major_locator(locator)

        # Add price analysis lines and indicators
        poc_line, va_high_line, va_low_line, current_line = style_price_indicators(
            axes["price"], poc_price, va_high, va_low, current_price
        )

        # Add resistance and support levels
        resistance_levels, support_levels = self._identify_price_levels(data)

        info(
            f"{ticker_symbol} - Found {len(resistance_levels)} resistance levels and {len(support_levels)} support levels"
        )

        # Draw resistance levels
        for i, level in enumerate(resistance_levels):
            label = f"Resistance {level:.2f}" if i == 0 else ""
            axes["price"].axhline(
                y=level,
                color="red",
                linestyle="-.",
                linewidth=1.5,
                alpha=0.8,
                label=label,
            )

        # Draw support levels
        for i, level in enumerate(support_levels):
            label = f"Support {level:.2f}" if i == 0 else ""
            axes["price"].axhline(
                y=level,
                color="green",
                linestyle="-.",
                linewidth=1.5,
                alpha=0.8,
                label=label,
            )

        # Add trend lines if we have enough data
        if len(data) >= 30:
            up_slope, up_intercept, down_slope, down_intercept, x_range, date_index = (
                self._calculate_trend_lines(data)
            )

            # Plot trend lines if they have meaningful slopes
            x_dates = mdates.date2num(date_index.to_pydatetime())

            if abs(up_slope) > 0.001:
                y_uptrend = up_slope * x_range + up_intercept
                axes["price"].plot(
                    x_dates, y_uptrend, "b-", linewidth=2, alpha=0.9, label="Uptrend"
                )
                info(f"{ticker_symbol} - Added uptrend line with slope {up_slope:.4f}")

            if abs(down_slope) > 0.001:
                y_downtrend = down_slope * x_range + down_intercept
                axes["price"].plot(
                    x_dates,
                    y_downtrend,
                    "r-",
                    linewidth=2,
                    alpha=0.9,
                    label="Downtrend",
                )
                info(
                    f"{ticker_symbol} - Added downtrend line with slope {down_slope:.4f}"
                )

        # Add volume profile visualization
        style_volume_profile(
            axes["profile"],
            price_levels,
            bin_size,
            buy_volume_by_price,
            sell_volume_by_price,
            poc_price,
        )

        # Finalize chart formatting
        plt.tight_layout()
        enhance_chart_aesthetics(axes["price"], price_levels)

        # Check if buy focus regions were detected
        buy_focus_detected = self.parent._detect_buy_focus_regions(
            price_levels, bin_size, buy_volume_by_price, poc_price
        )

        # Set the legend
        axes["price"].legend(loc="upper left")

        return fig


class ChartGenerator:
    """Class for generating various financial charts and visualizations"""

    def __init__(
        self,
        data_fetcher,
        market_profile_analyzer,
        sentiment_analyzer,
    ):
        """Initialize with required service dependencies"""
        self.data_fetcher = data_fetcher
        self.market_profile_analyzer = market_profile_analyzer
        self.sentiment_analyzer = sentiment_analyzer
        self.metrics_display = FinancialMetricsDisplay()
        self.candlestick_charts = CandlestickCharts(self)

    def _format_business_summary(self, summary):
        """Format business summary for display"""
        summary_no_colons = summary.replace(":", "\:")
        wrapped_summary = textwrap.fill(summary_no_colons)
        return wrapped_summary

    def _get_current_price(self, ticker_symbol, data=None):
        """Helper method to get current price with multiple fallbacks"""
        current_price = None

        # Try to fetch a fresh ticker to get the most updated price
        ticker = yf.Ticker(ticker_symbol)

        # Try multiple fields for getting current price with fallbacks
        if hasattr(ticker, "info") and ticker.info is not None:
            # First try the standard currentPrice field
            if (
                "currentPrice" in ticker.info
                and ticker.info["currentPrice"] is not None
            ):
                current_price = ticker.info["currentPrice"]
            # Then try regularMarketPrice as a fallback
            elif (
                "regularMarketPrice" in ticker.info
                and ticker.info["regularMarketPrice"] is not None
            ):
                current_price = ticker.info["regularMarketPrice"]
            # Then try the last closing price as a fallback
            elif (
                "previousClose" in ticker.info
                and ticker.info["previousClose"] is not None
            ):
                current_price = ticker.info["previousClose"]

        # If all else fails, try to get the most recent closing price from our data
        if current_price is None and data is not None and not data.empty:
            current_price = data["Close"].iloc[-1]

        # Ensure we have a numeric price
        if current_price is not None:
            try:
                current_price = float(current_price)
            except (ValueError, TypeError):
                current_price = None

        return current_price

    def _detect_buy_focus_regions(
        self, price_levels, bin_size, buy_volume_by_price, poc_price
    ):
        """Helper method to detect if buy focus regions exist"""
        # Find POC bin index
        poc_bin_idx = min(
            range(len(price_levels) - 1),
            key=lambda i: abs((price_levels[i] + price_levels[i + 1]) / 2 - poc_price),
        )

        # Check for buy focus regions
        for i in range(poc_bin_idx):
            if buy_volume_by_price[i] > 0.7 * buy_volume_by_price[poc_bin_idx]:
                return True
        return False

    def plot_with_volume_profile(
        self,
        ticker_symbol,
        start_date,
        end_date,
        combined_metrics,
        option,
    ):
        """Plot a candle chart with volume profile for a given ticker symbol"""
        info(f"Fetching data for {ticker_symbol} from {start_date} to {end_date}")
        data_fetcher = self.data_fetcher
        ticker = data_fetcher.fetch_ticker_data(ticker_symbol)
        data = data_fetcher.fetch_historical_data(ticker_symbol, start_date, end_date)

        # Add defensive check for ticker and ticker.info
        if ticker is None:
            warning(f"Ticker is None for symbol {ticker_symbol}")
            return None

        # Get website URL with fallback
        website = "#"
        if hasattr(ticker, "info") and ticker.info is not None:
            website = ticker.info.get("website", "#")

        if not data.empty:
            info(f"Calculating market profile for {ticker_symbol}")
            # Calculate market profile
            va_high, va_low, poc_price, _ = (
                self.market_profile_analyzer.calculate_market_profile(data)
            )

            # Get current price and check if it meets filtering criteria
            current_price = self._get_current_price(ticker_symbol, data)

            if not check_price_in_value_area(
                current_price, va_high, va_low, poc_price, option
            ):
                info(f"{ticker_symbol} - filtered out based on price criteria")
                return 0

            # Get company information
            if ticker and hasattr(ticker, "info"):
                website = ticker.info.get("website", "#")
                shortName = ticker.info.get("shortName", ticker_symbol)

                # Create a clean container for the entire ticker dashboard
                with st.expander(
                    f"{shortName} ({ticker_symbol}) \n\nP/E: {ticker.info.get('trailingPE', 'N/A')}, \n\nP/S: {ticker.info.get('priceToSalesTrailing12Months', 'N/A')}, \n\nP/B: {ticker.info.get('priceToBook', 'N/A')}, \n\nMCap: ${ticker.info.get('marketCap', 0)/1e9:.2f}B, \n\nOp Margin: {ticker.info.get('operatingMargins', 0)*100:.1f}%",
                    expanded=False,
                ):
                    # Header with company name and link
                    header_with_link = f"[🔗]({website}){shortName} - {ticker_symbol}"
                    st.markdown(f"## {header_with_link}", unsafe_allow_html=True)

                    # Display metrics in a clean dashboard at the top
                    self.metrics_display.display_ticker_metrics_dashboard(ticker)

                    # Main content area with two columns - ENSURE CONSISTENT SIZING
                    col1, col2 = st.columns([1, 1])  # Equal width columns

                    with col1:
                        # Display the chart in the left column (taking half the width)
                        try:
                            fig = self.candlestick_charts._display_market_profile_chart(
                                ticker_symbol, data, va_high, va_low, poc_price, option
                            )
                            if (
                                fig != 0
                            ):  # Check if figure was returned and not a filter code
                                st.pyplot(fig)
                        except TypeError:
                            fig = self.candlestick_charts._display_market_profile_chart(
                                ticker_symbol, data, va_high, va_low, poc_price
                            )
                            if (
                                fig != 0
                            ):  # Check if figure was returned and not a filter code
                                st.pyplot(fig)

                    with col2:
                        # Display news articles
                        self.sentiment_analyzer.display_news_without_sentiment(
                            ticker_symbol
                        )

            else:
                warning(f"No ticker info found for {ticker_symbol}")
                return 0

        else:
            warning(f"No data found for {ticker_symbol} in the given date range.")
            return 0

    def plot_candle_charts_per_symbol(self, start_date, end_date, df, price_option):
        """
        Plot candle charts for the filtered symbols

        Parameters:
        - start_date: Start date for chart data
        - end_date: End date for chart data
        - df: DataFrame containing all metrics data
        - price_option: String indicating which price display option to use
        """
        # Get the list of symbols to plot
        symbols = df["symbol"].tolist()
        info(f"Plotting charts for {len(symbols)} symbols")

        # For each symbol, create the chart
        for symbol in symbols:
            # Get the row for this symbol
            symbol_data = df[df["symbol"] == symbol]

            if symbol_data.empty:
                warning(f"No data found for symbol {symbol}")
                continue

            # Extract market profile data if available
            def safe_extract(df, column):
                """Safely extract a value from a DataFrame"""
                if column in df.columns and not df.empty:
                    val = df[column].iloc[0]
                    # Handle different data types
                    if isinstance(val, (list, tuple)) and len(val) > 0:
                        return val[0]
                    return val
                return None

            # Safely extract values using our helper function
            poc_price = safe_extract(symbol_data, "poc_price")
            va_high = safe_extract(symbol_data, "va_high")
            va_low = safe_extract(symbol_data, "va_low")

            debug(
                f"Symbol: {symbol}, POC: {poc_price}, VA High: {va_high}, VA Low: {va_low}"
            )

            try:
                # Plot with volume profile
                self.plot_with_volume_profile(
                    symbol, start_date, end_date, df, price_option
                )
            except Exception as e:
                error(f"Error creating chart for {symbol}: {str(e)}")
                st.error(f"Failed to create chart for {symbol}")
