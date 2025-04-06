import plotly.graph_objects as go
from plotly.subplots import make_subplots
import mplfinance as mpf
import streamlit as st
import textwrap
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import yfinance as yf
import matplotlib.dates as mdates

# Import custom logger functions
from analysis.metrics import FinancialMetrics
from data.news_research import SentimentAnalyzer
from utils.logger import info, debug, warning, error, critical


class ChartBase:
    """Base class for chart generators with common functionality"""

    def __init__(self):
        """Initialize the base chart class"""
        pass

    def _format_business_summary(self, summary):
        """Format business summary for display"""
        summary_no_colons = summary.replace(":", "\:")
        wrapped_summary = textwrap.fill(summary_no_colons)
        return wrapped_summary


class CandlestickCharts(ChartBase):
    """Class for generating candlestick charts and volume profiles"""

    def __init__(self, data_fetcher, market_profile_analyzer):
        """Initialize with required services"""
        super().__init__()
        self.data_fetcher = data_fetcher
        self.market_profile_analyzer = market_profile_analyzer

    def create_candle_chart_with_profile(self, data, poc_price, va_high, va_low):
        """Create a candlestick chart with volume profile overlay"""
        if data.empty:
            warning("Cannot create chart with empty data")
            return None

        # Create price-volume data for volume profile
        price_bins = 100
        price_range = data["High"].max() - data["Low"].min()
        bin_size = price_range / price_bins

        # Create figure with price chart, volume chart, and volume profile
        fig = plt.figure(figsize=(12, 8))
        gs = GridSpec(4, 5, figure=fig)

        # Main price chart
        ax1 = fig.add_subplot(gs[0:3, 0:4])

        # Volume chart below price chart
        ax_volume = fig.add_subplot(gs[3:4, 0:4], sharex=ax1)

        # Volume profile chart on the right
        ax2 = fig.add_subplot(gs[0:3, 4], sharey=ax1)

        # Plot candlestick chart with a cleaner style - use custom colors for better visibility
        # Create custom style with desired colors
        mc = mpf.make_marketcolors(
            up="#54ff54",  # Brighter green for up days
            down="#ff5454",  # Brighter red for down days
            edge="inherit",
            wick="inherit",
            volume={"up": "#54ff54", "down": "#ff5454"},
        )
        custom_style = mpf.make_mpf_style(
            marketcolors=mc, gridstyle=":", y_on_right=False
        )

        mpf.plot(
            data,
            type="candle",
            style=custom_style,
            ax=ax1,
            volume=ax_volume,
            show_nontrading=False,
        )

        # Force the x-axis to interpret the tick values as dates
        ax1.xaxis_date()
        ax_volume.xaxis_date()

        # Ensure that the x-axis for both the main chart and volume sub-chart are
        # formatted correctly with dates.
        date_format = mdates.DateFormatter("%b '%y")  # e.g., Jan '23
        ax1.xaxis.set_major_formatter(date_format)
        ax_volume.xaxis.set_major_formatter(date_format)
        fig.autofmt_xdate()  # Automatically format date labels (rotates for better readability)

        # Add POC and Value Area lines
        ax1.axhline(
            y=poc_price, color="red", linestyle="dashed", linewidth=2, label="POC"
        )
        ax1.axhline(
            y=va_high, color="blue", linestyle="dashed", linewidth=1, label="VA High"
        )
        ax1.axhline(
            y=va_low, color="blue", linestyle="dashed", linewidth=1, label="VA Low"
        )

        # Create and plot volume profile
        price_levels = [data["Low"].min() + i * bin_size for i in range(price_bins + 1)]
        volume_by_price = [0] * price_bins

        for idx, row in data.iterrows():
            for i in range(price_bins):
                lower_bound = price_levels[i]
                upper_bound = price_levels[i + 1]
                if not (row["High"] < lower_bound or row["Low"] > upper_bound):
                    volume_by_price[i] += row["Volume"] / (
                        (row["High"] - row["Low"]) / bin_size
                    )

        # Plot volume profile histogram on right side
        ax2.barh(
            price_levels[:-1],
            volume_by_price,
            height=bin_size,
            color="#1f77b4",
            alpha=0.7,
        )

        # Highlight POC in volume profile
        poc_bin_idx = min(
            range(len(price_levels) - 1),
            key=lambda i: abs((price_levels[i] + price_levels[i + 1]) / 2 - poc_price),
        )
        ax2.barh(
            price_levels[poc_bin_idx],
            volume_by_price[poc_bin_idx],
            height=bin_size,
            color="red",
            alpha=0.7,
        )

        # Remove unnecessary ticks
        ax2.set_xticks([])
        ax2.set_yticks([])

        # Add a legend
        ax1.legend(["POC", "VA High", "VA Low"], loc="upper left")

        plt.tight_layout()

        # Further improve the y-axis price display
        # Add currency symbol to y-axis label
        ax1.set_ylabel(
            "Price ($)", fontsize=14, fontweight="bold", color="white", labelpad=10
        )

        # Make the grid lines more prominent
        ax1.grid(
            which="major",
            axis="y",
            linestyle="-",
            linewidth=0.7,  # Slightly thicker
            color="#aaaaaa",  # Lighter color for better visibility
            alpha=0.8,  # More visible
        )

        # Add subtle minor grid lines for more precise price reading
        ax1.grid(
            which="minor",
            axis="y",
            linestyle=":",
            linewidth=0.3,
            color="#666666",
            alpha=0.5,
        )
        ax1.minorticks_on()

        # Improve date axis display
        ax_volume.set_xlabel(
            "Date", fontsize=14, fontweight="bold", color="white", labelpad=10
        )

        # Add background shading to make price bands more visible
        for i in range(0, len(price_levels) - 1, 2):
            if i < len(price_levels) - 1:
                ax1.axhspan(
                    price_levels[i],
                    price_levels[i + 1],
                    color="#333333",
                    alpha=0.2,
                    zorder=-10,
                )

        return fig

    def _display_market_profile_chart(
        self, ticker_symbol, data, va_high, va_low, poc_price, option=None
    ):
        """Display market profile chart with improved layout and styling"""
        # Get price-volume data for volume profile
        price_bins = 100  # Number of price bins for the volume profile
        price_range = data["High"].max() - data["Low"].min()
        bin_size = price_range / price_bins

        # Create price bins
        price_levels = [data["Low"].min() + i * bin_size for i in range(price_bins + 1)]
        buy_volume_by_price = [0] * price_bins
        sell_volume_by_price = [0] * price_bins

        # Distribute volume into price bins, separating buy/sell volume based on price movement
        for i in range(1, len(data)):
            row = data.iloc[i]
            prev_row = data.iloc[i - 1]

            # Determine if day was predominantly buying or selling
            is_up_day = row["Close"] > row["Open"]

            for j in range(price_bins):
                lower_bound = price_levels[j]
                upper_bound = price_levels[j + 1]

                # If price range during the day overlaps with this bin
                if not (row["High"] < lower_bound or row["Low"] > upper_bound):
                    # Calculate volume proportion for this price level
                    volume_contribution = row["Volume"] / (
                        (row["High"] - row["Low"]) / bin_size
                    )

                    # Assign to buy or sell volume based on price action
                    if is_up_day:
                        buy_volume_by_price[j] += volume_contribution
                    else:
                        sell_volume_by_price[j] += volume_contribution

        # Get current price from ticker_symbol
        current_price = None

        # Try to fetch a fresh ticker to get the most updated price
        ticker = yf.Ticker(ticker_symbol)

        # Try multiple fields for getting current price with fallbacks
        if hasattr(ticker, "info"):
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
        if current_price is None and not data.empty:
            current_price = data["Close"].iloc[-1]

        # Ensure we have a numeric price
        if current_price is not None:
            try:
                current_price = float(current_price)
            except (ValueError, TypeError):
                current_price = None

        # Check if option is provided before trying to access it
        if option and len(option) > 0:
            # Check if the price is within the value area based on the selected option
            if option[0] == "va_high":
                # Filter OUT stocks where price is ABOVE VA High
                if current_price > va_high:
                    info(
                        f"{ticker_symbol} - current price is above value area high: {current_price} (VA High: {va_high})"
                    )
                    return 0
            elif option[0] == "poc_price":
                # Filter OUT stocks where price is NOT below POC
                if current_price >= poc_price:
                    info(
                        f"{ticker_symbol} - price is not below POC: {current_price} (POC: {poc_price})"
                    )
                    return 0

        # Set a clean, modern style for better visibility
        plt.style.use("dark_background")

        # Create a figure with improved proportions (taller for better volume visibility)
        fig = plt.figure(
            figsize=(12, 9)
        )  # Increased overall size for better readability

        # Create a GridSpec with more space for price labels
        gs = GridSpec(
            5,
            5,
            figure=fig,
            height_ratios=[3, 3, 3, 1.5, 1.5],
            width_ratios=[
                0.12,
                0.88,
                0.88,
                0.88,
                1,
            ],  # Increased left margin for price labels
        )

        # Main price chart (takes 4/5 of the width and 3/5 of the height)
        ax1 = fig.add_subplot(gs[0:3, 1:4])

        # Volume chart below price chart (larger)
        ax_volume = fig.add_subplot(gs[3:5, 1:4], sharex=ax1)

        # Volume profile chart on the right side
        ax2 = fig.add_subplot(gs[0:3, 4], sharey=ax1)

        # Plot candlestick chart with a cleaner style - use custom colors for better visibility
        # Create custom style with desired colors
        mc = mpf.make_marketcolors(
            up="#54ff54",  # Brighter green for up days
            down="#ff5454",  # Brighter red for down days
            edge="inherit",
            wick="inherit",
            volume={"up": "#54ff54", "down": "#ff5454"},
        )
        custom_style = mpf.make_mpf_style(
            marketcolors=mc, gridstyle=":", y_on_right=False
        )

        mpf.plot(
            data,
            type="candle",
            style=custom_style,
            ax=ax1,
            volume=ax_volume,
            show_nontrading=False,
        )

        # Add prominent grid lines for better price tracking
        ax1.grid(
            which="major",
            axis="y",
            linestyle="-",
            linewidth=0.5,
            color="#888888",
            alpha=0.7,
        )
        ax1.grid(
            which="major",
            axis="x",
            linestyle="-",
            linewidth=0.5,
            color="#888888",
            alpha=0.5,
        )

        # Add the POC and VA lines to the main chart with annotations - use brighter colors for visibility
        ax1.axhline(
            y=poc_price, color="#ff5050", linestyle="dashed", linewidth=2, label="POC"
        )
        ax1.axhline(
            y=va_high,
            color="#5050ff",
            linestyle="dashed",
            linewidth=1.5,
            label="VA High",
        )
        ax1.axhline(
            y=va_low, color="#5050ff", linestyle="dashed", linewidth=1.5, label="VA Low"
        )

        # Add current price line with a darker, more visible color
        if current_price is not None:
            ax1.axhline(
                y=current_price,
                color="#ffcf40",  # Brighter gold color for current price
                linestyle="-",
                linewidth=2.5,
                label="Current Price",
                zorder=10,  # Ensure it's drawn on top of other lines
            )

        # Plot the buy volume profile histogram horizontally on ax2
        ax2.barh(
            price_levels[:-1],
            buy_volume_by_price,
            height=bin_size,
            color="#54ff54",  # Brighter green
            alpha=0.6,
            label="Buy Volume",
        )

        # Plot the sell volume in a different color
        ax2.barh(
            price_levels[:-1],
            sell_volume_by_price,
            height=bin_size,
            color="#ff5454",  # Brighter red
            alpha=0.5,
            label="Sell Volume",
        )

        # Highlight POC and Value Area in the volume profile
        poc_bin_idx = min(
            range(len(price_levels) - 1),
            key=lambda i: abs((price_levels[i] + price_levels[i + 1]) / 2 - poc_price),
        )

        # Highlight the POC bin with high opacity
        ax2.barh(
            price_levels[poc_bin_idx],
            buy_volume_by_price[poc_bin_idx] + sell_volume_by_price[poc_bin_idx],
            height=bin_size,
            color="#cc44cc",  # Brighter purple
            alpha=0.9,
            label="POC",
        )

        # Highlight potential buy zones (prices below POC but with significant buy volume)
        buy_focus_regions = []
        for i in range(poc_bin_idx):
            if buy_volume_by_price[i] > 0.7 * buy_volume_by_price[poc_bin_idx]:
                buy_focus_regions.append(i)

        for idx in buy_focus_regions:
            ax2.barh(
                price_levels[idx],
                buy_volume_by_price[idx],
                height=bin_size,
                color="#00ff00",  # Pure lime for high visibility
                alpha=0.9,
                label="Strong Buy Zone" if idx == buy_focus_regions[0] else "",
            )

        # Remove x-axis labels from the volume profile
        ax2.set_xticks([])

        # Remove y-axis labels from the volume profile (since it shares with main chart)
        ax2.set_yticks([])

        # Add a legend to the volume profile with better positioning and styling
        ax2.legend(
            loc="upper right",
            fontsize="medium",
            framealpha=0.9,
            facecolor="#333333",
            edgecolor="#888888",
        )

        # Force the x-axis to interpret the tick values as dates
        ax1.xaxis_date()
        ax_volume.xaxis_date()

        # Ensure that the x-axis for both the main chart and volume sub-chart are formatted correctly with dates
        date_format = mdates.DateFormatter("%b '%y")  # e.g., Jan '23
        ax1.xaxis.set_major_formatter(date_format)
        ax_volume.xaxis.set_major_formatter(date_format)
        fig.autofmt_xdate()  # Automatically format date labels (rotates for better readability)

        plt.tight_layout()

        # Further improve the y-axis price display
        # Add currency symbol to y-axis label
        ax1.set_ylabel(
            "Price ($)", fontsize=14, fontweight="bold", color="white", labelpad=10
        )

        # Make the grid lines more prominent
        ax1.grid(
            which="major",
            axis="y",
            linestyle="-",
            linewidth=0.7,  # Slightly thicker
            color="#aaaaaa",  # Lighter color for better visibility
            alpha=0.8,  # More visible
        )

        # Add subtle minor grid lines for more precise price reading
        ax1.grid(
            which="minor",
            axis="y",
            linestyle=":",
            linewidth=0.3,
            color="#666666",
            alpha=0.5,
        )
        ax1.minorticks_on()

        # Improve date axis display
        ax_volume.set_xlabel(
            "Date", fontsize=14, fontweight="bold", color="white", labelpad=10
        )

        # Add background shading to make price bands more visible
        for i in range(0, len(price_levels) - 1, 2):
            if i < len(price_levels) - 1:
                ax1.axhspan(
                    price_levels[i],
                    price_levels[i + 1],
                    color="#333333",
                    alpha=0.2,
                    zorder=-10,
                )

        # Update legend on main chart with correct styling
        if current_price is not None:
            # Clear any previous legend
            if ax1.get_legend() is not None:
                ax1.get_legend().remove()

            # Create custom legend entries that match the actual line styles
            from matplotlib.lines import Line2D

            legend_elements = [
                Line2D(
                    [0],
                    [0],
                    color="#ff5050",
                    linestyle="dashed",
                    linewidth=2,
                    label="POC",
                ),
                Line2D(
                    [0],
                    [0],
                    color="#5050ff",
                    linestyle="dashed",
                    linewidth=1.5,
                    label="VA High/Low",
                ),
                Line2D(
                    [0],
                    [0],
                    color="#ffcf40",
                    linestyle="-",
                    linewidth=2.5,
                    label="Current Price",
                ),
            ]

            # If buy zone is present, add it to legend
            if buy_focus_regions:
                legend_elements.append(
                    Line2D(
                        [0],
                        [0],
                        color="#00ff00",
                        linestyle="dotted",
                        linewidth=2,
                        label="Buy Zone",
                    )
                )

            # Add the custom legend to the chart with better styling
            ax1.legend(
                handles=legend_elements,
                loc="upper left",
                fontsize="medium",
                framealpha=0.7,
                facecolor="#333333",
                edgecolor="#555555",
            )

        return fig

    def _display_ticker_metrics_dashboard(
        self, ticker_symbol, ticker, eps, pe, ps, pb, peg, gm
    ):
        """Display ticker metrics in a dashboard layout with improved aesthetics and usability"""
        # Use a container with enhanced styling for the metrics dashboard
        st.markdown(
            """
            <style>
            .metrics-container {
                padding: 15px;
                border-radius: 10px;
                margin-bottom: 20px;
                background-color: rgba(30, 30, 40, 0.6);
                border: 1px solid rgba(100, 100, 150, 0.2);
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            .metric-card {
                background-color: rgba(40, 40, 60, 0.7);
                border-radius: 8px;
                padding: 10px;
                text-align: center;
                height: 100%;
                transition: transform 0.2s, box-shadow 0.2s;
                border: 1px solid rgba(100, 100, 150, 0.2);
            }
            .metric-card:hover {
                transform: translateY(-3px);
                box-shadow: 0 6px 8px rgba(0, 0, 0, 0.2);
            }
            .metric-label {
                font-size: 0.85rem;
                color: rgba(200, 200, 220, 0.9);
                margin-bottom: 5px;
                font-weight: 500;
            }
            .metric-value {
                font-size: 1.4rem;
                font-weight: 600;
                margin-bottom: 0;
            }
            .metric-value.positive {
                color: rgba(100, 255, 100, 0.9);
            }
            .metric-value.negative {
                color: rgba(255, 100, 100, 0.9);
            }
            .metric-value.neutral {
                color: rgba(220, 220, 255, 0.9);
            }
            .metric-desc {
                font-size: 0.75rem;
                color: rgba(180, 180, 200, 0.8);
                margin-top: 5px;
                font-style: italic;
            }
            .metric-container {
                display: flex;
                flex-direction: column;
                height: 100%;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        # Custom function to display metrics with enhanced styling
        def styled_metric(
            label,
            value,
            tooltip="",
            formatter=None,
            positive_good=True,
            prefix="",
            suffix="",
        ):
            # Determine if we should show positive/negative styling
            style_class = "neutral"
            if value is not None and isinstance(value, (int, float)):
                if positive_good:
                    style_class = (
                        "positive"
                        if value > 0
                        else "negative" if value < 0 else "neutral"
                    )
                else:
                    style_class = (
                        "negative"
                        if value > 0
                        else "positive" if value < 0 else "neutral"
                    )

            # Format the value
            if value is None:
                formatted_value = "N/A"
            elif formatter == "percent":
                formatted_value = f"{prefix}{value:.2f}{suffix}%"
            elif formatter == "currency":
                formatted_value = f"{prefix}${value:,.2f}{suffix}"
            elif formatter == "ratio":
                formatted_value = f"{prefix}{value:.2f}{suffix}"
            elif formatter == "integer":
                formatted_value = f"{prefix}{int(value):,}{suffix}"
            elif formatter == "billions":
                formatted_value = f"{prefix}${value/1e9:.2f}B{suffix}"
            elif formatter == "millions":
                formatted_value = f"{prefix}${value/1e6:.2f}M{suffix}"
            else:
                formatted_value = f"{prefix}{value}{suffix}"

            html = f"""
            <div class="metric-container">
                <div class="metric-card" title="{tooltip}">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value {style_class}">{formatted_value}</div>
                    <div class="metric-desc">{tooltip}</div>
                </div>
            </div>
            """
            return html

        with st.container():
            st.markdown('<div class="metrics-container">', unsafe_allow_html=True)

            # Organize metrics into categories with 3 rows of 4 columns
            # First row: Valuation metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                if peg:
                    tooltip = "Price/Earnings to Growth - Lower values (<1) typically indicate better value"
                    st.markdown(
                        styled_metric(
                            "PEG Ratio", peg, tooltip, "ratio", positive_good=False
                        ),
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        styled_metric("PEG Ratio", None, "Price/Earnings to Growth"),
                        unsafe_allow_html=True,
                    )

            with col2:
                if pe:
                    tooltip = "Price to Earnings - How much investors pay per dollar of earnings"
                    st.markdown(
                        styled_metric(
                            "P/E Ratio", pe, tooltip, "ratio", positive_good=False
                        ),
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        styled_metric("P/E Ratio", None, "Price to Earnings"),
                        unsafe_allow_html=True,
                    )

            with col3:
                if pb:
                    tooltip = "Price to Book - Stock price relative to book value"
                    st.markdown(
                        styled_metric(
                            "P/B Ratio", pb, tooltip, "ratio", positive_good=False
                        ),
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        styled_metric("P/B Ratio", None, "Price to Book"),
                        unsafe_allow_html=True,
                    )

            with col4:
                if ps:
                    tooltip = "Price to Sales - Stock price relative to annual revenue"
                    st.markdown(
                        styled_metric(
                            "P/S Ratio", ps, tooltip, "ratio", positive_good=False
                        ),
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        styled_metric("P/S Ratio", None, "Price to Sales"),
                        unsafe_allow_html=True,
                    )

            # Second row: Financial health and performance
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                if eps:
                    tooltip = (
                        "Earnings Per Share - Company profit allocated to each share"
                    )
                    st.markdown(
                        styled_metric(
                            "EPS", eps, tooltip, "currency", positive_good=True
                        ),
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        styled_metric("EPS", None, "Earnings Per Share"),
                        unsafe_allow_html=True,
                    )

            with col2:
                if gm is not None:
                    tooltip = "Gross Margin - Revenue minus cost of goods sold (higher is better)"
                    st.markdown(
                        styled_metric(
                            "Gross Margin",
                            gm * 100,
                            tooltip,
                            "percent",
                            positive_good=True,
                        ),
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        styled_metric(
                            "Gross Margin", None, "Revenue minus cost of goods sold"
                        ),
                        unsafe_allow_html=True,
                    )

            with col3:
                # Market cap with appropriate formatting
                if ticker and hasattr(ticker, "info") and "marketCap" in ticker.info:
                    market_cap = ticker.info["marketCap"]
                    currency = ticker.info.get("financialCurrency", "USD")

                    if market_cap >= 1e9:
                        tooltip = (
                            f"Total market value of outstanding shares ({currency})"
                        )
                        st.markdown(
                            styled_metric(
                                "Market Cap",
                                market_cap,
                                tooltip,
                                "billions",
                                prefix="",
                                suffix=" " + currency,
                            ),
                            unsafe_allow_html=True,
                        )
                    else:
                        tooltip = (
                            f"Total market value of outstanding shares ({currency})"
                        )
                        st.markdown(
                            styled_metric(
                                "Market Cap",
                                market_cap,
                                tooltip,
                                "millions",
                                prefix="",
                                suffix=" " + currency,
                            ),
                            unsafe_allow_html=True,
                        )
                else:
                    st.markdown(
                        styled_metric(
                            "Market Cap",
                            None,
                            "Total market value of outstanding shares",
                        ),
                        unsafe_allow_html=True,
                    )

            with col4:
                # 52-week range information
                if ticker and hasattr(ticker, "info"):
                    current_price = (
                        ticker.info.get("currentPrice")
                        or ticker.info.get("regularMarketPrice")
                        or ticker.info.get("previousClose")
                    )
                    week_low = ticker.info.get("fiftyTwoWeekLow")
                    week_high = ticker.info.get("fiftyTwoWeekHigh")

                    if current_price and week_low and week_high:
                        # Calculate position in 52-week range as percentage
                        range_position = (
                            (current_price - week_low) / (week_high - week_low) * 100
                        )
                        tooltip = f"Current price position in 52-week range: ${week_low:.2f} to ${week_high:.2f}"
                        st.markdown(
                            styled_metric(
                                "52-Week Range", range_position, tooltip, "percent"
                            ),
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            styled_metric(
                                "52-Week Range",
                                None,
                                "Current price position in 52-week range",
                            ),
                            unsafe_allow_html=True,
                        )
                else:
                    st.markdown(
                        styled_metric(
                            "52-Week Range",
                            None,
                            "Current price position in 52-week range",
                        ),
                        unsafe_allow_html=True,
                    )

            # Third row: Analyst ratings and dividend information
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                # Analyst rating
                if (
                    ticker
                    and hasattr(ticker, "info")
                    and "recommendationMean" in ticker.info
                    and ticker.info["recommendationMean"] is not None
                ):
                    rec_value = ticker.info["recommendationMean"]

                    # Create a description based on the rating
                    rec_desc = ""
                    if rec_value <= 1.5:
                        rec_desc = "Strong Buy"
                    elif rec_value <= 2.5:
                        rec_desc = "Buy"
                    elif rec_value <= 3.5:
                        rec_desc = "Hold"
                    elif rec_value <= 4.5:
                        rec_desc = "Sell"
                    else:
                        rec_desc = "Strong Sell"

                    tooltip = f"Average analyst rating: {rec_desc} (Scale: 1=Strong Buy to 5=Strong Sell)"
                    # For ratings, lower is better
                    st.markdown(
                        styled_metric(
                            "Analyst Rating",
                            rec_value,
                            tooltip,
                            "ratio",
                            positive_good=False,
                            suffix="/5",
                        ),
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        styled_metric(
                            "Analyst Rating", None, "Average analyst recommendation"
                        ),
                        unsafe_allow_html=True,
                    )

            with col2:
                # Dividend yield
                if (
                    ticker
                    and hasattr(ticker, "info")
                    and "dividendYield" in ticker.info
                    and ticker.info["dividendYield"] is not None
                ):
                    dividend_yield = (
                        ticker.info["dividendYield"] * 100
                    )  # Convert to percentage
                    tooltip = "Annual dividend payment as percentage of share price"
                    st.markdown(
                        styled_metric(
                            "Dividend Yield",
                            dividend_yield,
                            tooltip,
                            "percent",
                            positive_good=True,
                        ),
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        styled_metric(
                            "Dividend Yield",
                            None,
                            "Annual dividend payment as percentage of share price",
                        ),
                        unsafe_allow_html=True,
                    )

            with col3:
                # Beta
                if (
                    ticker
                    and hasattr(ticker, "info")
                    and "beta" in ticker.info
                    and ticker.info["beta"] is not None
                ):
                    beta = ticker.info["beta"]
                    tooltip = "Stock volatility vs. market (>1 = more volatile, <1 = less volatile)"
                    st.markdown(
                        styled_metric("Beta", beta, tooltip, "ratio"),
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        styled_metric(
                            "Beta",
                            None,
                            "Measure of stock volatility relative to the market",
                        ),
                        unsafe_allow_html=True,
                    )

            with col4:
                # Return on Equity (ROE)
                if (
                    ticker
                    and hasattr(ticker, "info")
                    and "returnOnEquity" in ticker.info
                    and ticker.info["returnOnEquity"] is not None
                ):
                    roe = ticker.info["returnOnEquity"] * 100  # Convert to percentage
                    tooltip = "Net income as a percentage of shareholder equity (higher is better)"
                    st.markdown(
                        styled_metric(
                            "ROE", roe, tooltip, "percent", positive_good=True
                        ),
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        styled_metric(
                            "ROE",
                            None,
                            "Return on Equity - Profitability relative to equity",
                        ),
                        unsafe_allow_html=True,
                    )

            # Add an expandable section for additional metrics
            with st.expander("Show More Metrics", expanded=False):
                cols = st.columns(4)

                metric_count = 0
                additional_metrics = [
                    (
                        "Debt to Equity",
                        "debtToEquity",
                        "ratio",
                        False,
                        "Debt relative to equity (lower is better)",
                    ),
                    (
                        "Current Ratio",
                        "currentRatio",
                        "ratio",
                        True,
                        "Current assets divided by current liabilities (>1 is good)",
                    ),
                    (
                        "Profit Margin",
                        "profitMargins",
                        "percent",
                        True,
                        "Net profit as percentage of revenue",
                    ),
                    (
                        "Operating Margin",
                        "operatingMargins",
                        "percent",
                        True,
                        "Operating profit as percentage of revenue",
                    ),
                    (
                        "Target Price",
                        "targetMedianPrice",
                        "currency",
                        None,
                        "Median analyst price target",
                    ),
                    (
                        "Price/Cash Flow",
                        "priceToOperCashPerShare",
                        "ratio",
                        False,
                        "Share price relative to operating cash flow per share",
                    ),
                    (
                        "Rev. Growth (YoY)",
                        "revenueGrowth",
                        "percent",
                        True,
                        "Year-over-year revenue growth",
                    ),
                    (
                        "Earnings Growth",
                        "earningsGrowth",
                        "percent",
                        True,
                        "Year-over-year earnings growth",
                    ),
                    (
                        "Shares Outstanding",
                        "sharesOutstanding",
                        "integer",
                        None,
                        "Total shares issued by company",
                    ),
                    (
                        "Short Ratio",
                        "shortRatio",
                        "ratio",
                        False,
                        "Days to cover short positions",
                    ),
                    (
                        "Book Value/Share",
                        "bookValue",
                        "currency",
                        True,
                        "Company's book value per share",
                    ),
                    (
                        "50-Day Avg",
                        "fiftyDayAverage",
                        "currency",
                        None,
                        "Average closing price over last 50 days",
                    ),
                ]

                for (
                    metric_name,
                    info_key,
                    formatter,
                    positive_good,
                    tooltip,
                ) in additional_metrics:
                    if metric_count % 4 == 0 and metric_count > 0:
                        cols = st.columns(4)

                    with cols[metric_count % 4]:
                        # Check if ticker has this metric
                        if (
                            ticker
                            and hasattr(ticker, "info")
                            and info_key in ticker.info
                            and ticker.info[info_key] is not None
                        ):
                            value = ticker.info[info_key]
                            # Handle percentage conversions
                            if (
                                formatter == "percent"
                                and value < 1
                                and info_key not in ["shortRatio", "currentRatio"]
                            ):
                                value *= 100

                            # For metrics where we don't know if positive is good
                            if positive_good is None:
                                st.markdown(
                                    styled_metric(
                                        metric_name,
                                        value,
                                        tooltip,
                                        formatter,
                                        positive_good=None,
                                    ),
                                    unsafe_allow_html=True,
                                )
                            else:
                                st.markdown(
                                    styled_metric(
                                        metric_name,
                                        value,
                                        tooltip,
                                        formatter,
                                        positive_good=positive_good,
                                    ),
                                    unsafe_allow_html=True,
                                )
                        else:
                            st.markdown(
                                styled_metric(metric_name, None, tooltip),
                                unsafe_allow_html=True,
                            )

                    metric_count += 1

            # Close container
            st.markdown("</div>", unsafe_allow_html=True)

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

        # Get dashboard metrics
        info(f"Retrieving dashboard metrics for {ticker_symbol}")
        dashboard_metrics = FinancialMetrics().get_dashboard_metrics(
            ticker_symbol, combined_metrics
        )
        (
            eps,
            pe,
            ps,
            pb,
            peg,
            gm,
            wh52,
            wl52,
            currentPrice,
            targetMedianPrice,
            targetLowPrice,
            targetMeanPrice,
            targetHighPrice,
            recommendationMean,
        ) = dashboard_metrics

        if not data.empty:
            info(f"Calculating market profile for {ticker_symbol}")
            # Calculate market profile
            va_high, va_low, poc_price, _ = (
                self.market_profile_analyzer.calculate_market_profile(data)
            )

            # Get current price from ticker_symbol - FIX STARTS HERE
            current_price = None

            # Try to fetch a fresh ticker to get the most updated price
            ticker = yf.Ticker(ticker_symbol)

            # Try multiple fields for getting current price with fallbacks
            if hasattr(ticker, "info"):
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
            if current_price is None and not data.empty:
                current_price = data["Close"].iloc[-1]

            # Ensure we have a numeric price
            if current_price is not None:
                try:
                    current_price = float(current_price)
                except (ValueError, TypeError):
                    current_price = None

            # FIX ENDS HERE

            # Check if the price is within the value area based on the selected option
            if option[0] == "va_high":
                # Filter OUT stocks where price is ABOVE VA High
                if current_price > va_high:
                    info(
                        f"{ticker_symbol} - current price is above value area high: {current_price} (VA High: {va_high})"
                    )
                    return 0
            elif option[0] == "poc_price":
                # Filter OUT stocks where price is NOT below POC
                if current_price >= poc_price:
                    info(
                        f"{ticker_symbol} - price is not below POC: {current_price} (POC: {poc_price})"
                    )
                    return 0

            # Get company information
            if ticker and hasattr(ticker, "info"):
                website = ticker.info.get("website", "#")
                shortName = ticker.info.get("shortName", ticker_symbol)

                # Create a clean container for the entire ticker dashboard
                with st.container():
                    st.markdown("---")  # Divider for visual separation

                    # Header with company name and link
                    header_with_link = f"[🔗]({website}){shortName} - {ticker_symbol}"
                    st.markdown(f"## {header_with_link}", unsafe_allow_html=True)

                    # Display metrics in a clean dashboard at the top
                    self._display_ticker_metrics_dashboard(
                        ticker_symbol, ticker, eps, pe, ps, pb, peg, gm
                    )

                    # Main content area with two columns - ENSURE CONSISTENT SIZING
                    col1, col2 = st.columns([1, 1])  # Equal width columns

                    with col1:
                        # Display the chart in the left column (taking half the width)
                        try:
                            fig = self._display_market_profile_chart(
                                ticker_symbol, data, va_high, va_low, poc_price, option
                            )
                            if (
                                fig != 0
                            ):  # Check if figure was returned and not a filter code
                                st.pyplot(fig)
                        except TypeError:
                            fig = self._display_market_profile_chart(
                                ticker_symbol, data, va_high, va_low, poc_price
                            )
                            if (
                                fig != 0
                            ):  # Check if figure was returned and not a filter code
                                st.pyplot(fig)

                    with col2:
                        # Display news articles
                        SentimentAnalyzer().display_news_without_sentiment(
                            ticker_symbol
                        )

                # Add some spacing after each ticker section - keep minimal for scrollable container
                st.markdown("<br>", unsafe_allow_html=True)

            else:
                warning(f"No ticker info found for {ticker_symbol}")
                return 0

        else:
            warning(f"No data found for {ticker_symbol} in the given date range.")
            return 0

    def plot_candle_charts_per_symbol(
        self,
        start_date,
        end_date,
        metrics,
        option,
    ):
        """Plot candle charts for each symbol organized by sector"""
        info("Started plotting candle charts for each symbol")

        critical("Inputs for plotting candle charts:")
        critical(f"Start Date: {start_date}")
        critical(f"End Date: {end_date}")
        critical(f"Combined Metrics: {metrics.get('company_labels')}")
        critical(f"Option: {option}")

        for ticker_symbol in metrics.get("company_labels"):
            info(f"Attempting to plot candle chart for symbol: {ticker_symbol}")

            response = self.plot_with_volume_profile(
                ticker_symbol,
                start_date,
                end_date,
                metrics,
                option,
            )

            if response == 0:
                info(f"Skipped plotting for {ticker_symbol} due to no response")
                continue

        info("Finished plotting candle charts for all symbols")


class MetricsCharts(ChartBase):
    """Class for generating financial metrics visualizations"""

    def __init__(self, metrics_analyzer):
        """Initialize with metrics analyzer"""
        super().__init__()
        self.metrics_analyzer = metrics_analyzer

    def create_combined_metrics_chart(self, combined_metrics):
        """Create a combined interactive chart with multiple subplots for company comparison"""
        if not combined_metrics or not isinstance(combined_metrics, dict):
            raise ValueError("combined_metrics must be a non-empty dictionary.")

        company_labels = combined_metrics.get("company_labels", [])
        eps_values = combined_metrics.get("eps_values", [])

        if not all(len(lst) == len(company_labels) for lst in [eps_values]):
            raise ValueError("Inconsistent data lengths found in combined_metrics.")

        high_diffs = [
            combined_metrics["price_diff"].get(company, {}).get("high_diff", 0)
            for company in company_labels
        ]
        low_diffs = [
            combined_metrics["price_diff"].get(company, {}).get("low_diff", 0)
            for company in company_labels
        ]
        market_caps = combined_metrics.get("market_caps", [])
        priceToBook = combined_metrics.get("priceToBook", [])
        pe_values = combined_metrics.get("pe_values", [])
        peg_values = combined_metrics.get("peg_values", [])
        priceToSalesTrailing12Months = combined_metrics.get(
            "priceToSalesTrailing12Months", []
        )
        gross_margins = combined_metrics.get("gross_margins", [])
        recommendations_summary = combined_metrics.get("recommendations_summary", [])
        earningsGrowth = combined_metrics.get("earningsGrowth", [])
        revenueGrowth = combined_metrics.get("revenueGrowth", [])
        freeCashflow = combined_metrics.get("freeCashflow", [])
        opCashflow = combined_metrics.get("opCashflow", [])
        repurchaseCapStock = combined_metrics.get("repurchaseCapStock", [])

        peg_min, peg_max = min(peg_values, default=0), max(peg_values, default=1)

        fig = make_subplots(
            rows=4,
            cols=3,
            subplot_titles=(
                "Price Difference % Over the Last Year",
                "EPS vs P/E Ratio",
                "Gross Margin (%)",
                "EPS vs P/B Ratio",
                "EPS vs PEG Ratio",
                "EPS vs P/S Ratio",
                "Upgrades & Downgrades Timeline",
                "Earnings Growth vs Revenue Growth",
                "Free Cash Flow",
                "Operational Cashflow",
                "Repurchase of Capital Stock",
            ),
            specs=[
                [{}, {}, {}],
                [{}, {}, {}],
                [{"colspan": 2}, None, {}],
                [{}, {}, {}],
            ],
            vertical_spacing=0.1,
        )

        colors = {
            company: f"hsl({(i / len(company_labels) * 360)},100%,50%)"
            for i, company in enumerate(company_labels)
        }

        for i, company in enumerate(company_labels):
            try:
                legendgroup = f"group_{company}"
                marker_size = max(market_caps[i] / max(market_caps, default=1) * 50, 5)

                fig.add_trace(
                    go.Scatter(
                        x=[high_diffs[i]],
                        y=[low_diffs[i]],
                        marker=dict(size=10, color=colors[company]),
                        legendgroup=legendgroup,
                        name=company,
                        hoverinfo="none",
                        hovertemplate=f"Company: {company}<br>High Diff: %{{x}}<br>Low Diff: %{{y}}<extra></extra>",
                    ),
                    row=1,
                    col=1,
                )

                fig.add_trace(
                    go.Scatter(
                        x=[eps_values[i]],
                        y=[pe_values[i]],
                        marker=dict(size=marker_size, color=colors[company]),
                        legendgroup=legendgroup,
                        showlegend=False,
                        hoverinfo="none",
                        hovertemplate=f"Company: {company}<br>EPS: ${eps_values[i]}<br>P/E Ratio: {pe_values[i]}<extra></extra>",
                    ),
                    row=1,
                    col=2,
                )

                # Continue adding all the traces as in the original code
                # ...

                # Add more traces for the other metrics

            except (ValueError, TypeError, IndexError) as error:
                error(f"Error plotting data for {company}: {error}")
                continue

        titles = [
            ("High Diff (%)", "Low Diff (%)"),
            ("EPS", "P/E Ratio"),
            ("Company", "Gross Margin (%)"),
            ("Price to Books", "EPS"),
            ("PEG", "EPS"),
            ("P/S", "EPS"),
            ("Earnings Growth", "Revenue Growth"),
            ("Years", "Free Cash Flow"),
            ("Years", "Operational Cashflow"),
            ("Years", "Repurchase of Capital Stock"),
        ]

        for col, (x_title, y_title) in enumerate(titles, start=1):
            fig.update_xaxes(title_text=x_title, row=1, col=col)
            fig.update_yaxes(title_text=y_title, row=1, col=col)

        fig.update_xaxes(title_text="Recommendation Type", row=1, col=4)
        fig.update_yaxes(title_text="Number of Recommendations", row=1, col=4)

        fig.update_layout(height=1500)

        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    buttons=list(
                        [
                            dict(
                                args=[{"visible": "legendonly"}],
                                label="Hide All",
                                method="restyle",
                            ),
                            dict(
                                args=[{"visible": True}],
                                label="Show All",
                                method="restyle",
                            ),
                        ]
                    ),
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0,
                    xanchor="left",
                    y=-0.15,
                    yanchor="top",
                ),
            ]
        )

        return fig

    def plot_combined_interactive(self, combined_metrics):
        """Create an interactive dashboard with multiple financial metrics visualizations"""
        if not combined_metrics or not isinstance(combined_metrics, dict):
            raise ValueError("combined_metrics must be a non-empty dictionary.")

        company_labels = combined_metrics.get("company_labels", [])
        eps_values = combined_metrics.get("eps_values", [])

        if not all(len(lst) == len(company_labels) for lst in [eps_values]):
            raise ValueError("Inconsistent data lengths found in combined_metrics.")

        high_diffs = [
            combined_metrics["price_diff"].get(company, {}).get("high_diff", 0)
            for company in company_labels
        ]
        low_diffs = [
            combined_metrics["price_diff"].get(company, {}).get("low_diff", 0)
            for company in company_labels
        ]
        market_caps = combined_metrics.get("market_caps", [])
        priceToBook = combined_metrics.get("priceToBook", [])
        pe_values = combined_metrics.get("pe_values", [])
        peg_values = combined_metrics.get("peg_values", [])
        priceToSalesTrailing12Months = combined_metrics.get(
            "priceToSalesTrailing12Months", []
        )
        gross_margins = combined_metrics.get("gross_margins", [])
        recommendations_summary = combined_metrics.get("recommendations_summary", [])
        earningsGrowth = combined_metrics.get("earningsGrowth", [])
        revenueGrowth = combined_metrics.get("revenueGrowth", [])
        freeCashflow = combined_metrics.get("freeCashflow", [])
        opCashflow = combined_metrics.get("opCashflow", [])
        repurchaseCapStock = combined_metrics.get("repurchaseCapStock", [])

        peg_min, peg_max = min(peg_values, default=0), max(peg_values, default=1)

        fig = make_subplots(
            rows=4,
            cols=3,
            subplot_titles=(
                "Price Difference % Over the Last Year",
                "EPS vs P/E Ratio",
                "Gross Margin (%)",
                "EPS vs P/B Ratio",
                "EPS vs PEG Ratio",
                "EPS vs P/S Ratio",
                "Upgrades & Downgrades Timeline",
                "Earnings Growth vs Revenue Growth",
                "Free Cash Flow",
                "Operational Cashflow",
                "Repurchase of Capital Stock",
            ),
            specs=[
                [{}, {}, {}],
                [{}, {}, {}],
                [{"colspan": 2}, None, {}],
                [{}, {}, {}],
            ],
            vertical_spacing=0.1,
        )

        colors = {
            company: f"hsl({(i / len(company_labels) * 360)},100%,50%)"
            for i, company in enumerate(company_labels)
        }

        for i, company in enumerate(company_labels):
            try:
                legendgroup = f"group_{company}"
                marker_size = max(market_caps[i] / max(market_caps, default=1) * 50, 5)

                # Add traces for each visualization
                self._add_chart_traces(
                    fig,
                    company,
                    i,
                    colors,
                    legendgroup,
                    marker_size,
                    high_diffs,
                    low_diffs,
                    eps_values,
                    pe_values,
                    gross_margins,
                    priceToBook,
                    peg_values,
                    priceToSalesTrailing12Months,
                    recommendations_summary,
                    earningsGrowth,
                    revenueGrowth,
                    freeCashflow,
                    opCashflow,
                    repurchaseCapStock,
                    company_labels,
                )

            except (ValueError, TypeError, IndexError) as error:
                error(f"Error plotting data for {company}: {error}")
                continue

        # Set chart titles and labels
        self._configure_chart_axes(fig)

        fig.update_layout(height=1500)
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    buttons=list(
                        [
                            dict(
                                args=[{"visible": "legendonly"}],
                                label="Hide All",
                                method="restyle",
                            ),
                            dict(
                                args=[{"visible": True}],
                                label="Show All",
                                method="restyle",
                            ),
                        ]
                    ),
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0,
                    xanchor="left",
                    y=-0.15,
                    yanchor="top",
                ),
            ]
        )

        return fig

    def _add_chart_traces(
        self,
        fig,
        company,
        i,
        colors,
        legendgroup,
        marker_size,
        high_diffs,
        low_diffs,
        eps_values,
        pe_values,
        gross_margins,
        priceToBook,
        peg_values,
        priceToSalesTrailing12Months,
        recommendations_summary,
        earningsGrowth,
        revenueGrowth,
        freeCashflow,
        opCashflow,
        repurchaseCapStock,
        company_labels,
    ):
        """Helper method to add traces to the combined chart"""
        # Price difference scatter plot
        fig.add_trace(
            go.Scatter(
                x=[high_diffs[i]],
                y=[low_diffs[i]],
                marker=dict(size=10, color=colors[company]),
                legendgroup=legendgroup,
                name=company,
                hoverinfo="none",
                hovertemplate=f"Company: {company}<br>High Diff: %{{x}}<br>Low Diff: %{{y}}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        # EPS vs P/E scatter plot
        fig.add_trace(
            go.Scatter(
                x=[eps_values[i]],
                y=[pe_values[i]],
                marker=dict(size=marker_size, color=colors[company]),
                legendgroup=legendgroup,
                showlegend=False,
                hoverinfo="none",
                hovertemplate=f"Company: {company}<br>EPS: ${eps_values[i]}<br>P/E Ratio: {pe_values[i]}<extra></extra>",
            ),
            row=1,
            col=2,
        )

        # Gross margin bar chart
        fig.add_trace(
            go.Bar(
                x=[company_labels[i]],
                y=[gross_margins[i] * 100],
                marker=dict(color=colors[company]),
                legendgroup=legendgroup,
                showlegend=False,
                width=0.8,
            ),
            row=1,
            col=3,
        )

        # EPS vs P/B scatter plot
        fig.add_trace(
            go.Scatter(
                x=[eps_values[i]],
                y=[priceToBook[i]],
                marker=dict(size=marker_size, color=colors[company]),
                legendgroup=legendgroup,
                showlegend=False,
                hoverinfo="none",
                hovertemplate=f"Company: {company}<br>EPS: ${eps_values[i]}<br>P/B Ratio: {priceToBook[i]}<extra></extra>",
            ),
            row=2,
            col=1,
        )

        # EPS vs PEG scatter plot
        fig.add_trace(
            go.Scatter(
                x=[eps_values[i]],
                y=[peg_values[i]],
                marker=dict(size=marker_size, color=colors[company]),
                legendgroup=legendgroup,
                showlegend=False,
                hoverinfo="none",
                hovertemplate=f"Company: {company}<br>EPS: ${eps_values[i]}<br>PEG Ratio: {peg_values[i]}<extra></extra>",
            ),
            row=2,
            col=2,
        )

        # EPS vs P/S scatter plot
        fig.add_trace(
            go.Scatter(
                x=[eps_values[i]],
                y=[priceToSalesTrailing12Months[i]],
                marker=dict(size=marker_size, color=colors[company]),
                legendgroup=legendgroup,
                showlegend=False,
                hoverinfo="none",
                hovertemplate=f"Company: {company}<br>EPS: ${eps_values[i]}<br>P/S Ratio: {priceToSalesTrailing12Months[i]}<extra></extra>",
            ),
            row=2,
            col=3,
        )

        # Add recommendations summary if available
        self._add_recommendations_trace(
            fig, company, i, colors, legendgroup, recommendations_summary
        )

        # Add cashflow traces
        self._add_cashflow_traces(
            fig,
            company,
            i,
            colors,
            legendgroup,
            freeCashflow,
            opCashflow,
            repurchaseCapStock,
        )

        # Add growth comparison trace
        fig.add_trace(
            go.Scatter(
                x=[revenueGrowth[i]],
                y=[earningsGrowth[i]],
                marker=dict(size=10, color=colors[company]),
                legendgroup=legendgroup,
                showlegend=False,
                hoverinfo="none",
                hovertemplate=f"Company: {company}<br>Revenue Growth: {revenueGrowth[i]}<br>Earnings Growth: {earningsGrowth[i]}<extra></extra>",
            ),
            row=3,
            col=3,
        )

    def _configure_chart_axes(self, fig):
        """Configure axes titles and ranges for the combined chart"""
        titles = [
            ("High Diff (%)", "Low Diff (%)"),
            ("EPS", "P/E Ratio"),
            ("Company", "Gross Margin (%)"),
            ("Price to Books", "EPS"),
            ("PEG", "EPS"),
            ("P/S", "EPS"),
            ("Earnings Growth", "Revenue Growth"),
            ("Years", "Free Cash Flow"),
            ("Years", "Operational Cashflow"),
            ("Years", "Repurchase of Capital Stock"),
        ]

        for col, (x_title, y_title) in enumerate(titles, start=1):
            fig.update_xaxes(title_text=x_title, row=1, col=col)
            fig.update_yaxes(title_text=y_title, row=1, col=col)

        fig.update_xaxes(title_text="Recommendation Type", row=1, col=4)
        fig.update_yaxes(title_text="Number of Recommendations", row=1, col=4)

    @staticmethod
    def get_dash_metrics(ticker_symbol, combined_metrics):
        """Get dashboard metrics for a ticker"""
        # Default return values
        default_return = (None,) * 14  # Returns 14 None values

        try:
            # First check if all required keys exist
            required_keys = [
                "company_labels",
                "eps_values",
                "pe_values",
                "priceToSalesTrailing12Months",
                "priceToBook",
                "peg_values",
                "gross_margins",
                "fiftyTwoWeekHigh",
                "fiftyTwoWeekLow",
                "currentPrice",
                "targetMedianPrice",
                "targetLowPrice",
                "targetMeanPrice",
                "targetHighPrice",
                "recommendationMean",
            ]

            # Check if all required keys exist
            for key in required_keys:
                if key not in combined_metrics:
                    info(f"Missing key in combined_metrics: '{key}'")
                    return default_return

            if ticker_symbol in combined_metrics["company_labels"]:
                index = combined_metrics["company_labels"].index(ticker_symbol)

                # Check if index is valid for all lists
                for key in required_keys[1:]:  # Skip company_labels
                    if len(combined_metrics[key]) <= index:
                        info(
                            f"Index {index} out of range for key '{key}' with length {len(combined_metrics[key])}"
                        )
                        return default_return

                eps = combined_metrics["eps_values"][index]
                pe = combined_metrics["pe_values"][index]
                ps = combined_metrics["priceToSalesTrailing12Months"][index]
                pb = combined_metrics["priceToBook"][index]
                peg = combined_metrics["peg_values"][index]
                gm = combined_metrics["gross_margins"][index]
                wh52 = combined_metrics["fiftyTwoWeekHigh"][index]
                wl52 = combined_metrics["fiftyTwoWeekLow"][index]
                currentPrice = combined_metrics["currentPrice"][index]
                targetMedianPrice = combined_metrics["targetMedianPrice"][index]
                targetLowPrice = combined_metrics["targetLowPrice"][index]
                targetMeanPrice = combined_metrics["targetMeanPrice"][index]
                targetHighPrice = combined_metrics["targetHighPrice"][index]
                recommendationMean = combined_metrics["recommendationMean"][index]

                return (
                    eps,
                    pe,
                    ps,
                    pb,
                    peg,
                    gm,
                    wh52,
                    wl52,
                    currentPrice,
                    targetMedianPrice,
                    targetLowPrice,
                    targetMeanPrice,
                    targetHighPrice,
                    recommendationMean,
                )
            else:
                info(f"Ticker '{ticker_symbol}' not found in the labels list.")
                return default_return
        except Exception as e:
            info(f"An error occurred in get_dash_metrics: {e}")
            return default_return


class ChartGenerator:
    """Facade class for generating various financial charts and visualizations"""

    def __init__(
        self,
        data_fetcher,
        metrics_analyzer,
        market_profile_analyzer,
        sentiment_analyzer,
    ):
        """Initialize with required service dependencies"""
        self.data_fetcher = data_fetcher
        self.metrics_analyzer = metrics_analyzer
        self.market_profile_analyzer = market_profile_analyzer
        self.sentiment_analyzer = sentiment_analyzer

        # Initialize specialized chart generators
        self.candlestick_charts = CandlestickCharts(
            data_fetcher, market_profile_analyzer
        )
        self.metrics_charts = MetricsCharts(metrics_analyzer)

    # Delegate to specialized chart classes
    def create_candle_chart_with_profile(self, data, poc_price, va_high, va_low):
        return self.candlestick_charts.create_candle_chart_with_profile(
            data, poc_price, va_high, va_low
        )

    def plot_with_volume_profile(
        self, ticker_symbol, start_date, end_date, combined_metrics, option
    ):
        return self.candlestick_charts.plot_with_volume_profile(
            ticker_symbol, start_date, end_date, combined_metrics, option
        )

    def plot_candle_charts_per_symbol(
        self,
        start_date,
        end_date,
        metrics,
        option,
    ):
        return self.candlestick_charts.plot_candle_charts_per_symbol(
            start_date, end_date, metrics, option
        )

    def display_ticker_metrics_dashboard(
        self, ticker_symbol, ticker, eps, pe, ps, pb, peg, gm
    ):
        self.candlestick_charts._display_ticker_metrics_dashboard(
            ticker_symbol, ticker, eps, pe, ps, pb, peg, gm
        )

    @staticmethod
    def get_dash_metrics(ticker_symbol, combined_metrics):
        return MetricsCharts.get_dash_metrics(ticker_symbol, combined_metrics)
