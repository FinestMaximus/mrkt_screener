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
        poc_bin_idx = max(range(len(volume_by_price)), key=volume_by_price.__getitem__)
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
        self, ticker_symbol, data, va_high, va_low, poc_price
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

        # Calculate appropriate price range for y-axis ticks
        price_min = data["Low"].min()
        price_max = data["High"].max()
        price_range = price_max - price_min

        # Create nice tick spacing based on price range
        if price_range > 200:
            tick_spacing = 20.0
        elif price_range > 100:
            tick_spacing = 10.0
        elif price_range > 50:
            tick_spacing = 5.0
        elif price_range > 20:
            tick_spacing = 2.0
        elif price_range > 10:
            tick_spacing = 1.0
        elif price_range > 5:
            tick_spacing = 0.5
        else:
            tick_spacing = 0.2

        # Create custom price ticks
        start_tick = np.floor(price_min / tick_spacing) * tick_spacing
        end_tick = np.ceil(price_max / tick_spacing) * tick_spacing
        price_ticks = np.arange(start_tick, end_tick + tick_spacing, tick_spacing)

        # Apply the custom ticks to the y-axis
        ax1.set_yticks(price_ticks)

        # Format the ticks as currency with appropriate decimals - LARGER FONT
        if tick_spacing >= 1:
            ax1.set_yticklabels(
                ["${:.0f}".format(p) for p in price_ticks],
                fontsize=12,
                fontweight="bold",
            )
        else:
            ax1.set_yticklabels(
                ["${:.2f}".format(p) for p in price_ticks],
                fontsize=12,
                fontweight="bold",
            )

        # Make the y-axis price labels stand out more
        ax1.tick_params(axis="y", colors="white", labelsize=12, width=1.5, length=6)
        ax1.tick_params(axis="x", colors="white", labelsize=10, width=1.5, length=6)
        ax_volume.tick_params(axis="both", colors="white", labelsize=10)

        # Add price annotations with better positioning and no overlap
        label_padding = price_range * 0.02  # 2% of price range for padding

        # Define a function for consistent price annotations
        def add_price_annotation(ax, label, price, color, offset_y=0):
            ax.annotate(
                f"{label}: ${price:.2f}",
                xy=(data.index[-1], price),
                xytext=(5, offset_y),
                textcoords="offset points",
                ha="left",
                va="center",
                color=color,
                fontweight="bold",
                fontsize=11,  # Larger font size for annotations
                bbox=dict(
                    facecolor="#333333",
                    alpha=0.9,
                    edgecolor="none",
                    pad=2,
                    boxstyle="round,pad=0.5",
                ),
                zorder=20,
            )

        # POC annotation with improved styling
        add_price_annotation(ax1, "POC", poc_price, "#ff5050")

        # VA High annotation
        add_price_annotation(ax1, "VA High", va_high, "#5050ff", offset_y=label_padding)

        # VA Low annotation
        add_price_annotation(ax1, "VA Low", va_low, "#5050ff", offset_y=-label_padding)

        # Current price annotation with improved placement
        if current_price is not None:
            offset = 7 if current_price > poc_price else -7
            add_price_annotation(
                ax1, "Current", current_price, "#ffcf40", offset_y=offset
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
        poc_bin_idx = max(
            range(len(buy_volume_by_price)),
            key=lambda i: buy_volume_by_price[i] + sell_volume_by_price[i],
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
                framealpha=0.9,
                facecolor="#333333",
                edgecolor="#888888",
            )

        # Add buy zone annotation to the chart with improved styling
        if buy_focus_regions:
            best_buy_level = price_levels[buy_focus_regions[0]]
            ax1.axhline(
                y=best_buy_level,
                color="#00ff00",  # Pure lime for high visibility
                linestyle="dotted",
                linewidth=2,
                label="Buy Zone",
            )
            add_price_annotation(
                ax1,
                "Buy Zone",
                best_buy_level,
                "#00ff00",
                offset_y=-label_padding * 1.5,
            )

        # Add title to chart axes for clarity - more prominent styling
        ax1.set_title(
            "Price History", fontsize=16, fontweight="bold", color="white", pad=10
        )
        ax_volume.set_title(
            "Volume", fontsize=14, fontweight="bold", color="white", pad=10
        )
        ax2.set_title(
            "Vol Profile", fontsize=14, fontweight="bold", color="white", pad=10
        )

        # Add overall chart title
        fig.suptitle(
            f"{ticker.info.get('shortName', ticker_symbol)} ({ticker_symbol})",
            fontsize=18,
            fontweight="bold",
            color="white",
            y=0.98,
        )

        # Improve x-axis date formatting - more readable dates
        date_format = mdates.DateFormatter(
            "%b '%y"
        )  # Format as 'Jan '23 - more compact
        ax1.xaxis.set_major_formatter(date_format)
        ax_volume.xaxis.set_major_formatter(date_format)

        # Set appropriate number of ticks based on data length
        data_span = (data.index[-1] - data.index[0]).days
        if data_span > 1095:  # > 3 years
            interval = 6
        elif data_span > 730:  # > 2 years
            interval = 4
        elif data_span > 365:  # > 1 year
            interval = 3
        elif data_span > 180:  # > 6 months
            interval = 2
        else:
            interval = 1

        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=interval))

        # Make x-axis date labels larger and more readable
        for label in ax_volume.get_xticklabels():
            label.set_fontsize(12)
            label.set_fontweight("bold")

        # Rotate x-axis date labels for better readability
        plt.setp(ax_volume.xaxis.get_majorticklabels(), rotation=45, ha="right")

        # Add ticker logo and symbol in bottom right corner
        try:
            # Create a new axes for the logo in the bottom right
            logo_ax = fig.add_axes([0.85, 0.02, 0.12, 0.12], frameon=False)
            logo_ax.axis("off")

            # Get company info
            short_name = ticker.info.get("shortName", ticker_symbol)

            # Try to get logo URL
            logo_url = None
            if "logo_url" in ticker.info and ticker.info["logo_url"]:
                logo_url = ticker.info["logo_url"]

            if logo_url:
                try:
                    import requests
                    from PIL import Image
                    from io import BytesIO

                    response = requests.get(logo_url, timeout=2)
                    img = Image.open(BytesIO(response.content))
                    logo_ax.imshow(img)
                except Exception as e:
                    # If logo loading fails, just show text
                    debug(f"Error loading logo: {e}")
                    logo_ax.text(
                        0.5,
                        0.5,
                        f"{ticker_symbol}",
                        ha="center",
                        va="center",
                        fontsize=18,
                        fontweight="bold",
                        color="white",
                    )
            else:
                # No logo URL, just show text
                logo_ax.text(
                    0.5,
                    0.5,
                    f"{ticker_symbol}",
                    ha="center",
                    va="center",
                    fontsize=18,
                    fontweight="bold",
                    color="white",
                )

            # Add company name text below the chart
            fig.text(
                0.85,
                0.01,
                short_name,
                ha="center",
                fontsize=12,
                fontweight="bold",
                color="white",
            )

        except Exception as e:
            debug(f"Error adding logo/symbol: {e}")

        # Ensure the price labels don't get cut off
        plt.subplots_adjust(left=0.12, right=0.95, bottom=0.14, top=0.92)

        # Display the chart with appropriate sizing
        st.pyplot(fig)

        # Add a small explanation of the chart elements below the chart in a nicer format
        st.markdown(
            """
        <style>
        .explanation-box {
            background-color: rgba(70, 70, 70, 0.2);
            border-radius: 5px;
            padding: 10px;
            margin-top: 10px;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )
        with st.expander("📊 Chart Guide", expanded=False):
            st.markdown(
                """
            <div class="explanation-box">
            <ul>
                <li><b style="color:#ff5050">POC (Point of Control)</b>: Price level with the highest trading volume</li>
                <li><b style="color:#5050ff">VA (Value Area)</b>: Range between VA High and VA Low where 70% of trading occurred</li>
                <li><b style="color:#00ff00">Buy Zone</b>: Price level with significant buying volume below POC - potential support</li>
                <li><b style="color:#ffcf40">Current Price</b>: The latest price of the stock</li>
                <li><b>Green/Red Bars</b>: Volume profile showing buy vs sell volume at each price level</li>
            </ul>
            </div>
            """,
                unsafe_allow_html=True,
            )

        # Business summary in an expander to save space
        if "longBusinessSummary" in ticker.info:
            with st.expander("Company Overview", expanded=False):
                summary_text = ticker.info["longBusinessSummary"]
                formatted_summary = self._format_business_summary(summary_text)
                st.markdown(formatted_summary)

    def _display_ticker_metrics_dashboard(
        self, ticker_symbol, ticker, eps, pe, ps, pb, peg, gm
    ):
        """Display ticker metrics in a dashboard layout with improved spacing"""
        # Use a container with border styling for the metrics dashboard
        st.markdown(
            """
        <style>
        .metrics-container {
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 20px;
            background-color: rgba(240, 242, 246, 0.1);
        }
        </style>
        """,
            unsafe_allow_html=True,
        )

        with st.container():
            st.markdown('<div class="metrics-container">', unsafe_allow_html=True)

            # Use 4 columns instead of 7 to give more space to each metric
            col1, col2, col3, col4 = st.columns(4)

            # First row of metrics
            with col1:
                if peg:
                    st.metric(label="PEG", value=f"{round(peg,2)}")
                else:
                    st.metric(label="PEG", value="-")

                if eps:
                    st.metric(label="EPS", value=f"{round(eps,2)}")
                else:
                    st.metric(label="EPS", value="-")

            with col2:
                if pe:
                    st.metric(label="P/E", value=f"{round(pe,2)}")
                else:
                    st.metric(label="P/E", value="-")

                if ps:
                    st.metric(label="P/S", value=f"{round(ps,2)}")
                else:
                    st.metric(label="P/S", value="-")

            with col3:
                if pb:
                    st.metric(label="P/B", value=f"{round(pb,2)}")
                else:
                    st.metric(label="P/B", value="-")

                if gm is not None:
                    st.metric(label="Gross Margin", value=f"{round(gm*100,1)}%")
                else:
                    st.metric(label="Gross Margin", value="-")

            with col4:
                if "marketCap" in ticker.info and "financialCurrency" in ticker.info:
                    market_cap = ticker.info["marketCap"]
                    currency = ticker.info["financialCurrency"]
                    if market_cap >= 1e9:
                        market_cap_display = f"{market_cap / 1e9:.2f} B"
                    elif market_cap >= 1e6:
                        market_cap_display = f"{market_cap / 1e6:.2f} M"
                    else:
                        market_cap_display = f"{market_cap:.2f}"
                    st.metric(
                        label=f"Market Cap ({currency})",
                        value=market_cap_display,
                    )
                else:
                    st.metric(label="Market Cap", value="-")

                # Add a recommendation metric if available
                if (
                    "recommendationMean" in ticker.info
                    and ticker.info["recommendationMean"] is not None
                ):
                    rec_value = ticker.info["recommendationMean"]
                    st.metric(label="Analyst Rating", value=f"{round(rec_value,1)}/5")
                else:
                    st.metric(label="Analyst Rating", value="-")

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
            if (
                current_price is not None
                and va_high is not None
                and poc_price is not None
            ):
                if option[0] == "va_high":
                    if current_price > va_high:
                        info(
                            f"{ticker_symbol} - current price is above value area: {current_price} (VA High: {va_high}, POC: {poc_price})"
                        )
                        return 0
                elif option[0] == "poc_price":
                    if current_price > poc_price:
                        info(
                            f"{ticker_symbol} - price is above price of control: {current_price} (VA High: {va_high}, POC: {poc_price})"
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

                    # Main content area with two columns
                    col1, col2 = st.columns([1, 1])  # Equal width columns

                    with col1:
                        # Display the chart in the left column (taking half the width)
                        self._display_market_profile_chart(
                            ticker_symbol, data, va_high, va_low, poc_price
                        )

                    with col2:

                        # Display news articles
                        SentimentAnalyzer().display_news_without_sentiment(
                            ticker_symbol
                        )

                # Add some spacing after each ticker section
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

    def plot_candle_charts_per_symbol(self, start_date, end_date, metrics, option):
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
