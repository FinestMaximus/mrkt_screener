import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from utils.logger import info, debug, warning, error


class VIXChartManager:
    """Class to handle VIX index data fetching and visualization"""

    def __init__(self):
        """Initialize the VIX chart manager"""
        info("Initializing VIXChartManager")

    def fetch_vix_data(self, days=2555):  # ~7 years (365*7)
        """Fetch VIX data for the specified number of days

        Args:
            days: Number of days of VIX data to fetch (default: 2555, ~7 years)

        Returns:
            DataFrame containing VIX data or None if fetch failed
        """
        try:
            info(f"Fetching VIX data for the last {days} days")
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            # Fetch VIX data using yfinance
            vix_data = yf.download(
                "^VIX",
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                progress=False,
            )

            if vix_data.empty:
                warning("No VIX data retrieved")
                return None

            debug(f"Retrieved {len(vix_data)} VIX data points")
            return vix_data
        except Exception as e:
            error(f"Error fetching VIX data: {str(e)}")
            return None

    def display_vix_chart(self, days=30):
        """Display VIX chart in the sidebar

        Args:
            days: Number of days of VIX data to show (default: 30)
        """
        vix_data = self.fetch_vix_data(days)

        if vix_data is None or vix_data.empty:
            st.warning("VIX data currently unavailable")
            return

        # Create figure with dark theme for consistency
        fig, ax = plt.subplots(figsize=(8, 3))
        fig.patch.set_facecolor("#1E1E28")
        ax.set_facecolor("#1E1E28")

        # Plot the VIX line
        ax.plot(
            vix_data.index, vix_data["Close"], color="#FF9800", linewidth=2, alpha=0.9
        )

        # Add range shading to indicate volatility levels
        ax.axhspan(0, 20, color="#4CAF50", alpha=0.1, label="Low Volatility")
        ax.axhspan(20, 30, color="#FF9800", alpha=0.1, label="Moderate Volatility")
        ax.axhspan(30, 100, color="#F44336", alpha=0.1, label="High Volatility")

        # Add current and peak values
        current_vix = vix_data["Close"].iloc[-1]
        peak_vix = vix_data["Close"].max()

        # Add horizontal line for current VIX price
        current_line = ax.axhline(
            y=current_vix,
            color="#FFFFFF",
            linestyle="-",
            linewidth=1.5,
            label=f"Current ({current_vix:.2f})",
            alpha=0.8,
        )

        # Format dates on x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        ax.tick_params(axis="x", colors="white", labelsize=8)
        ax.tick_params(axis="y", colors="white", labelsize=8)

        # Add grid for better readability
        ax.grid(axis="y", linestyle="--", alpha=0.3, color="#555555")

        # Add title and labels
        ax.set_title(
            f"VIX Index - Last {days} Days (Current: {current_vix:.2f})",
            color="white",
            fontsize=10,
        )

        # Highlight current value with a marker point
        ax.plot(vix_data.index[-1], current_vix, "o", color="#FFFFFF", markersize=5)

        # Style borders
        for spine in ax.spines.values():
            spine.set_color("#555555")

        # Add price annotation box
        bbox_props = dict(
            boxstyle="round,pad=0.3", fc="#222233", ec="#FFFFFF", alpha=0.8, lw=1
        )

        # Position the price label on the right side of the chart
        ax.annotate(
            f"VIX: {current_vix:.2f}",
            xy=(0.98, current_vix),
            xycoords=("axes fraction", "data"),
            verticalalignment="center",
            horizontalalignment="right",
            fontsize=9,
            fontweight="bold",
            color="#FFFFFF",
            bbox=bbox_props,
        )

        # Add legend with volatility levels
        ax.legend(
            loc="upper left",
            fontsize=8,
            framealpha=0.7,
            facecolor="#1E1E28",
            edgecolor="#555555",
        )

        # Adjust layout
        plt.tight_layout()

        # Display the chart in Streamlit
        st.pyplot(fig)

        # Add text description
        st.markdown(
            f"""
            <div style='background-color: rgba(0,0,0,0.05); padding: 8px; border-radius: 5px; font-size: 0.8rem;'>
            The VIX index measures market volatility expectations. 
            <span style='color: {"#F44336" if current_vix > 30 else "#FF9800" if current_vix > 20 else "#4CAF50"}'>
            Current volatility is <b>{"high" if current_vix > 30 else "moderate" if current_vix > 20 else "low"}</b>
            </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    def display_vix_chart_compact(self, days=2555):  # Default to ~7 years
        """Display a compact VIX chart in the sidebar with dynamic coloring"""
        vix_data = self.fetch_vix_data(days)

        if vix_data is None or vix_data.empty:
            st.warning("VIX data currently unavailable")
            return

        # Create a user-friendly time period description
        if days >= 365:
            years = days / 365
            time_desc = f"{years:.1f} Year{'s' if years > 1 else ''}"
        else:
            time_desc = f"{days} Day{'s' if days > 1 else ''}"

        # Create figure with dark theme but compact size
        fig, ax = plt.subplots(figsize=(7, 2))
        fig.patch.set_facecolor("#1E1E28")
        ax.set_facecolor("#1E1E28")

        # Plot the VIX line with segments colored by value
        for i in range(len(vix_data) - 1):
            # Extract as a float to ensure it's a scalar, not a Series
            value = float(vix_data["Close"].iloc[i])

            # Determine segment color based on value range
            if value < 20:  # Low volatility
                color = "#4CAF50"  # Green
            elif value < 30:  # Moderate volatility
                color = "#FF9800"  # Orange
            else:  # High volatility
                color = "#F44336"  # Red

            # Plot this segment with the appropriate color
            ax.plot(
                vix_data.index[i : i + 2],
                vix_data["Close"].iloc[i : i + 2],
                color=color,
                linewidth=2.5,
                solid_capstyle="round",
            )

        # Add range shading to indicate volatility levels
        ax.axhspan(0, 20, color="#4CAF50", alpha=0.1)  # Low Volatility - Green
        ax.axhspan(20, 30, color="#FF9800", alpha=0.1)  # Moderate Volatility - Orange
        ax.axhspan(30, 100, color="#F44336", alpha=0.1)  # High Volatility - Red

        # Add current values - convert to float to avoid pandas Series ambiguity error
        current_vix = float(vix_data["Close"].iloc[-1])

        # Determine color for current value marker
        if current_vix < 20:
            current_color = "#4CAF50"  # Green
        elif current_vix < 30:
            current_color = "#FF9800"  # Orange
        else:
            current_color = "#F44336"  # Red

        # Add horizontal line for current VIX price
        ax.axhline(
            y=current_vix,
            color="#FFFFFF",
            linestyle="-",
            linewidth=1,
            alpha=0.7,
        )

        # Highlight current value with a marker point
        ax.plot(
            vix_data.index[-1],
            current_vix,
            "o",
            color=current_color,
            markersize=6,
            markeredgecolor="#FFFFFF",
            markeredgewidth=1,
        )

        # Format dates on x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.tick_params(axis="x", colors="white", labelsize=7)
        ax.tick_params(axis="y", colors="white", labelsize=7)

        # Add grid for better readability
        ax.grid(axis="y", linestyle="--", alpha=0.2, color="#555555")

        # Add title
        ax.set_title(
            f"VIX Index - {time_desc} History ({current_vix:.1f})",
            color="white",
            fontsize=9,
            pad=2,
        )

        # Style borders
        for spine in ax.spines.values():
            spine.set_color("#555555")

        # Add price annotation
        bbox_props = dict(
            boxstyle="round,pad=0.2", fc="#222233", ec=current_color, alpha=0.8, lw=1.5
        )

        # Position the price label
        ax.annotate(
            f"VIX: {current_vix:.1f}",
            xy=(0.98, 0.90),
            xycoords="axes fraction",
            verticalalignment="top",
            horizontalalignment="right",
            fontsize=8,
            fontweight="bold",
            color=current_color,
            bbox=bbox_props,
        )

        # Adjust layout
        plt.tight_layout(pad=0.5)
        plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.15)

        # Display the chart in Streamlit
        st.pyplot(fig)
