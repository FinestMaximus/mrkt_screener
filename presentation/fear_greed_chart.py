import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import requests
import json
from utils.logger import info, debug, error
from data.fear_greed_indicator import FearGreedIndicator


class FearGreedChartManager:
    """Class to fetch and display Fear & Greed Index chart data"""

    def __init__(self):
        self.fear_greed_indicator = FearGreedIndicator()

    def fetch_historical_fear_greed_data(self, days=2555):  # ~7 years (365*7)
        """
        Fetch historical fear and greed index data

        Args:
            days (int): Number of days of historical data to fetch

        Returns:
            pd.DataFrame: DataFrame with date and value columns
        """
        try:
            # Use the CNN Money Fear & Greed API
            url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
            debug(f"Fetching historical fear and greed data from {url}")

            response = requests.get(
                url,
                headers={
                    "User-Agent": self.fear_greed_indicator.get_random_user_agent()
                },
            )
            response.raise_for_status()  # Raise an error for bad responses

            # Extract the data from the response
            data = response.json()
            debug(f"Response keys: {list(data.keys())}")

            fear_data = []

            # First try to use fear_and_greed_historical data
            if (
                isinstance(data, dict)
                and "fear_and_greed_historical" in data
                and isinstance(data["fear_and_greed_historical"], dict)
                and "data" in data["fear_and_greed_historical"]
                and isinstance(data["fear_and_greed_historical"]["data"], list)
            ):
                debug("Using fear_and_greed_historical data format")
                for item in data["fear_and_greed_historical"]["data"]:
                    if not isinstance(item, dict):
                        continue

                    timestamp_ms = item.get("x")
                    value = item.get("y")
                    rating = item.get("rating")

                    if timestamp_ms is None or value is None:
                        continue

                    # Convert timestamp to datetime
                    date = pd.to_datetime(timestamp_ms, unit="ms")
                    fear_data.append({"date": date, "value": value, "rating": rating})

            # If we couldn't extract data from the primary format, try alternative format
            if not fear_data and isinstance(data, dict) and "fear_and_greed" in data:
                fg_data = data["fear_and_greed"]

                if isinstance(fg_data, dict) and "score" in fg_data:
                    debug("Using individual data points from fear_and_greed")

                    # Extract current values
                    current_score = fg_data.get("score")
                    current_rating = fg_data.get("rating", "")
                    current_time = fg_data.get("timestamp")

                    # Extract historical reference points
                    previous_close = fg_data.get("previous_close")
                    previous_week = fg_data.get("previous_1_week")
                    previous_month = fg_data.get("previous_1_month")
                    previous_year = fg_data.get("previous_1_year")

                    # Create synthetic data points
                    today = datetime.now()
                    if current_time:
                        try:
                            today = pd.to_datetime(current_time)
                        except:
                            pass

                    # Add data points in chronological order (oldest to newest)
                    if previous_year is not None:
                        fear_data.append(
                            {
                                "date": today - timedelta(days=365),
                                "value": previous_year,
                            }
                        )

                    if previous_month is not None:
                        fear_data.append(
                            {
                                "date": today - timedelta(days=30),
                                "value": previous_month,
                            }
                        )

                    if previous_week is not None:
                        fear_data.append(
                            {"date": today - timedelta(days=7), "value": previous_week}
                        )

                    if previous_close is not None:
                        fear_data.append(
                            {"date": today - timedelta(days=1), "value": previous_close}
                        )

                    if current_score is not None:
                        fear_data.append(
                            {
                                "date": today,
                                "value": current_score,
                                "rating": current_rating,
                            }
                        )

            # If we still have no data, return None
            if not fear_data:
                error("No usable fear and greed data found in the response")
                debug(
                    f"Response content: {str(data)[:1000]}..."
                )  # Log first 1000 chars
                return None

            # Create DataFrame and sort by date
            df = pd.DataFrame(fear_data)
            df = df.sort_values("date")

            debug(f"Created dataframe with {len(df)} rows of fear and greed data")

            # Ensure date column is properly formatted as datetime
            df["date"] = pd.to_datetime(df["date"])

            # Remove duplicate dates by keeping the last value for each date
            # This prevents the "cannot reindex on an axis with duplicate labels" error
            df = df.drop_duplicates(subset=["date"], keep="last")
            debug(f"After removing duplicates: {len(df)} rows of fear and greed data")

            # Fill in missing days with linear interpolation if we have enough points
            if len(df) >= 2:
                full_date_range = pd.date_range(
                    start=df["date"].min(), end=df["date"].max()
                )
                df = (
                    df.set_index("date")
                    .reindex(full_date_range)
                    .infer_objects(copy=False)
                    .interpolate(method="linear")
                    .reset_index()
                )
                df = df.rename(columns={"index": "date"})
                # Ensure the date column is still datetime after reset_index
                df["date"] = pd.to_datetime(df["date"])

            # Limit to the requested number of days
            return df.tail(days)

        except Exception as e:
            error(f"An error occurred: {str(e)}")
            import traceback

            error(traceback.format_exc())
            return None

    def display_fear_greed_chart_compact(self, days=2555):  # Default to ~7 years
        """Display a compact Fear & Greed Index chart in the sidebar with dynamic coloring (inverted colors)"""
        try:
            # Fetch historical data
            df = self.fetch_historical_fear_greed_data(days=days)

            if df is None or len(df) == 0:
                st.warning("Unable to load Fear & Greed Index history")
                return

            # Create a user-friendly time period description
            if days >= 365:
                years = days / 365
                time_desc = f"{years:.1f} Year{'s' if years > 1 else ''}"
            else:
                time_desc = f"{days} Day{'s' if days > 1 else ''}"

            # Create figure with dark theme but compact size (to match PE/VIX charts)
            fig, ax = plt.subplots(figsize=(7, 2))
            fig.patch.set_facecolor("#1E1E28")
            ax.set_facecolor("#1E1E28")

            # Convert dates to matplotlib-compatible format and set as index for proper plotting
            df_plot = df.copy()
            df_plot = df_plot.set_index("date")

            # Plot the Fear & Greed line with segments colored by value
            # INVERTED COLOR SCHEME: Green for fear, Red for greed
            for i in range(len(df_plot) - 1):
                # Extract as a float to ensure it's a scalar
                value = float(df_plot["value"].iloc[i])

                # Determine segment color based on value range (INVERTED)
                if value < 25:  # Extreme Fear
                    color = "#4CAF50"  # Green
                elif value < 40:  # Fear
                    color = "#8BC34A"  # Light Green
                elif value < 60:  # Neutral
                    color = "#FFEB3B"  # Yellow
                elif value < 75:  # Greed
                    color = "#FF9800"  # Orange
                else:  # Extreme Greed
                    color = "#F44336"  # Red

                # Plot this segment with the appropriate color using index (dates)
                ax.plot(
                    df_plot.index[i : i + 2],
                    df_plot["value"].iloc[i : i + 2],
                    color=color,
                    linewidth=2.5,
                    solid_capstyle="round",
                )

            # Add range shading to indicate sentiment levels (INVERTED COLORS)
            ax.axhspan(0, 25, color="#4CAF50", alpha=0.1)  # Extreme Fear - Green
            ax.axhspan(25, 40, color="#8BC34A", alpha=0.1)  # Fear - Light Green
            ax.axhspan(40, 60, color="#FFEB3B", alpha=0.1)  # Neutral - Yellow
            ax.axhspan(60, 75, color="#FF9800", alpha=0.1)  # Greed - Orange
            ax.axhspan(75, 100, color="#F44336", alpha=0.1)  # Extreme Greed - Red

            # Get current value - convert to float to avoid pandas Series ambiguity error
            current_value = float(df["value"].iloc[-1])

            # Get current rating
            current_rating = "Unknown"
            if "rating" in df.columns and not pd.isna(df.iloc[-1]["rating"]):
                current_rating = df.iloc[-1]["rating"].capitalize()
            else:
                # Calculate rating based on value
                if current_value >= 75:
                    current_rating = "Extreme Greed"
                elif current_value >= 60:
                    current_rating = "Greed"
                elif current_value >= 40:
                    current_rating = "Neutral"
                elif current_value >= 25:
                    current_rating = "Fear"
                else:
                    current_rating = "Extreme Fear"

            # Determine color for current value marker (INVERTED)
            if current_value < 25:
                current_color = "#4CAF50"  # Green for Extreme Fear
            elif current_value < 40:
                current_color = "#8BC34A"  # Light Green for Fear
            elif current_value < 60:
                current_color = "#FFEB3B"  # Yellow for Neutral
            elif current_value < 75:
                current_color = "#FF9800"  # Orange for Greed
            else:
                current_color = "#F44336"  # Red for Extreme Greed

            # Add horizontal line for current value
            ax.axhline(
                y=current_value,
                color="#FFFFFF",
                linestyle="-",
                linewidth=1,
                alpha=0.7,
            )

            # Add marker point for current value
            ax.plot(
                df_plot.index[-1],
                current_value,
                "o",
                color=current_color,
                markersize=6,
                markeredgecolor="#FFFFFF",
                markeredgewidth=1,
            )

            # Format x-axis (dates)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.tick_params(axis="x", colors="white", labelsize=7)
            ax.tick_params(axis="y", colors="white", labelsize=7)

            # Add grid
            ax.grid(axis="y", linestyle="--", alpha=0.2, color="#555555")

            # Set title
            ax.set_title(
                f"Fear & Greed Index - {time_desc} History ({current_value:.1f})",
                color="white",
                fontsize=9,
                pad=2,
            )

            # Style borders
            for spine in ax.spines.values():
                spine.set_color("#555555")

            # Add annotation for current value
            bbox_props = dict(
                boxstyle="round,pad=0.2",
                fc="#222233",
                ec=current_color,
                alpha=0.8,
                lw=1.5,
            )

            # Position the value label
            ax.annotate(
                f"{current_rating}: {current_value:.1f}",
                xy=(0.98, 0.90),
                xycoords="axes fraction",
                verticalalignment="top",
                horizontalalignment="right",
                fontsize=8,
                fontweight="bold",
                color=current_color,
                bbox=bbox_props,
            )

            # Set y-axis limits
            ax.set_ylim(0, 100)

            # Adjust layout
            plt.tight_layout(pad=0.5)
            plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.15)

            # Display the chart in Streamlit
            st.pyplot(fig)

        except Exception as e:
            error(f"Error displaying fear and greed chart: {str(e)}")
            import traceback

            error(traceback.format_exc())
            st.error("Unable to display Fear & Greed Index chart")
