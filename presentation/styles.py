"""
Module for handling UI styling for the stock analysis dashboard.
Contains functions for:
- Core UI styling (CSS, containers, metrics)
- Chart configuration and styling (candlestick, volume profile)
- Data visualization elements
"""

import streamlit as st
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

# =============================================================================
# CORE UI STYLING
# =============================================================================


def apply_custom_styling():
    """Apply custom CSS styling to the Streamlit application.

    Enhances the overall look and feel with consistent spacing, colors, and
    interactive elements.
    """
    st.markdown(
        """
    <style>
    /* Main Layout Styles */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* Typography Styles */
    h1, h2, h3 {
        margin-top: 1rem;
        margin-bottom: 1rem;
        color: #f0f2f6;
        font-weight: 600;
    }
    h1 {
        font-size: 2.2rem;
    }
    h2 {
        font-size: 1.8rem;
    }
    h3 {
        font-size: 1.4rem;
    }
    p {
        line-height: 1.6;
    }
    
    /* Tab Navigation Styles */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        position: sticky !important;
        top: 0;
        background-color: #262730;
        z-index: 999;
        padding: 6px 0px;
        margin-top: -15px;
        box-shadow: 0 3px 10px rgba(0,0,0,0.3);
        border-radius: 0 0 8px 8px;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.1rem;
        font-weight: 500;
        padding: 8px 16px;
        transition: all 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(255, 255, 255, 0.2);
        border-radius: 6px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 6px;
    }
    .stTabs {
        padding-top: 15px;
    }
    
    /* Card Component Styles */
    .card {
        background-color: rgba(35, 38, 45, 0.8);
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        border: 1px solid rgba(100, 100, 150, 0.2);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.2);
    }
    
    /* Financial Metrics Dashboard Styles */
    .metrics-container {
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
        background-color: rgba(30, 30, 40, 0.7);
        border: 1px solid rgba(100, 100, 150, 0.2);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
    }
    .metric-card {
        background-color: rgba(40, 40, 60, 0.8);
        border-radius: 8px;
        padding: 12px;
        text-align: center;
        height: 100%;
        transition: transform 0.2s, box-shadow 0.2s;
        border: 1px solid rgba(100, 100, 150, 0.3);
        margin-bottom: 10px;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.25);
        background-color: rgba(45, 45, 65, 0.9);
    }
    .metric-label {
        font-size: 0.8rem;
        color: rgba(210, 210, 230, 0.95);
        margin-bottom: 5px;
        font-weight: 500;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 2px;
        letter-spacing: 0.5px;
    }
    .metric-value.positive {
        color: rgba(120, 255, 120, 0.95);
        text-shadow: 0 0 8px rgba(100, 255, 100, 0.4);
    }
    .metric-value.negative {
        color: rgba(255, 120, 120, 0.95);
        text-shadow: 0 0 8px rgba(255, 100, 100, 0.4);
    }
    .metric-value.neutral {
        color: rgba(230, 230, 255, 0.95);
    }
    .metric-desc {
        font-size: 0.7rem;
        color: rgba(190, 190, 210, 0.85);
        margin-top: 5px;
        font-style: italic;
    }
    .metric-container {
        display: flex;
        flex-direction: column;
        height: 100%;
    }
    
    /* Data Table Styles */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid rgba(100, 100, 150, 0.2);
    }
    .stDataFrame [data-testid="stTable"] {
        border-collapse: separate;
        border-spacing: 0;
    }
    .stDataFrame thead tr th {
        background-color: rgba(40, 40, 60, 0.8) !important;
        color: white !important;
        font-weight: 600;
        padding: 12px 8px !important;
        border-bottom: 2px solid rgba(100, 100, 200, 0.3) !important;
    }
    .stDataFrame tbody tr:nth-child(even) td {
        background-color: rgba(45, 45, 65, 0.6) !important;
    }
    .stDataFrame tbody tr:hover td {
        background-color: rgba(60, 60, 80, 0.7) !important;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )


def create_metrics_container(content_callable):
    """Create a container with proper styling for metrics.

    Creates a visually distinct container for displaying groups of financial metrics
    with consistent styling.

    Args:
        content_callable: A callable that will render the content inside the container
    """
    st.markdown("<div class='metrics-container'>", unsafe_allow_html=True)
    content_callable()
    st.markdown("</div>", unsafe_allow_html=True)


def get_dataframe_config():
    """Get standard configuration for dataframes in the dashboard.

    Provides consistent styling and behavior for all dataframes in the application.

    Returns:
        dict: Configuration parameters for st.dataframe
    """
    return {
        "height": 400,  # Limit height to ensure it doesn't take too much space
        "use_container_width": True,  # Use container width instead of fixed width
        "hide_index": True,  # Hide index for cleaner presentation
    }


# =============================================================================
# FINANCIAL METRICS UI COMPONENTS
# =============================================================================


def styled_metric(
    label,
    value,
    tooltip="",
    formatter=None,
    positive_good=True,
    prefix="",
    suffix="",
):
    """
    Create a styled metric HTML component for dashboard display.

    Creates a visually appealing metric card with appropriate styling based
    on value characteristics (positive/negative/neutral).

    Args:
        label (str): The metric label
        value: The metric value
        tooltip (str): Tooltip text for the metric
        formatter (str): How to format the value ('percent', 'currency', 'ratio', etc.)
        positive_good (bool): Whether positive values should be styled as good
        prefix (str): Text to display before the value
        suffix (str): Text to display after the value

    Returns:
        str: HTML string for the metric component
    """
    # Determine if we should show positive/negative styling
    style_class = "neutral"
    if value is not None and isinstance(value, (int, float)):
        if positive_good:
            style_class = (
                "positive" if value > 0 else "negative" if value < 0 else "neutral"
            )
        else:
            style_class = (
                "negative" if value > 0 else "positive" if value < 0 else "neutral"
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


def open_metrics_dashboard():
    """Start a financial metrics dashboard container."""
    st.markdown('<div class="metrics-container">', unsafe_allow_html=True)


def close_metrics_dashboard():
    """Close the financial metrics dashboard container."""
    st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# CHART STYLING AND CONFIGURATION
# =============================================================================


def create_candlestick_style():
    """Create a custom style for candlestick charts with better visibility

    Creates a visually appealing style with high contrast colors for price movement.

    Returns:
        tuple: (marketcolors, style) for use with mplfinance
    """
    import mplfinance as mpf

    # Create custom style with vibrant, high-contrast colors
    mc = mpf.make_marketcolors(
        up="#54ff54",  # Bright green for up days
        down="#ff5454",  # Bright red for down days
        edge="inherit",
        wick="inherit",
        volume={"up": "#54ff54", "down": "#ff5454"},
    )
    custom_style = mpf.make_mpf_style(
        marketcolors=mc,
        gridstyle=":",
        y_on_right=False,
        facecolor="#1E1E28",  # Dark background
        figcolor="#1E1E28",  # Dark figure background
        gridcolor="#555555",  # Medium-dark grid lines
    )

    return mc, custom_style


def configure_date_axis(ax1, ax_volume, fig):
    """Configure date axis formatting for price and volume charts

    Ensures consistent date formatting across all chart elements.

    Args:
        ax1: Main price chart axis
        ax_volume: Volume chart axis
        fig: Matplotlib figure
    """
    # Force the x-axis to interpret the tick values as dates
    ax1.xaxis_date()
    ax_volume.xaxis_date()

    # Ensure that the x-axis for both charts are formatted correctly with dates
    date_format = mdates.DateFormatter("%b '%y")  # e.g., Jan '23
    ax1.xaxis.set_major_formatter(date_format)
    ax_volume.xaxis.set_major_formatter(date_format)

    # Rotate date labels for better readability
    fig.autofmt_xdate(rotation=30)

    # Ensure the x-axis ticks are more visible
    ax1.tick_params(axis="x", colors="white", labelsize=9)
    ax_volume.tick_params(axis="x", colors="white", labelsize=9)


def setup_chart_grid(fig_size=(12, 9), is_volume_profile=True):
    """Create a grid layout for charts with appropriate spacing

    Sets up a professional chart grid layout optimized for financial data visualization.

    Args:
        fig_size: Size of the figure (width, height)
        is_volume_profile: Whether to include space for volume profile

    Returns:
        tuple: (fig, matplotlib figure, gs: GridSpec, axes: dict of chart axes)
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    # Set a clean, modern style for better visibility
    plt.style.use("dark_background")

    # Create a figure with specified proportions
    fig = plt.figure(figsize=fig_size, facecolor="#1E1E28")

    if is_volume_profile:
        # Create GridSpec with space for price labels, main chart, volume, and profile
        gs = GridSpec(
            5,
            5,
            figure=fig,
            height_ratios=[3, 3, 3, 1.5, 1.5],
            width_ratios=[0.12, 0.88, 0.88, 0.88, 1],
            hspace=0.05,  # Tighter vertical spacing
            wspace=0.05,  # Tighter horizontal spacing
        )

        # Create axes for each component
        axes = {
            "price": fig.add_subplot(gs[0:3, 1:4]),
            "volume": fig.add_subplot(gs[3:5, 1:4]),
            "profile": fig.add_subplot(gs[0:3, 4]),
        }

        # Share x and y axes as needed
        axes["volume"].sharex(axes["price"])
        axes["profile"].sharey(axes["price"])
    else:
        # Create GridSpec for price chart and volume only
        gs = GridSpec(
            4,
            1,
            figure=fig,
            height_ratios=[3, 3, 3, 1],
            hspace=0.05,  # Tighter spacing
        )

        # Create axes without volume profile
        axes = {
            "price": fig.add_subplot(gs[0:3, 0]),
            "volume": fig.add_subplot(gs[3, 0]),
        }

        # Share x axis
        axes["volume"].sharex(axes["price"])

    # Set background colors for all axes
    for ax in axes.values():
        ax.set_facecolor("#1E1E28")  # Dark background for each subplot
        ax.tick_params(colors="white")  # White tick labels

    return fig, gs, axes


def enhance_chart_aesthetics(ax1, price_levels=None):
    """Apply enhanced aesthetics to chart for better readability

    Args:
        ax1: Main price chart axis
        price_levels: Optional list of price levels for background shading
    """
    # Make the grid lines more prominent
    ax1.grid(
        which="major",
        axis="y",
        linestyle="-",
        linewidth=0.7,
        color="#aaaaaa",
        alpha=0.8,
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

    # Improve axis labels
    ax1.set_ylabel(
        "Price ($)", fontsize=14, fontweight="bold", color="white", labelpad=10
    )

    # Remove top and right spines for cleaner look
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Style the visible spines
    for spine in ["bottom", "left"]:
        ax1.spines[spine].set_color("#888888")
        ax1.spines[spine].set_linewidth(1.0)

    # Add background shading to make price bands more visible if we have price levels
    if price_levels is not None:
        for i in range(0, len(price_levels) - 1, 2):
            if i < len(price_levels) - 1:
                ax1.axhspan(
                    price_levels[i],
                    price_levels[i + 1],
                    color="#333333",
                    alpha=0.2,
                    zorder=-10,
                )


def style_price_indicators(ax, poc_price, va_high, va_low, current_price):
    """Add price analysis lines and indicators with clear, visible price labels

    Parameters:
    - ax: matplotlib axis for price chart
    - poc_price: Point of Control price
    - va_high: Value Area High price
    - va_low: Value Area Low price
    - current_price: Current price (can be None)

    Returns:
    - Tuple containing (poc_line, va_high_line, va_low_line, current_line)
    """
    # Create price indicator lines with improved styling and labels

    # POC Line (Point of Control)
    poc_line = ax.axhline(
        y=poc_price,
        color="#ff5454",
        linestyle="dashed",
        linewidth=2,
        label="POC",
        alpha=0.9,
    )

    # VA High Line
    va_high_line = ax.axhline(
        y=va_high,
        color="#5495ff",
        linestyle="dashed",
        linewidth=1,
        label="VA High",
        alpha=0.8,
    )

    # VA Low Line
    va_low_line = ax.axhline(
        y=va_low,
        color="#5495ff",
        linestyle="dashed",
        linewidth=1,
        label="VA Low",
        alpha=0.8,
    )

    # Current price line (if available)
    current_line = None
    if current_price is not None:
        current_line = ax.axhline(
            y=current_price,
            color="#54ff54",
            linestyle="solid",
            linewidth=1.5,
            label="Current",
            alpha=0.9,
        )

    # Add price labels with styled text boxes
    bbox_props = dict(
        boxstyle="round,pad=0.3", fc="#222233", ec="#888888", alpha=0.9, lw=1
    )

    # Position the labels on the right side of the chart
    x_position = 0.98  # Near right edge

    # POC price label
    ax.annotate(
        f"POC: ${poc_price:.2f}",
        xy=(x_position, poc_price),
        xycoords=("axes fraction", "data"),
        verticalalignment="center",
        horizontalalignment="right",
        fontsize=10,
        fontweight="bold",
        color="#ff5454",
        bbox=bbox_props,
    )

    # VA High price label
    ax.annotate(
        f"VA High: ${va_high:.2f}",
        xy=(x_position, va_high),
        xycoords=("axes fraction", "data"),
        verticalalignment="center",
        horizontalalignment="right",
        fontsize=9,
        color="#5495ff",
        bbox=bbox_props,
    )

    # VA Low price label
    ax.annotate(
        f"VA Low: ${va_low:.2f}",
        xy=(x_position, va_low),
        xycoords=("axes fraction", "data"),
        verticalalignment="center",
        horizontalalignment="right",
        fontsize=9,
        color="#5495ff",
        bbox=bbox_props,
    )

    # Current price label (if available)
    if current_price is not None:
        ax.annotate(
            f"Current: ${current_price:.2f}",
            xy=(x_position, current_price),
            xycoords=("axes fraction", "data"),
            verticalalignment="center",
            horizontalalignment="right",
            fontsize=10,
            fontweight="bold",
            color="#54ff54",
            bbox=bbox_props,
        )

    # Return the line objects for potential further customization
    return poc_line, va_high_line, va_low_line, current_line


def style_volume_profile(
    ax, price_levels, bin_size, buy_volume_by_price, sell_volume_by_price, poc_price
):
    """Style and draw the volume profile visualization with consistent colors

    Creates a visually informative volume profile with clear differentiation between
    buy and sell volumes, highlighting key price levels.

    Args:
        ax: Volume profile axis
        price_levels: List of price levels
        bin_size: Size of each price bin
        buy_volume_by_price: Buy volume distribution
        sell_volume_by_price: Sell volume distribution
        poc_price: Point of Control price
    """
    # Plot the volume profiles
    ax.barh(
        price_levels[:-1],
        buy_volume_by_price,
        height=bin_size,
        color="#54ff54",  # Bright green
        alpha=0.6,
        label="Buy Volume",
    )

    ax.barh(
        price_levels[:-1],
        sell_volume_by_price,
        height=bin_size,
        color="#ff5454",  # Bright red
        alpha=0.5,
        label="Sell Volume",
    )

    # Highlight POC in the volume profile
    poc_bin_idx = min(
        range(len(price_levels) - 1),
        key=lambda i: abs((price_levels[i] + price_levels[i + 1]) / 2 - poc_price),
    )

    # Highlight the POC bin with high opacity
    ax.barh(
        price_levels[poc_bin_idx],
        buy_volume_by_price[poc_bin_idx] + sell_volume_by_price[poc_bin_idx],
        height=bin_size,
        color="#cc44cc",  # Bright purple
        alpha=0.9,
        label="POC",
    )

    # Highlight potential buy zones
    buy_focus_regions = []
    for i in range(poc_bin_idx):
        if buy_volume_by_price[i] > 0.7 * buy_volume_by_price[poc_bin_idx]:
            buy_focus_regions.append(i)

    for idx in buy_focus_regions:
        ax.barh(
            price_levels[idx],
            buy_volume_by_price[idx],
            height=bin_size,
            color="#00ff00",  # Pure lime for high visibility
            alpha=0.9,
            label="Strong Buy Zone" if idx == buy_focus_regions[0] else "",
        )

    # Remove axis labels from the volume profile
    ax.set_xticks([])
    ax.set_yticks([])

    # Style the visible spines
    for spine in ["top", "right", "bottom", "left"]:
        if spine in ["left", "bottom"]:
            ax.spines[spine].set_color("#888888")
            ax.spines[spine].set_linewidth(0.5)
        else:
            ax.spines[spine].set_visible(False)

    # Add a legend to the volume profile
    ax.legend(
        loc="upper right",
        fontsize="medium",
        framealpha=0.9,
        facecolor="#222233",
        edgecolor="#888888",
    )


def style_candlestick_chart(price_ax, volume_ax, fig, data):
    """Style and draw the main candlestick chart with volume

    Creates a professional-looking candlestick chart with volume indicators
    and consistent styling.

    Args:
        price_ax: Main price chart axis
        volume_ax: Volume chart axis
        fig: Matplotlib figure
        data: DataFrame containing OHLCV data
    """
    import mplfinance as mpf

    # Get custom candlestick style
    _, custom_style = create_candlestick_style()

    # Plot candlestick chart
    mpf.plot(
        data,
        type="candle",
        style=custom_style,
        ax=price_ax,
        volume=volume_ax,
        show_nontrading=False,
    )

    # Add grid lines
    price_ax.grid(
        which="major",
        axis="y",
        linestyle="-",
        linewidth=0.5,
        color="#888888",
        alpha=0.7,
    )
    price_ax.grid(
        which="major",
        axis="x",
        linestyle="-",
        linewidth=0.5,
        color="#888888",
        alpha=0.5,
    )

    # Configure date axis
    configure_date_axis(price_ax, volume_ax, fig)

    # Ensure the volume subplot shares the same x-axis as the price subplot
    volume_ax.set_xlim(price_ax.get_xlim())

    # Share the x-axis between price and volume subplots
    volume_ax.sharex(price_ax)

    # Format the x-axis dates consistently
    date_format = mdates.DateFormatter("%Y-%m-%d")
    price_ax.xaxis.set_major_formatter(date_format)

    # Hide x-axis labels on the price chart to avoid duplication
    plt.setp(price_ax.get_xticklabels(), visible=False)

    # Ensure volume subplot has proper date formatting
    volume_ax.xaxis.set_major_formatter(date_format)

    # Synchronize the x-axis limits after all plotting is done
    volume_ax.set_xlim(price_ax.get_xlim())

    # Style the volume axis
    volume_ax.set_ylabel("Volume", fontsize=10, color="white")

    # Set background colors
    price_ax.set_facecolor("#1E1E28")
    volume_ax.set_facecolor("#1E1E28")


def create_chart_legend(ax1, current_price_exists=False, buy_focus_detected=False):
    """Create a standardized legend for the price chart

    Adds a clear, informative legend to help users understand chart elements.

    Args:
        ax1: Main price chart axis
        current_price_exists: Whether current price line exists
        buy_focus_detected: Whether buy focus regions were detected
    """
    from matplotlib.lines import Line2D

    # Create custom legend entries that match the actual line styles
    legend_elements = [
        Line2D(
            [0],
            [0],
            color="#ff5454",
            linestyle="dashed",
            linewidth=2,
            label="POC",
        ),
        Line2D(
            [0],
            [0],
            color="#5495ff",
            linestyle="dashed",
            linewidth=1.5,
            label="VA High/Low",
        ),
    ]

    if current_price_exists:
        legend_elements.append(
            Line2D(
                [0],
                [0],
                color="#54ff54",
                linestyle="-",
                linewidth=2.5,
                label="Current Price",
            )
        )

    if buy_focus_detected:
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
        framealpha=0.8,
        facecolor="#222233",
        edgecolor="#888888",
    )


# =============================================================================
# SIDEBAR STYLING
# =============================================================================


def apply_sidebar_styling():
    """Apply custom styling for the sidebar components"""
    st.markdown(
        """
        <style>
        .filter-section {
            background-color: rgba(0,0,0,0.03);
            border-radius: 8px;
            padding: 12px 15px;
            margin-bottom: 15px;
            border-left: 4px solid #ccc;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        }
        .filter-header {
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 10px;
            color: #333;
        }
        .filter-enabled {
            opacity: 1;
        }
        .filter-disabled {
            opacity: 0.6;
        }
        .filter-row {
            display: flex;
            align-items: center;
            margin-bottom: 8px;
            transition: all 0.2s ease;
        }
        .filter-checkbox {
            margin-right: 10px;
        }
        .info-icon {
            color: #888;
            font-size: 0.8rem;
            margin-left: 5px;
        }
        .filter-values {
            font-size: 0.8rem;
            color: #555;
            margin-top: 3px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def style_market_sentiment_dashboard():
    """Apply styling for the market sentiment section in the sidebar"""
    # No direct CSS needed here - we'll use this function to contain
    # the styling-related logic for this section
    pass


def style_metric_compact(label, value, color_code=None):
    """
    Create a compact metric display with label and value

    Args:
        label (str): The metric label
        value (str): The formatted value to display
        color_code (str): Optional color code for the value

    Returns:
        tuple: HTML for the label and value columns
    """
    if color_code:
        value_html = f"<span style='font-size: 0.9rem; font-weight: bold; color: {color_code};'>{value}</span>"
    else:
        value_html = (
            f"<span style='font-size: 0.9rem; font-weight: bold;'>{value}</span>"
        )

    label_html = f"<span style='font-size: 0.8rem; font-weight: 600;'>{label}:</span>"

    return label_html, value_html


def style_metric_value_with_note(value, note="", color_code=None):
    """
    Create a value display with an optional explanatory note underneath

    Args:
        value (str): The main value to display
        note (str): A smaller note to display under the value
        color_code (str): Optional color code for the value

    Returns:
        str: HTML for the formatted value with note
    """
    if color_code:
        html = f"<span style='font-size: 0.9rem; font-weight: bold; color: {color_code};'>{value}</span>"
    else:
        html = f"<span style='font-size: 0.9rem; font-weight: bold;'>{value}</span>"

    if note:
        html += f" <span style='font-size: 0.7rem; color: #999;'>({note})</span>"

    return html
