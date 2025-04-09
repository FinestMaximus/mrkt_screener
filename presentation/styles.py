"""
Module for handling UI styling for the stock analysis dashboard.
"""

import streamlit as st


def apply_custom_styling():
    """Apply custom CSS styling to the Streamlit application."""
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
        position: sticky !important;
        top: 0;
        background-color: #262730;
        z-index: 999;
        padding: 4px 0px;
        margin-top: -15px;
        box-shadow: 0 3px 10px rgba(0,0,0,0.2);
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1rem;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(255, 255, 255, 0.2);
        border-radius: 4px 4px 0 0;
    }
    .card {
        background-color: #f8f9fa;
        border-radius: 6px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    .stTabs {
        padding-top: 15px;
    }
    .metrics-container {
        margin-bottom: 20px;
    }
    
    /* Financial Metrics Dashboard Styles */
    .metrics-container {
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 15px;
        background-color: rgba(30, 30, 40, 0.6);
        border: 1px solid rgba(100, 100, 150, 0.2);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-card {
        background-color: rgba(40, 40, 60, 0.7);
        border-radius: 6px;
        padding: 8px;
        text-align: center;
        height: 100%;
        transition: transform 0.2s, box-shadow 0.2s;
        border: 1px solid rgba(100, 100, 150, 0.2);
        margin-bottom: 8px;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    }
    .metric-label {
        font-size: 0.75rem;
        color: rgba(200, 200, 220, 0.9);
        margin-bottom: 3px;
        font-weight: 500;
    }
    .metric-value {
        font-size: 1.1rem;
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
        font-size: 0.65rem;
        color: rgba(180, 180, 200, 0.8);
        margin-top: 3px;
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


def create_metrics_container(content_callable):
    """Create a container with proper styling for metrics.

    Args:
        content_callable: A callable that will render the content inside the container
    """
    st.markdown("<div class='metrics-container'>", unsafe_allow_html=True)
    content_callable()
    st.markdown("</div>", unsafe_allow_html=True)


def get_dataframe_config():
    """Get standard configuration for dataframes in the dashboard.

    Returns:
        dict: Configuration parameters for st.dataframe
    """
    return {
        "height": 400,  # Limit height to ensure it doesn't take too much space
        "use_container_width": True,  # Use container width instead of fixed width
        "hide_index": True,
    }


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


def create_candlestick_style():
    """Create a custom style for candlestick charts with better visibility

    Returns:
        tuple: (marketcolors, style) for use with mplfinance
    """
    import mplfinance as mpf

    # Create custom style with desired colors
    mc = mpf.make_marketcolors(
        up="#54ff54",  # Brighter green for up days
        down="#ff5454",  # Brighter red for down days
        edge="inherit",
        wick="inherit",
        volume={"up": "#54ff54", "down": "#ff5454"},
    )
    custom_style = mpf.make_mpf_style(marketcolors=mc, gridstyle=":", y_on_right=False)

    return mc, custom_style


def configure_date_axis(ax1, ax_volume, fig):
    """Configure date axis formatting for price and volume charts

    Args:
        ax1: Main price chart axis
        ax_volume: Volume chart axis
        fig: Matplotlib figure
    """
    import matplotlib.dates as mdates

    # Force the x-axis to interpret the tick values as dates
    ax1.xaxis_date()
    ax_volume.xaxis_date()

    # Ensure that the x-axis for both charts are formatted correctly with dates
    date_format = mdates.DateFormatter("%b '%y")  # e.g., Jan '23
    ax1.xaxis.set_major_formatter(date_format)
    ax_volume.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()  # Automatically format date labels


def setup_chart_grid(fig_size=(12, 9), is_volume_profile=True):
    """Create a grid layout for charts with appropriate spacing

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
    fig = plt.figure(figsize=fig_size)

    if is_volume_profile:
        # Create GridSpec with space for price labels, main chart, volume, and profile
        gs = GridSpec(
            5,
            5,
            figure=fig,
            height_ratios=[3, 3, 3, 1.5, 1.5],
            width_ratios=[0.12, 0.88, 0.88, 0.88, 1],
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
        gs = GridSpec(4, 1, figure=fig, height_ratios=[3, 3, 3, 1])

        # Create axes without volume profile
        axes = {
            "price": fig.add_subplot(gs[0:3, 0]),
            "volume": fig.add_subplot(gs[3, 0]),
        }

        # Share x axis
        axes["volume"].sharex(axes["price"])

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
        y=poc_price, color="red", linestyle="dashed", linewidth=2, label="POC"
    )

    # VA High Line
    va_high_line = ax.axhline(
        y=va_high, color="blue", linestyle="dashed", linewidth=1, label="VA High"
    )

    # VA Low Line
    va_low_line = ax.axhline(
        y=va_low, color="blue", linestyle="dashed", linewidth=1, label="VA Low"
    )

    # Current price line (if available)
    current_line = None
    if current_price is not None:
        current_line = ax.axhline(
            y=current_price,
            color="green",
            linestyle="solid",
            linewidth=1.5,
            label="Current",
        )

    # Add price labels with styled text boxes
    bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8, lw=1)

    # Position the labels on the right side of the chart
    x_position = 0.98  # Near right edge

    # POC price label
    ax.annotate(
        f"POC: ${poc_price:.2f}",
        xy=(x_position, poc_price),
        xycoords=("axes fraction", "data"),
        verticalalignment="center",
        horizontalalignment="right",
        fontsize=9,
        fontweight="bold",
        color="darkred",
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
        color="darkblue",
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
        color="darkblue",
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
            fontsize=9,
            fontweight="bold",
            color="darkgreen",
            bbox=bbox_props,
        )

    # Return the line objects for potential further customization
    return poc_line, va_high_line, va_low_line, current_line


def style_volume_profile(
    ax, price_levels, bin_size, buy_volume_by_price, sell_volume_by_price, poc_price
):
    """Style and draw the volume profile visualization with consistent colors

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
        color="#54ff54",  # Brighter green
        alpha=0.6,
        label="Buy Volume",
    )

    ax.barh(
        price_levels[:-1],
        sell_volume_by_price,
        height=bin_size,
        color="#ff5454",  # Brighter red
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
        color="#cc44cc",  # Brighter purple
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

    # Add a legend to the volume profile
    ax.legend(
        loc="upper right",
        fontsize="medium",
        framealpha=0.9,
        facecolor="#333333",
        edgecolor="#888888",
    )


def style_candlestick_chart(ax1, ax_volume, fig, data):
    """Style and draw the main candlestick chart with volume

    Args:
        ax1: Main price chart axis
        ax_volume: Volume chart axis
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
        ax=ax1,
        volume=ax_volume,
        show_nontrading=False,
    )

    # Add grid lines
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

    # Configure date axis
    configure_date_axis(ax1, ax_volume, fig)


def create_chart_legend(ax1, current_price_exists=False, buy_focus_detected=False):
    """Create a standardized legend for the price chart

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
    ]

    if current_price_exists:
        legend_elements.append(
            Line2D(
                [0],
                [0],
                color="#ffcf40",
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
        framealpha=0.7,
        facecolor="#333333",
        edgecolor="#555555",
    )
