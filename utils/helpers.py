import textwrap
import re
from datetime import datetime, timedelta
import pandas as pd
from utils.logger import info, debug, warning, error
import numpy as np


def get_date_range(days_back):
    """Helper function to compute start and end date strings."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    info(
        f"Calculating date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
    )
    return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")


def format_market_cap(value, currency="USD"):
    """Format market cap in billions or millions for display"""
    if value is None:
        warning("Market cap value is None, returning '-'")
        return "-"

    if value >= 1e9:
        formatted_value = f"{value / 1e9:.2f} B {currency}"
    elif value >= 1e6:
        formatted_value = f"{value / 1e6:.2f} M {currency}"
    else:
        formatted_value = f"{value:.2f} {currency}"

    debug(f"Formatted market cap: {formatted_value}")
    return formatted_value


def parse_ticker_symbols(input_string):
    """Parse a string of ticker symbols separated by commas, spaces or new lines"""
    if not input_string:
        info("Input string is empty, returning empty list.")
        return []

    # Replace common separators and split
    symbols = re.split(r"[,\s\n]+", input_string)
    # Filter out empty strings and uppercase
    symbols = [symbol.strip().upper() for symbol in symbols if symbol.strip()]
    # Remove duplicates while preserving order
    unique_symbols = []
    for symbol in symbols:
        if symbol not in unique_symbols:
            unique_symbols.append(symbol)

    debug(f"Parsed ticker symbols: {unique_symbols}")
    return unique_symbols


def safe_divide(numerator, denominator, default=None):
    """Safely perform division, returning default on error"""
    try:
        if denominator == 0:
            warning("Denominator is zero, returning default value.")
            return default
        result = numerator / denominator
        debug(f"Division result: {result}")
        return result
    except (TypeError, ZeroDivisionError) as e:
        error(f"Error during division: {e}")
        return default


def safe_get(data, key, default=None):
    """Safely get a key from a dictionary"""
    try:
        value = data.get(key, default)
        debug(f"Retrieved value for key '{key}': {value}")
        return value
    except (AttributeError, TypeError):
        error(f"Failed to get key '{key}' from data.")
        return default


def replace_with_zero(lst):
    """Replace NaN values in a list with zeros."""
    if lst is None:
        info("Input list is None, returning list with a single zero.")
        return [0.0]
    if not isinstance(lst, list):
        result = 0.0 if (pd.isna(lst) or str(lst).lower() == "nan") else lst
        debug(f"Single value processed: {result}")
        return result
    result = [0.0 if (pd.isna(x) or str(x).lower() == "nan") else x for x in lst]
    debug(f"Processed list with NaN replaced: {result}")
    return result


def calculate_price_bins(data, bin_size=0.5):
    """Calculate price bins for volume profile analysis with fixed bin size

    Args:
        data: DataFrame with OHLC data
        bin_size: Fixed size of each price bin (default: 0.5)

    Returns:
        tuple: (price_levels, bin_size, price_range)
    """
    price_min = data["Low"].min()
    price_max = data["High"].max()

    # Add a small buffer to ensure we capture all prices
    buffer = (price_max - price_min) * 0.05
    price_min -= buffer
    price_max += buffer

    # Round min down and max up to nearest 0.5 to get clean bins
    price_min = np.floor(price_min * 2) / 2  # Round down to nearest 0.5
    price_max = np.ceil(price_max * 2) / 2  # Round up to nearest 0.5

    price_range = price_max - price_min
    num_bins = int(np.ceil(price_range / bin_size))

    # Create price levels
    price_levels = [price_min + i * bin_size for i in range(num_bins + 1)]

    return price_levels, bin_size, price_range


def distribute_volume_by_price(data, price_levels, bin_size):
    """Distribute volume into price bins, separating buy/sell volume

    Args:
        data: DataFrame with OHLC data
        price_levels: List of price levels
        bin_size: Size of each price bin

    Returns:
        tuple: (buy_volume_by_price, sell_volume_by_price)
    """
    num_bins = len(price_levels) - 1
    buy_volume_by_price = [0] * num_bins
    sell_volume_by_price = [0] * num_bins

    # Distribute volume into price bins
    for i in range(1, len(data)):
        row = data.iloc[i]

        # Determine if day was predominantly buying or selling
        is_up_day = row["Close"] > row["Open"]

        for j in range(num_bins):
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

    return buy_volume_by_price, sell_volume_by_price


def check_price_in_value_area(current_price, va_high, va_low, poc_price, option=None):
    """
    Check if current price meets the selected criteria relative to value area

    Note: This now considers value areas and POC calculated from buy volume only
    """
    if current_price is None or va_high is None or va_low is None or poc_price is None:
        return False

    if option == "below_poc":
        return current_price < poc_price
    elif option == "at_poc":
        # Within 0.5% of POC
        return abs(current_price - poc_price) / poc_price < 0.005
    elif option == "above_poc":
        return current_price > poc_price
    elif option == "below_va":
        return current_price < va_low
    elif option == "in_va":
        return va_low <= current_price <= va_high
    elif option == "above_va":
        return current_price > va_high
    else:
        # No filtering if no option selected
        return True
