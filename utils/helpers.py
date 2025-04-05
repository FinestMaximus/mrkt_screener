import textwrap
import re
from datetime import datetime, timedelta
import pandas as pd
from utils.logger import info, debug, warning, error


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
