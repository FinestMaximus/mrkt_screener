import textwrap
import re
import logging
from datetime import datetime, timedelta
import pandas as pd


def get_date_range(days_back):
    """Helper function to compute start and end date strings."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")


def format_business_summary(summary):
    """Format business summary for display by escaping special characters"""
    if not summary:
        return "No summary available."
    summary_no_colons = summary.replace(":", "\:")
    wrapped_summary = textwrap.fill(summary_no_colons)
    return wrapped_summary


def format_market_cap(value, currency="USD"):
    """Format market cap in billions or millions for display"""
    if value is None:
        return "-"

    if value >= 1e9:
        return f"{value / 1e9:.2f} B {currency}"
    elif value >= 1e6:
        return f"{value / 1e6:.2f} M {currency}"
    else:
        return f"{value:.2f} {currency}"


def parse_ticker_symbols(input_string):
    """Parse a string of ticker symbols separated by commas, spaces or new lines"""
    if not input_string:
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

    return unique_symbols


def safe_divide(numerator, denominator, default=None):
    """Safely perform division, returning default on error"""
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (TypeError, ZeroDivisionError):
        return default


def safe_get(data, key, default=None):
    """Safely get a key from a dictionary"""
    try:
        return data.get(key, default)
    except (AttributeError, TypeError):
        return default


def replace_with_zero(lst):
    """Replace NaN values in a list with zeros."""
    if lst is None:
        return [0.0]
    if not isinstance(lst, list):
        return 0.0 if (pd.isna(lst) or str(lst).lower() == "nan") else lst
    return [0.0 if (pd.isna(x) or str(x).lower() == "nan") else x for x in lst]
