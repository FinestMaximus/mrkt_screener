from datetime import datetime, timedelta
import textwrap
import logging


def get_date_range(days_back):
    """Helper function to compute start and end date strings."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    logging.debug(
        f"[utils.py][get_date_range] Computed date range: {start_date} to {end_date}"
    )
    return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")


def format_business_summary(summary):
    summary_no_colons = summary.replace(":", "\:")
    wrapped_summary = textwrap.fill(summary_no_colons)
    logging.debug(
        f"[utils.py][format_business_summary] Formatted summary: {wrapped_summary}"
    )
    return wrapped_summary
