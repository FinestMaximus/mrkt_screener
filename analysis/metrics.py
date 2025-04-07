import pandas as pd
from utils.logger import info, debug, warning, error


class FinancialMetrics:
    """Class for calculating and analyzing financial metrics"""

    def calculate_combined_metrics(
        self, company_symbols, basic_metrics, additional_metrics
    ):
        """Build a combined metrics dictionary from multiple sources"""
        combined_metrics = {}

        # Add company symbols
        combined_metrics["company_labels"] = company_symbols
        info(f"Company symbols added: {company_symbols}")

        # Add other metrics from the metrics dictionary
        for key, value in basic_metrics.items():
            if key != "company_labels":  # Skip this as we've already added it
                combined_metrics[key] = value
                debug(f"Added basic metric '{key}' with value: {value}")

        # Add metrics from additional_metrics
        for key, value in additional_metrics.items():
            if key in combined_metrics:
                # Check if lengths match before combining
                if len(value) != len(company_symbols):
                    # Handle length mismatch by padding with None values
                    warning(
                        f"Length mismatch in combined metrics for key: {key}. Padding with None values."
                    )
                    # If the filtered data is shorter, extend it
                    if len(value) < len(company_symbols):
                        value = value + [None] * (len(company_symbols) - len(value))
                    # If the filtered data is longer, truncate it
                    else:
                        value = value[: len(company_symbols)]
                debug(f"Combined metric '{key}' adjusted to match company symbols.")
            combined_metrics[key] = value

        info("Combined metrics calculated successfully.")
        return combined_metrics
