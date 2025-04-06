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

    def get_dashboard_metrics(self, ticker_symbol, metrics):
        """Extract metrics for a specific ticker for dashboard display"""
        # Default return values
        default_return = (None,) * 14  # Returns 14 None values

        try:
            # First check if all required keys exist - updated to match actual available keys
            debug("combined_metrics keys: " + str(metrics.keys()))

            # Check if ticker exists in labels
            if ticker_symbol not in metrics.get("company_labels", []):
                warning(f"Ticker '{ticker_symbol}' not found in the labels list.")
                return default_return

            index = metrics["company_labels"].index(ticker_symbol)
            info(f"Ticker '{ticker_symbol}' found at index {index}.")

            # Extract values using available keys, with fallbacks for missing keys
            try:
                # Calculate EPS from price and PE ratio if available
                eps = None
                if "currentPrice" in metrics and "trailingPE" in metrics:
                    if metrics["trailingPE"][index] not in (None, 0):
                        eps = (
                            metrics["currentPrice"][index]
                            / metrics["trailingPE"][index]
                        )

                # Map keys to their available counterparts
                pe = metrics.get("trailingPE", [None] * len(metrics["company_labels"]))[
                    index
                ]
                ps = metrics.get(
                    "priceToSalesTrailing12Months",
                    metrics.get("forwardPE", [None] * len(metrics["company_labels"])),
                )[index]
                pb = metrics.get(
                    "priceToBook", [None] * len(metrics["company_labels"])
                )[index]

                # Calculate PEG if possible, otherwise None
                # PEG ratio: 0-1 indicates undervalued stock, >1 potentially overvalued
                peg = None
                if "forwardPE" in metrics and "revenueGrowth" in metrics:
                    if metrics["revenueGrowth"][index] not in (None, 0):
                        # Use absolute value of growth to handle negative growth scenarios
                        growth_value = abs(metrics["revenueGrowth"][index])
                        if growth_value > 0:  # Avoid division by zero
                            peg = metrics["forwardPE"][index] / growth_value

                gm = metrics.get(
                    "grossMargins", [None] * len(metrics["company_labels"])
                )[index]

                # 52-week high/low might not be available, use None as fallback
                wh52 = metrics.get(
                    "fiftyTwoWeekHigh", [None] * len(metrics["company_labels"])
                )[index]
                wl52 = metrics.get(
                    "fiftyTwoWeekLow", [None] * len(metrics["company_labels"])
                )[index]

                currentPrice = metrics.get(
                    "currentPrice", [None] * len(metrics["company_labels"])
                )[index]
                targetMedianPrice = metrics.get(
                    "targetMedianPrice",
                    [None] * len(metrics["company_labels"]),
                )[index]
                targetLowPrice = metrics.get(
                    "targetLowPrice", [None] * len(metrics["company_labels"])
                )[index]
                targetMeanPrice = metrics.get(
                    "targetMeanPrice", [None] * len(metrics["company_labels"])
                )[index]
                targetHighPrice = metrics.get(
                    "targetHighPrice", [None] * len(metrics["company_labels"])
                )[index]
                recommendationMean = metrics.get(
                    "recommendationMean",
                    [None] * len(metrics["company_labels"]),
                )[index]

                info(f"Dashboard metrics extracted for ticker '{ticker_symbol}'.")
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
            except IndexError as ie:
                error(f"Index error when retrieving metrics for {ticker_symbol}: {ie}")
                return default_return
        except Exception as e:
            error(f"An error occurred in get_dashboard_metrics: {e}")
            return default_return
