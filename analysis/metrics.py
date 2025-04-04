import pandas as pd
import logging


class FinancialMetrics:
    """Class for calculating and analyzing financial metrics"""

    def calculate_combined_metrics(
        self, company_symbols, basic_metrics, additional_metrics
    ):
        """Build a combined metrics dictionary from multiple sources"""
        combined_metrics = {}

        # Add company symbols
        combined_metrics["company_labels"] = company_symbols

        # Add other metrics from the metrics dictionary
        for key, value in basic_metrics.items():
            if key != "company_labels":  # Skip this as we've already added it
                combined_metrics[key] = value

        # Add metrics from additional_metrics
        for key, value in additional_metrics.items():
            if key in combined_metrics:
                # Check if lengths match before combining
                if len(value) != len(company_symbols):
                    # Handle length mismatch by padding with None values
                    logging.warning(
                        f"Length mismatch in combined metrics for key: {key}. Padding with None values."
                    )
                    # If the filtered data is shorter, extend it
                    if len(value) < len(company_symbols):
                        value = value + [None] * (len(company_symbols) - len(value))
                    # If the filtered data is longer, truncate it
                    else:
                        value = value[: len(company_symbols)]
            combined_metrics[key] = value

        return combined_metrics

    def get_dashboard_metrics(self, ticker_symbol, combined_metrics):
        """Extract metrics for a specific ticker for dashboard display"""
        # Default return values
        default_return = (None,) * 14  # Returns 14 None values

        try:
            # First check if all required keys exist
            required_keys = [
                "company_labels",
                "eps_values",
                "pe_values",
                "priceToSalesTrailing12Months",
                "priceToBook",
                "peg_values",
                "gross_margins",
                "fiftyTwoWeekHigh",
                "fiftyTwoWeekLow",
                "currentPrice",
                "targetMedianPrice",
                "targetLowPrice",
                "targetMeanPrice",
                "targetHighPrice",
                "recommendationMean",
            ]

            # Check if all required keys exist
            for key in required_keys:
                if key not in combined_metrics:
                    logging.warning(f"Missing key in combined_metrics: '{key}'")
                    return default_return

            if ticker_symbol in combined_metrics["company_labels"]:
                index = combined_metrics["company_labels"].index(ticker_symbol)

                # Check if index is valid for all lists
                for key in required_keys[1:]:  # Skip company_labels
                    if len(combined_metrics[key]) <= index:
                        logging.warning(
                            f"Index {index} out of range for key '{key}' with length {len(combined_metrics[key])}"
                        )
                        return default_return

                eps = combined_metrics["eps_values"][index]
                pe = combined_metrics["pe_values"][index]
                ps = combined_metrics["priceToSalesTrailing12Months"][index]
                pb = combined_metrics["priceToBook"][index]
                peg = combined_metrics["peg_values"][index]
                gm = combined_metrics["gross_margins"][index]
                wh52 = combined_metrics["fiftyTwoWeekHigh"][index]
                wl52 = combined_metrics["fiftyTwoWeekLow"][index]
                currentPrice = combined_metrics["currentPrice"][index]
                targetMedianPrice = combined_metrics["targetMedianPrice"][index]
                targetLowPrice = combined_metrics["targetLowPrice"][index]
                targetMeanPrice = combined_metrics["targetMeanPrice"][index]
                targetHighPrice = combined_metrics["targetHighPrice"][index]
                recommendationMean = combined_metrics["recommendationMean"][index]

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
            else:
                logging.warning(
                    f"Ticker '{ticker_symbol}' not found in the labels list."
                )
                return default_return
        except Exception as e:
            logging.error(f"An error occurred in get_dashboard_metrics: {e}")
            return default_return

    def filter_companies(
        self,
        list_metrics_all_tickers,
        eps_threshold,
        peg_threshold_low,
        peg_threshold_high,
        gross_margin_threshold,
    ):
        """Filter companies based on financial criteria"""
        try:
            if not isinstance(list_metrics_all_tickers, list):
                raise ValueError("list_metrics_all_tickers must be a list")

            # Convert list of dictionaries to DataFrame directly
            df = pd.DataFrame(list_metrics_all_tickers)

            # Apply filters using the original column names
            if not df.empty:
                # Calculate EPS if not directly available
                if "currentPrice" in df.columns and "trailingPE" in df.columns:
                    df["calculatedEPS"] = df["currentPrice"] / df["trailingPE"]
                    eps_column = "calculatedEPS"
                else:
                    eps_column = None

                # Keep track of valid criteria
                valid_criteria = []

                # EPS filter
                if eps_column:
                    valid_criteria.append(df[eps_column] > eps_threshold)

                # Profit margins filter (instead of gross_margin)
                if "profitMargins" in df.columns:
                    margin_criteria = df["profitMargins"] * 100 > gross_margin_threshold
                    valid_criteria.append(margin_criteria)

                # PEG filter - use trailingPE and earningsGrowth
                if "trailingPE" in df.columns and "earningsGrowth" in df.columns:
                    df["calculatedPEG"] = df["trailingPE"] / df["earningsGrowth"]
                    peg_criteria = (df["calculatedPEG"] > peg_threshold_low) & (
                        df["calculatedPEG"] <= peg_threshold_high
                    )
                    valid_criteria.append(peg_criteria)

                # Only apply filtering if we have criteria
                if valid_criteria:
                    combined_criteria = valid_criteria[0]
                    for criterion in valid_criteria[1:]:
                        combined_criteria = combined_criteria & criterion

                    filtered_df = df[combined_criteria]

                    # Sort by trailingPE if available
                    if "trailingPE" in filtered_df.columns:
                        filtered_df_sorted = filtered_df.sort_values(
                            by="trailingPE", ascending=True
                        )
                    else:
                        filtered_df_sorted = filtered_df

                    logging.info(
                        f"Filtered down to {len(filtered_df_sorted)} companies based on criteria."
                    )
                    return filtered_df_sorted
                else:
                    logging.warning("No valid criteria could be applied.")
                    return df
            else:
                logging.warning("DataFrame is empty.")
                return df

        except Exception as e:
            logging.error(f"An error occurred in filter_companies: {e}")
            import traceback

            traceback.print_exc()
            return pd.DataFrame()

    # ... other metrics analysis methods ...
