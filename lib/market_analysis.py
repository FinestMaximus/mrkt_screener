from market_profile import MarketProfile
import pandas as pd
import logging


def calculate_market_profile(data):
    try:
        mp = MarketProfile(data)
        mp_slice = mp[data.index.min() : data.index.max()]
        va_low, va_high = mp_slice.value_area
        poc_price = mp_slice.poc_price
        profile_range = mp_slice.profile_range
        return va_high, va_low, poc_price, profile_range
    except Exception as e:
        logging.error(f"[market_analysis.py][calculate_market_profile] Error: {str(e)}")
        return None, None, None, None


def filter_companies(
    metrics,
    eps_threshold,
    peg_threshold_low,
    peg_threshold_high,
    gross_margin_threshold,
):
    """Filter companies based on various financial metrics."""
    logging.debug("[market_analysis.py][filter_companies] Starting function.")

    try:
        # Create DataFrame for filtering
        df = pd.DataFrame(
            {
                "company": metrics["company_labels"],
                "eps": metrics["eps_values"],
                "peg": metrics[
                    "trailingPegRatio"
                ],  # Use trailingPegRatio instead of peg_values
                "gross_margin": metrics["gross_margins"],
                "short_name": metrics["short_name"],
            }
        )

        # Log initial state
        logging.info(f"Initial companies count: {len(df)}")
        logging.debug("Initial companies data:\n%s", df)

        # Apply filters one by one and log results
        eps_filter = df["eps"] >= eps_threshold
        peg_filter = df["peg"].between(peg_threshold_low, peg_threshold_high)
        margin_filter = df["gross_margin"] >= gross_margin_threshold

        # Log which companies are filtered out by each criterion
        logging.info(
            f"Companies passing EPS filter ({eps_threshold}): {df[eps_filter]['company'].tolist()}"
        )
        logging.info(
            f"Companies passing PEG filter ({peg_threshold_low}-{peg_threshold_high}): {df[peg_filter]['company'].tolist()}"
        )
        logging.info(
            f"Companies passing margin filter ({gross_margin_threshold}): {df[margin_filter]['company'].tolist()}"
        )

        # Apply all filters
        filtered_df = df[eps_filter & peg_filter & margin_filter]

        logging.info(f"Final filtered companies count: {len(filtered_df)}")
        if not filtered_df.empty:
            logging.info(
                "Final filtered companies: %s", filtered_df["company"].tolist()
            )
        else:
            logging.warning("No companies passed all filters")

        return filtered_df

    except Exception as e:
        logging.error(
            "[market_analysis.py][filter_companies] An unexpected error occurred: %s",
            str(e),
        )
        logging.error("Stack trace:", exc_info=True)
        return pd.DataFrame()  # Return empty DataFrame on error
