from utils.logger import info, debug, warning, error
import pandas as pd
import numpy as np


class MarketProfileAnalyzer:
    """Class for analyzing market profiles and value areas"""

    def __init__(self):
        """Initialize the Market Profile Analyzer"""
        pass

    def calculate_market_profile(self, data, value_area_volume_percent=0.7):
        """
        Calculate the market profile including Point of Control and Value Area

        Parameters:
        - data: DataFrame with OHLC and volume data
        - value_area_volume_percent: Percentage of volume to include in value area (default: 0.7 or 70%)

        Returns:
        - va_high: Value Area High price
        - va_low: Value Area Low price
        - poc_price: Point of Control price
        - volume_profile: Dictionary with volume profile data
        """
        if data.empty:
            warning("Cannot calculate market profile with empty data")
            return None, None, None, None

        try:
            info("Calculating market profile metrics.")

            # Calculate price bins and volume distribution
            price_min = min(data["Low"].min(), data["Close"].min())
            price_max = max(data["High"].max(), data["Close"].max())

            # Create price bins with appropriate granularity
            price_range = price_max - price_min
            # Adjust bin_count for appropriate granularity based on price
            bin_count = max(50, min(200, int(price_range * 100)))

            # Create price bins
            price_bins = np.linspace(price_min, price_max, bin_count)
            bin_size = price_bins[1] - price_bins[0]

            # Initialize volume arrays
            volume_by_price = np.zeros(len(price_bins) - 1)

            # Distribute volume across price bins
            for i, row in data.iterrows():
                # For each candle, distribute its volume across the price range it covered
                candle_low = row["Low"]
                candle_high = row["High"]
                candle_volume = row["Volume"]

                # Find which bins this candle spans
                low_bin = max(0, int((candle_low - price_min) / bin_size))
                high_bin = min(
                    len(price_bins) - 2, int((candle_high - price_min) / bin_size)
                )

                # Distribute volume proportionally across the spanned bins
                span = high_bin - low_bin + 1
                if span > 0:
                    volume_per_bin = candle_volume / span
                    for bin_idx in range(low_bin, high_bin + 1):
                        volume_by_price[bin_idx] += volume_per_bin

            # Find POC (Point of Control) - price with highest volume
            poc_bin_idx = np.argmax(volume_by_price)
            poc_price = price_bins[poc_bin_idx] + bin_size / 2

            # Calculate Value Area using the standard method
            total_volume = np.sum(volume_by_price)
            target_volume = total_volume * value_area_volume_percent

            # Start with POC
            current_volume = volume_by_price[poc_bin_idx]
            va_bins = [poc_bin_idx]

            # Expand outward until we reach target volume
            above_idx = poc_bin_idx + 1
            below_idx = poc_bin_idx - 1

            while current_volume < target_volume and (
                below_idx >= 0 or above_idx < len(volume_by_price)
            ):
                # Check volume above and below to decide which direction to expand
                vol_above = (
                    volume_by_price[above_idx]
                    if above_idx < len(volume_by_price)
                    else 0
                )
                vol_below = volume_by_price[below_idx] if below_idx >= 0 else 0

                # Add the side with higher volume
                if vol_above > vol_below:
                    va_bins.append(above_idx)
                    current_volume += vol_above
                    above_idx += 1
                else:
                    va_bins.append(below_idx)
                    current_volume += vol_below
                    below_idx -= 1

            # Determine VA high and low from bins
            va_high_bin = max(va_bins)
            va_low_bin = min(va_bins)

            va_high = price_bins[va_high_bin] + bin_size
            va_low = price_bins[va_low_bin]

            info(
                f"POC: {poc_price:.2f}, VA High: {va_high:.2f}, VA Low: {va_low:.2f}, "
                f"Volume %: {(current_volume/total_volume)*100:.1f}%"
            )

            # Create volume profile structure to return
            volume_profile = {
                "price_bins": price_bins,
                "volume_by_price": volume_by_price,
                "bin_size": bin_size,
                "total_volume": total_volume,
            }

            return va_high, va_low, poc_price, volume_profile
        except Exception as e:
            error(f"Error calculating market profile: {e}")
            return None, None, None, None

    def _calculate_price_bins(self, data):
        """Calculate price bins for volume distribution with fixed 0.5 price range"""
        # Determine price range
        price_min = data["Low"].min()
        price_max = data["High"].max()

        # Price range buffer (add 5% padding)
        price_range = price_max - price_min
        buffer = price_range * 0.05
        price_min -= buffer
        price_max += buffer

        # Use fixed bin size of 0.5
        bin_size = 0.5

        # Calculate number of bins based on fixed bin size
        num_bins = int(np.ceil((price_max - price_min) / bin_size))

        # Create price levels
        price_levels = [price_min + i * bin_size for i in range(num_bins + 1)]

        # Make sure bin_size is used by checking it's valid
        if bin_size <= 0:
            bin_size = price_range / max(1, num_bins)

        return price_levels, bin_size, num_bins

    def _distribute_volume_by_price(self, data, price_levels, bin_size):
        """Distribute volume by price level, separating buy and sell volume"""
        buy_volume_by_price = np.zeros(len(price_levels) - 1)
        sell_volume_by_price = np.zeros(len(price_levels) - 1)

        for idx, row in data.iterrows():
            # Determine if this candle is bullish (Close > Open) or bearish (Close < Open)
            is_bullish = row["Close"] >= row["Open"]

            # Rough estimation of buy/sell volume based on candle direction
            # In bullish candles, consider more volume as buying
            # In bearish candles, consider more volume as selling
            if is_bullish:
                buy_volume = row["Volume"] * 0.7  # Assume 70% of volume is buying
                sell_volume = row["Volume"] * 0.3  # Assume 30% of volume is selling
            else:
                buy_volume = row["Volume"] * 0.3  # Assume 30% of volume is buying
                sell_volume = row["Volume"] * 0.7  # Assume 70% of volume is selling

            # Distribute volume across price levels covered by this candle
            candle_low = row["Low"]
            candle_high = row["High"]

            # Find bins that this candle spans
            for i in range(len(price_levels) - 1):
                bin_low = price_levels[i]
                bin_high = price_levels[i + 1]

                # Check if this price bin overlaps with the candle's range
                if candle_high >= bin_low and candle_low <= bin_high:
                    # Calculate overlap and distribute volume proportionally
                    overlap_low = max(candle_low, bin_low)
                    overlap_high = min(candle_high, bin_high)
                    overlap_ratio = (
                        (overlap_high - overlap_low) / (candle_high - candle_low)
                        if candle_high > candle_low
                        else 1.0
                    )

                    # Add volume to this price bin
                    buy_volume_by_price[i] += buy_volume * overlap_ratio
                    sell_volume_by_price[i] += sell_volume * overlap_ratio

        return buy_volume_by_price, sell_volume_by_price

    def compare_price_to_value_area(
        self, ticker_symbol, current_price, va_high, va_low, poc_price
    ):
        """Compare current price to value area and POC"""
        if None in (current_price, va_high, va_low, poc_price):
            warning(f"Cannot compare price for {ticker_symbol}: Missing values")
            return None

        info(
            f"Comparing current price {current_price} to value area for {ticker_symbol}."
        )
        results = {
            "above_value_area": current_price > va_high,
            "in_value_area": va_low <= current_price <= va_high,
            "below_value_area": current_price < va_low,
            "above_poc": current_price > poc_price,
            "at_poc": current_price == poc_price,
            "below_poc": current_price < poc_price,
            "distance_to_poc": current_price - poc_price,
            "distance_to_va_high": current_price - va_high,
            "distance_to_va_low": current_price - va_low,
            "price": current_price,
            "va_high": va_high,
            "va_low": va_low,
            "poc_price": poc_price,
        }

        if results["above_value_area"]:
            results["position"] = "above_va"
            results["distance_percent"] = (current_price - va_high) / va_high * 100
            info(f"{ticker_symbol} is above the value area.")
        elif results["below_value_area"]:
            results["position"] = "below_va"
            results["distance_percent"] = (current_price - va_low) / va_low * 100
            info(f"{ticker_symbol} is below the value area.")
        else:
            results["position"] = "in_va"
            mid_va = (va_high + va_low) / 2
            results["distance_percent"] = (current_price - mid_va) / mid_va * 100
            info(f"{ticker_symbol} is within the value area.")

        debug(f"Comparison results: {results}")
        return results
