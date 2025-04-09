from market_profile import MarketProfile
from utils.logger import info, debug, warning, error
import pandas as pd
import numpy as np


class MarketProfileAnalyzer:
    """Class for analyzing market profiles and value areas"""

    def calculate_market_profile(self, data):
        """Calculate market profile metrics from price data"""
        if data.empty:
            warning("Cannot calculate market profile with empty data")
            return None, None, None, None

        try:
            info("Calculating market profile metrics.")

            # Original approach using the MarketProfile library for total volume
            # mp = MarketProfile(data)
            # mp_slice = mp[data.index.min() : data.index.max()]
            # va_low, va_high = mp_slice.value_area
            # poc_price = mp_slice.poc_price
            # profile_range = mp_slice.profile_range

            # NEW APPROACH: Calculate POC and value areas using sell volume only
            price_levels, bin_size, _ = self._calculate_price_bins(data)
            buy_volume_by_price, sell_volume_by_price = (
                self._distribute_volume_by_price(data, price_levels, bin_size)
            )

            # Find POC price (price level with highest SELL volume)
            max_vol_idx = np.argmax(sell_volume_by_price)
            poc_price = (price_levels[max_vol_idx] + price_levels[max_vol_idx + 1]) / 2

            # Calculate value area (70% of total sell volume)
            total_sell_volume = sum(sell_volume_by_price)
            target_volume = total_sell_volume * 0.7

            # Start from POC and work outward
            cum_volume = sell_volume_by_price[max_vol_idx]
            lower_idx = max_vol_idx
            upper_idx = max_vol_idx

            while cum_volume < target_volume and (
                lower_idx > 0 or upper_idx < len(sell_volume_by_price) - 1
            ):
                # Compare volumes at next higher and lower levels
                lower_vol = sell_volume_by_price[lower_idx - 1] if lower_idx > 0 else 0
                upper_vol = (
                    sell_volume_by_price[upper_idx + 1]
                    if upper_idx < len(sell_volume_by_price) - 1
                    else 0
                )

                # Add the level with higher volume
                if lower_vol > upper_vol and lower_idx > 0:
                    lower_idx -= 1
                    cum_volume += lower_vol
                elif upper_idx < len(sell_volume_by_price) - 1:
                    upper_idx += 1
                    cum_volume += upper_vol
                else:
                    break

            # Calculate value area high and low based on sell volume
            va_high = price_levels[upper_idx + 1]
            va_low = price_levels[lower_idx]
            profile_range = (price_levels[0], price_levels[-1])

            info(
                f"Market profile calculated using sell volume: VA High: {va_high}, VA Low: {va_low}, POC Price: {poc_price}, Profile Range: {profile_range}"
            )
            return va_high, va_low, poc_price, profile_range
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
