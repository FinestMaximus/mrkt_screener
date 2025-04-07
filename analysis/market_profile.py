from market_profile import MarketProfile
from utils.logger import info, debug, warning, error
import pandas as pd


class MarketProfileAnalyzer:
    """Class for analyzing market profiles and value areas"""

    def calculate_market_profile(self, data):
        """Calculate market profile metrics from price data"""
        if data.empty:
            warning("Cannot calculate market profile with empty data")
            return None, None, None, None

        try:
            info("Calculating market profile metrics.")
            mp = MarketProfile(data)
            mp_slice = mp[data.index.min() : data.index.max()]

            va_low, va_high = mp_slice.value_area
            poc_price = mp_slice.poc_price
            profile_range = mp_slice.profile_range

            info(
                f"Market profile calculated: VA High: {va_high}, VA Low: {va_low}, POC Price: {poc_price}, Profile Range: {profile_range}"
            )
            return va_high, va_low, poc_price, profile_range
        except Exception as e:
            error(f"Error calculating market profile: {e}")
            return None, None, None, None

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
