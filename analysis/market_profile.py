from market_profile import MarketProfile
import logging
import pandas as pd


class MarketProfileAnalyzer:
    """Class for analyzing market profiles and value areas"""

    def calculate_market_profile(self, data):
        """Calculate market profile metrics from price data"""
        if data.empty:
            logging.warning("Cannot calculate market profile with empty data")
            return None, None, None, None

        try:
            mp = MarketProfile(data)
            mp_slice = mp[data.index.min() : data.index.max()]

            va_low, va_high = mp_slice.value_area
            poc_price = mp_slice.poc_price
            profile_range = mp_slice.profile_range

            return va_high, va_low, poc_price, profile_range
        except Exception as e:
            logging.error(f"Error calculating market profile: {e}")
            return None, None, None, None

    def compare_price_to_value_area(
        self, ticker_symbol, current_price, va_high, va_low, poc_price
    ):
        """Compare current price to value area and POC"""
        if None in (current_price, va_high, va_low, poc_price):
            logging.warning(f"Cannot compare price for {ticker_symbol}: Missing values")
            return None

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
        elif results["below_value_area"]:
            results["position"] = "below_va"
            results["distance_percent"] = (current_price - va_low) / va_low * 100
        else:
            results["position"] = "in_va"
            mid_va = (va_high + va_low) / 2
            results["distance_percent"] = (current_price - mid_va) / mid_va * 100

        return results
