import streamlit as st
import textwrap

# Import custom logger functions
from utils.logger import info, debug, warning, error, critical

# Import styles module
from presentation.styles import (
    styled_metric,
    open_metrics_dashboard,
    close_metrics_dashboard,
)


class FinancialMetricsDisplay:
    """Class for displaying financial metrics dashboards"""

    def __init__(self):
        """Initialize the metrics display class"""
        pass

    def _format_business_summary(self, summary):
        """Format business summary for display"""
        summary_no_colons = summary.replace(":", "\:")
        wrapped_summary = textwrap.fill(summary_no_colons)
        return wrapped_summary

    def _get_metric_value(self, ticker_info, key, multiply_by_100=False):
        """Helper method to get a metric value from ticker info"""
        if ticker_info and key in ticker_info and ticker_info[key] is not None:
            value = ticker_info[key]
            if multiply_by_100 and isinstance(value, float) and value <= 1:
                value *= 100
            return value
        return None

    def _display_metric(
        self,
        col,
        ticker_info,
        key,
        label,
        tooltip,
        format_type="ratio",
        positive_good=None,
        multiply_by_100=False,
        prefix="",
        suffix="",
    ):
        """Helper method to display a metric in a column"""
        value = self._get_metric_value(ticker_info, key, multiply_by_100)
        with col:
            st.markdown(
                styled_metric(
                    label,
                    value,
                    tooltip,
                    format_type,
                    positive_good=positive_good,
                    prefix=prefix,
                    suffix=suffix,
                ),
                unsafe_allow_html=True,
            )

    def _display_financial_metric(
        self, col, ticker_info, key, label, tooltip, positive_good=True
    ):
        """Helper method to display financial values that could be in billions or millions"""
        value = self._get_metric_value(ticker_info, key)
        currency = ticker_info.get("financialCurrency", "USD")

        format_type = "billions" if value is not None and value >= 1e9 else "millions"
        self._display_metric(
            col,
            ticker_info,
            key,
            label,
            tooltip,
            format_type=format_type,
            positive_good=positive_good,
            suffix=" " + currency,
        )

    def _display_metrics_row(self, ticker_info, metrics_config):
        """Display a row of metrics based on configuration"""
        cols = st.columns(len(metrics_config))

        for i, config in enumerate(metrics_config):
            display_method = config.pop("display_method", self._display_metric)
            display_method(cols[i], ticker_info, **config)

    def display_ticker_metrics_dashboard(self, ticker):
        """Display ticker metrics focused on value investing metrics"""
        # Get values directly from ticker.info when available
        ticker_info = {}
        if ticker and hasattr(ticker, "info") and ticker.info is not None:
            ticker_info = ticker.info

        # Open the metrics dashboard container
        open_metrics_dashboard()

        # First row: Core Valuation Metrics
        row1_metrics = [
            {
                "key": "trailingPE",
                "label": "P/E (TTM)",
                "tooltip": "Trailing P/E - Price to last 12 months earnings",
                "positive_good": False,
            },
            {
                "key": "forwardPE",
                "label": "P/E (Fwd)",
                "tooltip": "Forward P/E - Price to projected earnings",
                "positive_good": False,
            },
            {
                "key": "priceToBook",
                "label": "P/B Ratio",
                "tooltip": "Price to Book - Lower values typically suggest undervaluation",
                "positive_good": False,
            },
            {
                "key": "priceToSalesTrailing12Months",
                "label": "P/S Ratio",
                "tooltip": "Price to Sales - Lower values may indicate better value",
                "positive_good": False,
            },
            {
                "key": "trailingPegRatio",
                "label": "PEG Ratio",
                "tooltip": "Price/Earnings to Growth - <1 typically indicates undervaluation",
                "positive_good": False,
            },
        ]
        self._display_metrics_row(ticker_info, row1_metrics)

        # Second row: Enterprise Value Metrics and Cash Flows
        row2_metrics = [
            {
                "key": "enterpriseToEbitda",
                "label": "EV/EBITDA",
                "tooltip": "Enterprise Value to EBITDA - Key valuation metric",
                "positive_good": False,
            },
            {
                "key": "enterpriseToRevenue",
                "label": "EV/Revenue",
                "tooltip": "Enterprise Value to Revenue - Alternative valuation metric",
                "positive_good": False,
            },
            {
                "key": "priceToOperCashPerShare",
                "label": "P/CF",
                "tooltip": "Price to Cash Flow - Lower values may indicate better value",
                "positive_good": False,
            },
            {
                "key": "freeCashflow",
                "label": "FCF",
                "tooltip": "Free Cash Flow - Cash after capex",
                "display_method": self._display_financial_metric,
                "positive_good": True,
            },
            {
                "key": "operatingCashflow",
                "label": "Op CF",
                "tooltip": "Operating Cash Flow - Cash from operations",
                "display_method": self._display_financial_metric,
                "positive_good": True,
            },
        ]
        self._display_metrics_row(ticker_info, row2_metrics)

        # Third row: Profitability and Returns
        row3_metrics = [
            {
                "key": "trailingEps",
                "label": "EPS",
                "tooltip": "Earnings Per Share - Company's profit per outstanding share",
                "format_type": "currency",
                "positive_good": True,
            },
            {
                "key": "returnOnEquity",
                "label": "ROE",
                "tooltip": "Return on Equity - Measures profitability relative to equity",
                "format_type": "percent",
                "positive_good": True,
                "multiply_by_100": True,
            },
            {
                "key": "returnOnAssets",
                "label": "ROA",
                "tooltip": "Return on Assets - Measures efficiency of asset utilization",
                "format_type": "percent",
                "positive_good": True,
                "multiply_by_100": True,
            },
            {
                "key": "grossMargins",
                "label": "Gross Margin",
                "tooltip": "Gross Margin - Indicates pricing power and production efficiency",
                "format_type": "percent",
                "positive_good": True,
                "multiply_by_100": True,
            },
            {
                "key": "profitMargins",
                "label": "Profit Margin",
                "tooltip": "Net Profit Margin - Measures overall profitability",
                "format_type": "percent",
                "positive_good": True,
                "multiply_by_100": True,
            },
        ]
        self._display_metrics_row(ticker_info, row3_metrics)

        # Fourth row: Financial Health and Book Value
        row4_metrics = [
            {
                "key": "debtToEquity",
                "label": "D/E Ratio",
                "tooltip": "Debt to Equity - Lower is better for financial stability",
                "positive_good": False,
            },
            {
                "key": "currentRatio",
                "label": "Current Ratio",
                "tooltip": "Current Ratio - Values >1 indicate good short-term financial health",
                "positive_good": True,
            },
            {
                "key": "quickRatio",
                "label": "Quick Ratio",
                "tooltip": "Quick Ratio - Liquidity excluding inventory (>1 is good)",
                "positive_good": True,
            },
            {
                "key": "bookValue",
                "label": "Book Value",
                "tooltip": "Book Value per Share - Theoretical value if company was liquidated",
                "format_type": "currency",
                "positive_good": True,
            },
            {
                "key": "marketCap",
                "label": "Market Cap",
                "tooltip": "Market Cap - Total market value of company",
                "display_method": self._display_financial_metric,
            },
        ]
        self._display_metrics_row(ticker_info, row4_metrics)

        # Fifth row: Income, Growth, and Dividends
        row5_metrics = [
            {
                "key": "dividendYield",
                "label": "Div Yield",
                "tooltip": "Dividend Yield - Annual dividend as percentage of share price",
                "format_type": "percent",
                "positive_good": True,
                "multiply_by_100": True,
            },
            {
                "key": "payoutRatio",
                "label": "Payout Ratio",
                "tooltip": "Payout Ratio - Percentage of earnings paid as dividends",
                "format_type": "percent",
                "positive_good": None,
                "multiply_by_100": True,
            },
            {
                "key": "earningsGrowth",
                "label": "EPS Growth",
                "tooltip": "Earnings Growth - Year-over-year growth in earnings",
                "format_type": "percent",
                "positive_good": True,
                "multiply_by_100": True,
            },
            {
                "key": "revenueGrowth",
                "label": "Rev Growth",
                "tooltip": "Revenue Growth - Year-over-year growth in revenue",
                "format_type": "percent",
                "positive_good": True,
                "multiply_by_100": True,
            },
            {
                "key": "beta",
                "label": "Beta",
                "tooltip": "Beta - Volatility vs. market (<1 = less volatile, >1 = more volatile)",
            },
        ]
        self._display_metrics_row(ticker_info, row5_metrics)

        # Close the metrics dashboard container
        close_metrics_dashboard()
