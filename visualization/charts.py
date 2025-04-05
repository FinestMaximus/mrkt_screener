import plotly.graph_objects as go
from plotly.subplots import make_subplots
import mplfinance as mpf
import pandas as pd
import streamlit as st
from datetime import datetime
import textwrap
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import yfinance as yf

from streamlit.delta_generator import DeltaGenerator

# Import custom logger functions
from analysis.metrics import FinancialMetrics
from data.news_research import SentimentAnalyzer
from utils.logger import info, debug, warning, error, critical


class ChartGenerator:
    """Class for generating various financial charts and visualizations"""

    def __init__(
        self,
        data_fetcher,
        metrics_analyzer,
        market_profile_analyzer,
        sentiment_analyzer,
    ):
        """Initialize with required service dependencies"""
        self.data_fetcher = data_fetcher
        self.metrics_analyzer = metrics_analyzer
        self.market_profile_analyzer = market_profile_analyzer
        self.sentiment_analyzer = sentiment_analyzer

    def create_sector_distribution_chart(self, industries, title):
        """Create a pie chart showing distribution of tickers by sector"""
        sector_counts = {sector: len(tickers) for sector, tickers in industries.items()}

        labels = list(sector_counts.keys())
        sizes = list(sector_counts.values())

        fig = go.Figure(data=[go.Pie(labels=labels, values=sizes, hole=0.3)])

        fig.update_layout(
            title_text=title,
            annotations=[
                dict(text="Sectors", x=0.50, y=0.5, font_size=20, showarrow=False)
            ],
        )

        return fig

    def create_candle_chart_with_profile(self, data, poc_price, va_high, va_low):
        """Create a candlestick chart with volume profile overlay"""
        if data.empty:
            warning("Cannot create chart with empty data")
            return None

        # Create price-volume data for volume profile
        price_bins = 100
        price_range = data["High"].max() - data["Low"].min()
        bin_size = price_range / price_bins

        # Create figure with price chart, volume chart, and volume profile
        fig = plt.figure(figsize=(12, 8))
        gs = GridSpec(4, 5, figure=fig)

        # Main price chart
        ax1 = fig.add_subplot(gs[0:3, 0:4])

        # Volume chart below price chart
        ax_volume = fig.add_subplot(gs[3:4, 0:4], sharex=ax1)

        # Volume profile chart on the right
        ax2 = fig.add_subplot(gs[0:3, 4], sharey=ax1)

        # Plot candlestick chart
        mpf.plot(
            data,
            type="candle",
            style="yahoo",
            ax=ax1,
            volume=ax_volume,
            show_nontrading=False,
        )

        # Add POC and Value Area lines
        ax1.axhline(
            y=poc_price, color="red", linestyle="dashed", linewidth=2, label="POC"
        )
        ax1.axhline(
            y=va_high, color="blue", linestyle="dashed", linewidth=1, label="VA High"
        )
        ax1.axhline(
            y=va_low, color="blue", linestyle="dashed", linewidth=1, label="VA Low"
        )

        # Create and plot volume profile
        price_levels = [data["Low"].min() + i * bin_size for i in range(price_bins + 1)]
        volume_by_price = [0] * price_bins

        for idx, row in data.iterrows():
            for i in range(price_bins):
                lower_bound = price_levels[i]
                upper_bound = price_levels[i + 1]
                if not (row["High"] < lower_bound or row["Low"] > upper_bound):
                    volume_by_price[i] += row["Volume"] / (
                        (row["High"] - row["Low"]) / bin_size
                    )

        # Plot volume profile histogram on right side
        ax2.barh(
            price_levels[:-1],
            volume_by_price,
            height=bin_size,
            color="#1f77b4",
            alpha=0.7,
        )

        # Highlight POC in volume profile
        poc_bin_idx = max(range(len(volume_by_price)), key=volume_by_price.__getitem__)
        ax2.barh(
            price_levels[poc_bin_idx],
            volume_by_price[poc_bin_idx],
            height=bin_size,
            color="red",
            alpha=0.7,
        )

        # Remove unnecessary ticks
        ax2.set_xticks([])
        ax2.set_yticks([])

        # Add a legend
        ax1.legend(["POC", "VA High", "VA Low"], loc="upper left")

        plt.tight_layout()

        return fig

    def create_combined_metrics_chart(self, combined_metrics):
        """Create a combined interactive chart with multiple subplots for company comparison"""
        if not combined_metrics or not isinstance(combined_metrics, dict):
            raise ValueError("combined_metrics must be a non-empty dictionary.")

        company_labels = combined_metrics.get("company_labels", [])
        eps_values = combined_metrics.get("eps_values", [])

        if not all(len(lst) == len(company_labels) for lst in [eps_values]):
            raise ValueError("Inconsistent data lengths found in combined_metrics.")

        high_diffs = [
            combined_metrics["price_diff"].get(company, {}).get("high_diff", 0)
            for company in company_labels
        ]
        low_diffs = [
            combined_metrics["price_diff"].get(company, {}).get("low_diff", 0)
            for company in company_labels
        ]
        market_caps = combined_metrics.get("market_caps", [])
        priceToBook = combined_metrics.get("priceToBook", [])
        pe_values = combined_metrics.get("pe_values", [])
        peg_values = combined_metrics.get("peg_values", [])
        priceToSalesTrailing12Months = combined_metrics.get(
            "priceToSalesTrailing12Months", []
        )
        gross_margins = combined_metrics.get("gross_margins", [])
        recommendations_summary = combined_metrics.get("recommendations_summary", [])
        earningsGrowth = combined_metrics.get("earningsGrowth", [])
        revenueGrowth = combined_metrics.get("revenueGrowth", [])
        freeCashflow = combined_metrics.get("freeCashflow", [])
        opCashflow = combined_metrics.get("opCashflow", [])
        repurchaseCapStock = combined_metrics.get("repurchaseCapStock", [])

        peg_min, peg_max = min(peg_values, default=0), max(peg_values, default=1)

        fig = make_subplots(
            rows=4,
            cols=3,
            subplot_titles=(
                "Price Difference % Over the Last Year",
                "EPS vs P/E Ratio",
                "Gross Margin (%)",
                "EPS vs P/B Ratio",
                "EPS vs PEG Ratio",
                "EPS vs P/S Ratio",
                "Upgrades & Downgrades Timeline",
                "Earnings Growth vs Revenue Growth",
                "Free Cash Flow",
                "Operational Cashflow",
                "Repurchase of Capital Stock",
            ),
            specs=[
                [{}, {}, {}],
                [{}, {}, {}],
                [{"colspan": 2}, None, {}],
                [{}, {}, {}],
            ],
            vertical_spacing=0.1,
        )

        colors = {
            company: f"hsl({(i / len(company_labels) * 360)},100%,50%)"
            for i, company in enumerate(company_labels)
        }

        for i, company in enumerate(company_labels):
            try:
                legendgroup = f"group_{company}"
                marker_size = max(market_caps[i] / max(market_caps, default=1) * 50, 5)

                fig.add_trace(
                    go.Scatter(
                        x=[high_diffs[i]],
                        y=[low_diffs[i]],
                        marker=dict(size=10, color=colors[company]),
                        legendgroup=legendgroup,
                        name=company,
                        hoverinfo="none",
                        hovertemplate=f"Company: {company}<br>High Diff: %{{x}}<br>Low Diff: %{{y}}<extra></extra>",
                    ),
                    row=1,
                    col=1,
                )

                fig.add_trace(
                    go.Scatter(
                        x=[eps_values[i]],
                        y=[pe_values[i]],
                        marker=dict(size=marker_size, color=colors[company]),
                        legendgroup=legendgroup,
                        showlegend=False,
                        hoverinfo="none",
                        hovertemplate=f"Company: {company}<br>EPS: ${eps_values[i]}<br>P/E Ratio: {pe_values[i]}<extra></extra>",
                    ),
                    row=1,
                    col=2,
                )

                # Continue adding all the traces as in the original code
                # ...

                # Add more traces for the other metrics

            except (ValueError, TypeError, IndexError) as error:
                error(f"Error plotting data for {company}: {error}")
                continue

        titles = [
            ("High Diff (%)", "Low Diff (%)"),
            ("EPS", "P/E Ratio"),
            ("Company", "Gross Margin (%)"),
            ("Price to Books", "EPS"),
            ("PEG", "EPS"),
            ("P/S", "EPS"),
            ("Earnings Growth", "Revenue Growth"),
            ("Years", "Free Cash Flow"),
            ("Years", "Operational Cashflow"),
            ("Years", "Repurchase of Capital Stock"),
        ]

        for col, (x_title, y_title) in enumerate(titles, start=1):
            fig.update_xaxes(title_text=x_title, row=1, col=col)
            fig.update_yaxes(title_text=y_title, row=1, col=col)

        fig.update_xaxes(title_text="Recommendation Type", row=1, col=4)
        fig.update_yaxes(title_text="Number of Recommendations", row=1, col=4)

        fig.update_layout(height=1500)

        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    buttons=list(
                        [
                            dict(
                                args=[{"visible": "legendonly"}],
                                label="Hide All",
                                method="restyle",
                            ),
                            dict(
                                args=[{"visible": True}],
                                label="Show All",
                                method="restyle",
                            ),
                        ]
                    ),
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0,
                    xanchor="left",
                    y=-0.15,
                    yanchor="top",
                ),
            ]
        )

        return fig

    def _display_sentiment_gauge(self, news_data, total_polarity):
        """Display sentiment gauge chart"""
        if len(news_data) > 0:
            average_sentiment = total_polarity / len(news_data)

            if average_sentiment >= 0.5:
                color = "green"
            elif average_sentiment >= 0:
                color = "orange"
            else:
                color = "red"

            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=average_sentiment,
                    domain={"x": [0, 1], "y": [0, 1]},
                    gauge={"axis": {"range": [-1, 1]}, "bar": {"color": color}},
                )
            )

            fig.update_layout(width=300, height=300)

            st.plotly_chart(fig)
        else:
            info("No news data available for sentiment analysis.")
            st.write("No sentiment data available.")

    def _display_news_articles(self, news_data):
        """Display formatted news articles data"""
        if not news_data:
            st.write("No news data available.")
            return

        for news_item in news_data:
            title = news_item["Title"]
            if len(title) > 70:
                title = title[:67] + "..."
            rounded_sentiment = round(news_item["Sentiment"], 2)
            days_ago = news_item["Days Ago"]
            st.markdown(
                f"{rounded_sentiment} - [{title.replace(':', '')}]({news_item['Link']}) - ({days_ago} days ago)"
            )

    def plot_sector_distribution_interactive(self, industries, title):
        """Create a pie chart showing distribution of stocks by sector"""
        sector_counts = {sector: len(tickers) for sector, tickers in industries.items()}

        labels = list(sector_counts.keys())
        sizes = list(sector_counts.values())

        fig = go.Figure(data=[go.Pie(labels=labels, values=sizes, hole=0.3)])

        fig.update_layout(
            title_text=title,
            annotations=[
                dict(text="Sectors", x=0.50, y=0.5, font_size=20, showarrow=False)
            ],
        )

        debug(
            "[charts.py][plot_sector_distribution_interactive] Successfully created sector distribution plot."
        )

        return fig

    def plot_combined_interactive(self, combined_metrics):
        """Create an interactive dashboard with multiple financial metrics visualizations"""
        if not combined_metrics or not isinstance(combined_metrics, dict):
            raise ValueError("combined_metrics must be a non-empty dictionary.")

        company_labels = combined_metrics.get("company_labels", [])
        eps_values = combined_metrics.get("eps_values", [])

        if not all(len(lst) == len(company_labels) for lst in [eps_values]):
            raise ValueError("Inconsistent data lengths found in combined_metrics.")

        high_diffs = [
            combined_metrics["price_diff"].get(company, {}).get("high_diff", 0)
            for company in company_labels
        ]
        low_diffs = [
            combined_metrics["price_diff"].get(company, {}).get("low_diff", 0)
            for company in company_labels
        ]
        market_caps = combined_metrics.get("market_caps", [])
        priceToBook = combined_metrics.get("priceToBook", [])
        pe_values = combined_metrics.get("pe_values", [])
        peg_values = combined_metrics.get("peg_values", [])
        priceToSalesTrailing12Months = combined_metrics.get(
            "priceToSalesTrailing12Months", []
        )
        gross_margins = combined_metrics.get("gross_margins", [])
        recommendations_summary = combined_metrics.get("recommendations_summary", [])
        earningsGrowth = combined_metrics.get("earningsGrowth", [])
        revenueGrowth = combined_metrics.get("revenueGrowth", [])
        freeCashflow = combined_metrics.get("freeCashflow", [])
        opCashflow = combined_metrics.get("opCashflow", [])
        repurchaseCapStock = combined_metrics.get("repurchaseCapStock", [])

        peg_min, peg_max = min(peg_values, default=0), max(peg_values, default=1)

        fig = make_subplots(
            rows=4,
            cols=3,
            subplot_titles=(
                "Price Difference % Over the Last Year",
                "EPS vs P/E Ratio",
                "Gross Margin (%)",
                "EPS vs P/B Ratio",
                "EPS vs PEG Ratio",
                "EPS vs P/S Ratio",
                "Upgrades & Downgrades Timeline",
                "Earnings Growth vs Revenue Growth",
                "Free Cash Flow",
                "Operational Cashflow",
                "Repurchase of Capital Stock",
            ),
            specs=[
                [{}, {}, {}],
                [{}, {}, {}],
                [{"colspan": 2}, None, {}],
                [{}, {}, {}],
            ],
            vertical_spacing=0.1,
        )

        colors = {
            company: f"hsl({(i / len(company_labels) * 360)},100%,50%)"
            for i, company in enumerate(company_labels)
        }

        for i, company in enumerate(company_labels):
            try:
                legendgroup = f"group_{company}"
                marker_size = max(market_caps[i] / max(market_caps, default=1) * 50, 5)

                # Add traces for each visualization
                self._add_chart_traces(
                    fig,
                    company,
                    i,
                    colors,
                    legendgroup,
                    marker_size,
                    high_diffs,
                    low_diffs,
                    eps_values,
                    pe_values,
                    gross_margins,
                    priceToBook,
                    peg_values,
                    priceToSalesTrailing12Months,
                    recommendations_summary,
                    earningsGrowth,
                    revenueGrowth,
                    freeCashflow,
                    opCashflow,
                    repurchaseCapStock,
                    company_labels,
                )

            except (ValueError, TypeError, IndexError) as error:
                error(f"Error plotting data for {company}: {error}")
                continue

        # Set chart titles and labels
        self._configure_chart_axes(fig)

        fig.update_layout(height=1500)
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    buttons=list(
                        [
                            dict(
                                args=[{"visible": "legendonly"}],
                                label="Hide All",
                                method="restyle",
                            ),
                            dict(
                                args=[{"visible": True}],
                                label="Show All",
                                method="restyle",
                            ),
                        ]
                    ),
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0,
                    xanchor="left",
                    y=-0.15,
                    yanchor="top",
                ),
            ]
        )

        return fig

    def _add_chart_traces(
        self,
        fig,
        company,
        i,
        colors,
        legendgroup,
        marker_size,
        high_diffs,
        low_diffs,
        eps_values,
        pe_values,
        gross_margins,
        priceToBook,
        peg_values,
        priceToSalesTrailing12Months,
        recommendations_summary,
        earningsGrowth,
        revenueGrowth,
        freeCashflow,
        opCashflow,
        repurchaseCapStock,
        company_labels,
    ):
        """Helper method to add traces to the combined chart"""
        # Price difference scatter plot
        fig.add_trace(
            go.Scatter(
                x=[high_diffs[i]],
                y=[low_diffs[i]],
                marker=dict(size=10, color=colors[company]),
                legendgroup=legendgroup,
                name=company,
                hoverinfo="none",
                hovertemplate=f"Company: {company}<br>High Diff: %{{x}}<br>Low Diff: %{{y}}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        # EPS vs P/E scatter plot
        fig.add_trace(
            go.Scatter(
                x=[eps_values[i]],
                y=[pe_values[i]],
                marker=dict(size=marker_size, color=colors[company]),
                legendgroup=legendgroup,
                showlegend=False,
                hoverinfo="none",
                hovertemplate=f"Company: {company}<br>EPS: ${eps_values[i]}<br>P/E Ratio: {pe_values[i]}<extra></extra>",
            ),
            row=1,
            col=2,
        )

        # Gross margin bar chart
        fig.add_trace(
            go.Bar(
                x=[company_labels[i]],
                y=[gross_margins[i] * 100],
                marker=dict(color=colors[company]),
                legendgroup=legendgroup,
                showlegend=False,
                width=0.8,
            ),
            row=1,
            col=3,
        )

        # EPS vs P/B scatter plot
        fig.add_trace(
            go.Scatter(
                x=[eps_values[i]],
                y=[priceToBook[i]],
                marker=dict(size=marker_size, color=colors[company]),
                legendgroup=legendgroup,
                showlegend=False,
                hoverinfo="none",
                hovertemplate=f"Company: {company}<br>EPS: ${eps_values[i]}<br>P/B Ratio: {priceToBook[i]}<extra></extra>",
            ),
            row=2,
            col=1,
        )

        # EPS vs PEG scatter plot
        fig.add_trace(
            go.Scatter(
                x=[eps_values[i]],
                y=[peg_values[i]],
                marker=dict(size=marker_size, color=colors[company]),
                legendgroup=legendgroup,
                showlegend=False,
                hoverinfo="none",
                hovertemplate=f"Company: {company}<br>EPS: ${eps_values[i]}<br>PEG Ratio: {peg_values[i]}<extra></extra>",
            ),
            row=2,
            col=2,
        )

        # EPS vs P/S scatter plot
        fig.add_trace(
            go.Scatter(
                x=[eps_values[i]],
                y=[priceToSalesTrailing12Months[i]],
                marker=dict(size=marker_size, color=colors[company]),
                legendgroup=legendgroup,
                showlegend=False,
                hoverinfo="none",
                hovertemplate=f"Company: {company}<br>EPS: ${eps_values[i]}<br>P/S Ratio: {priceToSalesTrailing12Months[i]}<extra></extra>",
            ),
            row=2,
            col=3,
        )

        # Add recommendations summary if available
        self._add_recommendations_trace(
            fig, company, i, colors, legendgroup, recommendations_summary
        )

        # Add cashflow traces
        self._add_cashflow_traces(
            fig,
            company,
            i,
            colors,
            legendgroup,
            freeCashflow,
            opCashflow,
            repurchaseCapStock,
        )

        # Add growth comparison trace
        fig.add_trace(
            go.Scatter(
                x=[revenueGrowth[i]],
                y=[earningsGrowth[i]],
                marker=dict(size=10, color=colors[company]),
                legendgroup=legendgroup,
                showlegend=False,
                hoverinfo="none",
                hovertemplate=f"Company: {company}<br>Revenue Growth: {revenueGrowth[i]}<br>Earnings Growth: {earningsGrowth[i]}<extra></extra>",
            ),
            row=3,
            col=3,
        )

    def _add_recommendations_trace(
        self, fig, company, i, colors, legendgroup, recommendations_summary
    ):
        """Helper to add recommendation summary traces"""
        current_recommendations = recommendations_summary[i]

        if (
            isinstance(current_recommendations, dict)
            and "0m" in current_recommendations
        ):
            ratings = current_recommendations["0m"]
            rating_categories = ["strongBuy", "buy", "hold", "sell", "strongSell"]
            rating_values = [ratings.get(category, 0) for category in rating_categories]

            bar_heights = rating_values

            fig.add_trace(
                go.Bar(
                    x=rating_categories,
                    y=bar_heights,
                    marker=dict(color=colors[company]),
                    name=company,
                    legendgroup=legendgroup,
                    showlegend=False,
                    text=company,
                    hoverinfo="y+text",
                ),
                row=3,
                col=1,
            )

    def _add_cashflow_traces(
        self,
        fig,
        company,
        i,
        colors,
        legendgroup,
        freeCashflow,
        opCashflow,
        repurchaseCapStock,
    ):
        """Helper to add cashflow related traces"""
        now = datetime.now()
        year = now.year

        if now.month < 4:
            year -= 1

        years = [str(year - i) for i in range(3, -1, -1)]

        # Free Cash Flow
        if isinstance(freeCashflow[i], list):
            fig.add_trace(
                go.Scatter(
                    x=years[: len(freeCashflow[i])],
                    y=[cf for cf in freeCashflow[i]],
                    name=company,
                    hoverinfo="none",
                    legendgroup=legendgroup,
                    showlegend=False,
                    hovertemplate=f"Company: {company}<br>Year: %{{x}}<br> Free Cashflow: %{{y}}<extra></extra>",
                ),
                row=4,
                col=3,
            )

        # Operational Cashflow
        if isinstance(opCashflow[i], list):
            fig.add_trace(
                go.Scatter(
                    x=years[: len(opCashflow[i])],
                    y=[cf for cf in opCashflow[i]],
                    mode="lines",
                    name=company,
                    hoverinfo="none",
                    legendgroup=legendgroup,
                    showlegend=False,
                    hovertemplate=f"Company: {company}<br>Year: %{{x}}<br> Operational Cashflow: %{{y}}<extra></extra>",
                ),
                row=4,
                col=1,
            )

        # Repurchase of Capital Stock
        if isinstance(repurchaseCapStock[i], list):
            fig.add_trace(
                go.Scatter(
                    x=years[: len(repurchaseCapStock[i])],
                    y=[-cf for cf in repurchaseCapStock[i]],
                    mode="lines",
                    name=company,
                    hoverinfo="none",
                    legendgroup=legendgroup,
                    showlegend=False,
                    hovertemplate=f"Company: {company}<br>Year: %{{x}}<br> Repurchase of Capital Stock: %{{y}}<extra></extra>",
                ),
                row=4,
                col=2,
            )

    def _configure_chart_axes(self, fig):
        """Configure axes titles and ranges for the combined chart"""
        titles = [
            ("High Diff (%)", "Low Diff (%)"),
            ("EPS", "P/E Ratio"),
            ("Company", "Gross Margin (%)"),
            ("Price to Books", "EPS"),
            ("PEG", "EPS"),
            ("P/S", "EPS"),
            ("Earnings Growth", "Revenue Growth"),
            ("Years", "Free Cash Flow"),
            ("Years", "Operational Cashflow"),
            ("Years", "Repurchase of Capital Stock"),
        ]

        for col, (x_title, y_title) in enumerate(titles, start=1):
            fig.update_xaxes(title_text=x_title, row=1, col=col)
            fig.update_yaxes(title_text=y_title, row=1, col=col)

        fig.update_xaxes(title_text="Recommendation Type", row=1, col=4)
        fig.update_yaxes(title_text="Number of Recommendations", row=1, col=4)

    def show_calendar_data(self, data):
        """Display calendar data in a structured format"""
        if data:
            st.write("### Key Dates")
            for key, value in data.items():
                if "Date" in key:
                    if isinstance(value, list):
                        dates = ", ".join([date.strftime("%Y-%m-%d") for date in value])
                        st.write(f"**{key}**: {dates}")
                    else:
                        st.write(f"**{key}**: {value.strftime('%Y-%m-%d')}")

            st.write("\n### Financial Metrics")
            st.metric(label="Earnings High", value=f"${data['Earnings High']:.2f}")
            st.metric(label="Earnings Low", value=f"${data['Earnings Low']:.2f}")
            st.metric(
                label="Earnings Average", value=f"${data['Earnings Average']:.2f}"
            )
            revenue_fmt = lambda x: f"${x:,}"
            st.metric(label="Revenue High", value=revenue_fmt(data["Revenue High"]))
            st.metric(label="Revenue Low", value=revenue_fmt(data["Revenue Low"]))
            st.metric(
                label="Revenue Average", value=revenue_fmt(data["Revenue Average"])
            )
        else:
            st.write("No calendar data available.")

    def plot_with_volume_profile(
        self,
        ticker_symbol,
        start_date,
        end_date,
        combined_metrics,
        option,
    ):
        """Plot a candle chart with volume profile for a given ticker symbol"""
        info(f"Fetching data for {ticker_symbol} from {start_date} to {end_date}")
        data_fetcher = self.data_fetcher
        ticker = data_fetcher.fetch_ticker_data(ticker_symbol)
        data = data_fetcher.fetch_historical_data(ticker_symbol, start_date, end_date)

        # Get dashboard metrics
        info(f"Retrieving dashboard metrics for {ticker_symbol}")
        dashboard_metrics = FinancialMetrics().get_dashboard_metrics(
            ticker_symbol, combined_metrics
        )
        (
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
        ) = dashboard_metrics

        if not data.empty:
            info(f"Calculating market profile for {ticker_symbol}")
            # Calculate market profile
            va_high, va_low, poc_price, _ = (
                self.market_profile_analyzer.calculate_market_profile(data)
            )

            if ticker and hasattr(ticker, "info"):
                price = ticker.info.get("currentPrice")
                info(f"Current price for {ticker_symbol} is {price}")
            else:
                price = None
                warning(f"No ticker info found for {ticker_symbol}")

            # Check if the price is within the value area based on the selected option
            if price is not None and va_high is not None and poc_price is not None:
                if option[0] == "va_high":
                    if price > va_high:
                        info(
                            f"{ticker_symbol} - current price is above value area: {price} (VA High: {va_high}, POC: {poc_price})"
                        )
                        return 0
                elif option[0] == "poc_price":
                    if price > poc_price:
                        info(
                            f"{ticker_symbol} - price is above price of control: {price} (VA High: {va_high}, POC: {poc_price})"
                        )
                        return 0

            # Get company information
            if ticker and hasattr(ticker, "info"):
                website = ticker.info.get("website", "#")
                shortName = ticker.info.get("shortName", ticker_symbol)

                # Create a clean container for the entire ticker dashboard
                with st.container():
                    st.markdown("---")  # Divider for visual separation

                    # Header with company name and link
                    header_with_link = f"[🔗]({website}){shortName} - {ticker_symbol}"
                    st.markdown(f"## {header_with_link}", unsafe_allow_html=True)

                    # Display metrics in a clean dashboard at the top
                    self._display_ticker_metrics_dashboard(
                        ticker_symbol, ticker, eps, pe, ps, pb, peg, gm
                    )

                    # Main content area with two columns
                    col1, col2 = st.columns([1, 1])  # Equal width columns

                    with col1:
                        # Display the chart in the left column (taking half the width)
                        self._display_market_profile_chart(
                            ticker_symbol, data, va_high, va_low, poc_price
                        )

                    with col2:

                        # Display news articles
                        SentimentAnalyzer().display_news_without_sentiment(
                            ticker_symbol
                        )

                # Add some spacing after each ticker section
                st.markdown("<br>", unsafe_allow_html=True)

            else:
                warning(f"No ticker info found for {ticker_symbol}")
                return 0

        else:
            warning(f"No data found for {ticker_symbol} in the given date range.")
            return 0

    def _display_ticker_metrics_dashboard(
        self, ticker_symbol, ticker, eps, pe, ps, pb, peg, gm
    ):
        """Display ticker metrics in a dashboard layout with improved spacing"""
        # Use a container with border styling for the metrics dashboard
        st.markdown(
            """
        <style>
        .metrics-container {
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 20px;
            background-color: rgba(240, 242, 246, 0.1);
        }
        </style>
        """,
            unsafe_allow_html=True,
        )

        with st.container():
            st.markdown('<div class="metrics-container">', unsafe_allow_html=True)

            # Use 4 columns instead of 7 to give more space to each metric
            col1, col2, col3, col4 = st.columns(4)

            # First row of metrics
            with col1:
                if peg:
                    st.metric(label="PEG", value=f"{round(peg,2)}")
                else:
                    st.metric(label="PEG", value="-")

                if eps:
                    st.metric(label="EPS", value=f"{round(eps,2)}")
                else:
                    st.metric(label="EPS", value="-")

            with col2:
                if pe:
                    st.metric(label="P/E", value=f"{round(pe,2)}")
                else:
                    st.metric(label="P/E", value="-")

                if ps:
                    st.metric(label="P/S", value=f"{round(ps,2)}")
                else:
                    st.metric(label="P/S", value="-")

            with col3:
                if pb:
                    st.metric(label="P/B", value=f"{round(pb,2)}")
                else:
                    st.metric(label="P/B", value="-")

                if gm is not None:
                    st.metric(label="Gross Margin", value=f"{round(gm*100,1)}%")
                else:
                    st.metric(label="Gross Margin", value="-")

            with col4:
                if "marketCap" in ticker.info and "financialCurrency" in ticker.info:
                    market_cap = ticker.info["marketCap"]
                    currency = ticker.info["financialCurrency"]
                    if market_cap >= 1e9:
                        market_cap_display = f"{market_cap / 1e9:.2f} B"
                    elif market_cap >= 1e6:
                        market_cap_display = f"{market_cap / 1e6:.2f} M"
                    else:
                        market_cap_display = f"{market_cap:.2f}"
                    st.metric(
                        label=f"Market Cap ({currency})",
                        value=market_cap_display,
                    )
                else:
                    st.metric(label="Market Cap", value="-")

                # Add a recommendation metric if available
                if (
                    "recommendationMean" in ticker.info
                    and ticker.info["recommendationMean"] is not None
                ):
                    rec_value = ticker.info["recommendationMean"]
                    st.metric(label="Analyst Rating", value=f"{round(rec_value,1)}/5")
                else:
                    st.metric(label="Analyst Rating", value="-")

            st.markdown("</div>", unsafe_allow_html=True)

    def _display_market_profile_chart(
        self, ticker_symbol, data, va_high, va_low, poc_price
    ):
        """Display market profile chart with improved layout and styling"""
        # Get price-volume data for volume profile
        price_bins = 100  # Number of price bins for the volume profile
        price_range = data["High"].max() - data["Low"].min()
        bin_size = price_range / price_bins

        # Create price bins
        price_levels = [data["Low"].min() + i * bin_size for i in range(price_bins + 1)]
        buy_volume_by_price = [0] * price_bins
        sell_volume_by_price = [0] * price_bins

        # Distribute volume into price bins, separating buy/sell volume based on price movement
        for i in range(1, len(data)):
            row = data.iloc[i]
            prev_row = data.iloc[i - 1]

            # Determine if day was predominantly buying or selling
            is_up_day = row["Close"] > row["Open"]

            for j in range(price_bins):
                lower_bound = price_levels[j]
                upper_bound = price_levels[j + 1]

                # If price range during the day overlaps with this bin
                if not (row["High"] < lower_bound or row["Low"] > upper_bound):
                    # Calculate volume proportion for this price level
                    volume_contribution = row["Volume"] / (
                        (row["High"] - row["Low"]) / bin_size
                    )

                    # Assign to buy or sell volume based on price action
                    if is_up_day:
                        buy_volume_by_price[j] += volume_contribution
                    else:
                        sell_volume_by_price[j] += volume_contribution

        # Create a figure with improved proportions (wider than tall)
        fig = plt.figure(figsize=(10, 6))  # Reduced height for better UI proportions
        gs = GridSpec(4, 5, figure=fig)

        # Main price chart (takes 4/5 of the width and 3/4 of the height)
        ax1 = fig.add_subplot(gs[0:3, 0:4])

        # Volume chart below price chart (smaller)
        ax_volume = fig.add_subplot(gs[3:4, 0:4], sharex=ax1)

        # Volume profile chart on the right side (narrower)
        ax2 = fig.add_subplot(gs[0:3, 4], sharey=ax1)

        # Plot candlestick chart with a cleaner style
        mpf.plot(
            data,
            type="candle",
            style="yahoo",
            ax=ax1,
            volume=ax_volume,
            show_nontrading=False,
        )

        # Add the POC and VA lines to the main chart with annotations
        poc_line = ax1.axhline(
            y=poc_price, color="red", linestyle="dashed", linewidth=2, label="POC"
        )
        va_high_line = ax1.axhline(
            y=va_high, color="blue", linestyle="dashed", linewidth=1, label="VA High"
        )
        va_low_line = ax1.axhline(
            y=va_low, color="blue", linestyle="dashed", linewidth=1, label="VA Low"
        )

        # Add price annotations in a more readable format
        ax1.annotate(
            f"POC: {poc_price:.2f}",
            xy=(data.index[-1], poc_price),
            xytext=(10, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            color="red",
            fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=0),
        )

        ax1.annotate(
            f"VA High: {va_high:.2f}",
            xy=(data.index[-1], va_high),
            xytext=(10, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            color="blue",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=0),
        )

        ax1.annotate(
            f"VA Low: {va_low:.2f}",
            xy=(data.index[-1], va_low),
            xytext=(10, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            color="blue",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=0),
        )

        # Plot the buy volume profile histogram horizontally on ax2
        buy_bars = ax2.barh(
            price_levels[:-1],
            buy_volume_by_price,
            height=bin_size,
            color="green",
            alpha=0.6,
            label="Buy Volume",
        )

        # Plot the sell volume in a different color
        sell_bars = ax2.barh(
            price_levels[:-1],
            sell_volume_by_price,
            height=bin_size,
            color="red",
            alpha=0.4,
            label="Sell Volume",
        )

        # Highlight POC and Value Area in the volume profile
        poc_bin_idx = max(
            range(len(buy_volume_by_price)),
            key=lambda i: buy_volume_by_price[i] + sell_volume_by_price[i],
        )

        # Highlight the POC bin with high opacity
        ax2.barh(
            price_levels[poc_bin_idx],
            buy_volume_by_price[poc_bin_idx] + sell_volume_by_price[poc_bin_idx],
            height=bin_size,
            color="purple",
            alpha=0.9,
            label="POC",
        )

        # Highlight potential buy zones (prices below POC but with significant buy volume)
        buy_focus_regions = []
        for i in range(poc_bin_idx):
            if buy_volume_by_price[i] > 0.7 * buy_volume_by_price[poc_bin_idx]:
                buy_focus_regions.append(i)

        for idx in buy_focus_regions:
            ax2.barh(
                price_levels[idx],
                buy_volume_by_price[idx],
                height=bin_size,
                color="lime",
                alpha=0.9,
                label="Strong Buy Zone" if idx == buy_focus_regions[0] else "",
            )

        # Remove x-axis labels from the volume profile
        ax2.set_xticks([])

        # Remove y-axis labels from the volume profile (since it shares with main chart)
        ax2.set_yticks([])

        # Add a legend to the volume profile with better positioning
        ax2.legend(loc="upper right", fontsize="small")

        # Add a legend to the main chart
        ax1.legend(["POC", "VA High", "VA Low"], loc="upper left", fontsize="small")

        # Add buy zone annotation to the chart with improved styling
        if buy_focus_regions:
            best_buy_level = price_levels[buy_focus_regions[0]]
            ax1.axhline(
                y=best_buy_level,
                color="lime",
                linestyle="dotted",
                linewidth=2,
                label="Buy Zone",
            )
            ax1.annotate(
                f"Buy Zone: {best_buy_level:.2f}",
                xy=(data.index[-1], best_buy_level),
                xytext=(10, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                color="lime",
                fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=0),
            )

        # Add title to chart axes for clarity
        ax1.set_title("Price History", fontsize=12)
        ax_volume.set_title("Volume", fontsize=10)
        ax2.set_title("Vol Profile", fontsize=10)

        # Tighten layout with better margins
        plt.tight_layout()

        # Display the chart with appropriate sizing
        st.pyplot(fig)

        # Add a small explanation of the chart elements below the chart
        with st.expander("Chart Explanation"):
            st.markdown(
                """
            - **POC (Point of Control)**: Price level with the highest trading volume
            - **VA (Value Area)**: Range between VA High and VA Low where 70% of trading occurred
            - **Buy Zone**: Price level with significant buying volume below POC
            - **Green/Red Bars**: Volume profile showing buy vs sell volume at each price level
            """
            )

        # Business summary in an expander to save space
        if "longBusinessSummary" in yf.Ticker(ticker_symbol).info:
            with st.expander("Business Summary", expanded=False):
                summary_text = yf.Ticker(ticker_symbol).info["longBusinessSummary"]
                formatted_summary = self._format_business_summary(summary_text)
                st.markdown(formatted_summary)

    def _format_business_summary(self, summary):
        """Format business summary for display"""
        summary_no_colons = summary.replace(":", "\:")
        wrapped_summary = textwrap.fill(summary_no_colons)
        return wrapped_summary

    def plot_candle_charts_per_symbol(
        self,
        start_date,
        end_date,
        metrics,
        option,
    ):
        """Plot candle charts for each symbol organized by sector"""
        info("Started plotting candle charts for each symbol")

        critical("Inputs for plotting candle charts:")
        critical(f"Start Date: {start_date}")
        critical(f"End Date: {end_date}")
        critical(f"Combined Metrics: {metrics.get('company_labels')}")
        critical(f"Option: {option}")

        for ticker_symbol in metrics.get("company_labels"):
            info(f"Attempting to plot candle chart for symbol: {ticker_symbol}")

            response = self.plot_with_volume_profile(
                ticker_symbol,
                start_date,
                end_date,
                metrics,
                option,
            )

            if response == 0:
                info(f"Skipped plotting for {ticker_symbol} due to no response")
                continue

        info("Finished plotting candle charts for all symbols")

    def get_dash_metrics(ticker_symbol, combined_metrics):
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
                    info(f"Missing key in combined_metrics: '{key}'")
                    return default_return

            if ticker_symbol in combined_metrics["company_labels"]:
                index = combined_metrics["company_labels"].index(ticker_symbol)

                # Check if index is valid for all lists
                for key in required_keys[1:]:  # Skip company_labels
                    if len(combined_metrics[key]) <= index:
                        info(
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
                info(f"Ticker '{ticker_symbol}' not found in the labels list.")
                return default_return
        except Exception as e:
            info(f"An error occurred in get_dash_metrics: {e}")
            return default_return
