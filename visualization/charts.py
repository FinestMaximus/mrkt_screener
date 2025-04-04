import plotly.graph_objects as go
from plotly.subplots import make_subplots
import mplfinance as mpf
import pandas as pd
import streamlit as st
import logging
from datetime import datetime
import textwrap


class ChartGenerator:
    """Class for generating various financial charts and visualizations"""

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
            logging.warning("Cannot create chart with empty data")
            return None

        poc_line = pd.Series(poc_price, index=data.index)
        va_high_line = pd.Series(va_high, index=data.index)
        va_low_line = pd.Series(va_low, index=data.index)

        apds = [
            mpf.make_addplot(
                poc_line, type="line", color="red", linestyle="dashed", width=3
            ),
            mpf.make_addplot(
                va_high_line, type="line", color="blue", linestyle="dashed", width=0.7
            ),
            mpf.make_addplot(
                va_low_line, type="line", color="blue", linestyle="dashed", width=0.7
            ),
        ]

        fig, ax = mpf.plot(
            data,
            type="candle",
            addplot=apds,
            style="yahoo",
            volume=True,
            show_nontrading=False,
            returnfig=True,
        )

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
                logging.error(f"Error plotting data for {company}: {error}")
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

    def display_ticker_dashboard(
        self,
        ticker_symbol,
        ticker_info,
        data,
        va_high,
        va_low,
        poc_price,
        eps,
        pe,
        ps,
        pb,
        peg,
        gm,
        market_cap,
        currency,
        news_data,
        total_polarity,
    ):
        """Display a comprehensive dashboard for a ticker"""
        if not ticker_info:
            st.error(f"No data found for {ticker_symbol}")
            return

        website = ticker_info.get("website", "#")
        shortName = ticker_info.get("shortName", ticker_symbol)

        header_with_link = f"[🔗]({website}){shortName} - {ticker_symbol}"

        st.markdown(f"### {header_with_link}", unsafe_allow_html=True)

        col1, col2, col3, col4, col5, col6, col7 = st.columns(7)

        with col1:
            if peg:
                st.metric(label="PEG", value=f"{round(peg,2)}")
            else:
                st.metric(label="PEG", value="-")

        with col2:
            if eps:
                st.metric(label="EPS", value=f"{round(eps,2)}")
            else:
                st.metric(label="EPS", value="-")

        with col3:
            if pe:
                st.metric(label="P/E", value=f"{round(pe,2)}")
            else:
                st.metric(label="P/E", value="-")

        with col4:
            if ps:
                st.metric(label="P/S", value=f"{round(ps,2)}")
            else:
                st.metric(label="P/S", value="-")

        with col5:
            if pb:
                st.metric(label="P/B", value=f"{round(pb,2)}")
            else:
                st.metric(label="P/B", value="-")

        with col6:
            if market_cap:
                if market_cap >= 1e9:
                    market_cap_display = f"{market_cap / 1e9:.2f} B"
                elif market_cap >= 1e6:
                    market_cap_display = f"{market_cap / 1e6:.2f} M"
                else:
                    market_cap_display = f"{market_cap:.2f}"
                st.metric(label=f"Market Cap ({currency})", value=market_cap_display)
            else:
                st.metric(label="Market Cap", value="-")

        with col7:
            if gm is not None:
                st.metric(label="Gross Margin", value=f"{round(gm*100,1)}%")
            else:
                st.metric(label="Gross Margin", value="-")

        if "longBusinessSummary" in ticker_info:
            summary_text = ticker_info["longBusinessSummary"]
            formatted_summary = self._format_business_summary(summary_text)

            with st.container():
                st.markdown(formatted_summary)

        col1_weight, col2_weight, col3_weight = 1, 2, 1
        col1, col2, col3 = st.columns([col1_weight, col2_weight, col3_weight])

        with col1:
            self._display_sentiment_gauge(news_data, total_polarity)

        with col2:
            self._display_news_articles(news_data)

        with col3:
            if not data.empty and va_high and va_low and poc_price:
                fig = self.create_candle_chart_with_profile(
                    data, poc_price, va_high, va_low
                )
                st.pyplot(fig)

    def _format_business_summary(self, summary):
        """Format business summary for display"""
        summary_no_colons = summary.replace(":", "\:")
        wrapped_summary = textwrap.fill(summary_no_colons)
        return wrapped_summary

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
            logging.info("No news data available for sentiment analysis.")
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

    # ... other chart generation methods ...
