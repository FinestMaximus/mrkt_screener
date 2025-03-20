import plotly.graph_objects as go
import streamlit as st
import logging
import pandas as pd
import yfinance as yf
import mplfinance as mpf

from lib.data_fetching import fetch_historical_data
from lib.metrics_handling import get_dash_metrics
from lib.market_analysis import calculate_market_profile
from lib.news_analysis import get_news_data, display_news_articles
from lib.utils import format_business_summary

from datetime import datetime


def plot_sector_distribution_interactive(industries, title):
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
    logging.debug(
        "[plotting.py][plot_sector_distribution_interactive] Successfully created sector distribution plot."
    )
    return fig


def plot_combined_interactive(combined_metrics):
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

        current_recommendations = recommendations_summary[i]
        if (
            isinstance(current_recommendations, dict)
            and "0m" in current_recommendations
        ):
            ratings = current_recommendations["0m"]
            rating_categories = [
                "strongBuy",
                "buy",
                "hold",
                "sell",
                "strongSell",
            ]
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

            fig.update_yaxes(range=[0, max(peg_values)], row=2, col=2)
            for row in range(1, 3):
                for col in range(1, 3):
                    fig.update_yaxes(range=[0, "auto"], row=row, col=col)

        now = datetime.now()
        year = now.year
        if now.month < 4:
            year -= 1
        years = [str(year - i) for i in range(3, -1, -1)]

        if isinstance(freeCashflow[i], list):
            fig.add_trace(
                go.Scatter(
                    x=years[: len(freeCashflow[i])],
                    y=[cf for cf in freeCashflow[i]],
                    name=company_labels[i],
                    hoverinfo="none",
                    legendgroup=legendgroup,
                    showlegend=False,
                    hovertemplate=f"Company: {company}<br>Year: %{{x}}<br> Free Cashflow: %{{y}}<extra></extra>",
                ),
                row=4,
                col=3,
            )

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

        if isinstance(opCashflow[i], list):
            fig.add_trace(
                go.Scatter(
                    x=years[: len(opCashflow[i])],
                    y=[cf for cf in opCashflow[i]],
                    mode="lines",
                    name=company_labels[i],
                    hoverinfo="none",
                    legendgroup=legendgroup,
                    showlegend=False,
                    hovertemplate=f"Company: {company}<br>Year: %{{x}}<br> Operational Cashflow: %{{y}}<extra></extra>",
                ),
                row=4,
                col=1,
            )

        if isinstance(repurchaseCapStock[i], list):
            fig.add_trace(
                go.Scatter(
                    x=years[: len(repurchaseCapStock[i])],
                    y=[-cf for cf in repurchaseCapStock[i]],
                    mode="lines",
                    name=company_labels[i],
                    hoverinfo="none",
                    legendgroup=legendgroup,
                    showlegend=False,
                    hovertemplate=f"Company: {company}<br>Year: %{{x}}<br> Repurchase of Capital Stock: %{{y}}<extra></extra>",
                ),
                row=4,
                col=2,
            )

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
    logging.debug(
        "[plotting.py][plot_combined_interactive] Successfully created combined interactive plot."
    )
    st.plotly_chart(fig)


def plot_with_volume_profile(
    ticker_symbol,
    start_date,
    end_date,
    combined_metrics,
    final_shortlist_labels,
    option,
):
    ticker = yf.Ticker(ticker_symbol)
    data = fetch_historical_data(ticker_symbol, start_date, end_date)
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
    ) = get_dash_metrics(ticker_symbol, combined_metrics)

    if data.empty:
        logging.warning(
            f"[visualization.py][plot_with_volume_profile] No data found for {ticker_symbol} in the given date range."
        )
        return

    va_high, va_low, poc_price, _ = calculate_market_profile(data)
    price = ticker.info["currentPrice"]

    if option[0] == "va_high" and price > va_high:
        logging.info(
            f"[visualization.py][plot_with_volume_profile] {ticker_symbol} - current price is above value area: {price} {va_high} {poc_price}"
        )
        return
    elif option[0] == "poc_price" and price > poc_price:
        logging.info(
            f"[visualization.py][plot_with_volume_profile] {ticker_symbol} - price is above price of control: {price} {va_high} {poc_price}"
        )
        return

    website = ticker.info["website"]
    shortName = ticker.info["shortName"]
    header_with_link = f"[🔗]({website}){shortName} - {ticker_symbol}"
    st.markdown(f"### {header_with_link}", unsafe_allow_html=True)
    final_shortlist_labels.append(ticker_symbol)

    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    st.metric(label="PEG", value=f"{round(peg,2)}" if peg else "-")
    st.metric(label="EPS", value=f"{round(eps,2)}" if eps else "-")
    st.metric(label="P/E", value=f"{round(pe,2)}" if pe else "-")
    st.metric(label="P/S", value=f"{round(ps,2)}" if ps else "-")
    st.metric(label="P/B", value=f"{round(pb,2)}" if pb else "-")

    if "marketCap" in ticker.info:
        market_cap = ticker.info["marketCap"]
        market_cap_display = (
            f"{market_cap / 1e9:.2f} B"
            if market_cap >= 1e9
            else (
                f"{market_cap / 1e6:.2f} M"
                if market_cap >= 1e6
                else f"{market_cap:.2f}"
            )
        )
        st.metric(
            label=f"Market Cap ({ticker.info['financialCurrency']})",
            value=market_cap_display,
        )
    else:
        st.metric(label="Market Cap", value="-")

    st.metric(label="Gross Margin", value=f"{round(gm*100,1)}%" if pb else "-")

    summary_text = ticker.info["longBusinessSummary"]
    formatted_summary = format_business_summary(summary_text)
    st.markdown(formatted_summary)

    news_data, total_polarity = get_news_data(ticker_symbol)
    col1_weight, col2_weight, col3_weight = 1, 2, 1
    col1, col2, col3 = st.columns([col1_weight, col2_weight, col3_weight])

    if news_data:
        average_sentiment = total_polarity / len(news_data)
        color = (
            "green"
            if average_sentiment >= 0.5
            else "orange" if average_sentiment >= 0 else "red"
        )
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
        logging.info(
            f"[visualization.py][plot_with_volume_profile] No news data available for {ticker_symbol}."
        )
        st.write("No sentiment or news data available.")

    display_news_articles(news_data)

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
    st.pyplot(fig)


def plot_candle_charts_per_symbol(
    industries, start_date, end_date, combined_metrics, final_shortlist_labels, option
):
    logging.info(
        "[visualization.py][plot_candle_charts_per_symbol] Started plotting candle charts for each symbol"
    )
    for sector, symbol_list in industries.items():
        logging.info(
            f"[visualization.py][plot_candle_charts_per_symbol] Processing sector: {sector} with symbols: {len(symbol_list)}"
        )
        container = st.container()
        all_skipped = True
        for ticker_symbol in symbol_list:
            logging.debug(
                f"[visualization.py][plot_candle_charts_per_symbol] Attempting to plot candle chart for symbol: {ticker_symbol}"
            )
            response = plot_with_volume_profile(
                ticker_symbol,
                start_date,
                end_date,
                combined_metrics,
                final_shortlist_labels,
                option,
            )
            if response == 0:
                logging.warning(
                    f"[visualization.py][plot_candle_charts_per_symbol] Skipped plotting for {ticker_symbol} due to no response"
                )
                continue
            all_skipped = False
        if not all_skipped:
            with container.expander(f"Sector: {sector}", expanded=False):
                st.write("Charts for the sector.")
    logging.info(
        "[visualization.py][plot_candle_charts_per_symbol] Finished plotting candle charts for all symbols"
    )


def show_calendar_data(data):
    if data:
        st.write("### Key Dates")
        for key, value in data.items():
            if "Date" in key:
                dates = (
                    ", ".join([date.strftime("%Y-%m-%d") for date in value])
                    if isinstance(value, list)
                    else value.strftime("%Y-%m-%d")
                )
                st.write(f"**{key}**: {dates}")
        st.write("\n### Financial Metrics")
        st.metric(label="Earnings High", value=f"${data['Earnings High']:.2f}")
        st.metric(label="Earnings Low", value=f"${data['Earnings Low']:.2f}")
        st.metric(label="Earnings Average", value=f"${data['Earnings Average']:.2f}")
        revenue_fmt = lambda x: f"${x:,}"
        st.metric(label="Revenue High", value=revenue_fmt(data["Revenue High"]))
        st.metric(label="Revenue Low", value=revenue_fmt(data["Revenue Low"]))
        st.metric(label="Revenue Average", value=revenue_fmt(data["Revenue Average"]))
    else:
        st.write("No calendar data available.")
