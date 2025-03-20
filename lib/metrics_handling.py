import yfinance as yf
import streamlit as st
import logging


def populate_metrics(ticker, metrics):
    if hasattr(ticker, "info"):
        try:
            stock_info = ticker.info
            logging.debug(
                "[metrics_handling.py][populate_metrics] Populating metrics for ticker: {}".format(
                    ticker.ticker
                )
            )
            metrics["eps_values"].append(stock_info.get("trailingEps", 0))
            metrics["pe_values"].append(stock_info.get("trailingPE", 0))
            metrics["peg_values"].append(stock_info.get("pegRatio", 0))
            metrics["gross_margins"].append(stock_info.get("grossMargins", 0))
            metrics["sector"].append(stock_info.get("sector", ""))
            metrics["short_name"].append(stock_info.get("shortName", ""))
            metrics["fullTimeEmployees"].append(stock_info.get("fullTimeEmployees", ""))
            metrics["boardRisk"].append(stock_info.get("boardRisk", ""))
            metrics["industry"].append(stock_info.get("industry", ""))
            metrics["compensationRisk"].append(stock_info.get("compensationRisk", ""))
            metrics["shareHolderRightsRisk"].append(
                stock_info.get("shareHolderRightsRisk", "")
            )
            metrics["overallRisk"].append(stock_info.get("overallRisk", ""))
            metrics["exDividendDate"].append(stock_info.get("exDividendDate", ""))
            metrics["dividendYield"].append(stock_info.get("dividendYield", ""))
            metrics["dividendRate"].append(stock_info.get("dividendRate", ""))
            metrics["priceHint"].append(stock_info.get("priceHint", ""))
            metrics["fiftyTwoWeekLow"].append(stock_info.get("fiftyTwoWeekLow", ""))
            metrics["forwardPE"].append(stock_info.get("forwardPE", 0))
            metrics["marketCap"].append(stock_info.get("marketCap", 0))
            metrics["beta"].append(stock_info.get("beta", 0))
            metrics["fiveYearAvgDividendYield"].append(
                stock_info.get("fiveYearAvgDividendYield", 0)
            )
            metrics["payoutRatio"].append(stock_info.get("payoutRatio", 0))
            metrics["ebitdaMargins"].append(stock_info.get("ebitdaMargins", 0))
            metrics["website"].append(stock_info.get("website", ""))
            metrics["operatingMargins"].append(stock_info.get("operatingMargins", 0))
            metrics["financialCurrency"].append(stock_info.get("financialCurrency", ""))
            metrics["trailingPegRatio"].append(stock_info.get("trailingPegRatio", 0))
            metrics["fiftyTwoWeekHigh"].append(stock_info.get("fiftyTwoWeekHigh", 0))
            metrics["priceToSalesTrailing12Months"].append(
                stock_info.get("priceToSalesTrailing12Months", 0)
            )
            metrics["fiftyDayAverage"].append(stock_info.get("fiftyDayAverage", 0))
            metrics["twoHundredDayAverage"].append(
                stock_info.get("twoHundredDayAverage", 0)
            )
            metrics["trailingAnnualDividendRate"].append(
                stock_info.get("trailingAnnualDividendRate", 0)
            )
            metrics["trailingAnnualDividendYield"].append(
                stock_info.get("trailingAnnualDividendYield", 0)
            )
            metrics["currency"].append(stock_info.get("currency", ""))
            metrics["enterpriseValue"].append(stock_info.get("enterpriseValue", 0))
            metrics["profitMargins"].append(stock_info.get("profitMargins", 0))
            metrics["floatShares"].append(stock_info.get("floatShares", 0))
            metrics["sharesOutstanding"].append(stock_info.get("sharesOutstanding", 0))
            metrics["sharesShort"].append(stock_info.get("sharesShort", 0))
            metrics["sharesShortPriorMonth"].append(
                stock_info.get("sharesShortPriorMonth", 0)
            )
            metrics["sharesShortPreviousMonthDate"].append(
                stock_info.get("sharesShortPreviousMonthDate", 0)
            )
            metrics["dateShortInterest"].append(stock_info.get("dateShortInterest", 0))
            metrics["sharesPercentSharesOut"].append(
                stock_info.get("sharesPercentSharesOut", 0)
            )
            metrics["heldPercentInsiders"].append(
                stock_info.get("heldPercentInsiders", 0)
            )
            metrics["heldPercentInstitutions"].append(
                stock_info.get("heldPercentInstitutions", 0)
            )
            metrics["shortRatio"].append(stock_info.get("shortRatio", 0))
            metrics["shortPercentOfFloat"].append(
                stock_info.get("shortPercentOfFloat", 0)
            )
            metrics["bookValue"].append(stock_info.get("bookValue", 0))
            metrics["priceToBook"].append(stock_info.get("priceToBook", 0))
            metrics["lastFiscalYearEnd"].append(stock_info.get("lastFiscalYearEnd", 0))
            metrics["nextFiscalYearEnd"].append(stock_info.get("nextFiscalYearEnd", 0))
            metrics["mostRecentQuarter"].append(stock_info.get("mostRecentQuarter", 0))
            metrics["earningsQuarterlyGrowth"].append(
                stock_info.get("earningsQuarterlyGrowth", 0)
            )
            metrics["netIncomeToCommon"].append(stock_info.get("netIncomeToCommon", 0))
            metrics["forwardEps"].append(stock_info.get("forwardEps", 0))
            metrics["lastSplitFactor"].append(stock_info.get("lastSplitFactor", ""))
            metrics["lastSplitDate"].append(stock_info.get("lastSplitDate", 0))
            metrics["enterpriseToRevenue"].append(
                stock_info.get("enterpriseToRevenue", 0)
            )
            metrics["enterpriseToEbitda"].append(
                stock_info.get("enterpriseToEbitda", 0)
            )
            metrics["exchange"].append(stock_info.get("exchange", ""))
            metrics["quoteType"].append(stock_info.get("quoteType", ""))
            metrics["symbol"].append(stock_info.get("symbol", ""))
            metrics["underlyingSymbol"].append(stock_info.get("underlyingSymbol", ""))
            metrics["shortName"].append(stock_info.get("shortName", ""))
            metrics["longName"].append(stock_info.get("longName", ""))
            metrics["firstTradeDateEpochUtc"].append(
                stock_info.get("firstTradeDateEpochUtc", 0)
            )
            metrics["timeZoneFullName"].append(stock_info.get("timeZoneFullName", ""))
            metrics["timeZoneShortName"].append(stock_info.get("timeZoneShortName", ""))
            metrics["uuid"].append(stock_info.get("uuid", ""))
            metrics["gmtOffSetMilliseconds"].append(
                stock_info.get("gmtOffSetMilliseconds", 0)
            )
            metrics["currentPrice"].append(stock_info.get("currentPrice", 0))
            metrics["targetHighPrice"].append(stock_info.get("targetHighPrice", 0))
            metrics["targetLowPrice"].append(stock_info.get("targetLowPrice", 0))
            metrics["targetMeanPrice"].append(stock_info.get("targetMeanPrice", 0))
            metrics["targetMedianPrice"].append(stock_info.get("targetMedianPrice", 0))
            metrics["recommendationMean"].append(
                stock_info.get("recommendationMean", 0)
            )
            metrics["recommendationKey"].append(stock_info.get("recommendationKey", ""))
            metrics["numberOfAnalystOpinions"].append(
                stock_info.get("numberOfAnalystOpinions", 0)
            )
            metrics["totalCash"].append(stock_info.get("totalCash", 0))
            metrics["totalCashPerShare"].append(stock_info.get("totalCashPerShare", 0))
            metrics["ebitda"].append(stock_info.get("ebitda", 0))
            metrics["totalDebt"].append(stock_info.get("totalDebt", 0))
            metrics["quickRatio"].append(stock_info.get("quickRatio", 0))
            metrics["currentRatio"].append(stock_info.get("currentRatio", 0))
            metrics["totalRevenue"].append(stock_info.get("totalRevenue", 0))
            metrics["debtToEquity"].append(stock_info.get("debtToEquity", 0))
            metrics["revenuePerShare"].append(stock_info.get("revenuePerShare", 0))
            metrics["returnOnAssets"].append(stock_info.get("returnOnAssets", 0))
            metrics["returnOnEquity"].append(stock_info.get("returnOnEquity", 0))
            metrics["freeCashflow"].append(stock_info.get("freeCashflow", 0))
            metrics["operatingCashflow"].append(stock_info.get("operatingCashflow", 0))
            metrics["earningsGrowth"].append(stock_info.get("earningsGrowth", 0))
            metrics["revenueGrowth"].append(stock_info.get("revenueGrowth", 0))
            metrics["company_labels"].append(ticker.ticker)
            logging.debug(
                "[metrics_handling.py][populate_metrics] Successfully populated metrics for ticker: {}".format(
                    ticker.ticker
                )
            )
        except Exception as e:
            logging.error(
                f"[metrics_handling.py][populate_metrics] Failed to process ticker {ticker.ticker}: {e}"
            )
            st.error(f"Failed to process ticker {ticker.ticker}: {e}")
    else:
        logging.warning(
            "[metrics_handling.py][populate_metrics] Skipped a company ticker due to missing info or an invalid object."
        )
        st.write("Skipped a company ticker due to missing info or an invalid object.")


def populate_additional_metrics(ticker_symbol, metrics):
    try:
        ticker = yf.Ticker(ticker_symbol)
        if not hasattr(ticker, "info") or not hasattr(ticker, "cashflow"):
            raise AttributeError(
                "The ticker object must have 'info' and 'cashflow' attributes"
            )
        from lib.data_fetching import fetch_recommendations_summary

        recommendations_summary = fetch_recommendations_summary(ticker_symbol)
        metrics["recommendations_summary"].append(recommendations_summary)
    except Exception as e:
        logging.error(
            f"[metrics_handling.py][populate_additional_metrics] Failed to fetch recommendations summary: {str(e)}"
        )
        metrics["recommendations_summary"].append(None)
    fields_to_add = {
        "freeCashflow": None,
        "opCashflow": None,
        "repurchaseCapStock": None,
    }
    get_cash_flows(ticker_symbol, fields_to_add, metrics)


def get_cash_flows(ticker_symbol, fields_to_add, metrics):
    try:
        ticker = yf.Ticker(ticker_symbol)
        df = ticker.cashflow
    except Exception as e:
        logging.error(
            f"[metrics_handling.py][get_cash_flows] Failed to fetch cashflow data: {str(e)}"
        )
        df = None
    for key, value in fields_to_add.items():
        if key not in metrics:
            metrics[key] = []
        if df is not None and key in [
            "freeCashflow",
            "opCashflow",
            "repurchaseCapStock",
        ]:
            try:
                if key == "freeCashflow":
                    free_cash_flow = df.iloc[0, :].tolist()
                    metrics[key].append(free_cash_flow)
                elif key == "opCashflow":
                    op_cash_flow = df.iloc[33, :].tolist()
                    metrics[key].append(op_cash_flow)
                elif key == "repurchaseCapStock":
                    repurchase_capital_stock = df.iloc[1, :].tolist()
                    metrics[key].append(repurchase_capital_stock)
            except Exception as e:
                logging.error(
                    f"[metrics_handling.py][get_cash_flows] {ticker.ticker} Failed to process {key}: {str(e)}"
                )
                metrics[key].append(None)


def build_combined_metrics(filtered_company_symbols, metrics, metrics_filtered):
    if not isinstance(filtered_company_symbols, list):
        raise ValueError("filtered_company_symbols must be a list")
    if not isinstance(metrics, dict):
        raise ValueError("metrics must be a dictionary")
    if not isinstance(metrics_filtered, dict):
        raise ValueError("metrics_filtered must be a dictionary")
    metrics.pop("companies_fetched", None)
    metrics_filtered.pop("companies_fetched", None)
    combined_keys = set(metrics.keys()).union(metrics_filtered.keys()) - {
        "company_labels",
        "companies_fetched",
    }
    combined_metrics = {key: [] for key in combined_keys}
    combined_metrics["company_labels"] = filtered_company_symbols
    for symbol in filtered_company_symbols:
        if "company_labels" in metrics and not isinstance(
            metrics["company_labels"], list
        ):
            raise ValueError("'company_labels' in metrics must be a list")
        metrics_index = (
            metrics["company_labels"].index(symbol)
            if "company_labels" in metrics and symbol in metrics["company_labels"]
            else -1
        )
        for key in combined_metrics:
            if key == "company_labels":
                continue
            if key in metrics and metrics_index >= 0:
                if isinstance(metrics[key][metrics_index], list):
                    value = metrics[key][metrics_index]
                else:
                    value = (
                        metrics[key][metrics_index]
                        if len(metrics[key]) > metrics_index
                        else None
                    )
            elif key in metrics_filtered:
                filtered_index = filtered_company_symbols.index(symbol)
                value = (
                    metrics_filtered[key][filtered_index]
                    if len(metrics_filtered[key]) > filtered_index
                    else None
                )
            else:
                value = None
            # Append or extend based on the type of value
            if (
                isinstance(value, list)
                and not isinstance(value, str)
                and key == "freeCashflow"
            ):
                # Assuming we want to extend to flatten the list of lists where key is 'freeCashflow'
                (
                    combined_metrics[key].extend(value)
                    if isinstance(combined_metrics[key], list)
                    else combined_metrics[key].append([value])
                )
            else:
                combined_metrics[key].append(value)

    expected_length = len(filtered_company_symbols)
    for key, values_list in combined_metrics.items():
        if len(values_list) != expected_length:
            raise ValueError(f"Length mismatch in combined metrics for key: {key}")
    return combined_metrics


def get_dash_metrics(ticker_symbol, combined_metrics):
    try:
        if ticker_symbol in combined_metrics["company_labels"]:
            index = combined_metrics["company_labels"].index(ticker_symbol)
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
                f"[metrics_handling.py][get_dash_metrics] Ticker '{ticker_symbol}' not found in the labels list."
            )
            return (
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
    except Exception as e:
        logging.error(
            f"[metrics_handling.py][get_dash_metrics] An error occurred: {str(e)}"
        )
        return (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
