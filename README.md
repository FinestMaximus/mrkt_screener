# Stock Analysis Notebook

## Overview

This app is designed for stock analysis and investment decision-making. It follows a comprehensive approach by taking an initial set of tickers, extracting key financial indicators, applying user-defined thresholds for filtering, and obtaining additional data for the filtered tickers, such as news, charts, technical analysis, etc.

## Contents

1. **Initialization:**
    - Configure thresholds.
    - Tool will apply predefined thresholds to filter tickers based on financial indicators.

4. **Detailed Analysis:**
    - For the filtered tickers retrieve additional data, including news, charts, technical analysis, etc.

## Requirements

- Python 3.10.x
- Streamlit 1.29.0

## Next

- Add a SOCK5 Proxy.

- Total Debt: The lower the company's total debt, the less risk it carries. If the total debt grows over time, it might indicate a potentially risky situation.

- Net Debt: Similar to total debt, a lower net debt is more preferable. It represents the company’s total debt minus its cash and cash equivalents.

- Total Assets: This is the sum of current and non-current assets and can give an idea of the company's size and health.

- Total Equity: This figure represents the net value of the company (assets minus liabilities). An increasing total equity might indicate a financially healthy company.

- Retained Earnings: Positive and growing retained earnings generally indicate that the company is profitable and reinvests its profits back into the business.

- Current Assets and Current Liabilities: These figures give you an idea of the company's liquidity and ability to cover its short-term obligations.

- Book Value: It’s the total tangible assets minus total liabilities. A company trading at a low multiple of its book value could be undervalued.

## Example Ouput

![image](https://github.com/FinestMaximus/mrkt_screener/assets/21218173/a36233b1-45b7-4c3d-8919-b0e23e0c0410)
