# Stock Analysis Notebook

## Overview

This Jupyter notebook is designed for stock analysis and investment decision-making. It follows a comprehensive approach by taking an initial set of tickers, extracting key financial indicators, applying user-defined thresholds for filtering, and obtaining additional data for the filtered tickers, such as news, charts, technical analysis, etc.

## Contents

1. **Initialization:**
    - Import necessary libraries.
    - Define the initial set of tickers.

2. **Financial Indicators:**
    - Extract key financial indicators for each ticker.
        - EPS (Earnings Per Share)
        - P/E (Price-to-Earnings ratio)
        - DFC (Dividend Cash Flow)
        - WACC (Weighted Average Cost of Capital)
        - Fair Value
        - P/B (Price-to-Book ratio)
        - P/S (Price-to-Sales ratio)
        - Debt to Equity ratio
        - PEG (Price/Earnings-to-Growth ratio)

3. **Filtering:**
    - Apply predefined thresholds to filter tickers based on financial indicators.

4. **Detailed Analysis:**
    - For the filtered tickers retrieve additional data, including news, charts, technical analysis, etc.

## Usage

2. **Ticker Input:**
    - Modify the list of tickers in the initialization section to match your analysis requirements.

## Requirements

- Python 3.10.x
- Jupyter Notebook / Colab

## Next

- Total Debt: The lower the company's total debt, the less risk it carries. If the total debt grows over time, it might indicate a potentially risky situation.

- Net Debt: Similar to total debt, a lower net debt is more preferable. It represents the company’s total debt minus its cash and cash equivalents.

- Total Assets: This is the sum of current and non-current assets and can give an idea of the company's size and health.

- Total Equity: This figure represents the net value of the company (assets minus liabilities). An increasing total equity might indicate a financially healthy company.

- Retained Earnings: Positive and growing retained earnings generally indicate that the company is profitable and reinvests its profits back into the business.

- Current Assets and Current Liabilities: These figures give you an idea of the company's liquidity and ability to cover its short-term obligations.

- Book Value: It’s the total tangible assets minus total liabilities. A company trading at a low multiple of its book value could be undervalued.