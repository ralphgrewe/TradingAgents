---
name: fundamental-analyst
description: "Call this skill when the user asks for a structured fundamental analysis of a stock, company or symbol"
---

Delivering a structured fundamental analysis follows steps and the output is given in a structured format. Apply the following rules to the analysis:
- Collect all metrics first, evaluate second - never skip Phase 1.
- Value and Growth perspectives use the same dataset - no duplicate data fetching.
- Use only fundamental data sources. No technical charts, no sentiment, no social media.

## Step 1 - Data Collection:

Create a table of historic fundamentals for {ticker}, yearly from five years ago till today and quarterly for the last four quarters. The resulting table shall have the years in columns and the following data in rows: 
VALUATION: PE Ratio, PB Ratio, EV/EBITDA, PCF Ratio, PEG Ratio, Price/Sales
PROFITABILITY: Gross Margin, Operating Margin, Net Margin, ROE, ROIC, ROA
BALANCE SHEET: Debt-to-Equity, Current Ratio, Quick Ratio, Equity Ratio, Interest Coverage
CASHFLOW: Free Cash Flow, Operating Cash Flow, FCF Yield, FCF Margin, Capex/Revenue
DIVIDENDS: Dividend Yield, Payout Ratio
CONTEXT: Market Cap, Sector, Industry, 52W High/Low, Analyst Consensus

Create a separate insider transactions table covering the last 12 months. Columns: Date, Insider, Role, Type (Buy/Sell), Shares, Value (USD). Below the table add a one-line net-sentiment summary: net shares bought/sold by insiders and whether the balance is net buying (bullish signal) or net selling (bearish signal).

Create a prediction for the next two years. Use data sources but also create an own prediction, based on the historic data and the expectation for the sector and industry given in the context. Ensure that the data in the prediction is consistent considering the rules of accounting. Create the prediction for the following data:
Forward PE, Forward Free Cash Flow, Forward PB Ratio, Forward ROE, Forward Debt-to-Equity, Dividend Yield

## Step 2 - Value Evaluation:
Evaluate the collected metrics strictly according to Value Investing principles (Benjamin Graham, Warren Buffett). Assess: Is the company trading below intrinsic value? Is the balance sheet solid? Is FCF sustainable? Does it have a durable competitive moat?

## Step 3 - Growth Evaluation:
Evaluate the same metrics strictly according to Growth Investing principles (Philip Fisher, Peter Lynch). Assess: Is revenue and earnings growth strong and accelerating? Is the business scalable? Are margins expanding? Is forward guidance positive?

## Step 4 - JSON Output:
Output a single JSON block named `fundamentals_report` containing the following fields:
insider_sentiment: BULLISH, NEUTRAL, BEARISH # Net insider buying/selling signal from Step 1
value_fundamentals_confidence: HIGH, MEDIUM, LOW # Confidence in the data quality for the value evaluation
value_signal: BUY, HOLD, SELL # Verdict applying value investing principles
value_confidence: HIGH, MEDIUM, LOW # Reliability of the value verdict
growth_fundamentals_confidence: HIGH, MEDIUM, LOW # Confidence in the data quality for the growth evaluation
growth_signal: BUY, HOLD, SELL # Verdict applying growth investing principles
growth_confidence: HIGH, MEDIUM, LOW # Reliability of the growth verdict