---
id: mohanram-g-score-low-bm
title: Mohanram G_SCORE for Low Book-to-Market (Glamour) Stocks
tags:
- fundamental-analysis
- financial-statement-analysis
- growth-stocks
- low-book-to-market
- glamour-stocks
- mispricing
signals:
- G_SCORE
- profitability
- cash-flow-quality
- earnings-stability
- growth-stability
- accounting-conservatism
asset_classes:
- equity
horizon:
- swing
- position
source:
  authors: Partha S. Mohanram
  title: Separating Winners from Losers among Low Book-to-Market Stocks using Financial
    Statement Analysis
  year: 2004
  file: paper/Mohanram_SeparatingWinnersFromLosersAmongLowBookToMarketStocks.pdf
---
## Summary
Mohanram’s G_SCORE is a nine-point binary index that combines three categories of signals—traditional profitability and cash-flow fundamentals, measures of naïve earnings/growth extrapolation risk, and indicators of conservative accounting—to separate future winners from losers among low book-to-market (“glamour”) stocks. A long–short strategy that buys high G_SCORE firms and shorts low G_SCORE firms earns economically large and statistically significant abnormal returns that persist for at least two years, are robust across size, analyst coverage, exchange listing, and IPO inclusion, and survive controls for momentum, book-to-market, accruals, and equity issuance. The evidence supports a mispricing explanation: markets underreact to the implications of current growth fundamentals for future performance, producing predictable earnings surprises and return drift.

## Signal — what it is
G_SCORE is an additive index of eight binary signals (range 0–8) designed specifically for low book-to-market (growth) stocks. Each signal equals 1 if a firm passes a conservative criterion relative to its industry peers; otherwise it equals 0. The eight signals are grouped into three categories:
- Profitability and cash-flow quality (G1–G3)
- Earnings and growth stability (G4–G5)
- Accounting conservatism via R&D, capex, and advertising intensity (G6–G8)

A ninth value (G_SCORE = 8) is possible but rare in practice.

## How to compute
1. Identify the low book-to-market sample (bottom BM quintile, including negative BM) and match each firm to its 2-digit SIC industry peers.
2. Compute the eight signals using trailing annual financials (scaled by beginning-of-period assets where applicable):
   - G1 = 1 if ROA (net income before extraordinary items) > contemporaneous industry median ROA; else 0
   - G2 = 1 if cash-flow ROA > contemporaneous industry median cash-flow ROA; else 0
   - G3 = 1 if cash-flow ROA > ROA (i.e., accruals are negative); else 0
   - G4 = 1 if the variance of ROA over the prior 3–5 years < contemporaneous industry median variance; else 0
   - G5 = 1 if the variance of sales growth over the prior 3–5 years < contemporaneous industry median variance; else 0
   - G6 = 1 if R&D intensity (R&D/assets) > contemporaneous industry median; else 0
   - G7 = 1 if capital-expenditure intensity (capex/assets) > contemporaneous industry median; else 0
   - G8 = 1 if advertising intensity (advertising/assets) > contemporaneous industry median; else 0
3. Sum the eight binary indicators to obtain G_SCORE ∈ {0,1,…,8}.
4. Form portfolios by grouping firms into low (G_SCORE 0–1), medium (2–5), and high (6–8) buckets; the canonical strategy is long high and short low.

## Empirical evidence
In a 1979–1999 U.S. sample of 20,866 low-BM firm-years, the long–short G_SCORE strategy produced:
- One-year size-adjusted returns: +3.3% (high) vs. −17.9% (low), a 21.2% spread (t = 11.07)
- Two-year size-adjusted returns: +2.4% (high) vs. −13.3% (low), a 15.8% spread (t = 7.94)
- Positive returns in every sample year (21/21), with 16/21 years showing statistically significant spreads
- Effects robust to size, analyst following, exchange listing, IPO inclusion, and controls for momentum, BM, accruals, and equity issuance; the coefficient on G_SCORE in pooled regressions is 0.039 (t = 10.07), implying a ≈3.9% annual return per point.

## When to apply / regime
Apply to low book-to-market (“glamour”) equities where traditional value screens would otherwise be uninformative. The strategy is most effective when:
- Growth firms are heterogeneous in fundamentals (e.g., high vs. low profitability, stable vs. volatile growth, conservative vs. aggressive accounting)
- Markets are prone to naïve extrapolation of recent performance or underappreciate the future benefits of conservative accounting
- Liquidity and short-selling constraints are manageable (the strategy works best for larger, exchange-listed stocks)

## Caveats
- Requires reliable, timely financial statement data and industry peers; firms with missing history or zero expenditures default to 0 on relevant signals.
- Shorting low G_SCORE (“torpedo”) stocks can be challenging due to delistings and liquidity; the extreme negative returns concentrate in the bottom tail.
- The accrual signal (G3) overlaps with Sloan’s accrual anomaly; however, the composite G_SCORE survives explicit controls for accruals, suggesting incremental information.
- Industry adjustment is essential; using raw cross-sectional thresholds rather than industry-relative medians weakens performance.
