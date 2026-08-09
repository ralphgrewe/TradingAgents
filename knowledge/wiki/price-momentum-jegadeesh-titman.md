---
id: price-momentum-jegadeesh-titman
title: Price Momentum Strategy (Jegadeesh and Titman)
tags:
- momentum
- cross-sectional
- relative-strength
- equity
signals:
- price_momentum
- relative_strength
asset_classes:
- equity
horizon:
- swing
source:
  authors: Narasimhan Jegadeesh, Sheridan Titman
  title: 'Returns to Buying Winners and Selling Losers: Implications for Stock Market
    Efficiency'
  year: 1993
  file: paper/JegadeeshTitman_Momentum.pdf
---
## Summary
A cross-sectional price momentum strategy that buys past winners and sells past losers generates consistent positive returns across U.S. equities and most developed markets over 3–12 month formation and holding periods. The effect persists despite extensive academic and practitioner attention, and is robust to standard risk adjustments. The strategy exploits underreaction to information rather than compensation for systematic risk, and is strongest among smaller, lower-analyst-coverage, high-turnover, and growth stocks. Long-horizon evidence shows partial reversal after 12–60 months, consistent with delayed overreaction in some subperiods.

## Signal — what it is
The signal is a cross-sectional relative-strength score: a stock’s cumulative return over a prior formation window (J months) minus the contemporaneous cross-sectional mean return. Stocks are ranked by this score; the top decile (“winners”) are bought and the bottom decile (“losers”) are sold, creating a zero-cost, dollar-neutral portfolio. The strategy relies on the empirical regularity that high past returns tend to predict high future returns over the subsequent 3–12 months, while low past returns tend to predict low future returns.

## How to compute
1. Formation window: Choose J ∈ {3, 6, 9, 12} months.
2. At the start of each month t, compute each stock’s cumulative return over months t−J to t−1.
3. Compute the cross-sectional mean of these cumulative returns and subtract it from each stock’s return to obtain a relative-strength score.
4. Rank all stocks by this score and assign to deciles. The top decile (P1) is the winner portfolio; the bottom decile (P10) is the loser portfolio.
5. Construct a zero-cost portfolio: long P1, short P10, equally weighted within each decile.
6. Hold the portfolio for K months (K ∈ {3, 6, 9, 12}), skipping one week between formation and holding to mitigate microstructure effects.
7. Rebalance monthly.

## Empirical evidence
Using NYSE/AMEX stocks from 1965–1989, the 12-month formation/3-month holding strategy yields 1.31% per month (t = 3.74) without a skip week, and 1.49% per month (t = 4.28) with a one-week skip. The 6-month/6-month strategy earns ~1.0% per month (t ≈ 3.07). Risk adjustment via CAPM or Fama–French three-factor models increases the spread: the zero-cost portfolio’s Fama–French alpha is 1.36% per month over 1965–1998. The effect is pervasive globally: Rouwenhorst (1998) replicates 1.16% monthly returns (t = 4.02) in 12 European countries for the 6/6 strategy. Japan is a notable exception, showing no significant momentum.

## When to apply / regime
Apply when:
- The market is not in a pronounced crash or liquidity crisis (momentum crashes are documented in crisis periods).
- Stocks have sufficient liquidity and price above $5 to avoid microstructure frictions.
- The formation window aligns with the 3–12 month horizon; shorter windows may be more sensitive to noise, longer windows to drift.
- The strategy performs best among small-cap, low-analyst-coverage, high-turnover, and growth (low book-to-market) stocks, suggesting higher efficacy in less efficiently priced segments.

## Caveats
- January seasonality: momentum returns are significantly negative in January (−1.55% in one sample), reducing annualized performance.
- Long-horizon reversal: cumulative profits peak at ~12 months and decline thereafter, turning negative by month 60 in the 1965–1998 sample, with partial reversal concentrated in years 4–5.
- Risk of momentum crashes during market stress; the strategy’s negative skewness is higher than standard benchmarks.
- Industry momentum contributes, but individual-stock momentum persists even when controlling for industry effects; however, skipping a month between formation and holding can attenuate industry momentum profits.
