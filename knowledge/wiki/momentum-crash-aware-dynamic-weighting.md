---
id: momentum-crash-aware-dynamic-weighting
title: Momentum Crash-Aware Dynamic Weighting
tags:
- momentum
- crash-risk
- dynamic-volatility
- beta-hedging
- volatility-risk
signals:
- momentum_wml
- bear_market_indicator
- market_variance
- dynamic_weighting
asset_classes:
- equity
- futures
- commodities
- currencies
- fixed-income
horizon:
- swing
- position
source:
  authors: Kent Daniel, Tobias J. Moskowitz
  title: Momentum crashes
  year: 2016
  file: paper/DanielMoskowitz_MomentumCrashes.pdf
---
## Summary
Momentum strategies—long past winners and short past losers—deliver strong average returns across many asset classes but suffer infrequent yet severe crashes concentrated in panic states. These crashes are predictable: they occur following market declines and during periods of high ex-ante volatility, and coincide with sharp market rebounds. A dynamic strategy that scales the winner-minus-loser (WML) portfolio based on forecasts of momentum’s conditional mean and variance approximately doubles the Sharpe ratio and alpha of a static momentum strategy, and the improvement is robust across US equities, international markets, and multiple asset classes.

## Signal — what it is
The signal is a dynamic weight applied to a zero-investment momentum portfolio (WML) that is long past winners and short past losers. The weight is proportional to the conditional Sharpe ratio of the WML portfolio, computed as the ratio of the forecasted mean return to the forecasted volatility. The forecasted mean is derived from the interaction of a bear-market indicator (negative trailing two-year market return) and recent market variance, while the forecasted volatility is estimated using a GJR-GARCH model and recent realized volatility. The strategy leverages the fact that momentum crashes are concentrated in bear markets with high volatility and that the WML portfolio behaves like a short call option on the market during these states.

## How to compute
1. Construct the WML portfolio by ranking assets on past 12- to 2-month returns (skipping the most recent month), going long the top decile (winners) and short the bottom decile (losers), value-weighted within each decile.
2. Compute the bear-market indicator \(I_{B,t-1}\): 1 if the cumulative market return over the past 24 months is negative, else 0.
3. Compute the market variance \(\hat{\sigma}^2_{m,t-1}\): the annualized variance of daily market returns over the prior 126 trading days.
4. Forecast the conditional mean of WML returns:
   \[
   \hat{\mu}_{t-1} = \hat{\gamma}_0 + \hat{\gamma}_{int} \cdot I_{B,t-1} \cdot \hat{\sigma}^2_{m,t-1}
   \]
   where \(\hat{\gamma}_0\) and \(\hat{\gamma}_{int}\) are estimated from a time-series regression of WML returns on \(I_{B,t-1}\) and \(I_{B,t-1} \cdot \hat{\sigma}^2_{m,t-1}\).
5. Forecast the conditional volatility of WML returns using a GJR-GARCH(1,1) model with daily returns, then combine the model forecast with the realized volatility over the prior 126 days to obtain \(\hat{\sigma}_{t-1}\).
6. Compute the dynamic weight:
   \[
   w^*_t = \frac{1}{2\lambda} \cdot \frac{\hat{\mu}_{t-1}}{\hat{\sigma}^2_{t-1}}
   \]
   where \(\lambda\) is a scalar chosen to target a desired portfolio volatility.
7. Scale the WML portfolio by \(w^*_t\) each period to form the dynamic momentum strategy.

## Empirical evidence
Over 1927–2013, the static WML portfolio in US equities had a Sharpe ratio of 0.71 and annualized alpha of 22.2% relative to the market. The dynamic strategy more than doubled the Sharpe ratio to 1.20 and maintained a significant alpha after controlling for market, Fama–French factors, and conditional betas. In out-of-sample tests starting in 1934, the dynamic strategy achieved a Sharpe ratio of 1.19 versus 0.68 for the static WML. Across international equity markets (US, UK, Europe, Japan) and asset classes (equity index futures, commodities, currencies, fixed income), the dynamic strategy consistently outperformed both static and constant-volatility momentum strategies, with Sharpe ratios ranging from 0.84 to 1.22 depending on the market. The worst momentum crashes—e.g., July–August 1932 (WML −74.36% and −60.98% in consecutive months) and March–May 2009 (WML −42.28%, −45.52%, −30.54%)—occurred in bear markets with high volatility and coincided with sharp market rebounds.

## When to apply / regime
Apply the dynamic momentum strategy during regimes characterized by:
- Bear markets: trailing two-year market returns are negative.
- Elevated market variance: recent realized volatility is high.
- Market rebounds: contemporaneous market returns are positive.

These conditions coincide with momentum crashes and the WML portfolio’s negative conditional beta and option-like losses. The strategy scales back or even shorts the WML portfolio in these states, while scaling up exposure in calm or bullish regimes where momentum premia are higher and volatility is lower.

## Caveats
- The dynamic strategy requires accurate forecasts of conditional mean and volatility; estimation error can reduce performance.
- The strategy employs leverage and can take negative weights, increasing transaction costs and implementation complexity.
- The worst crashes are rare and clustered, so robustness checks across multiple regimes and asset classes are essential.
- While the strategy mitigates crashes, it does not eliminate all downside risk; risk management remains critical.
