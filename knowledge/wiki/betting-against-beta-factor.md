---
id: betting-against-beta-factor
title: Betting Against Beta (BAB) Factor
tags:
- low-beta
- leverage-constraints
- anomaly
- cross-asset
signals:
- beta-sorted-portfolios
- BAB-factor
asset_classes:
- equity
- fixed-income
- credit
- futures
horizon:
- swing
- position
source:
  authors: Andrea Frazzini, Lasse H. Pedersen
  title: Betting Against Beta
  year: 2013
  file: paper/BettingAgainstBeta.pdf
---
## Summary
The Betting Against Beta (BAB) factor exploits a pervasive anomaly where high-beta assets deliver lower risk-adjusted returns than low-beta assets due to leverage constraints faced by many investors. The strategy constructs a self-financing, market-neutral portfolio that is long leveraged low-beta securities and short high-beta securities, earning positive risk-adjusted returns across U.S. equities, international equities, Treasury bonds, corporate bonds, and futures. The effect is robust across 20 developed equity markets, multiple asset classes, and decades of data, with the U.S. BAB factor achieving a Sharpe ratio of 0.78 (1926–2012) and international BAB factors delivering similarly strong performance. The strategy’s returns are inversely related to funding liquidity conditions: tightening constraints compress betas toward 1 and reduce contemporaneous BAB returns, while funding liquidity risk compresses cross-sectional beta dispersion.

## Signal — what it is
BAB is a zero-beta, market-neutral factor that systematically buys low-beta assets (leveraged to a beta of 1) and sells high-beta assets (de-leveraged to a beta of 1), creating a self-financing portfolio with unit exposure to the beta anomaly. The factor’s payoff arises because constrained investors (e.g., mutual funds, individuals) overweight high-beta assets, bidding up their prices and depressing future returns, while unconstrained investors (e.g., hedge funds, LBO firms) exploit the mispricing by leveraging low-beta assets. The strategy’s returns are driven by the spread in betas between the long and short legs and the tightness of funding constraints in the market.

## How to compute
1. **Beta estimation**: For each asset in the universe, estimate ex-ante beta using rolling regressions of excess returns on market excess returns. Use daily data where available (1-year rolling volatility, 5-year rolling correlation) and shrink estimates toward the cross-sectional mean to reduce noise (Vasicek shrinkage with weight w = 0.6).
2. **Portfolio construction**:
   - **Sorting**: Rank assets by estimated beta and split into two portfolios: low-beta (below median) and high-beta (above median).
   - **Weighting**: Within each portfolio, weight assets by their ranked betas (lower-beta assets receive larger weights in the low-beta portfolio; higher-beta assets receive larger weights in the high-beta portfolio).
   - **Rescaling**: Rescale both portfolios to have a beta of 1 at formation.
   - **BAB factor**: Construct a self-financing portfolio that is long the low-beta portfolio (financed by borrowing at the risk-free rate) and short the high-beta portfolio (with proceeds invested at the risk-free rate).
3. **Rebalancing**: Rebalance portfolios monthly to maintain target betas and weights.

Formally, the BAB factor return at time t is:
\[
r_{t}^{\text{BAB}} = \left( \frac{1}{\beta_{L,t}} r_{L,t} - r_{f,t} \right) - \left( \frac{1}{\beta_{H,t}} r_{H,t} - r_{f,t} \right)
\]
where \(r_{L,t}\) and \(r_{H,t}\) are the returns of the low-beta and high-beta portfolios, \(\beta_{L,t}\) and \(\beta_{H,t}\) are their ex-ante betas, and \(r_{f,t}\) is the risk-free rate.

## Empirical evidence
- **U.S. equities (1926–2012)**: The BAB factor achieves a Sharpe ratio of 0.78, with Fama-French 5-factor alpha of 0.55% per month (t-stat = 4.09). The strategy’s returns are robust to controlling for size, value, momentum, and liquidity factors.
- **International equities (1984–2012)**: Pooled international BAB factor earns 0.64% monthly alpha (t-stat = 4.68) with a Sharpe ratio of 0.95. Country-level BAB factors are positive in 18 of 19 developed markets, with 6 countries showing statistically significant alphas.
- **Treasury bonds (1952–2012)**: BAB factor delivers 0.17% monthly alpha (t-stat = 6.26) and a Sharpe ratio of 0.81, exploiting the term structure anomaly where short-maturity bonds offer higher risk-adjusted returns than long-maturity bonds.
- **Corporate bonds (1973–2012)**: BAB factors across maturities and ratings deliver 0.11–0.57% monthly alphas with Sharpe ratios up to 0.82, confirming the anomaly extends to credit markets.
- **Futures (1963–2012)**: BAB factors across equity indexes, country bonds, currencies, and commodities generate positive returns, with diversified combinations yielding 0.25–0.26% monthly alphas (t-stats = 2.42–2.53).

## When to apply / regime
- **Favorable regimes**: Apply BAB when funding liquidity is abundant (low TED spread volatility) and beta dispersion is high. The strategy performs best in periods of stable funding conditions, as tightening constraints compress betas and reduce contemporaneous returns.
- **Unfavorable regimes**: Reduce exposure during periods of funding stress (high TED spread volatility), as the BAB factor tends to underperform when margin requirements tighten. The strategy’s conditional beta rises in volatile credit environments, increasing market sensitivity.
- **Cross-asset applicability**: The anomaly is pervasive across equities, bonds, credit, and futures, making BAB a robust multi-asset signal. The strategy’s performance is consistent across time periods, sub-samples, and global markets.

## Caveats
- **Beta estimation noise**: Ex-ante beta estimates are noisy, especially for illiquid assets. Shrinkage helps but does not eliminate estimation error, which can affect portfolio construction and realized betas.
- **Funding liquidity risk**: The strategy is exposed to funding liquidity shocks, which can compress betas and reduce returns. The TED spread is a proxy for funding conditions and may not capture all relevant constraints.
- **Implementation costs**: Leveraging low-beta assets and short-selling high-beta assets incurs transaction costs, borrowing costs, and potential short-sale constraints. The strategy’s self-financing nature mitigates some costs but not all.
- **Regime shifts**: The anomaly’s persistence relies on leverage constraints remaining binding for a significant fraction of investors. Structural changes in market structure or regulation could alter the strategy’s edge.
