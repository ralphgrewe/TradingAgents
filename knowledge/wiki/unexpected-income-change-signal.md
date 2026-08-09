---
id: unexpected-income-change-signal
title: Unexpected Income Change as a Trading Signal
tags:
- accounting-based
- earnings-announcement
- anomaly
- equity-selection
signals:
- unexpected_net_income_change
- unexpected_eps_change
asset_classes:
- equity
horizon:
- swing
- position
source:
  authors: Ray Ball, Philip Brown
  title: An Empirical Evaluation of Accounting Income Numbers
  year: 1968
  file: paper/BallBrown_AnEmpiciralEvaluationOfAccountingIncomeNumbers.pdf
---
## Summary
The paper demonstrates that unexpected changes in accounting income (net income or EPS) contain information that is rapidly reflected in stock prices. The authors construct a forecast error model to separate expected and unexpected income changes and show that the sign and magnitude of the forecast error are associated with abnormal stock returns around earnings announcements. The signal is strongest when the unexpected component is large and persists over months prior to the announcement, indicating that markets anticipate earnings information before its official release.

## Signal — what it is
The signal is the forecast error from a model predicting a firm’s income change, defined as the difference between the actual change in net income or EPS and the expected change conditional on economy-wide and firm-specific historical patterns. A positive forecast error (actual > expected) is interpreted as good news; a negative forecast error (actual < expected) is bad news. The signal can be computed for either net income or EPS using either a regression-based expectation model or a naive model (persistence of prior year’s EPS).

## How to compute
1. **Compute expected income change (regression model):**
   - For each firm j and year t, estimate the linear regression using prior years (T = 1, 2, ..., t−1):
     ΔI_{j,T} = α_{j,t} + β_{j,t} ΔM_{T} + ε_{j,T}
     where ΔI_{j,T} is the change in firm j’s income (net income or EPS) and ΔM_{T} is the change in the market income index (excluding firm j).
   - Use the estimated coefficients to predict the expected income change for year t:
     Ī_{j,t} = α̂_{j,t} + β̂_{j,t} ΔM_{t}
   - Compute the forecast error (unexpected income change):
     u_{j,t} = ΔI_{j,t} − Ī_{j,t}

2. **Compute expected income change (naive model):**
   - Define the forecast error as the change in EPS from the prior year:
     u_{j,t} = EPS_{j,t} − EPS_{j,t−1}

3. **Interpret the signal:**
   - u_{j,t} > 0 → buy signal (good news)
   - u_{j,t} < 0 → sell/short signal (bad news)

## Empirical evidence
The authors analyze 261 firms over 1957–1965 and find that the sign of the income forecast error is strongly associated with abnormal stock returns. For firms with positive forecast errors, the Abnormal Performance Index (API) rises from 1.000 at month −12 to 1.071 by month 0 (announcement month), while for firms with negative forecast errors, the API declines from 1.000 to 0.907 over the same period. The chi-square statistics for the association between forecast error sign and residual return sign are significant at the 1% level in most months from −11 to +2 relative to the announcement. The effect size indicates that approximately 25% of the net information value over the 12 months preceding the report is captured by the income number, with about 85–90% of the signal anticipated by the market before the announcement.

## When to apply / regime
Apply the signal in equity markets where accounting income data is available and earnings announcements are scheduled. The strategy is most effective when:
- Earnings announcements are imminent (within 1–3 months), as the signal’s predictive power peaks around the announcement month.
- The market has not yet fully priced the information (i.e., before the earnings report is released).
- The signal is used in conjunction with other timely information sources (e.g., interim reports, dividend announcements) to refine expectations, as the paper suggests that interim reports and other disclosures contribute to the market’s anticipation of earnings.

## Caveats
- The signal is backward-looking and relies on historical relationships that may change due to structural shifts in the economy or accounting standards.
- The naive model may misclassify firms during periods of broad market growth, as it does not account for economy-wide income trends.
- The study’s sample is limited to large, publicly traded firms with December 31 fiscal years and excludes firms without complete data, which may limit generalizability.
- Transaction costs and market efficiency imply that the signal’s exploitable edge may be small after accounting for trading frictions, especially around announcement dates when liquidity can be thin.
