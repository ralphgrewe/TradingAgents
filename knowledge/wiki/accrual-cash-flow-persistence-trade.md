---
id: accrual-cash-flow-persistence-trade
title: Accrual vs Cash-Flow Persistence Trade
tags:
- accruals
- cash-flow
- earnings-quality
- persistence
- anomaly
signals:
- accrual_component
- cash_flow_component
- earnings_persistence
asset_classes:
- equity
horizon:
- swing
- position
source:
  authors: P.J. Heyns, W.D. Hamman, E. vd M. Smit
  title: Do share prices fully reflect the information about future earnings in accruals
    and cash flow?
  year: 1999
  file: paper/HeynsHammanSmit_DoSharePricesFullyReflectFutureEarnings.pdf
---
## Summary
The paper tests whether share prices fully reflect the differential persistence of earnings attributable to accruals versus cash flow. Using South African data (1974–1996, 3,244 firm-years), it finds that earnings performance driven by cash flow is more persistent than that driven by accruals. While markets appear to recognize the higher persistence of cash flow, they systematically underestimate the influence of both components, leading to a rejection of market efficiency under the tested framework.

## Signal — what it is
A trading signal that exploits the documented difference in persistence between the accrual and cash-flow components of current earnings. The accrual component is less persistent (i.e., mean-reverts faster), while the cash-flow component is more persistent (i.e., more likely to recur in future earnings). Markets underreact to both components, but especially to the relative persistence gap, creating a predictable drift in returns.

## How to compute
1. Compute earnings components scaled by total assets:
   - Accruals = ΔInventories + ΔDebtors − ΔCreditors − Depreciation
   - Cash Flow = Earnings − Accruals
2. Standardize each component by end-of-year total assets.
3. Form a long-short portfolio ranked annually on the ratio of accruals to cash flow (high accrual/low cash flow vs. low accrual/high cash flow).
4. Hold for the subsequent year and rebalance annually.

## Empirical evidence
Using 3,244 firm-years from 1974–1996 on the JSE, the study finds:
- The persistence coefficient for cash flow (≈0.84) exceeds that for accruals (≈0.76) in forecasting next-year earnings (F-test rejects equality at p<0.01).
- Market prices underreact to both components, but especially to the higher persistence of cash flow. The likelihood-ratio test rejects market efficiency (p<0.10 across specifications).
- Time-series plots show mean reversion is slower for cash-flow-driven earnings than for accrual-driven earnings.

## When to apply / regime
Apply in equity markets where accrual and cash-flow data are reliably reported and audited. Best suited to liquid, large-cap segments where accounting quality is high. Avoid during periods of accounting regime shifts or macro shocks that disrupt earnings persistence (e.g., financial crises, regulatory changes). Prefer in stable economic environments where earnings are informative about future cash flows.

## Caveats
- Survivorship bias in the sample (delisted firms excluded).
- Results are strongest for operating earnings (profit before tax, excluding interest and non-recurring items).
- Market underreaction is subtle and may be arbitraged away in highly efficient markets.
- Requires careful scaling and standardization to avoid size effects.
- The signal is sensitive to accounting definitions; consistency in accrual computation is critical.
