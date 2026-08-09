---
id: piotroski-f-score
title: Piotroski F-Score
tags: [value, fundamentals, quality, low-book-to-market]
signals: [f_score, roa, delta_roa, cfo, accrual, delta_lever, delta_liquid, eq_offer, delta_margin, delta_turn]
asset_classes: [equity]
horizon: [position]
source: {authors: "Joseph D. Piotroski", title: "Value Investing: The Use of Historical Financial Statement Information to Separate Winners from Losers", year: 2000, file: paper/Piotroski_ValueInvestingTheUseOfHistoricalFinancialStatementInformation.pdf}
---
## Summary

Piotroski (2000) shows that within the universe of high book-to-market ("value") firms,
a simple 9-point score built entirely from historical financial-statement data — the
**F-Score** — separates future winners from future losers. Value investing as a class
earns a premium on average, but the premium is concentrated almost entirely in the
subset of value firms whose fundamentals are actually improving; a large share of
nominally cheap stocks are cheap because they are genuinely deteriorating businesses
("value traps"). The F-Score is a low-cost, purely mechanical fundamentals screen — no
forecasting, no valuation model — that a value investor can use to tilt a portfolio
away from those traps.

## Signal — what it is

`f_score` is the unweighted sum of nine independent binary (0/1) signals, each testing
whether one dimension of the firm's fundamentals moved in a favorable direction over the
past fiscal year. The nine signals fall into three groups:

**Profitability**
- `roa` — return on assets is positive (current-year net income before extraordinary
  items, scaled by beginning-of-year total assets).
- `cfo` — operating cash flow is positive.
- `delta_roa` — ROA improved relative to the prior year.
- `accrual` — operating cash flow exceeds ROA (i.e. cash-flow-based profitability
  exceeds accrual-based profitability), a quality-of-earnings check: firms whose
  earnings are backed by cash rather than accruals tend to have more persistent
  profitability.

**Leverage, liquidity, and source of funds**
- `delta_lever` — the ratio of long-term debt to average total assets decreased
  year-over-year (lower long-term leverage).
- `delta_liquid` — the current ratio (current assets / current liabilities) improved
  year-over-year (better short-term liquidity).
- `eq_offer` — the firm did **not** issue new common equity during the year (a proxy for
  not needing to raise external capital, and for management not viewing the stock as
  currently overvalued).

**Operating efficiency**
- `delta_margin` — gross margin improved year-over-year (a proxy for improving product
  differentiation/cost control).
- `delta_turn` — asset turnover (sales / average total assets) improved year-over-year
  (more sales generated per dollar of assets, i.e. improving productivity or demand).

Each signal contributes exactly one point when true, zero otherwise, so
`f_score` ranges from 0 (uniformly deteriorating fundamentals) to 9 (uniformly
improving fundamentals).

## How to compute

For a firm with fiscal-year-end financials for year `t` (current) and `t-1` (prior),
using standard line items (net income before extraordinary items `NI`, total assets
`TA`, operating cash flow `CFO`, long-term debt `LTD`, current assets `CA`, current
liabilities `CL`, common shares outstanding `SHARES`, sales `SALES`, cost of goods sold
`COGS`):

```
ROA_t        = NI_t / TA_{t-1}
ACCRUAL_t    = CFO_t / TA_{t-1} - ROA_t                      # CFO scaled minus ROA
LEVER_t      = LTD_t / ((TA_t + TA_{t-1}) / 2)
LIQUID_t     = CA_t / CL_t
MARGIN_t     = (SALES_t - COGS_t) / SALES_t
TURN_t       = SALES_t / ((TA_t + TA_{t-1}) / 2)

roa          = 1 if ROA_t > 0 else 0
cfo          = 1 if CFO_t > 0 else 0
delta_roa    = 1 if ROA_t > ROA_{t-1} else 0
accrual      = 1 if ACCRUAL_t > 0 else 0                     # CFO_t/TA_{t-1} > ROA_t
delta_lever  = 1 if LEVER_t < LEVER_{t-1} else 0
delta_liquid = 1 if LIQUID_t > LIQUID_{t-1} else 0
eq_offer     = 1 if SHARES_t <= SHARES_{t-1} else 0           # no new equity issued
delta_margin = 1 if MARGIN_t > MARGIN_{t-1} else 0
delta_turn   = 1 if TURN_t > TURN_{t-1} else 0

f_score = roa + cfo + delta_roa + accrual + delta_lever + delta_liquid
          + eq_offer + delta_margin + delta_turn
```

All nine components need two consecutive fiscal years of financial-statement data
(current + prior); this is a pure fundamentals computation with no market-price inputs
(book-to-market is used only to define the *universe* the score is applied within, not
in the score itself), so — consistent with this repo's convention of precomputing
numeric signals in Python rather than asking an LLM to do the arithmetic (see
`tradingagents/agents/analysts/fundamentals_computation.py` and `LEARNINGS.md`) — this
belongs in a fundamentals computation module fed by the configured `fundamental_data`
vendor, not computed inline by an agent.

## Empirical evidence

Piotroski (2000) tests the score on U.S. firms in the highest book-to-market quintile
(the "value" universe) from 1976–1996. Key findings:

- A strategy that goes long high-F_Score (F_Score ≥ 8) firms and short low-F_Score
  (F_Score ≤ 1) firms within the high book-to-market quintile earns a mean annual
  return of roughly 23% over the sample period, with the long side alone
  outperforming the value-quintile average by about 7.5 percentage points annually and
  the short side underperforming it by a comparable margin.
- The effect is driven disproportionately by the short side and by small,
  less-followed firms: the market appears slower to price deteriorating fundamentals
  into low-F_Score value stocks than to price improving fundamentals into high-F_Score
  ones.
- The strategy is most effective among firms with low share turnover, low institutional
  ownership, and no analyst coverage — i.e. where information about fundamentals is
  least likely to already be reflected in price.
- Because the underlying signals are simple accounting ratios, the result is a natural
  test of (weak-form) semi-efficiency: Piotroski interprets the return spread as
  evidence that the market does not fully and immediately incorporate historical
  financial-statement information into price, particularly for neglected value stocks.

## When to apply / regime

- Designed for **equities**, applied **within a value universe** (high book-to-market /
  low price-to-book firms) — the original paper explicitly does not claim the score
  adds value uniformly across the market; its power comes from separating winners and
  losers *among stocks that already look cheap*.
- A **position-horizon** signal: F-Score is recomputed once per fiscal year (annual
  financial statements), so it is not a timing signal for swing-length holding
  periods — it is a screen for which cheap stocks to hold over a multi-quarter-to-annual
  horizon, and is far more relevant to portfolio-manager-style stock selection than to
  short-horizon swing-trade entries.
- Works best in a **broad, liquid market with regular fundamental disclosure** and is
  most powerful among **small-cap, low-coverage names** where mispricing of
  fundamentals is more likely to persist; the edge is expected to be weaker (and harder
  to trade net of costs) in large, heavily-covered mega-caps.
- Should be read as a **relative-ranking tool within a peer/value universe**, not an
  absolute buy/sell threshold applicable to any single stock in isolation.

## Caveats

- **Data requirements**: needs two consecutive years of clean fundamental data
  (net income, operating cash flow, total assets, long-term debt, current
  assets/liabilities, shares outstanding, sales, COGS); missing or restated financials
  make individual components unreliable or unavailable, especially around M&A,
  spin-offs, and accounting-standard changes.
- **Look-ahead bias**: financial statements must be dated by their *public filing*
  date, not fiscal period end, when used historically — using period-end dates as if
  they were available immediately overstates backtested performance.
- **Sample/era specificity**: the original result is from U.S. equities, 1976–1996; the
  effect's magnitude in more recent, more efficient, and non-U.S. markets is not
  guaranteed to replicate at the same size, and small/low-coverage names (where the
  effect concentrates) can be costlier to trade (wider spreads, less liquidity) than
  the raw score-based returns suggest.
- **Crowding/decay**: to the extent the F-Score itself becomes a widely-known
  screening criterion, its predictive power among the most-followed value names should
  be expected to erode over time — the original edge specifically relies on limited
  market attention.
- **Binary scoring loses magnitude information**: each component is a strict 0/1 test,
  so a firm that barely improved ROA scores identically to one that improved it
  dramatically; the score is a coarse ranking tool, not a magnitude-weighted signal.
- **Not a timing signal**: because it updates only with annual fundamentals, F-Score
  says nothing about near-term price action and should not be treated as an entry/exit
  trigger for short-horizon trades.
