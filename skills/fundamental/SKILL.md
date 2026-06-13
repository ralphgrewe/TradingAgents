---
name: fundamental-analyst
description: "Call this skill when the user asks for a structured fundamental analysis of a stock, company or symbol"
---

# Fundamental Analyst

Produce a structured fundamental analysis for `<TICKER>`. Output is the shared JSON envelope defined in `../SCHEMA.md`. **No narrative, tables, or markdown to chat.** Final chat message is one line per the schema's chat-output rule.

Rules:
- Collect all metrics first, evaluate second — never skip Step 1.
- Value and Growth evaluations use the same dataset — no duplicate fetching.
- Fundamental data sources only. No technical indicators, no sentiment, no social media.

## Step 1 — Data collection + ratio computation

**1a. Fetch raw data** using these yfinance MCP tools for `<TICKER>`:
- `yfinance_get_ticker_info` — save result to `<run_dir>/raw_ticker_info.json`
- `yfinance_get_financials` with `report_type=income_statement`, `frequency=annual` → `raw_income_annual.json`
- `yfinance_get_financials` with `report_type=balance_sheet`, `frequency=annual` → `raw_balance_annual.json`
- `yfinance_get_financials` with `report_type=cash_flow`, `frequency=annual` → `raw_cashflow_annual.json`
- `yfinance_get_holders` → `raw_holders.json`

Write each tool result to the run directory using the `Write` tool.

**1b. Compute ratios** by running `compute_ratios.py` in the shell:

> **Shell path note (important).** The shell is a Linux sandbox, **not** Windows. Do not pass
> `C:\...` paths to bash — they fail, and a failed script run must **not** be worked around with
> computer-use / "use my computer". The project folder
> `C:\Users\ralph\Documents\Claude\Projects\trading-skills` is mounted at the Linux path given in
> your environment (e.g. `/sessions/<id>/mnt/trading-skills`). `cd` into that mount; `<run_dir>`
> below is `runs/<YYYYMMDD>/<TICKER>` relative to it.

```bash
cd <trading-skills-mount>   # Linux mount of the project folder (see environment)
python TradingAgents/skills/fundamental/compute_ratios.py \
  <run_dir>/raw_ticker_info.json \
  <run_dir>/raw_income_annual.json \
  <run_dir>/raw_balance_annual.json \
  <run_dir>/raw_cashflow_annual.json \
  <run_dir>/raw_holders.json
```

The script prints JSON with: `context`, `annual` (ratio tables for the 3 most recent years),
`insider_sentiment`, `forecast` (empty — model fills this).

Parse the output — these fields populate the `details` payload directly.

For `forecast`: extend two years beyond the latest annual data point. Use analyst
forward estimates from `ticker_info` where available; otherwise project from historic
trend + sector/industry context. Keep accounting consistency (FCF must align with
margins and revenue).

## Step 2 — Value evaluation

Apply Value Investing principles (Graham, Buffett). Is the company below intrinsic value? Solid balance sheet? Sustainable FCF? Durable moat? Output `value.signal` ∈ `{BUY, HOLD, SELL}` with `value.confidence` ∈ `{HIGH, MEDIUM, LOW}` and `value.data_confidence` (quality of the underlying data).

## Step 3 — Growth evaluation

Apply Growth Investing principles (Fisher, Lynch). Strong/accelerating revenue and earnings growth? Scalable? Expanding margins? Positive forward guidance? Output `growth.signal`, `growth.confidence`, `growth.data_confidence`.

## Step 4 — Derive top-level envelope fields

- `signal`: if `value.signal == growth.signal`, take that. Otherwise `HOLD`.
- `confidence`: if both signals agree, take the higher of the two confidences; if they disagree, take the lower.
- `summary`: one line, e.g. `"Below intrinsic value but margins compressing — value BUY, growth HOLD"`.

## Step 5 — Write artifact

Write the envelope to `C:\Users\ralph\Documents\Claude\Projects\trading-skills\runs\<YYYYMMDD>\<TICKER>\fundamental-analyst.json` using the `Write` tool.

## `details` payload

```json
{
  "context": {
    "market_cap": null, "sector": null, "industry": null,
    "52w_high": null, "52w_low": null, "analyst_consensus": null
  },
  "annual": {
    "<YYYY>": {
      "valuation":    { "pe": null, "pb": null, "ev_ebitda": null, "pcf": null, "peg": null, "ps": null },
      "profitability":{ "gross_margin": null, "op_margin": null, "net_margin": null, "roe": null, "roic": null, "roa": null },
      "balance_sheet":{ "debt_to_equity": null, "current_ratio": null, "quick_ratio": null, "equity_ratio": null, "interest_coverage": null },
      "cashflow":     { "fcf": null, "op_cf": null, "fcf_yield": null, "fcf_margin": null, "capex_to_revenue": null },
      "dividends":    { "yield": null, "payout_ratio": null }
    }
  },
  "insider_sentiment": "BULLISH | NEUTRAL | BEARISH",
  "forecast": {
    "<YYYY>": {
      "forward_pe": null, "forward_fcf": null, "forward_pb": null,
      "forward_roe": null, "forward_debt_to_equity": null, "dividend_yield": null
    }
  },
  "value":  { "signal": "BUY|HOLD|SELL", "confidence": "HIGH|MEDIUM|LOW", "data_confidence": "HIGH|MEDIUM|LOW" },
  "growth": { "signal": "BUY|HOLD|SELL", "confidence": "HIGH|MEDIUM|LOW", "data_confidence": "HIGH|MEDIUM|LOW" }
}
```

## Step 6 — Chat output

Exactly one line, no more:

```
<TICKER> fundamental-analyst: <signal> (<confidence>) → [fundamental-analyst.json](computer://C:\Users\ralph\Documents\Claude\Projects\trading-skills\runs\<YYYYMMDD>\<TICKER>\fundamental-analyst.json)
```
