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

This skill is wired into the shared memory core (issue #9), mechanically repeating the pattern
validated by the quant pilot (issue #8, `skills/quant/SKILL.md`). Unlike quant, `value.signal` /
`growth.signal` are not produced by a deterministic script — they are your own judgment, applied
in Steps 2–3 to the ratio table `compute_ratios.py` computes in Step 1b. That ratio table (like
quant's indicator computation) takes no memory-context input at all — `compute_ratios.py`'s
`compute()` is a pure function of the raw yfinance files only. The equivalent determinism
guarantee here is structural at the prompt level: Past Context loaded in Step 0 must not be read
until *after* Steps 2–3 have already fixed `value.signal` / `value.confidence` / `value.key_ratios`
/ `growth.signal` / `growth.confidence` / `growth.key_ratios` from Step 1's data alone — it may
only be woven into `summary` in Step 4.

## Step 0 — Memory: resolve pending + load past context (mandatory, before Step 1)

Agent id for all memory calls in this skill: `"fundamental-analyst"` (matches the `name:` in this
file's frontmatter).

1. Call `memory_resolve_pending` with `agent="fundamental-analyst"`, `ticker=<TICKER>` to fill in
   `forward_return`/`lesson` on any of this skill's own past decisions for `<TICKER>` whose horizon
   has elapsed. Ignore the return value (a list of resolved row ids, possibly empty) — it is a side
   effect, not an input to this run.
2. Call `memory_get_past_context` with `agent="fundamental-analyst"`, `ticker=<TICKER>` to retrieve
   prior resolved lessons (same-ticker + cross-ticker) as a markdown block.
3. Carry that markdown forward as **"Past Context"** and inject it in Step 4 when writing
   `summary`: if it contains prior lessons (i.e. it is not just the "no prior lessons yet"
   placeholder), let it inform the tone/emphasis of `summary` (e.g. flag if a similar value/growth
   call previously missed). It must not change `value.signal`, `value.confidence`,
   `value.key_ratios`, `growth.signal`, `growth.confidence`, `growth.key_ratios`, or any other
   `details` field — those are decided from Step 1's ratio table alone in Steps 2–3, before Past
   Context is woven in anywhere.

If either call errors (returns a string starting with `"ERROR:"`), proceed without past context —
do not abort the run over a memory-core failure.

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

Apply Value Investing principles (Graham, Buffett). Is the company below intrinsic value? Solid balance sheet? Sustainable FCF? Durable moat? Output `value.signal` ∈ `{BUY, HOLD, SELL}` with `value.confidence` ∈ `{HIGH, MEDIUM, LOW}`, `value.data_confidence` (quality of the underlying data), and `value.key_ratios` — the specific ratio field name(s) from Step 1b's `annual` table (e.g. `["pe", "roic", "debt_to_equity"]`) that most drove this signal. Decide `key_ratios` from Step 1's ratio table only — never from Past Context (Step 0).

## Step 3 — Growth evaluation

Apply Growth Investing principles (Fisher, Lynch). Strong/accelerating revenue and earnings growth? Scalable? Expanding margins? Positive forward guidance? Output `growth.signal`, `growth.confidence`, `growth.data_confidence`, and `growth.key_ratios` — the specific ratio field name(s) from Step 1b's `annual` table (e.g. `["gross_margin", "fcf_margin", "peg"]`) that most drove this signal, decided from Step 1's ratio table only, same rule as `value.key_ratios`.

## Step 4 — Derive top-level envelope fields

- `signal`: if `value.signal == growth.signal`, take that. Otherwise `HOLD`.
- `confidence`: if both signals agree, take the higher of the two confidences; if they disagree, take the lower.
- `summary`: one line, e.g. `"Below intrinsic value but margins compressing — value BUY, growth HOLD"`. This
  is the one place Past Context (Step 0) may be woven in — a clause noting a relevant prior lesson,
  still consistent with `value.signal`/`growth.signal` as already decided in Steps 2–3.

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
  "value":  { "signal": "BUY|HOLD|SELL", "confidence": "HIGH|MEDIUM|LOW", "data_confidence": "HIGH|MEDIUM|LOW", "key_ratios": [] },
  "growth": { "signal": "BUY|HOLD|SELL", "confidence": "HIGH|MEDIUM|LOW", "data_confidence": "HIGH|MEDIUM|LOW", "key_ratios": [] }
}
```

## Step 6 — Memory: store decision

Call `memory_store_decision` (agent id `"fundamental-analyst"`, same as Step 0) with:

| Argument | Value |
|---|---|
| `agent` | `"fundamental-analyst"` |
| `ticker` | `<TICKER>` |
| `date` | the envelope's `date` (verbatim, `YYYY-MM-DD`) |
| `signal` | the envelope's `signal` (verbatim) |
| `confidence` | the envelope's `confidence` mapped to a numeric score via `compute_ratios.py`'s `confidence_to_score` (`HIGH -> 1.0`, `MEDIUM -> 0.6`, `LOW -> 0.3` — the same convention `skills/quant/compute_indicators.py` and `skills/trader/score_trader.py`'s `conf_weight` use) |
| `key_drivers` | `compute_ratios.py`'s `build_key_drivers(details)` — `{"value": {"signal", "confidence", "key_ratios"}, "growth": {"signal", "confidence", "key_ratios"}, "insider_sentiment"}` taken verbatim from `details.value` / `details.growth` / `details.insider_sentiment` |
| `thesis` | the envelope's `summary` |

`compute_ratios.py` prints only `{context, annual, insider_sentiment, forecast}` to stdout — compute
`confidence_to_score(confidence)` and `build_key_drivers(details)` yourself (either by importing
those two functions from `compute_ratios.py` if running in the same Python process, or by
reproducing the trivial mapping/reshaping shown above) before calling `memory_store_decision`.

This call is idempotent on `(agent, ticker, date)` — a duplicate call for a ticker/date already
recorded today is a harmless no-op, so it is safe to call unconditionally every run. Do not let a
memory-core error (a string starting with `"ERROR:"`) block writing the envelope or the chat
output — the envelope file (Step 5) is already on disk by this point; log/ignore the error and
continue.

## Step 7 — Chat output

Exactly one line, no more:

```
<TICKER> fundamental-analyst: <signal> (<confidence>) → [fundamental-analyst.json](computer://C:\Users\ralph\Documents\Claude\Projects\trading-skills\runs\<YYYYMMDD>\<TICKER>\fundamental-analyst.json)
```
