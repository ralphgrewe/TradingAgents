---
name: quant-indicator-analyst
description: "Call this skill when the user asks for a structured technical analysis of a stock, company or symbol"
---

# Quant Indicator Analyst

Produce a structured technical/indicator analysis for `<TICKER>`. Output is the shared JSON envelope defined in `../SCHEMA.md`. **No markdown report, no narrative, no per-indicator commentary in chat.** Final chat message is one line per the schema's chat-output rule.

Scope: indicators only — no fundamentals, no news, no macro speculation.

This skill is the **memory-system pilot** (issue #8): it is the first skill wired to the shared
SQLite memory core (`tradingagents/memory/`, exposed via the `memory_*` MCP tools in
`mcp_server.py`) because it is the most mechanical analyst — its `signal`/`confidence` come
entirely from `compute_indicators.py`, a deterministic script with no LLM call in the loop. That
determinism is load-bearing: past-lesson context injected in Step 0 is **informational only** —
it may shape how you phrase `summary`/commentary, but it must never change the `signal` /
`confidence` values the script computes in Steps 2–3. Do not feed prior lessons back into the
script or let them alter which indicators are read.

## Step 0 — Memory: resolve pending + load past context (mandatory, before Step 1)

Agent id for all memory calls in this skill: `"quant-indicator-analyst"` (matches the `name:` in
this file's frontmatter).

1. Call `memory_resolve_pending` with `agent="quant-indicator-analyst"`, `ticker=<TICKER>` to fill
   in `forward_return`/`lesson` on any of this skill's own past decisions for `<TICKER>` whose
   horizon has elapsed. Ignore the return value (a list of resolved row ids, possibly empty) — it
   is a side effect, not an input to this run.
2. Call `memory_get_past_context` with `agent="quant-indicator-analyst"`, `ticker=<TICKER>` to
   retrieve prior resolved lessons (same-ticker + cross-ticker) as a markdown block.
3. Carry that markdown forward as **"Past Context"** and inject it in Step 4 when writing
   `summary`: if it contains prior lessons (i.e. it is not just the "no prior lessons yet"
   placeholder), let it inform the tone/emphasis of `summary` (e.g. flag if a similar setup
   previously missed). It must not change `signal`, `confidence`, or any `details` field — those
   come from the script's output in Steps 2–3, verbatim, computed before Past Context is even
   read.

If either call errors (returns a string starting with `"ERROR:"`), proceed without past context —
do not abort the run over a memory-core failure.

## Step 1 — Fetch OHLCV

Call `yfinance_get_price_history` with:
- `symbol`: ticker (uppercase)
- `period`: `"1y"` (enough for SMA-50 and all other indicators)
- `interval`: `"1d"`

Save the returned JSON; it feeds Step 2.

## Steps 2–3 — Compute indicators + interpret (run script)

Save the price history JSON from Step 1 to a file, then run `compute_indicators.py`:

> **Shell path note (important).** The shell is a Linux sandbox, **not** Windows. Do not pass
> `C:\...` paths to bash — they fail, and a failed script run must **not** be worked around with
> computer-use / "use my computer". The project folder
> `C:\Users\ralph\Documents\Claude\Projects\trading-skills` is mounted at the Linux path given in
> your environment (e.g. `/sessions/<id>/mnt/trading-skills`). `cd` into that mount; `<run_dir>`
> below is `runs/<YYYYMMDD>/<TICKER>` relative to it.

```bash
# Save Step 1 output first: Write tool → <run_dir>/raw_price_history.json

cd <trading-skills-mount>   # Linux mount of the project folder (see environment)
python TradingAgents/skills/quant/compute_indicators.py \
  <run_dir>/raw_price_history.json \
  <TICKER>
```

The script installs `pandas`/`pandas-ta` if needed, computes all indicator values,
derives per-indicator `trend` and `signal`, builds `convergence` and `trade_setup`,
and emits JSON with: `signal`, `confidence`, `summary`, `details` (matching the
payload schema below exactly).

Parse the output — use `details` directly as the `details` payload, and `signal` /
`confidence` / `summary` as the envelope top-level fields.

Do not estimate values; the script puts `null` and lists missing indicators in
`details.convergence.missing`.

## Indicator catalogue

| Indicator | Dimension | Role |
|-----------|-----------|------|
| `sma_50`  | Trend | Medium-term trend, dynamic support/resistance |
| `macdh`   | Momentum | Direction + strength + divergence |
| `rsi`     | Overbought/Oversold | 70/30 thresholds |
| `boll_ub` | Volatility | Upper ~2σ band — breakout / resistance |
| `boll_lb` | Volatility | Lower ~2σ band — breakdown / support |
| `atr`     | Risk sizing | Required for stop-loss in `trade_setup` |
| `vwma`    | Volume | Volume-weighted trend confirmation |

## Step 4 — Derive top-level envelope fields

`signal` and `confidence` come **directly and only** from the script output — never overridden or
adjusted using Past Context (Step 0) or anything else. Optionally refine `summary` to a more
descriptive one-liner before writing, and this is the one place Past Context may be woven in (see
Step 0.3) — a sentence noting a relevant prior lesson, still consistent with the script's `signal`.

## Step 5 — Write artifact

Write the envelope to `C:\Users\ralph\Documents\Claude\Projects\trading-skills\runs\<YYYYMMDD>\<TICKER>\quant-indicator-analyst.json` via the `Write` tool.

## `details` payload

```json
{
  "as_of": "YYYY-MM-DD",
  "close": 0.0,
  "market_regime": "trending_up | trending_down | ranging | volatile",
  "indicators": [
    {
      "indicator": "sma_50",
      "value": 0.0,
      "prev":  0.0,
      "trend": "Rising | Falling | Flat",
      "signal": "Bullish | Bearish | Neutral",
      "role": "<one sentence>"
    }
  ],
  "convergence": {
    "confirms":  ["<bullet>"],
    "conflicts": ["<bullet>"],
    "missing":   ["<indicator-name-or-empty>"]
  },
  "trade_setup": {
    "bias": "BUY | HOLD | SELL",
    "entry_trigger": "<exact condition>",
    "stop_loss":     "close - 1.5 × ATR",
    "take_profit":   "<price or condition>",
    "risk_reward":   "1:2.3"
  }
}
```

## Step 6 — Memory: store decision

Call `memory_store_decision` (agent id `"quant-indicator-analyst"`, same as Step 0) with:

| Argument | Value |
|---|---|
| `agent` | `"quant-indicator-analyst"` |
| `ticker` | `<TICKER>` |
| `date` | `<YYYY-MM-DD>` — the `details.as_of` value from the envelope just written |
| `signal` | the envelope's `signal` (verbatim) |
| `confidence` | the envelope's `confidence` mapped to a numeric score via `compute_indicators.py`'s `confidence_to_score` (`HIGH -> 1.0`, `MEDIUM -> 0.6`, `LOW -> 0.3` — the same convention `skills/trader/score_trader.py` uses) |
| `key_drivers` | `compute_indicators.py`'s `build_key_drivers(details)` — `{"market_regime", "confirms", "conflicts", "trade_setup"}` taken verbatim from `details.market_regime` / `details.convergence.confirms` / `details.convergence.conflicts` / `details.trade_setup` |
| `thesis` | the envelope's `summary` |

`compute_indicators.py` prints only `{signal, confidence, summary, details}` to stdout — compute
`confidence_to_score(confidence)` and `build_key_drivers(details)` yourself (either by importing
those two functions from `compute_indicators.py` if running in the same Python process, or by
reproducing the trivial mapping/reshaping shown above) before calling `memory_store_decision`.

This call is idempotent on `(agent, ticker, date)` — a duplicate call for a ticker/date already
recorded today is a harmless no-op, so it is safe to call unconditionally every run. Do not let a
memory-core error (a string starting with `"ERROR:"`) block writing the envelope or the chat
output — the envelope file (Step 5) is already on disk by this point; log/ignore the error and
continue.

## Step 7 — Chat output

Exactly one line:

```
<TICKER> quant-indicator-analyst: <signal> (<confidence>) → [quant-indicator-analyst.json](computer://C:\Users\ralph\Documents\Claude\Projects\trading-skills\runs\<YYYYMMDD>\<TICKER>\quant-indicator-analyst.json)
```
