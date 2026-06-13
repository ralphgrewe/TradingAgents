---
name: quant-indicator-analyst
description: "Call this skill when the user asks for a structured technical analysis of a stock, company or symbol"
---

# Quant Indicator Analyst

Produce a structured technical/indicator analysis for `<TICKER>`. Output is the shared JSON envelope defined in `../SCHEMA.md`. **No markdown report, no narrative, no per-indicator commentary in chat.** Final chat message is one line per the schema's chat-output rule.

Scope: indicators only — no fundamentals, no news, no macro speculation.

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

`signal`, `confidence`, and `summary` come directly from the script output.
Optionally refine `summary` to a more descriptive one-liner before writing.

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

## Step 6 — Chat output

Exactly one line:

```
<TICKER> quant-indicator-analyst: <signal> (<confidence>) → [quant-indicator-analyst.json](computer://C:\Users\ralph\Documents\Claude\Projects\trading-skills\runs\<YYYYMMDD>\<TICKER>\quant-indicator-analyst.json)
```
