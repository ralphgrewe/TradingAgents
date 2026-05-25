---
name: portfolio-manager
description: "Call this skill when the user wants to create, rebalance, or manage a simulated stock portfolio. Triggers: 'run the portfolio manager', 'rebalance my portfolio', 'aggressive portfolio', 'conservative portfolio', 'manage my depot', 'build a portfolio from these stocks'."
---

# Portfolio Manager

Manage a simulated portfolio in a named depot. Given a universe of tickers and a style, run the `trader` skill on each ticker, compute a target allocation, and execute rebalancing trades via the trading-simulation MCP.

**No narrative to chat. Final chat output is exactly one line (see Step 9).**

---

## Inputs

| Parameter | Description | Example |
|---|---|---|
| `style` | `aggressive` or `conservative` | `aggressive` |
| `depot_id` | Named depot in the simulation | `mag7-aggressive` |
| `tickers` | Space/comma-separated list **or** a file path | `AAPL MSFT NVDA` or `TradingAgents/universes/mag7.txt` |
| `mode` | `fresh` (re-run trader) or `cached` (use today's existing trader.json) | `fresh` (default) |

---

## Style parameters

### Aggressive
| Signal | Conviction | Raw weight |
|---|---|---|
| BUY | HIGH | 2× slot |
| BUY | MEDIUM | 1× slot |
| BUY | LOW | 0.5× slot |
| HOLD | any | 0 (exit) |
| SELL | any | 0 (exit) |

- `max_positions = 5`, `cash_reserve = 0.05`
- `slot = (1 - cash_reserve) / max_positions = 0.19`
- Single-position cap: **35%**
- Min trade size: **1 share**
- HOLD = exit: any position not actively rated BUY is sold

### Conservative
| Signal | Conviction | Raw weight |
|---|---|---|
| BUY | HIGH | 1× slot |
| BUY | MEDIUM | 0.75× slot |
| BUY | LOW | 0.4× slot |
| HOLD | any | keep current weight (up to slot cap) |
| SELL | HIGH | 0 (full exit) |
| SELL | MEDIUM or LOW | 50% of current weight (trim only) |

- `max_positions = 10`, `cash_reserve = 0.15`
- `slot = (1 - cash_reserve) / max_positions = 0.085`
- Single-position cap: **15%**
- Min trade size: **2 shares** (suppresses rounding noise)
- HOLD = keep: no trade placed for HOLD signals

---

## Step 0 — Resolve inputs

1. Determine `YYYYMMDD` via bash: `date +%Y%m%d`
2. Resolve tickers:
   - If the tickers input contains `/` or `\` or ends in `.txt` or `.json` → treat as file path, `Read` the file, parse one ticker per line (skip blank lines and `#` comments).
   - Otherwise → split on whitespace and/or commas.
   - Uppercase all tickers. Deduplicate.
3. Validate: at least 1 ticker, `style` ∈ `{aggressive, conservative}`, `depot_id` non-empty.
4. Set style parameters from the table above.

---

## Step 1 — Get or create depot

Call `mcp__trading-simulation__list_depots`. If `depot_id` is not in the result, call `mcp__trading-simulation__create_depot` with `depot_id` and `initial_cash = 100000`.

Call `mcp__trading-simulation__get_portfolio` with `depot_id`. Record:
- `pre_cash` — cash balance
- `pre_equity` — total equity
- `pre_positions` — map of `symbol → { shares, price, market_value }`

---

## Step 2 — Run trader skills

For each ticker in the universe:

1. Check if `C:\Users\ralph\Documents\Claude\Projects\trading-skills\runs\<YYYYMMDD>\<TICKER>\trader.json` exists (use `Read` — if it errors, file is absent).
2. If absent **or** `mode = fresh`: invoke the `trader` skill for `<TICKER>`. Wait for it to complete and verify the file exists before moving to the next ticker.
3. `Read` the `trader.json`. Extract:
   - `signal` ∈ `{BUY, HOLD, SELL}`
   - `confidence` ∈ `{HIGH, MEDIUM, LOW}`

Build a signals map: `{ TICKER: { signal, confidence } }`.

---

## Step 3 — Fetch current prices

For each ticker in the universe, call `mcp__trading-simulation__get_quote` with `symbol`. Record `price`. If `get_quote` returns an error for a ticker, log a warning and remove that ticker from the universe for this run.

For tickers already in `pre_positions`, use the portfolio's recorded price as a fallback if `get_quote` fails.

---

## Step 4 — Compute target allocation

**4a. Raw weights**

For each ticker apply the style table to get `raw_weight`. Tickers not in the universe but present in the portfolio: see Step 4d.

**4b. HOLD handling**
- Aggressive: set `raw_weight = 0` for any HOLD ticker.
- Conservative: for HOLD tickers, set `raw_weight = min(current_weight, slot)` where `current_weight = pre_positions[ticker].market_value / pre_equity`. If ticker is not currently held, `raw_weight = 0` (no new buys on HOLD).

**4c. Normalise**

Sum all non-zero raw weights → `total_raw`. Scale each: `target_weight[ticker] = raw_weight / total_raw × (1 - cash_reserve)`. Clamp each to the single-position cap. Re-normalise after clamping if any cap was hit (iterate once).

**4d. Tickers in portfolio but NOT in universe**

Leave them untouched — this portfolio manager only manages tickers it knows about. Do not place orders for out-of-universe positions.

**4e. Target shares**

```
target_dollars[ticker] = pre_equity × target_weight[ticker]
target_shares[ticker]  = floor(target_dollars[ticker] / price[ticker])
```

**4f. Deltas**

```
delta[ticker] = target_shares[ticker] - pre_positions.get(ticker, 0)
```

Positive = buy, negative = sell, zero = no action.

Apply min trade size filter: if `|delta| < min_trade_size`, set `delta = 0`.

---

## Step 5 — Execute trades

Order of execution: **sells first** (to free cash), then **buys** (highest conviction first — sort buys by `raw_weight` descending).

For each trade:
1. Call `mcp__trading-simulation__place_order` with `symbol`, `side` (`"buy"` or `"sell"`), `quantity = abs(delta)`, `depot_id`.
2. Record the result: `{ symbol, side, quantity, status, message, fill_price }`.
3. If status is `"rejected"`, log the rejection but continue with remaining orders. Do not abort the run.

---

## Step 6 — Post-trade snapshot

Call `mcp__trading-simulation__get_portfolio` again. Record `post_cash`, `post_equity`, `post_positions`.

Call `mcp__trading-simulation__get_trades` with `limit = len(universe) * 2` and `depot_id`. Record fills for this run.

---

## Step 7 — Append performance log

Append one CSV row to `C:\Users\ralph\Documents\Claude\Projects\trading-skills\runs\performance-<depot_id>.csv`.

If the file does not exist yet, write the header first:
```
date,equity,cash,num_positions,tickers_held
```

Row format:
```
<YYYY-MM-DD>,<post_equity>,<post_cash>,<count of post_positions>,<space-separated list of held tickers>
```

Use the `Edit` tool to append (read the file first, then append the row). If the file is new, use `Write`.

---

## Step 8 — Write portfolio-manager envelope

Write to `C:\Users\ralph\Documents\Claude\Projects\trading-skills\runs\<YYYYMMDD>\portfolio-manager-<depot_id>.json` using the `Write` tool.

```json
{
  "skill": "portfolio-manager",
  "ticker": null,
  "date": "YYYY-MM-DD",
  "signal": null,
  "confidence": null,
  "summary": "<one-line: e.g. 'aggressive rebalance of mag7-aggressive: 3 buys, 1 sell — equity $102,450'>",
  "details": {
    "style": "aggressive | conservative",
    "depot_id": "<depot_id>",
    "universe": ["TICKER", "..."],
    "mode": "fresh | cached",
    "style_params": {
      "max_positions": 5,
      "cash_reserve": 0.05,
      "slot": 0.19,
      "position_cap": 0.35,
      "min_trade_size": 1
    },
    "signals": {
      "TICKER": { "signal": "BUY|HOLD|SELL", "confidence": "HIGH|MEDIUM|LOW" }
    },
    "allocation": {
      "TICKER": {
        "raw_weight": 0.0,
        "target_weight": 0.0,
        "target_shares": 0,
        "current_shares": 0,
        "delta": 0,
        "price": 0.0
      }
    },
    "trades_executed": [
      {
        "symbol": "TICKER",
        "side": "buy | sell",
        "quantity": 0,
        "status": "executed | pending | rejected",
        "message": "...",
        "fill_price": 0.0
      }
    ],
    "rejected_orders": [],
    "pre_snapshot": {
      "equity": 0.0,
      "cash": 0.0,
      "positions": {}
    },
    "post_snapshot": {
      "equity": 0.0,
      "cash": 0.0,
      "positions": {}
    },
    "equity_change": 0.0,
    "performance_log": "computer://C:\\Users\\ralph\\Documents\\Claude\\Projects\\trading-skills\\runs\\performance-<depot_id>.csv"
  }
}
```

---

## Step 9 — Chat output

Exactly one line:

```
portfolio-manager (<style>, <depot_id>): <N_buys> buys, <N_sells> sells, <N_holds> holds — equity $<post_equity> → [portfolio-manager-<depot_id>.json](computer://C:\Users\ralph\Documents\Claude\Projects\trading-skills\runs\<YYYYMMDD>\portfolio-manager-<depot_id>.json)
```

Do not print allocation tables, trade lists, or signal summaries. The JSON and CSV are the deliverables.
