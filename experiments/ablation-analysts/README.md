# Ablation Study: Analyst Agents and Risk Stage

This directory contains the configuration for the ablation-study harness defined in issue #81.

## Overview

The ablation study measures the value-per-cost contribution of each agent (the four analysts and the risk-debate stage) by running leave-one-out variants and comparing their decisions and performance against a control baseline.

## Structure

- **`stocks.json`** — Shared ticker universe for all main arms. Replace the placeholder with your own list of tickers to analyze.
- **Six main arm config files** — Each defines one experiment variant:
  - `control.json` — Baseline (all analysts, risk debate enabled)
  - `no-market.json` — Ablates the market analyst
  - `no-social.json` — Ablates the social/sentiment analyst
  - `no-news.json` — Ablates the news analyst
  - `no-fundamentals.json` — Ablates the fundamentals analyst
  - `no-risk.json` — Bypasses the risk-debate stage
- **`stability-stocks.json`** — Small (3-ticker) placeholder ticker universe for the
  repeat-stability block. Replace with your own 3-ticker subset if desired.
- **Six stability-block config files** (`*.stability.json`) — One per main arm, same ablated
  setting, portfolio mode off, isolated memory/report paths. See "Stability Block" below.

## Run Configuration

Each arm config file specifies:
- **`stocks_file`** — Path to the shared ticker list (relative to the config file's directory)
- **`portfolio`** — Enabled (true) for portfolio-mode trading
- **`style`** — "aggressive" (aligned portfolio mode)
- **`depot_id`** — Isolated simulated depot per arm
- **`memory_id`** — Isolated SQLite decision history (runs/memory/<id>/memory.db)
- **`report_dir`** — Isolated report output per arm
- **`config.memory_log_path`** — Isolated markdown memory log per arm
- **`config.results_dir`** — Isolated full-state-log tree per arm (`runs/results/<arm>`).
  Added alongside #121's KPI report script: `results_dir` is a separate `DEFAULT_CONFIG` key
  from `report_dir` (`tradingagents/graph/trading_graph.py`'s `_log_state` writes
  `full_states_log_<date>.json` under `results_dir`, not `report_dir`) and defaults to the
  shared `~/.tradingagents/logs` if left unset — every arm setting its own `results_dir` is
  what actually isolates the per-arm full-state-log tree KPI 4 reads.
- **`config.selected_analysts`** (arms 2–5 only) — The ablated analyst list
- **`config.risk_stage`** (no-risk arm only) — Set to "none" to bypass risk debate

## Launching an Experiment

To run a single arm:
```bash
./venv/bin/python run_trading_agents.py experiments/ablation-analysts/control.json
```

To run all six arms sequentially (continues on failure, prints a summary):
```bash
./venv/bin/python scripts/run_ablation.py
```

Use `--arms-dir` to run arms from a custom location:
```bash
./venv/bin/python scripts/run_ablation.py --arms-dir /path/to/custom/arms
```

## Stability Block

For repeat-stability analysis (KPI 5 in issue #81), each of the six arms has a matching
`*.stability.json` config (`control.stability.json`, `no-market.stability.json`,
`no-social.stability.json`, `no-news.stability.json`, `no-fundamentals.stability.json`,
`no-risk.stability.json`), all six mirroring the same ablated setting as their main-arm
counterpart. Each stability config:

- Points at a separate, small `stability-stocks.json` (3 tickers, per #81's "3 tickers"
  cadence dimension) instead of the main `stocks.json`
- Has portfolio mode OFF (`"portfolio": false`, with no `depot_id`/`style` — no portfolio
  trades per #81)
- Has its own isolated `memory_id` (`<arm>-stability`) and `report_dir`
  (`reports/stability/<arm>`), distinct from both the main arm's and every other stability
  config's, so repeats don't contaminate the main arms' decision history or each other's
  history across the 3 repeats
- Carries the same `config.selected_analysts` / `config.risk_stage` override as its
  corresponding main arm, so the repeat is actually testing the same ablation under repeat
  conditions

Per #81's cadence ("3 days × 3 repeats × 3 tickers, per-ticker mode, no portfolio trades"),
the "3 tickers" dimension is `stability-stocks.json`, and the "no portfolio trades" dimension
is `portfolio: false`. `scripts/run_ablation.py` has no built-in `--repeats` flag or
scheduling — the "3 days × 3 repeats" dimension is realized simply by invoking the driver
repeatedly (e.g. 3 times on 3 different trading days, or 3 times in one sitting for a same-day
repeat check), each invocation appending fresh rows to the per-arm SQLite decision history at
`runs/memory/<arm>-stability/memory.db`. The KPI report script (#121) computes rating
variance across those repeated rows.

Run the stability block with:
```bash
./venv/bin/python scripts/run_ablation.py --stability
```

## Isolation Guarantees

Each arm runs with isolated:
- **Decision history** — SQLite DB at `runs/memory/<memory_id>/memory.db`
- **Simulated depot** — Portfolio mode trades against a named depot
- **Report output** — Per-ticker reports under `report_dir`
- **Full-state logs** — `full_states_log_<date>.json` under `config.results_dir`
  (`runs/results/<arm>`) — a *separate* tree from `report_dir` (see "Run Configuration" above)
- **Markdown memory log** — Historical decisions and reflections

This isolation prevents cross-arm contamination in decision history, portfolio state, and performance metrics.

## KPI Report

After arms have run (and, for KPI 2, after enough trading days have elapsed for
`resolve_pending` to fill in forward returns), generate the KPI report with:

```bash
./venv/bin/python scripts/ablation_report.py
```

`scripts/ablation_report.py` (issue #121) reads each arm's artifacts directly off disk
(no live MCP server required) and writes a markdown report (default:
`experiments/ablation-analysts/ablation_report.md`, override with `--out`):
1. Compares decision agreement/flip rate vs. control per ticker-day (exact and ±1 tier)
2. Pairs 10-trading-day forward returns on disagreement ticker-days only, separating
   resolved rows from pending ones
3. Reports depot performance per arm (final equity, max drawdown, daily equity path)
4. Measures token cost per run per arm (char/4 proxy over the full-state logs)
5. Assesses repeat-run rating variance per ticker from the stability-block memory DBs

Missing/partial artifacts (an arm that hasn't run yet, unresolved forward returns, a
missing depot DB) are reported as explicit gaps in the report rather than raising. See the
script's module docstring for the exact artifact access paths and the KPI 3 equity-curve
approximation it uses (no live price feed is available offline).

The keep/merge/drop call for each ablated agent is human judgment made on top of this
report — the script deliberately encodes no decision rule.

## References

- **Issue #81** — Ablation-study harness design: arms, isolation, KPIs
- **Issue #115** — Run-config file support
- **Issue #118** — Configurable `selected_analysts`
- **Issue #119** — Configurable `risk_stage` (bypass risk debate)
- **Issue #120** — Ablation arm run-config files + driver script
- **Issue #121** — KPI report script
