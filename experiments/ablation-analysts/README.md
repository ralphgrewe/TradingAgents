# Ablation Study: Analyst Agents and Risk Stage

This directory contains the configuration for the ablation-study harness defined in issue #81.

## Overview

The ablation study measures the value-per-cost contribution of each agent (the four analysts and the risk-debate stage) by running leave-one-out variants and comparing their decisions and performance against a control baseline.

## Structure

- **`stocks.json`** — Shared ticker universe for all arms. Replace the placeholder with your own list of tickers to analyze.
- **Six arm config files** — Each defines one experiment variant:
  - `control.json` — Baseline (all analysts, risk debate enabled)
  - `no-market.json` — Ablates the market analyst
  - `no-social.json` — Ablates the social/sentiment analyst
  - `no-news.json` — Ablates the news analyst
  - `no-fundamentals.json` — Ablates the fundamentals analyst
  - `no-risk.json` — Bypasses the risk-debate stage

## Run Configuration

Each arm config file specifies:
- **`stocks_file`** — Path to the shared ticker list (relative to the config file's directory)
- **`portfolio`** — Enabled (true) for portfolio-mode trading
- **`style`** — "aggressive" (aligned portfolio mode)
- **`depot_id`** — Isolated simulated depot per arm
- **`memory_id`** — Isolated SQLite decision history (runs/memory/<id>/memory.db)
- **`report_dir`** — Isolated report output per arm
- **`config.memory_log_path`** — Isolated markdown memory log per arm
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

## Stability Block (Optional)

For repeat-stability analysis (KPI 5 in issue #81), a separate set of per-arm configs can be created with:
- Portfolio mode OFF (portfolio: false)
- Separate `memory_id` (e.g., "no-market-stability")
- Separate `report_dir` (e.g., "reports/stability/no-market")
- Small ticker list (3 tickers, e.g., `stability-stocks.json`)

Run the stability block with:
```bash
./venv/bin/python scripts/run_ablation.py --stability
```

## Isolation Guarantees

Each arm runs with isolated:
- **Decision history** — SQLite DB at `runs/memory/<memory_id>/memory.db`
- **Simulated depot** — Portfolio mode trades against a named depot
- **Report output** — Full-state logs in a separate directory tree
- **Markdown memory log** — Historical decisions and reflections

This isolation prevents cross-arm contamination in decision history, portfolio state, and performance metrics.

## KPI Report

After all arms complete, the KPI report script (issue #121) will:
1. Compare decision agreement/flip rate vs. control per ticker-day
2. Analyze 10-trading-day forward returns (from the memory DB resolution path)
3. Report depot performance per arm (equity, drawdown, daily path)
4. Measure token cost per run per arm
5. Assess repeat-run rating variance (stability block only)

## References

- **Issue #81** — Ablation-study harness design: arms, isolation, KPIs
- **Issue #115** — Run-config file support
- **Issue #118** — Configurable `selected_analysts`
- **Issue #119** — Configurable `risk_stage` (bypass risk debate)
