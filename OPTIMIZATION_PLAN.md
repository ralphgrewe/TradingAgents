# Trading Skills — Optimization Plan

Working spec for reducing compute/effort across the trading skills **without degrading trading
performance**. Each item is independently implementable. Tick boxes as completed.

Evidence base: 5 skills (`trader`, `fundamental`, `news`, `quant`, `portfolio-manager`),
the shared `SCHEMA.md`, and 30 ticker runs + 22 PDFs + 3 portfolio rebalances under `runs/`
(20260525, 20260529, 20260604, 20260605).

Guiding principle (from `LEARNINGS.md`): move **computation** into Python for stability and
lower token use, downshift the model only on mechanical stages, and **do not strip context
the reasoning stages need** — short prompts made the system unstable.

Legend — Impact: ⭐⭐⭐ high / ⭐⭐ medium / ⭐ low. Risk: how likely to affect decisions.

---

## A. Move deterministic work into Python scripts

Biggest stability + effort win. These stages are arithmetic/rule logic currently done by the
model, which is the main source of run-to-run variance.

- [x] **A1 — Trader scoring script** ⭐⭐⭐ · risk: none
  `skills/trader/score_trader.py` — reads 3 analyst JSONs from the run directory, performs
  Steps 2–5 (signal extraction, weighted scoring, conflict checks, confidence derivation),
  prints JSON with `signals[]`, `composite_score`, `analyst_aggregates`, `conflicts`,
  `signal`, `confidence`. SKILL.md Steps 2–5 replaced with a single `python score_trader.py`
  shell call; model writes `rationale` + `risk_note` only. Smoke-tested against AAPL
  20260605 — all 9 signal rows and final signal/confidence match exactly.

- [x] **A2 — Fundamental ratio script** ⭐⭐⭐ · risk: low
  `skills/fundamental/compute_ratios.py` — accepts 4–5 raw yfinance JSON files (ticker_info,
  income_annual, balance_annual, cashflow_annual, holders), computes all valuation /
  profitability / balance-sheet / cashflow ratios, derives insider sentiment from net shares,
  prints the full `details` payload (minus forecast). SKILL.md Step 1 updated: fetch raw
  files → run script → model adds forecast + value/growth judgment only.

- [x] **A3 — Extend quant script to full interpretation** ⭐⭐ · risk: low
  `skills/quant/compute_indicators.py` — replaces the old inline Step 2 snippet.
  Computes all indicators, derives per-indicator `trend` and `signal`, builds `convergence`
  and `trade_setup` (stop = close ± 1.5 × ATR, take_profit, risk_reward), determines
  `market_regime`, and emits the complete envelope `signal`/`confidence`/`summary`/`details`.
  Model role: optionally refine the one-line summary. SKILL.md Steps 2–3 collapsed.

- [x] **A4 — Portfolio-manager allocation script** ⭐⭐⭐ · risk: none
  `skills/portfolio-manager/rebalance.py` — takes a `params.json` with signals, prices,
  snapshots, and style; computes raw weights → HOLD handling → normalise → clamp →
  target shares → deltas → min-trade filter; appends the performance CSV; assembles and
  prints the full envelope JSON. SKILL.md Steps 4/7/8 replaced with two script invocations
  (pre- and post-trades). Smoke-tested against mag7-aggressive 20260605 — all 7 ticker
  deltas match exactly.

---

## A-fix — Script paths must use the Linux sandbox mount

- [x] **A-fix — Wrong path style triggered "use my computer"** ⭐⭐⭐ · risk: none
  The A-scripts were invoked in `SKILL.md` with Windows paths (`python "C:\Users\ralph\...\
  score_trader.py"`). The shell is a **Linux sandbox** where `C:\...` doesn't exist (the project
  is mounted at `/sessions/<id>/mnt/trading-skills`), so the call failed — and the model then
  reached for the computer-use tool ("make use of my computer") to run Python on the real
  machine, which stalled on the permission dialog. None of the skills need desktop control.
  *Fixed:* all 5 invocation blocks (trader, fundamental, quant, portfolio-manager ×2) now `cd`
  into the Linux mount and run with relative paths, plus an explicit "never pass `C:\` paths to
  bash / never fall back to computer-use" note. Verified the script runs via the mount path.

## B. Drop unused outputs & fetches

Confirmed against actual run files — none of these feed the decision, so performance is
unaffected.

- [ ] **B1 — Stop fetching quarterly fundamentals** ⭐⭐ · risk: none
  `quarterly` populated in **0 / 30** runs and unused by trader. Remove the quarterly
  `yfinance_get_financials` calls and the `quarterly` block from the fundamental payload.

- [ ] **B2 — Drop insider transaction list** ⭐ · risk: none
  `insider_transactions` populated in **0 / 30** runs; trader uses only the derived
  `insider_sentiment` (F3). Keep `insider_sentiment`; drop the `yfinance_get_holders` fetch
  and the `insider_transactions[]` array. *(Decide: can sentiment be derived without the
  holders call? If not, keep the call but discard the list.)*

- [ ] **B3 — Reduce annual history 5 → 2–3 years** ⭐ · risk: low
  Only ~2 years ever populated; trader uses only the latest. Lower the fetch/storage window.

- [ ] ~~**B4 — Drop news `top_*` item lists**~~ — **DROPPED.** Trader never consumes the
  `top_*` lists (it isn't in the json the trader reads, and the trader doesn't read the PDF —
  by design it uses only counts N3/N4 and ratings N1/N2). But since the PDF is being **kept**
  (C1 revised), those lists are the PDF's human-readable News section, so we keep generating
  them. Decision unaffected either way; no action.

- [ ] **B5 — Fix schema drift** ⭐ · risk: none
  - `articles_analyzed`: count `0` in `SCHEMA`/skill but emitted as array `[]` in output — pick one.
  - `top_positive_sentiment` (schema) vs `top_positive_sentiments` (output) — align the key.
  Clean up so downstream parsing is deterministic.

---

## C. PDF report

- [x] **C1 — Keep PDF, drop the redundant JSON appendix** ⭐⭐ · risk: none
  **Revised per Ralph:** keep the per-ticker PDF (useful for human debugging), but stop
  duplicating the JSON inside it. JSON lives only in the four separate `.json` files; the PDF
  carries only the pretty-printed summaries and tables.
  *Done:* removed the four "Appendix: raw envelope JSON in monospace" lines (sections 1–4) and
  the "Decision summary: envelope JSON" line from `trader/SKILL.md` Step 7; added an explicit
  "do not embed raw JSON in the PDF" rule and dropped the `Preformatted` JSON note from the
  reportlab fallback. PDF still generated every run.

---

## D. Model / effort routing (structural)

Skills invoked via the `Skill` tool run inline on the **session model** — you cannot set a
model in `SKILL.md` frontmatter. To assign per-task models, the orchestrators must dispatch
analysts as **subagents** with a `model` override.

- [ ] **D1 — Dispatch analysts as subagents with model overrides** ⭐⭐ · risk: medium · *implemented, pending validation*
  Restructure `trader` (and `portfolio-manager`) to spawn each analyst via the subagent
  mechanism with:
  | Stage | Model | Why |
  |---|---|---|
  | quant | Haiku | mechanical once A3 lands |
  | fundamental | Sonnet | judgment over scripted ratios |
  | news | Sonnet/Opus | hardest reasoning — keep strong |
  | trader rationale | Haiku | scoring is scripted (A1) |
  | portfolio orchestration | Haiku | math is scripted (A4) |
  *Validate:* re-run a known ticker set (e.g. mag7) before/after and diff signals — the
  decision must not move. This is the only item with non-trivial regression risk; do it last.

---

## Suggested order

1. **A1, A4** (scoring + allocation scripts) — kills variance, no signal change.
2. **C1** then **B4** (PDF off-demand unlocks dropping news text lists).
3. **A2, A3** (fundamental + quant scripts).
4. **B1, B2, B3, B5** (trim fetches/schema).
5. **D1** (model routing) — last, with a before/after signal diff as the gate.

## Validation gate (every item)

Re-run one representative ticker and the mag7 universe; diff `trader.json` `signal` +
`confidence` and the portfolio `allocation` against the pre-change run. Any change in a
decision must be explained by the edit, not by added noise.
