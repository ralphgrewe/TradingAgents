# Trading Skills — Optimization Plan 2

Follow-up to `OPTIMIZATION_PLAN.md`. These items came out of the post-optimization review of the
mag7-aggressive and europe01 portfolio runs (before ≤ `20260605`, after `20260606`–`20260607`).
They are written to be implemented directly with **Claude Code**: each item names the exact files,
the concrete change, and a verification command/gate.

Guiding principle (unchanged, from `LEARNINGS.md`): keep deterministic work in Python, keep the
reasoning stages (fundamental, news) on a strong model, and **never strip context the reasoning
stages need**. Every item below is decision-neutral — re-run the validation gate and confirm the
`signal`/`confidence`/`allocation` do not move except where explicitly intended.

Legend — Impact: ⭐⭐⭐ high / ⭐⭐ medium / ⭐ low. Risk: likelihood of affecting a decision.

Paths are relative to the project root (`C:\Users\ralph\Documents\Claude\Projects\trading-skills`,
mounted in the Linux sandbox at `/sessions/<id>/mnt/trading-skills`). Never pass `C:\...` paths to
bash.

---

## P1. Remove the fragile `pandas-ta` dependency from the quant skill ⭐⭐⭐ · risk: none

**Problem.** `TradingAgents/skills/quant/compute_indicators.py` hard-installs and imports
`pandas-ta`. Recent `pandas-ta` requires Python ≥ 3.12; the sandbox runs 3.10, so
`pip install pandas-ta` fails and the script crashes. The `20260607/GOOGL` run shows a manual
work-around (`fetch_and_analyze.py`, `quant_output.json`) — i.e. the skill was bypassed by hand.

**Fix.** A dependency-free calculator already exists at repo root
(`compute_indicators_standalone.py`, imports only `pandas`/`numpy`) and produces **identical**
output (verified: GOOGL 20260607 signal, confidence, and all 7 indicator values match the saved
file exactly). Promote it into the skill:

- [ ] Replace the body of `TradingAgents/skills/quant/compute_indicators.py` with the logic from
  `compute_indicators_standalone.py` (or have the skill script import/exec the standalone), so the
  skill computes SMA-50, MACDH, RSI, Bollinger bands, ATR, VWMA in pure pandas/numpy.
- [ ] Remove the `pip install ... pandas-ta` subprocess block and the `import pandas_ta as ta` line.
  Keep the `pandas`/`numpy` install guard (the standalone's `try/except ImportError`).
- [ ] Delete the now-redundant root copies once merged: `compute_indicators_standalone.py`,
  `runs/20260607/GOOGL/fetch_and_analyze.py`, `runs/20260607/GOOGL/quant_output.json`
  (confirm with Ralph before deleting run artifacts).

**Verify.**
```bash
cd <trading-skills-mount>
python TradingAgents/skills/quant/compute_indicators.py runs/20260607/GOOGL/raw_price_history.json GOOGL > /tmp/q.json
python - <<'PY'
import json
n=json.load(open('/tmp/q.json')); s=json.load(open('runs/20260607/GOOGL/quant-indicator-analyst.json'))
assert n['signal']==s['signal']
assert {i['indicator']:round(i['value'],2) for i in n['details']['indicators'] if i['value']} \
    == {i['indicator']:round(i['value'],2) for i in s['details']['indicators'] if i['value']}
print('OK — quant reproduces saved output with no pandas-ta')
PY
```
Run the same on TSLA and one EU ticker (e.g. `runs/20260606/A0LR9`) to confirm.

---

## P2. Remove the performance-log CSV entirely ⭐⭐ · risk: none

**Rationale (per Ralph).** The CSV duplicates state the trading-simulation MCP already holds — the
equity/cash time series is reconstructable from `get_portfolio` + `get_trades`. The log also caused
the EU re-run crash (`runs/20260606/rebalance_params.json` hardcoded an absolute, session-specific
path → `PermissionError`) and has **drifted schemas** between depots (mag7: 10 columns incl. `note`;
europe: 5 columns). Drop it rather than patch the path. The only thing lost is the free-text `note`
column (e.g. "orders pending — market closed") — acceptable; capture such context in the run’s
envelope `summary` if needed.

**Fix.**
- [ ] In `TradingAgents/skills/portfolio-manager/rebalance.py`: remove the Step 7 performance-log
  block (the `if perf_log:` append, the `performance_log_path` param read, and the
  `performance_log` field in the envelope output — set it to `null` or drop the key).
- [ ] In `TradingAgents/skills/portfolio-manager/SKILL.md`: delete the `performance_log_path` line
  from the Step 4 `params.json` build, retitle "Steps 7–8 — Performance log + envelope" to just
  "envelope", and remove the `performance_log` field from the Step 7–8 envelope schema.
- [ ] Leave the existing `runs/performance-*.csv` files in place as historical artifacts (or archive
  them) — confirm with Ralph before deleting.

**Verify.**
```bash
cd <trading-skills-mount>
python TradingAgents/skills/portfolio-manager/rebalance.py runs/20260607/rebalance_params.json > /tmp/r.json
python - <<'PY'
import json
d=json.load(open('/tmp/r.json'))
a={k:v['target_shares'] for k,v in d['details']['allocation'].items()}
s={k:v['target_shares'] for k,v in json.load(open('runs/20260607/portfolio-manager-mag7-aggressive-test.json'))['details']['allocation'].items()}
assert a==s, 'allocation changed!'
assert 'performance' not in json.dumps(d).lower() or d['details'].get('performance_log') in (None,''), 'perf log still emitted'
print('OK — allocation unchanged, no CSV written')
PY
```

---

## P3. Finish the fetch/payload trims (B1, B2, B3) ⭐ · risk: none

Token savings only; none of these feed the trader decision (confirmed: trader reads derived
`insider_sentiment`, latest annual only, and never `quarterly`).

- [ ] **B1 — drop `quarterly`.** Already empty in every after-run. Remove the `quarterly` block
  from the `details` payload schema in `fundamental/SKILL.md` and stop emitting it in
  `compute_ratios.py`.
- [ ] **B2 — drop the `insider_transactions[]` list.** Keep the derived `insider_sentiment`
  (the trader's only consumer). In `compute_ratios.py`, compute `insider_sentiment` from the
  holders data but do **not** emit the per-transaction array; remove `insider_transactions` from
  the `fundamental/SKILL.md` payload. (If sentiment can't be derived without
  `yfinance_get_holders`, keep the fetch but discard the list.)
- [ ] **B3 — cap annual history at 3 years.** In `compute_ratios.py` change `annual_dates =
  all_dates[:5]` to `[:3]`. Trader uses only the latest year; 3 gives a small trend buffer.

**Verify.** Re-run `compute_ratios.py` on `runs/20260607/AAPL/raw_*.json`; confirm output has no
`quarterly`/`insider_transactions` keys, ≤ 3 annual years, and `insider_sentiment` still present.
Then re-run `score_trader.py runs/20260607/AAPL` and confirm `signal`/`composite_score` are
unchanged from the saved `trader.json` (BUY, 1.5068).

---

## P4. Dispatch the three analysts in parallel ⭐⭐ · risk: low

**Problem.** `trader/SKILL.md` Step 0 dispatches fundamental → news → quant **sequentially**
("wait for each to complete before starting the next"). They are independent — each writes its own
JSON and none reads another's output — so the wall-clock is ~3× longer than necessary.

**Fix.**
- [ ] In `trader/SKILL.md` Step 0, change the instruction to dispatch all three analyst subagents
  **in a single batch** (parallel) and then wait for all three files to exist before Step 1. Keep
  the per-skill model overrides (fundamental=Sonnet, news=Sonnet, quant=Haiku).
- [ ] Keep the existing "verify each JSON exists, re-invoke once on failure" guard, applied after
  the batch returns.

**Risk note.** Parallel dispatch is wall-clock/UX only; outputs are independent so the decision is
unaffected. Sequential was likely chosen for stability — validate with the gate below before
keeping.

**Verify.** Re-run the trader on a known ticker (e.g. AAPL `20260607`) and diff `signal` +
`confidence` against the saved run — must be identical.

---

## Suggested order

1. **P1** (quant dependency) — unblocks reproducible quant runs; highest robustness win.
2. **P2** (remove performance log) — drops simulator-redundant CSV; restores EU re-runs.
3. **P3** (B1/B2/B3 trims) — mechanical, decision-neutral token savings.
4. **P4** (parallel analysts) — wall-clock win, validate decision is unchanged.

## Validation gate (every item)

Re-run one representative ticker **and** the mag7 universe; diff `trader.json` `signal` +
`confidence` and the portfolio `allocation` (target shares) against the pre-change run. The four
scripts are deterministic, so any change must be explained by the edit, not by noise. P1/P3 must
reproduce saved outputs exactly; P4 must not move any decision.
