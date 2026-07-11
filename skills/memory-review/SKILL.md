---
name: memory-review
description: "Call this skill to periodically review the shared trading-memory core — pull hit-rate/calibration statistics, surface systematic per-agent/per-ticker patterns, reason about why they happen, and distill/update each agent's curated heuristics.md. Triggers: 'run a memory review', 'review the trading memory', 'update the heuristics', 'what has the memory core learned', 'derive findings from memory stats'."
---

# Memory Review

Periodic review of the shared SQLite memory core (`tradingagents/memory/`, see `CLAUDE.md`
"Persistence") that turns raw decision history into a small number of curated, ExpeL-style
heuristics ([Zhao et al., 2023 — *ExpeL: LLM Agents Are Experiential Learners*](https://arxiv.org/abs/2308.10144))
per agent. This is the step that turns "remembers individual lessons" (the per-decision
`resolve_pending` lesson already written for each row) into genuine closed-loop
self-improvement: instead of re-reading dozens of past decisions every run, an agent's future
runs read a handful of durable, already-reasoned-about heuristics.

**Manually invoked for now** — there is no scheduler wired up yet (scheduling is Memory #13, a
separate future issue). Run this skill by hand whenever you want a fresh look at how the memory
core's agents are doing.

**Scope**: this skill reviews and writes `heuristics.md` files. It does **not** itself change
what any analyst/trader skill does at run time — no consumer skill reads these files yet (see
Step 5, "Future wiring", for what that would take). It also does not produce a per-ticker JSON
envelope (unlike the analyst/trader skills) — there is no `<TICKER>`/`<date>` scope here, the
review is over the whole memory core (optionally filtered).

> **Execution environment.** Unlike the other `skills/` (`quant`, `fundamental`, `news`,
> `trader`, `portfolio-manager`), this skill does **not** run against the Windows-mounted
> `trading-skills` Cowork project — it runs from *this* repository, against its own shared
> memory core. Run all shell commands from this repo's root using the project virtualenv, per
> `CLAUDE.md`: `./venv/bin/python ...`, never system Python and never a `C:\...` path.
> `find_patterns.py` reaches the memory core the same way `trading_graph.py` does (issue #54) —
> over the networked memory MCP server (`memory_get_statistics`/`memory_get_decisions`, via
> `MemoryMCPClient`), not by opening the SQLite DB in-process — so the memory MCP server must be
> running first (`./start_server.sh`; see `CLAUDE.md` "Commands").

---

## Inputs (all optional)

| Parameter | Description | Default |
|---|---|---|
| `agent` | Restrict the review to one agent id (e.g. `fundamental`, `quant-indicator-analyst`) | all agents |
| `ticker` | Restrict to one exact ticker (no normalization — matches the stored column verbatim) | all tickers |
| `since` | Inclusive lower bound on `decision_date` (`YYYY-MM-DD`) | all history |

If the user doesn't specify any of these, review everything.

---

## Step 1 — Pull the hard numbers

Call the `memory_get_statistics` MCP tool (agent/ticker/since from Inputs, all optional) to get
hit-rates per agent/ticker and calibration-by-confidence. This is the factual backbone — read it,
but don't stop here; a raw stats table is not a finding.

---

## Step 2 — Detect divergent patterns (run script)

Run `find_patterns.py` to flag which `(agent, ticker)` cells in that statistics table diverge
sharply from that same agent's performance on its *other* tickers — the shape called out in the
issue this skill implements: *"fundamental was wrong 7/9 times on TSLA — the misses cluster
around earnings surprises the ratios can't see."* `find_patterns.py` finds the "wrong 7/9 times
on TSLA, right elsewhere" half mechanically; it does not (and cannot) find the "misses cluster
around earnings surprises" half — that's Step 3.

```bash
cd <repo-root>
./venv/bin/python skills/memory-review/find_patterns.py \
  --json \
  [--agent <agent>] [--ticker <ticker>] [--since <YYYY-MM-DD>]
```

This prints `{"findings": [...], "context": {...}}`:
- `findings`: one entry per flagged `(agent, ticker)` cell — `n`, `hit_rate`, `baseline_hit_rate`
  (that agent's hit-rate on its *other* tickers), `baseline_n`, `delta`, `direction`
  (`underperforms` / `overperforms`). A cell is only flagged once it has at least 3 resolved
  decisions, its baseline also has at least 3, and the gap is at least 30 percentage points — see
  `find_patterns.py`'s module docstring for the exact rationale and how to override those knobs
  (`--min-n`, `--divergence`) if a quick look with looser/tighter bars is useful.
- `context`: for every flagged `(agent, ticker)`, up to 10 of its most recent resolved decision
  rows (`decision_date`, `signal`, `confidence`, `key_drivers`, `thesis`, `lesson`,
  `forward_return`, `correct`) — the raw material for Step 3.

If `findings` is empty, say so in the final chat output (Step 6) and stop — there is nothing to
distill this run.

---

## Step 3 — Reason about *why* (this is the part the script cannot do)

For each finding, read its `context` rows — specifically the misses (`"correct": false`) versus
the hits. Look at `key_drivers` and `thesis` for a common thread across the misses that isn't
present in the hits: a recurring event type the underlying data can't see (earnings surprises,
guidance changes, litigation, a regime shift), a data-quality gap, a confidence miscalibration
(cross-check against the `calibration` section from Step 1 — is this agent's HIGH-confidence
bucket actually more accurate than its LOW bucket for this pattern, or is confidence not tracking
correctness at all?), or a structural reason a given signal type doesn't apply to this ticker
(e.g. a fundamentals-only agent has no way to see a sentiment-driven meme-stock move). Write one
to three sentences per finding explaining the *why*, not just restating the *what* the script
already printed.

If the misses show no discernible common thread, say so plainly rather than forcing a
just-so explanation — a weak or absent finding is more useful to the next reader than a
fabricated one.

---

## Step 4 — Curate `heuristics.md` per agent

For every agent with at least one finding:

1. Determine that agent's heuristics file path: `runs/memory/heuristics/<agent>.md`, or
   `$TRADINGAGENTS_HEURISTICS_DIR/<agent>.md` if that env var is set (mirrors
   `TRADINGAGENTS_MEMORY_DB_PATH`'s precedence — see `find_patterns.py`'s
   `resolve_heuristics_dir`). If the file already exists, `Read` it first — an update should
   preserve any still-relevant prior heuristics, not blindly clobber accumulated understanding.
2. For each of this agent's findings, write **one short, concrete heuristic bullet** grounded in
   the Step 3 reasoning — not the raw statistic. Bad: *"fundamental is 22% accurate on TSLA."*
   Good: *"On TSLA, weight fundamentals-driven BUY calls down (or require corroborating
   news/quant signal) heading into earnings — 7 of 9 past misses were earnings-surprise driven,
   which ratio-based analysis can't anticipate."*
3. Keep the file **short and curated** — this is the point of the ExpeL-style distillation, not a
   log. As a rule of thumb, cap each agent's file at roughly 8-10 bullets; when adding a new one
   would exceed that, either merge it with a closely related existing heuristic or drop the
   weakest/stalest one (e.g. one whose pattern no longer shows up in the latest statistics).
4. Write the result with the `Write` tool (or `Edit` if only touching a few lines). Suggested
   shape — not a rigid schema, just keep it scannable:

   ```markdown
   # Heuristics: <agent>

   *Curated from tradingagents/memory statistics — see skills/memory-review/SKILL.md. Last
   reviewed: <YYYY-MM-DD>.*

   - <ticker/topic>: <one to three sentence curated heuristic, grounded in the "why">
   - ...
   ```

`find_patterns.py --write-draft` (adding that flag to the Step 2 command) will mechanically write
a starting-point file per flagged agent (one templated line per finding, no reasoning) to the
same path — useful as scaffolding for a first-ever review of an agent, but always replace its
templated wording with the real Step 3 reasoning before treating the file as done; the template
line explicitly is not a substitute for it.

---

## Step 5 — Future wiring (do not implement yet)

Once this exists, analyst/trader skills should inject the relevant agent's `heuristics.md` into
their own prompt context every run — e.g. a new step in `skills/quant/SKILL.md` /
`skills/fundamental/SKILL.md` / `skills/trader/SKILL.md` analogous to their existing "Step 0 —
Memory: resolve pending + load past context" (see `skills/quant/SKILL.md`), reading
`heuristics/<agent-id>.md` right alongside the `memory_get_past_context` call and folding it into
the same informational (never signal/confidence-altering) context those skills already treat
`past_context` as. Two things that future issue will need to resolve and this one deliberately
does not:

- **Cross-environment file access.** This skill writes heuristics files into *this* repo's own
  filesystem (`runs/memory/heuristics/`), but `quant`/`fundamental`/`trader` run against the
  Windows-mounted `trading-skills` Cowork project and only touch the shared memory core through
  MCP tool calls (`memory_store_decision` etc.), never by reading this repo's files directly. A
  consumer skill will need either a new `memory_get_heuristics`-style MCP tool (mirroring
  `memory_get_past_context`'s pass-through shape) or a documented shared mount, not a bare
  `Read` of a path only this repo's sandbox can see.
- **Legacy graph parity.** The legacy LangGraph pipeline (`tradingagents/graph/`) has its own
  prompt-injection point (`get_past_context` -> Portfolio Manager prompt); if heuristics.md should
  also reach it, that's a `trading_graph.py` change, not a `skills/` change.

---

## Step 6 — Chat output

No JSON envelope, no per-run artifact file beyond the `heuristics.md` updates themselves. Report
back in chat, concisely:

- Which agents/tickers were reviewed (filters used, if any).
- How many findings were surfaced, and for each: the one-line pattern (agent, ticker, direction,
  magnitude) plus the one-line *why*.
- Which agents' `heuristics.md` were updated (path + how many bullets now).
- If `findings` was empty: say so and stop; do not touch any heuristics file.

Do not dump the full statistics table or the raw `context` rows to chat — those are the working
material for Steps 1-4, not the deliverable.
