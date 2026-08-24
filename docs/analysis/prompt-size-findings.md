# Prompt-size findings: PM prompt composition, risk-debate/analyst-report segments, memory-growth trajectory

Status: measurement report (2026-08-24), issue #144, part of the #142 tracking issue. Produced with
`scripts/analyze_llm_calls.py` (#143, commit `17c6d37`) against the `reports/` corpus checked into this
working tree at the time of writing. Every number below is reproducible with the commands in
"Reproducibility".

## Reproducibility

Corpus: **33 run directories, 521 LLM calls**, all with prompt dumps (`llm_call_log_prompts=true` was on
for every run counted here). 19 additional directories under `reports/` were skipped (empty/partial —
either only a `prompts/` subdirectory with no `llm_calls.jsonl`, or duplicate re-runs from the same batch
that hadn't finished writing their log at measurement time); the skip count is reported by the tool itself,
not hand-filtered.

- **Trade dates covered**: 2026-08-22 and 2026-08-23.
- **Run wall-clock range**: 2026-08-22T17:50:57Z to 2026-08-23T07:23:17Z.
- **Model mix**: `ministral-3:3b` (359 calls), `ministral-3:8b` (162 calls), served locally via Ollama.
- **Measurement run date**: 2026-08-24.

Commands used (from the repo root, `./venv/bin/python`):

```bash
# Corpus summary + per-agent ranking (critical-prompt ranking table)
./venv/bin/python scripts/analyze_llm_calls.py --reports-dir reports --format json \
  --sort-by estimated_prompt_tokens_p95 --top 10

# Same data, human-readable
./venv/bin/python scripts/analyze_llm_calls.py --reports-dir reports --format text \
  --sort-by duration_total --top 10

# Truncation-candidate surfacing (cited, not diagnosed — see "Token-accuracy caveat" below)
./venv/bin/python scripts/analyze_llm_calls.py --reports-dir reports --format json \
  --truncation-threshold 0.8 --top 1000

# Segment composition — must cover all 96 Portfolio Manager calls, not just the
# largest few, so --top must be at least the full corpus size (521) and the
# Portfolio Manager rows filtered out of the result afterwards.
./venv/bin/python scripts/analyze_llm_calls.py --reports-dir reports --segment \
  --format json --top 521
```

`--segment --top 96` alone is **not** sufficient to get all 96 Portfolio Manager prompts: segment mode
ranks the globally largest prompts across *all* agents, and 5 Neutral Analyst prompts are large enough to
displace 5 Portfolio Manager ones from a top-96 cut. Every table below that claims "all 96 PM calls" was
built from the `--top 521` run, filtered to `agent == "Portfolio Manager"` in post-processing.

## Critical-prompt ranking

"Critical" is used here on two independent criteria — a prompt can be critical because it's the largest
input, or because it dominates wall-clock cost — and they don't always agree:

| Agent | Calls | Mean est. tok | P95 est. tok | Max est. tok | Max reported `input_tokens` | Total duration (s) | Mean duration (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Portfolio Manager | 96 | 7,666 | 11,011 | 12,289 | 2,472 | 1,994.7 | 20.8 |
| Researcher | 66 | 4,706 | 5,012 | 5,116 | 2,051 | 1,794.7 | 27.2 |
| Trader | 63 | 3,214 | 3,453 | 3,724 | 4,094 | 977.1 | 15.5 |
| Neutral Analyst | 32 | 5,087 | 7,508 | 7,770 | 4,089 | 894.5 | 28.0 |
| Conservative Analyst | 33 | 3,559 | 5,467 | 5,685 | 4,089 | 620.6 | 18.8 |
| Aggressive Analyst | 33 | 2,907 | 3,201 | 3,792 | 4,091 | 448.0 | 13.6 |
| Sentiment Analyst | 33 | 3,382 | 5,215 | 5,520 | 3,899 | 428.6 | 13.0 |
| Fundamentals Analyst | 66 | 475 | 964 | 969 | 1,279 | 297.0 | 4.5 |
| News Analyst | 66 | 388 | 696 | 697 | 992 | 135.2 | 2.0 |
| Market Analyst | 33 | 180 | 182 | 183 | 194 | 53.2 | 1.6 |

**By prompt size**, the Portfolio Manager is unambiguously the critical agent: highest mean, P95, and max
estimated prompt tokens of any agent, by a wide margin over the second-place Neutral Analyst.

**By total wall time**, the Portfolio Manager is again first (1,994.7s of the corpus's 7,643.5s total —
26%), but the Researcher is close behind (1,794.7s, 23%) despite having a much smaller prompt (mean 4,706
vs. 7,666 tokens) — its 66 calls and long per-call generation (mean 27.2s, driven by synthesis output
length, not input size) make it the second time sink on a different axis than prompt size. The Trader
(977.1s) and the three risk analysts combined (aggressive + conservative + neutral: 1,963.1s) round out
the next tier.

**Conclusion**: the Portfolio Manager is the critical prompt on both criteria and is this report's primary
subject. The risk-analyst trio, taken together, spends more wall time than the Researcher and is where the
history the Portfolio Manager later re-reads (see "Segment 1" below) is generated.

## PM prompt composition (all 96 calls, corpus-wide)

Each Portfolio Manager call's **first message** (the constructed prompt; a second message — the model's
own prior turn in a tool-calling round — is present on 64/96 calls and handled separately below) was split
at the five anchor lines `scripts/analyze_llm_calls.py` defines (`SEGMENT_ANCHORS`,
`scripts/analyze_llm_calls.py:54`): `Past analyses of `, `Recent cross-ticker lessons:`,
`Analyst Reports:`, `**Risk Analysts Debate History:**`, `**Available Tools:**`.

First-message total: **median 26,192 chars** (mean 27,617; min 16,667; max 43,669 — the 43,669-char case is
the AAPL run discussed in "Memory-growth trajectory" below, a 4-entry memory outlier, not the median case).

| Segment | Median chars | Median share | Max share | Min share |
|---|---:|---:|---:|---:|
| Risk-analysts debate history | 8,938.5 | **34.4%** | 49.6% | 1.3% |
| Analyst reports | 8,543.5 | **32.0%** | 53.6% | 20.2% |
| Header + rating scale + research/trader plan | 3,960.0 | 15.4% | 29.5% | 8.7% |
| Past-context (memory) injection — combined | 3,573.5 | 14.3% | **49.4%** | — |
| — same-ticker sub-block | 850.0 | 3.4% | 43.3% | 1.8% |
| — cross-ticker sub-block | 2,754.0 | 10.6% | 16.5% | 6.1% |
| Available-tools note | 334.0 | 1.3% | 2.0% | 0.8% |

These figures reproduce the corpus-wide numbers already cited in issue #144's "What the data shows"
section exactly (verified independently rather than trusted), confirming that description is accurate for
this corpus. The memory row is reported both combined (as it appears in one contiguous prompt region: the
`Past analyses of <ticker>` heading immediately followed by `Recent cross-ticker lessons:`) and split into
its two constituent blocks, since they scale on entirely different axes (see "Memory-growth trajectory").

**The follow-up tool-response messages** — present when the Portfolio Manager's bounded wiki tool-calling
loop (`run_structured_with_tools`, gated by `knowledge_base_tool_max_rounds`, default 2 rounds) makes a
second LLM call in the same decision — add a **median 3,892 chars** (mean 3,049; max 7,085) on top of the
first message, on 64 of the 96 calls (67%). These are not part of the five-segment breakdown above (they
are a different message in the `chat_messages` list, not part of the constructed prompt), but they are real
input the model reprocesses on the second call, so they belong in the accounting: a Portfolio Manager call
that hits this path pays for its first-message prompt *and* ~3.9K more characters of prior tool-loop
content on the follow-up call.

**Caveat on the largest prompt**: the single largest prompt in the corpus (12,289 estimated tokens,
49,160 total dump chars including the follow-up message) is `reports/AAPL_2026-08-23_20260823_091531/`,
where memory injection is 49.4% of the first message — a `n_same=4` outlier, not representative of the
median (14.3%). All headline percentages above are medians for exactly this reason; do not quote the max
row as if it were typical.

## Segment 1: risk-analysts debate history (~34% of the PM prompt, the largest segment)

**Mechanism**: `risk_debate_state["history"]` (`tradingagents/agents/utils/agent_states.py`) is a single
accumulating string. Each of the three risk analysts (`tradingagents/agents/risk_mgmt/aggressive_debator.py`,
`conservative_debator.py`, `neutral_debator.py`) reads the current `history`, appends its own argument, and
writes back `history + "\n" + argument` — **plain string concatenation, never trimmed or summarized**. Every
analyst's turn embeds every prior turn's full text verbatim, so `history`'s length grows monotonically
across the debate with no cap.

**Round accounting**: `ConditionalLogic.__init__` (`tradingagents/graph/conditional_logic.py:9`) defaults
`max_risk_discuss_rounds=1`; the debate terminates when `risk_debate_state["count"] >= 3 *
max_risk_discuss_rounds` (`conditional_logic.py:50`) — 3 speaker turns (one full round: aggressive →
conservative → neutral) at the default setting. The Portfolio Manager receives this fully-accumulated
`history` string verbatim via `risk_debate_history_section` in
`tradingagents/agents/managers/portfolio_manager.py` (assembled a few lines above the `prompt = f"""..."""`
f-string, gated on `is_present_text(history)`). At the default round count, the corpus's median 8,938.5-char
segment is the concatenation of exactly 3 arguments (~2,980 chars each on average) — increasing
`max_risk_discuss_rounds` would multiply this segment directly, since each additional round adds 3 more
full-length arguments on top of what's already there.

**Cited example**: `reports/AAPL_2026-08-23_20260823_091531/prompts/01a02d7d-1089-7622-bf52-1cb91084e18d.json`,
first message, chars `[34332:43335]` (9,003 chars, this call's instance of the segment), opening:

> **Risk Analysts Debate History:**
>
> Aggressive Analyst: Alright, listen up—this *is* the moment you either sit back or step into the fight
> for AAPL's potential upside. The market's whispering caution, the "SELL with low confidence" flag from
> the analyst, and all that noise about weak momentum isn't just overvalued fea[r...]

## Segment 2: analyst reports (~32% of the PM prompt)

**Mechanism**: `format_analyst_reports_section` (`tradingagents/agents/utils/agent_utils.py`) renders the
market, sentiment, news, and fundamentals JSON envelopes (plus optional macro envelopes when those
analysts are selected) into one text block. The Portfolio Manager calls this directly
(`portfolio_manager.py`) and embeds the result as `reports_line` in its prompt — the `Analyst Reports:`
anchor.

**Fan-out — the same content is paid for multiple times per run**: `format_analyst_reports_section` is also
called from `tradingagents/agents/trader/trader.py` (Trader prompt) and
`tradingagents/agents/trader/swing_trader.py`. Separately, the three risk analysts
(`aggressive_debator.py`, `conservative_debator.py`, `neutral_debator.py`) each read
`state["market_report"]`, `state["sentiment_report"]`, `state["news_report"]`, `state["fundamentals_report"]`
directly and interpolate all four verbatim into their own prompts (not via the shared formatter, but the
same underlying report text). So the same four analyst-report envelopes, generated once by the analyst
team, are paid for again at minimum **five more times per run**: once each for the Trader, the three risk
analysts, and the Portfolio Manager. With `research_stage="researcher"` (the default), the Researcher also
consumes the analyst envelopes as its synthesis input — a sixth consumer, though its prompt wasn't
segmented here since the anchor set is PM-specific.

**Cited example**: same dump as above, chars `[25499:34332]` (8,833 chars for this call), opening:

> Analyst Reports:
> Market research report (JSON envelope): {
>   "skill": "market-analyst",
>   "ticker": "AAPL",
>   "date": "2026-08-23",
>   "signal": "SELL",
>   "confidence": "LOW",
>   "summary": "**Sell with caution amid light momentum shift: Weak balance (3[...]"

## Memory-growth trajectory

This is the forward-looking finding and the reason memory injection stays in scope for follow-up work
despite ranking fourth (14.3% median) today: it is the **only segment with no size cap of any kind**, and
it is currently observed at less than a quarter of the way to its own steady state in this corpus.

**Memory injection is Portfolio-Manager-only.** Checked directly against all 425 non-PM prompt dumps in the
corpus (every Researcher, Trader, analyst, and risk-analyst first message): **zero** contain a `Past
analyses of` or `Recent cross-ticker lessons:` block. This matches the code: `trading_graph.py:457` calls
`self.memory_log.get_past_context(company_name)` unconditionally and passes it into `past_context`, which
only `portfolio_manager.py` reads. The Researcher, Trader, all four analysts, and all three risk analysts
receive no injected memory of any kind — a change to how memory is injected or budgeted affects one agent
in the pipeline, not the other nine.

**Producing code**: `TradingMemoryLog.get_past_context` (`tradingagents/agents/utils/memory.py:70`) collects
up to `n_same=5` most-recent same-ticker entries (formatted **in full** via `_format_full`,
`memory.py:283` — tag line + entire `DECISION:` text + entire `REFLECTION:` text, no truncation) and up to
`n_cross=3` most-recent other-ticker entries (formatted as a one-line reflection excerpt via
`_format_reflection_only`, `memory.py:293` — capped at 300 chars of decision text if no reflection exists,
but *not* capped when a reflection exists). **No character or token budget exists on either path.** The
SQLite-core equivalent, `tradingagents/memory/query.py:142` (used by the swing trader and macro-agent
memory paths, not by the Portfolio Manager in this corpus, since those features are off by default), is
structurally cheaper per entry — `_format_same_ticker_entry` (`query.py:126`) renders one markdown bullet
(date, signal, confidence, lesson) rather than the full decision+reflection prose `_format_full` embeds —
but it likewise has no `n_same`/`n_cross`-independent size cap.

**Observed scaling** (same-ticker sub-block only, one value per run; 33 runs, deduplicated by ticker+run):

| Entries observed (`n_same` so far) | Runs | Same-ticker block size (chars) |
|---|---:|---|
| 1 | 30 tickers | 558 – 1,555 |
| 4 | 3 tickers (AAPL, MSFT, NVDA) | 15,982 – 18,922 |

The cross-ticker sub-block is already saturated at `n_cross=3` across the whole corpus (2,640 – 2,754
chars, median 2,754, essentially flat) since every run already has ≥3 prior decisions on *some* other
ticker to draw from — it will not grow further regardless of how much longer the memory log runs.

**Projected steady state**: fitting a line through the two observed data points (n=1 → n=4) gives a
per-additional-entry cost of roughly 5,100 – 5,800 chars (low end anchored to the cheapest observed
entries, high end to the priciest). This is a two-point linear extrapolation from real per-entry decision
text whose length varies run to run, so treat the range as directional, not exact:

- At `n_same=5` (the configured cap — every ticker's memory eventually saturates here, same as
  `n_cross` already has), the same-ticker block alone projects to roughly **21,100 – 24,700 chars**.
- Adding the already-saturated ~2,754-char cross-ticker block: a steady-state memory injection of roughly
  **23,850 – 27,450 chars** per Portfolio Manager call, for any ticker that has accumulated 5 resolved
  decisions.
- **Where it overtakes today's largest segments**: the same-ticker block alone already crosses the
  analyst-reports median (8,543 chars) and the risk-debate median (8,938 chars) individually somewhere
  around `n_same=3` on this projection — before any ticker in the current corpus has even reached that
  count. By `n_same=4` (already observed for 3 tickers), the same-ticker block (15,982 – 18,922 chars) is
  comparable to the risk-debate and analyst-report medians *combined* (17,481.5 chars) — AAPL's 18,922-char
  block already exceeds that combined figure. At the `n_same=5` steady state, the projected memory block
  (23,850 – 27,450 chars total) would exceed the current combined risk-debate + analyst-report total by
  roughly 35–55%, making memory injection the single largest segment of the Portfolio Manager prompt once
  a ticker has been analyzed 5 times — a threshold this corpus will cross for its most-frequently-run
  tickers well before most other tickers reach even a second entry.

**Cited examples spanning the range**:

- `reports/AAPL_2026-08-23_20260823_091531/` — 4 same-ticker entries, 18,922-char same-ticker block, 49.4%
  of that call's first message. Opening of the block (chars `[3908:4200]` of
  `prompts/01a02d7d-1089-7622-bf52-1cb91084e18d.json`):

  > Past analyses of AAPL (most recent first):
  >
  > [2026-08-15 | AAPL | Hold | +1.4% | +2.2% | 4d]
  >
  > DECISION:
  > ---
  > Need refinements? Narrow focus by telling me:
  > - How long have you been positioned?
  > - Do you expect more growth volatility (beta) or debt stabilization?
  > ---
  >
  > REFLECTION:
  > The directional [...]

- `reports/APLD_2026-08-23_20260823_075709/` — 1 same-ticker entry, 558-char same-ticker block, the
  contrast case. Opening of the block (chars `[3592:3950]` of
  `prompts/01a02d34-e9e6-7421-9231-61ccdffce468.json`):

  > Past analyses of APLD (most recent first):
  >
  > [2026-08-15 | APLD | Hold | -11.8% | -10.9% | 4d]
  >
  > DECISION:
  > ---
  >
  > REFLECTION:
  > The directional short was slightly correct at 4% against the underlying futures move of $45 (SPY was
  > +6%).
  >
  > The thesis failed on leverage exhaustion in late-March when margin calls forced aggressive unwind—but
  > the macro divergence (risi[...]

## Token-accuracy caveat

Every token figure in this report (mean/median/P95/max estimated prompt tokens, the chars/4-derived
columns) rests on `prompt_tokens_estimated = prompt_chars // 4` (`_CHARS_PER_TOKEN_ESTIMATE`,
`tradingagents/llm_call_log.py`) — a heuristic the module's own docstring calls "explicitly an estimate,
not a real tokenizer count." Measured against this corpus's provider-reported `input_tokens`, the two
numbers disagree systematically and in both directions: 266 of 521 calls (51%, all 96 Portfolio Manager
calls among them) fall below the tool's default truncation-candidate ratio of 0.8 (reported far below
estimated), while a separate cluster reports *above* the estimate by roughly 30%. **Issue #146/#148
(diagnosis: recurring reported value of 2,051; a ceiling near 4,096) is the place investigating this
belongs — as of this report neither is resolved (#148 open), so this report does not attempt to say
whether prompts were truncated in this corpus.** #147 (also open) is where the chars/4 estimator itself
gets replaced or calibrated.

**What this caveat does and doesn't affect**: the estimated-token columns above (per-agent ranking table)
inherit this ±30%-class uncertainty and should be read as "roughly this many tokens," not exact counts.
**The segment-composition percentages are unaffected** — they are computed directly from character counts
(chars per segment ÷ total chars), never from the token estimate, so the 34.4%/32.0%/15.4%/14.3%/1.3%
breakdown and the memory-growth character projections carry no token-estimator error at all.

## Implications

The risk-debate history and analyst reports together are ~66% of today's Portfolio Manager prompt and are
the largest lever available right now; the risk-debate segment in particular is trivially controllable via
the existing `max_risk_discuss_rounds` knob, and the analyst-report fan-out means a saving applied at the
formatting layer would compound across up to six consumers, not just the Portfolio Manager. Memory
injection, despite ranking fourth today, is the one segment with no size cap on any path and is on a
trajectory to become the largest segment in the prompt as soon as any given ticker accumulates a handful
more resolved decisions — which, for a handful of frequently-run tickers, is close. Ranking and proposing
concrete optimizations across both findings is #145's job, not this report's; the numbers above are the
input it depends on.
