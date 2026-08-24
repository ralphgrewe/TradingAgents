# Prompt optimization options: ranked, with a selection checklist

Status: options/ranking report (2026-08-24), issue #145, part of the #142 tracking issue. Builds
directly on the measurements in [`docs/analysis/prompt-size-findings.md`](prompt-size-findings.md)
(#144, commit `6b8fac9`) — every saving estimate below cites that report rather than re-deriving a
measurement. Verification tooling: [`scripts/analyze_llm_calls.py`](../../scripts/analyze_llm_calls.py)
(#143, commit `17c6d37`).

**This report proposes options and ranks them. It implements none of them.** No config keys, no
code, no prompt text changes ship here. The user ticks the checklist at the end; each ticked
option becomes its own follow-up issue.

## Bottom line up front

Tier 1 alone looks sufficient. #144 found that two segments — the risk-debate history (34.4%
median) and the analyst reports (32.0% median) — are ~66% of today's Portfolio Manager prompt, and
both are directly reachable by shortening work with existing or trivially-new levers (an existing
config knob for one, a formatting-layer budget for the other). Tier 2 (chunking, batching,
compaction steps, larger context windows, caching) is the fallback if Tier 1 turns out not to move
the numbers enough once tried — it is not needed to make a first cut at this problem, and nothing
here should be read as recommending it before Tier 1 has been tried and measured.

**Start here:** set `max_risk_discuss_rounds=1` (already the default — verify it, don't change it)
and, as the first *new* piece of work, ship the risk-debate compaction option (Tier 1, #1 below)
behind an opt-in config key. It targets the single largest segment (34.4%), the mechanism is
narrow (one function, three call sites all touching `risk_debate_state["history"]`), and it is
independently verifiable with `analyze_llm_calls.py --segment` before/after. See "Explicit
ranking" for why it beats the memory budget option despite the memory segment's steeper growth
curve.

## Scope note on memory

Memory injection (`past_context`, the `Past analyses of <ticker>` / `Recent cross-ticker lessons:`
blocks) reaches **only the Portfolio Manager**. #144 checked all 425 non-PM prompt dumps in the
corpus directly: zero contain a memory block. `trading_graph.py:457` calls
`self.memory_log.get_past_context(company_name)` unconditionally, but only
`portfolio_manager.py` reads the resulting `past_context` state field. Any option below that
targets memory injection (Tier 1 #3, #4) changes only the Portfolio Manager's prompt — it has no
effect on the Trader, Researcher, the four analysts, or the three risk analysts. Note also that
the Portfolio Manager's memory comes from the **legacy markdown log**
(`tradingagents/agents/utils/memory.py`, `TradingMemoryLog.get_past_context`), not the SQLite core
(`tradingagents/memory/query.py`) — the two memory systems documented in `CLAUDE.md` are not
interchangeable here, and an option touching one does not automatically touch the other (the
SQLite path is used by the swing trader and macro-agent memory injections, which are off by
default and were not part of #144's corpus).

## Opt-in constraint

`LEARNINGS.md` (May 1st, 2026 entry) records that shortening prompts has previously made this
system *more* unstable: cutting context the agent actually needed produced worse decisions, not
just shorter ones, in earlier work on this codebase. Every option below is written to ship behind
a new config key defaulting to **today's behavior unchanged** — no proposal here silently changes
what a default run sees. Each entry states the config key it would need and, in its "Verification"
line, how re-running `scripts/analyze_llm_calls.py --segment` before/after (same corpus shape:
same tickers/dates, dumps enabled via `llm_call_log_prompts=true`) would confirm the saving lands
where predicted without an unexpected side effect elsewhere in the prompt.

---

## Tier 1 — make the prompts shorter

Ordered by measured share of the Portfolio Manager prompt (the critical prompt on both size and
wall-time per #144's "Critical-prompt ranking" table), except where trajectory overrides current
share (called out explicitly).

### 1. Compact the risk-debate history carried into the Portfolio Manager

- **Measured saving**: risk-analysts debate history, **34.4% median** (8,938.5 of 26,192 median
  chars), the largest single segment (#144 "PM prompt composition"). At the default
  `max_risk_discuss_rounds=1`, this is exactly 3 full-length arguments (~2,980 chars each)
  concatenated verbatim.
- **Mechanism**: `risk_debate_state["history"]` (`tradingagents/agents/utils/agent_states.py`) is a
  single accumulating string. Each of the three risk analysts
  (`tradingagents/agents/risk_mgmt/{aggressive,conservative,neutral}_debator.py`) does
  `history + "\n" + argument` — plain concatenation, never trimmed or summarized (#144 "Segment
  1"). The Portfolio Manager embeds the full string verbatim via `risk_debate_history_section`
  (`tradingagents/agents/managers/portfolio_manager.py:77-81`).
- **What it would take**: replace (or offer as an alternative to) the raw `history` string with a
  synthesized version before it reaches the Portfolio Manager — either (a) an LLM-free structural
  compaction (each analyst's *own* per-viewpoint history — `aggressive_history` /
  `conservative_history` / `neutral_history`, already tracked separately in `RiskDebateState` —
  condensed to its final position rather than replaying every round), or (b) a short LLM synthesis
  step reusing the pattern the Portfolio Manager itself already uses for the wiki tool loop
  (`run_structured_with_tools`). Whether the Portfolio Manager needs the full transcript (to judge
  *how* the debate moved) or only the final positions (to judge *where* it landed) is the open
  design question this option should resolve first, since the two approaches trade off
  differently: (a) is free (no LLM call, no risk of losing content the PM needs) but assumes final
  position is enough; (b) preserves more nuance but adds a call and a new place for prompt
  instability to enter.
- **Files/config touched**: `tradingagents/agents/managers/portfolio_manager.py` (read path),
  `tradingagents/agents/utils/agent_states.py` (if a new state field is added rather than
  transforming in place), a new config key.
- **Risk to decision quality**: medium. This is the segment `LEARNINGS.md` warns about most
  directly — the risk debate exists specifically to surface disconfirming views before the PM
  decides, and compacting it risks cutting the exact content the PM needs (the debate's internal
  disagreement, not just its endpoint). Ship as opt-in and A/B against uncompacted history before
  ever defaulting it on.
- **Effort**: medium (one function, three well-isolated call sites already point at the same field;
  the LLM-synthesis variant adds a new prompt to get right).
- **Config key**: `TRADINGAGENTS_PM_RISK_HISTORY_MODE` (e.g. `"full"` default / `"compact"` /
  `"synthesis"`) — naming and exact values are a follow-up-issue decision, not fixed here.
- **Verification**: re-run `analyze_llm_calls.py --segment --top 521`, filter to Portfolio Manager,
  compare the risk-analysts-debate-history row's median share/chars before vs. after with the same
  ticker/date corpus.

### 2. Budget analyst reports where they fan out to trader / risk team / portfolio manager

- **Measured saving**: analyst reports, **32.0% median** (8,543.5 of 26,192 median chars), the
  second-largest PM segment (#144 "Segment 2"). Combined with segment 1, ~66% of the PM prompt.
- **Mechanism**: `format_analyst_reports_section` (`tradingagents/agents/utils/agent_utils.py:65`)
  renders the market/sentiment/news/fundamentals (+ optional macro) JSON envelopes into one text
  block. It is called from **four** sites — `portfolio_manager.py:57`, `trader/trader.py:72`,
  `trader/swing_trader.py:61`, `researchers/researcher.py:187` — and the three risk analysts
  (`risk_mgmt/{aggressive,conservative,neutral}_debator.py`) separately interpolate the same four
  raw report fields (`state["market_report"]` etc.) directly into their own prompts, not through
  the shared formatter but the same underlying text. So the same analyst-report text, generated
  once, is paid for again at minimum **five more times per run** (Trader, three risk analysts,
  Portfolio Manager), plus a sixth time in the Researcher when `research_stage="researcher"` (the
  default) — #144 confirms this fan-out from the call-site count, not from the PM segmentation
  alone.
- **What it would take**: a budget applied inside `format_analyst_reports_section` itself (e.g. cap
  each envelope's `summary`+`details` rendering to N characters, or drop `details` sub-fields not
  cited by the reading-instructions framing) — because it's the shared formatting layer, a saving
  applied here compounds across every one of its four call sites automatically. The three risk
  analysts read the raw state fields directly rather than through the formatter, so they would need
  a separate change (or a refactor to route them through the same formatter) to see the same
  saving — note this explicitly in the follow-up issue so the risk-analyst prompts aren't assumed
  covered for free.
- **Files/config touched**: `tradingagents/agents/utils/agent_utils.py`
  (`format_analyst_reports_section`); `tradingagents/agents/risk_mgmt/*.py` if extended to the risk
  analysts too; a new config key.
- **Risk to decision quality**: medium — the analyst envelopes are the primary evidence base every
  downstream agent grounds its reasoning in; over-aggressive truncation risks cutting a `details`
  field a later stage specifically cites (the risk-analyst prompts explicitly instruct "cite
  specific `details` fields... as evidence"). Budget by field priority (keep `summary` and
  `signal`/`confidence` in full; budget `details` sub-fields) rather than a blind character cap.
- **Effort**: low-medium for the formatter-only change (one function); medium if extended to the
  risk analysts (three more call sites, currently not routed through the shared formatter at all).
- **Config key**: `TRADINGAGENTS_ANALYST_REPORT_CHAR_BUDGET` (or per-field budgets) — exact shape a
  follow-up-issue decision.
- **Verification**: `analyze_llm_calls.py --segment` before/after on Portfolio Manager (analyst
  reports row) and, if extended, on Trader/risk-analyst prompts via their own `--sort-by
  estimated_prompt_tokens_median` rows in the main (non-segment) report, since risk-analyst prompts
  aren't PM-anchor-segmented today.

### 3. Character/token budget and entry-count limit on past-context (memory) injection

- **Measured saving today**: 14.3% median (3,573.5 of 26,192 median chars) — ranks fourth by
  today's share, but **must be ranked on trajectory, not today's share**, per #144's
  "Memory-growth trajectory" section: it is the only segment in the prompt with **no size cap of
  any kind** on either the same-ticker or cross-ticker path.
  - Observed scaling (same-ticker sub-block): 558–1,555 chars at `n_same=1` (30 tickers), 15,982–
    18,922 chars at `n_same=4` (3 tickers: AAPL, MSFT, NVDA) — roughly 5,100–5,800 chars per
    additional entry.
  - **Projected steady state** (at the configured `n_same=5` cap): same-ticker block alone ≈
    21,100–24,700 chars; combined with the already-saturated ~2,754-char cross-ticker block, total
    memory injection ≈ **23,850–27,450 chars** — this would exceed today's *combined* risk-debate +
    analyst-report median (17,481.5 chars) by roughly 35–55%, making memory the single largest PM
    segment once any ticker has 5 resolved decisions. #144 notes this threshold is close for the
    corpus's most-frequently-run tickers.
  - This is why memory ranks where it does in "Explicit ranking" below despite trailing segments 1
    and 2 on today's numbers: it is not optional to eventually address, only currently smaller.
- **Mechanism**: `TradingMemoryLog.get_past_context` (`tradingagents/agents/utils/memory.py:70`)
  formats up to `n_same=5` same-ticker entries **in full** via `_format_full`
  (`memory.py:283` — entire `DECISION:` and `REFLECTION:` text, uncapped) and up to `n_cross=3`
  other-ticker entries via `_format_reflection_only` (`memory.py:293` — capped at 300 chars only
  when no reflection exists; **uncapped when a reflection does exist**, which is the common case
  for a resolved entry). Neither `n_same` nor `n_cross` themselves are exposed as config today —
  they're call-site defaults in `get_past_context`'s signature, called with no override from
  `trading_graph.py:457`.
- **What it would take**: (a) a character/token cap on `_format_full`'s DECISION/REFLECTION text
  (truncate or ellipsize past a threshold), (b) exposing `n_same`/`n_cross` as config so a user can
  lower the entry count directly, or both. (a) bounds the worst case per entry; (b) bounds the
  number of entries multiplying that worst case.
- **Files/config touched**: `tradingagents/agents/utils/memory.py` (`_format_full`,
  `_format_reflection_only`, `get_past_context` signature); `trading_graph.py:457` (the call site,
  to pass through new config); two new config keys (or one budget + reuse of a lowered `n_same`).
- **Risk to decision quality**: medium-high — memory injection is literally the reflection/lesson
  mechanism the whole "learn from past decisions" design in `CLAUDE.md`'s Persistence section
  exists for; capping DECISION/REFLECTION text risks cutting the specific sentence that made a
  past lesson actionable. Character budgets should bias toward keeping REFLECTION (the distilled
  lesson) over DECISION (the raw output) if a cap must drop one before the other.
- **Effort**: low (two formatting functions, one call-site signature change; no new call sites since
  this is PM-only).
- **Config key**: `TRADINGAGENTS_MEMORY_CONTEXT_CHAR_BUDGET` and/or
  `TRADINGAGENTS_MEMORY_N_SAME` / `TRADINGAGENTS_MEMORY_N_CROSS`.
- **Verification**: re-run `analyze_llm_calls.py --segment` on a ticker with `n_same>=3` before/
  after (the corpus already has AAPL/MSFT/NVDA at `n_same=4` as ready-made high-signal cases);
  confirm the same-ticker sub-block median/max chars drop as configured.

### 4. Condensed past entries — reflection-only instead of `_format_full` prose

- **Measured saving**: a variant of #3 rather than a separate segment — same 14.3%-today /
  unbounded-trajectory profile from #144, but a different mechanism than a raw character cap.
- **Mechanism**: same-ticker entries currently use `_format_full` (full DECISION + REFLECTION
  prose, `memory.py:283`); cross-ticker entries already use the cheaper `_format_reflection_only`
  (`memory.py:293` — tag + reflection only, no DECISION text). This option asks: could same-ticker
  entries use the *same* cheaper formatter cross-ticker entries already use, rather than a
  character truncation of the full formatter's output? The SQLite core's equivalent
  (`tradingagents/memory/query.py:126`, `_format_same_ticker_entry`) already does something similar
  — a single markdown bullet (date, signal, confidence, lesson) — but that path isn't used by the
  Portfolio Manager today (see "Scope note on memory").
- **Files/config touched**: `tradingagents/agents/utils/memory.py` (a new formatter, or reuse of
  `_format_reflection_only` for same-ticker entries too, gated by config).
- **Risk to decision quality**: medium — same-ticker entries specifically include the full DECISION
  text today, which may carry information the reflection alone doesn't (e.g. the specific
  entry/stop/target levels that were tried). Switching wholesale to reflection-only for
  same-ticker entries is a bigger behavior change than a character budget (#3) and should be
  proposed as an alternative to #3, not stacked with it, until one is shown to preserve decision
  quality at least as well.
- **Effort**: low (reuses an existing formatter, or a small new one modeled on it).
- **Config key**: `TRADINGAGENTS_MEMORY_SAME_TICKER_FORMAT` (`"full"` default / `"reflection_only"`).
- **Verification**: same as #3 — `analyze_llm_calls.py --segment` on `n_same>=3` runs, before/after.

### 5. Knobs that already exist — no new code required

These are achievable **today**, by configuration alone, with each entry's measured share of context
it actually controls (from #144 / the code paths above) so the reader can see what's already on the
table before any new code is written.

| Knob | Default | What it controls | Measured/structural share |
|---|---|---|---|
| `max_risk_discuss_rounds` | `1` | Number of full risk-debate rounds (3 speaker turns each) before the Portfolio Manager. `ConditionalLogic.should_continue_risk_analysis` (`tradingagents/graph/conditional_logic.py:47-57`) terminates at `count >= 3 * max_risk_discuss_rounds`. | Directly multiplies the **34.4%-median risk-debate segment** — each additional round adds 3 more full-length arguments on top of what's already there (#144 "Segment 1"). Already at its minimum (1) by default; this knob's leverage is in *not* raising it, and in the fact that lowering below 1 isn't possible (0 disables the whole stage — see `risk_stage` below instead). |
| `risk_stage` | `"debate"` | Whether the risk-debate stage runs at all. `"none"` (`tradingagents/graph/setup.py:106-248`) routes the Trader's plan straight to the Portfolio Manager, skipping all three risk analysts entirely. | Eliminates the **entire 34.4%-median segment** — the most drastic version of the risk-debate lever, at the cost of removing the risk-debate stage's function altogether (not a shortening, a stage removal; treat as a different decision than the other rows here, not a tuning knob). |
| `research_evidence_token_budget` | `3000` | Token budget for the Researcher's web-search evidence pack assembly (`researcher.py`, plan-execute-synthesize pipeline). | Bounds the Researcher's own prompt (not PM-segmented in #144, since the anchor set is PM-specific), and indirectly bounds what enters `investment_plan`, which the PM's `research_trader_line` embeds (part of the 15.4%-median header+plan segment). |
| `max_debate_rounds` | `1` | Bull/Bear debate rounds when `research_stage="debate"` (not the default). | Same shape as `max_risk_discuss_rounds` but for the alternate research-stage mode — irrelevant at the default `research_stage="researcher"`. |
| `selected_analysts` | `["market","social","news","fundamentals"]` | Which analysts run, and therefore how many envelopes `format_analyst_reports_section` has to render everywhere it's called. | Directly scales the **32.0%-median analyst-reports segment** and its five-to-six-way fan-out (#144 "Segment 2") — fewer selected analysts means less text repeated at every consumer, at the cost of losing that analyst's signal entirely. Opt-in analysts (`macro_fundamentals`, `macro_news`) already default *out* for this reason (per `CLAUDE.md`). |
| `research_stage` | `"researcher"` | `"none"` skips the research stage entirely, sending analyst reports directly to the Trader. | Removes the Researcher's own prompt cost (mean 4,706 est. tokens, #144 "Critical-prompt ranking") and the header+plan segment's `research_trader_line` contribution on the PM side, at the cost of no research synthesis at all. |

Rows for `max_risk_discuss_rounds` and `risk_stage` are called out first since they bear directly
on the largest segment, per the issue's acceptance criteria.

---

## Tier 2 — long-prompt handling (fallback)

Considered only if Tier 1 turns out not to be promising once tried and measured. Shorter
treatment by design — these are architectural changes to *how* long prompts are handled, not to
*how long* the prompts are, and several of them are better read as mitigations for a truncation
problem (if #148 confirms one exists) than as optimizations in their own right.

- **Sequential/chunked processing of large inputs.** Split a large input (e.g. the full analyst-
  report set) into chunks processed across multiple calls, combined at the end. Applicability
  here is limited: the PM's inputs are already individually-sized envelopes, not one large
  document that needs splitting — chunking would add calls (cost, latency) to solve a problem
  that's already addressable by budgeting at the source (Tier 1 #2). Worth it only if a single
  segment's *irreducible* content genuinely can't be budgeted down, which #144's data doesn't
  currently show for any PM segment.
- **Map-reduce summarization.** Summarize each risk-analyst's argument (or each analyst report)
  independently, then combine summaries. This is closely related to Tier 1 #1's "LLM synthesis"
  variant, generalized — the map-reduce framing adds a formal per-source summarization step rather
  than one combined synthesis call. Trade-off: more LLM calls (cost, latency, another place for the
  `LEARNINGS.md` instability risk to enter) versus finer-grained control over what's kept per
  source. Only worth the extra complexity if Tier 1 #1's single-synthesis approach is tried first
  and found to lose too much nuance.
- **An explicit compaction step between pipeline stages.** A dedicated LangGraph node (e.g.
  between the risk-debate stage and the Portfolio Manager, per `tradingagents/graph/setup.py`'s
  stage boundaries) that rewrites `risk_debate_state["history"]` before the PM ever sees it. This is
  architecturally the same idea as Tier 1 #1's synthesis variant, but framed as a first-class graph
  node rather than an inline transform inside `portfolio_manager.py` — cleaner separation of
  concerns, testable independently, but a new node in `CLAUDE.md`'s documented pipeline stages
  (a design surface change, not just a prompt change). Consider this framing if Tier 1 #1 is
  selected and the inline-transform approach proves awkward to test in isolation.
- **Batching.** Not applicable to a single ticker/date decision — this pipeline processes one
  ticker+date through the graph at a time (`run_trading_agents.py` iterates a stock list, it
  doesn't batch multiple tickers into one LLM call). No lever here without a fundamentally
  different call pattern.
- **Context-length and KV-cache knobs (`num_ctx`, flash attention).** Per `docs/local-models.md`,
  Ollama's default context window is 4,096 tokens, configurable via `OLLAMA_CONTEXT_LENGTH` /
  Modelfile `PARAMETER num_ctx` — server/model-level, not exposed through TradingAgents config
  (`docs/local-models.md` "API Request Parameter" section: "This repo does not expose such per-call
  overrides"). This doesn't shorten anything — it raises the ceiling before truncation risk
  appears, and is the most direct mitigation for #148's open truncation question if that issue
  confirms truncation is happening (see the token-accuracy caveat in #144, and #146/#148). Not a
  prompt optimization; a capacity increase.
- **Prompt caching.** Provider-side reuse of a previously-processed prefix (e.g. Anthropic's
  prompt caching) to avoid re-paying for stable prefix content across calls. Most applicable to the
  analyst-report fan-out (Tier 1 #2) — the same envelope text is sent to six consumers per run —
  if those consumers shared a stable prefix structure and the provider in use supports it. Ollama's
  OpenAI-compatible endpoint (this repo's default local-model path, per `CLAUDE.md`) has no
  documented prompt-caching support; this option is provider-dependent and would need per-provider
  gating, unlike every Tier 1 option which is provider-agnostic. A cost/latency mitigation, not a
  token-count reduction — the model still processes the same tokens, so it doesn't move the
  numbers `analyze_llm_calls.py` measures at all.
- **Retrieval-instead-of-injection for the memory block.** Instead of injecting all `n_same`/
  `n_cross` entries unconditionally (Tier 1 #3/#4), retrieve only the entries most relevant to the
  current decision (e.g. via the wiki's BM25 retrieval pattern, `data_vendors["knowledge_base"]` in
  `tradingagents/dataflows/config.py`, adapted to memory rows) on demand, mirroring how the
  Portfolio Manager already consults the strategy wiki via `search_strategy_wiki` rather than
  having it injected. This is architecturally the most different option here — it changes memory
  from "always in the prompt" to "a tool the PM can call" — and is a bigger design change than
  Tier 1 #3/#4's budget-in-place approach; worth it long-term given memory's unbounded trajectory,
  but Tier 1 #3 (a budget) is the smaller, faster first step toward the same problem.

---

## Explicit ranking

One ordered list across both tiers, by (measured saving ÷ effort, adjusted for risk). Where an
option's rank depends on trajectory rather than current share, that's called out in its rationale.

1. **Tier 1 #1 — compact the risk-debate history.** Highest measured saving (34.4%) at
   medium effort, narrow blast radius (one segment, one consuming agent). **Start here.**
2. **Tier 1 #5 — `max_risk_discuss_rounds` verification (no-op check).** Zero effort (it's already
   at its minimum default, 1) but worth an explicit line: confirm no deployment has silently raised
   it, since doing so would directly multiply the largest segment. Free to check, not free to fix
   if found misconfigured.
3. **Tier 1 #2 — budget analyst reports at the formatting layer.** Second-largest measured saving
   (32.0%) and the saving compounds across up to six consumers per the fan-out finding — the
   highest saving-per-line-changed of any option here, since one function change touches four call
   sites automatically (the three risk analysts need a separate follow-up to see the same benefit,
   noted above).
4. **Tier 1 #3 — memory character/entry-count budget.** Ranks here **on trajectory, not today's
   14.3% share** — #144 projects it overtakes the risk-debate + analyst-report combined total by
   the time any ticker reaches `n_same=5`, and the corpus's most-frequently-run tickers are already
   partway there (`n_same=4` observed for 3 tickers). Ranked above Tier 1 #4 because a budget is a
   smaller, more mechanical change than reformatting entries, and above Tier 2 entirely because
   it's cheap (low effort) even though its current share is smaller than #1/#2.
5. **Tier 1 #4 — condensed past entries (reflection-only).** A variant/alternative to #3 rather than
   additive; ranked just below it because it's a bigger behavior change (drops DECISION text
   entirely rather than budgeting it) for a similar saving.
6. **Tier 1 #5 — `selected_analysts` / `research_stage` / `research_evidence_token_budget` as
   available lower-effort levers.** Zero new code, real but coarser-grained savings (dropping an
   entire analyst or research stage is a bigger decision-quality trade than budgeting text within
   one), useful as an immediate stopgap while #1/#2/#3 are built.
7. **Tier 1 #5 — `risk_stage="none"`.** Eliminates the largest segment outright but removes the
   risk-debate stage's function; ranked last among the "existing knobs" because it's a stage
   removal, not a shortening — only appropriate if the risk-debate stage itself is judged
   low-value, a decision independent of this report's prompt-size scope.
8. **Tier 2 — retrieval-instead-of-injection for memory.** The Tier 2 option most directly
   addressing the same unbounded-trajectory problem as #4/#5 above, and the best long-term answer
   to it, but a bigger architectural change (memory becomes a tool call, not an injection) than
   Tier 1 #3's budget — try Tier 1 #3 first and revisit this only if a static budget proves
   insufficient (e.g. it turns out entries need to be *selected*, not merely *shortened*, to stay
   useful).
9. **Tier 2 — an explicit compaction step / map-reduce summarization.** Architecturally related to
   Tier 1 #1; only worth building as a separate graph node if the inline-transform version of #1 is
   selected first and found to need cleaner separation.
10. **Tier 2 — `num_ctx` / KV-cache knobs.** Not a prompt shortener; a capacity increase and the
    direct mitigation if #148 confirms truncation. Independent of this ranking's savings axis —
    pursue based on #148's outcome, not on this list's ordering.
11. **Tier 2 — prompt caching.** Provider-dependent (no support on this repo's default Ollama path),
    doesn't reduce measured token counts, lowest priority of the Tier 2 entries for this codebase
    specifically.
12. **Tier 2 — sequential/chunked processing, batching.** Not applicable to this pipeline's current
    call pattern (single ticker+date, already-sized envelopes) — listed for completeness, not
    because either has a clear use here today.

---

## Selection checklist

Tick the rows to file as follow-up issues. Each carries its expected saving and risk so the choice
can be made from this table alone.

- [ ] **1. Compact risk-debate history into the Portfolio Manager prompt** — saves ~34% of the PM
      prompt (largest lever) — risk: medium (the debate's disagreement is the point; must not
      collapse to endpoints only without checking decision quality).
- [ ] **2. Budget analyst reports in `format_analyst_reports_section`** — saves ~32% of the PM
      prompt, compounds across up to 6 consumers — risk: medium (must preserve `details` fields
      that downstream prompts explicitly cite as evidence).
- [ ] **2b. Extend the analyst-report budget to the three risk analysts** (currently read raw state
      fields, not the shared formatter) — saves the same ~32% at those three additional call sites
      — risk: medium, same caveat as #2, plus a refactor to route them through the shared
      formatter.
- [ ] **3. Character/token budget + entry-count limit on memory injection** — saves up to the full
      unbounded trajectory (projected ~24-27K chars at steady state, currently 14.3%/~3.6K median)
      — risk: medium-high (the reflection/lesson mechanism is the point of this memory system;
      must bias toward keeping REFLECTION over DECISION text if forced to drop one).
- [ ] **4. Condensed (reflection-only) same-ticker memory entries** — alternative to #3, similar
      saving, bigger behavior change (drops DECISION text entirely) — risk: medium-high, same
      concern as #3 but less tunable (binary format switch vs. a budget).
- [ ] **5. Verify `max_risk_discuss_rounds` stays at its default (1)** — zero-effort audit, prevents
      silently multiplying the largest segment — risk: none (a check, not a change).
- [ ] **6. Document/surface the existing coarse-grained knobs** (`selected_analysts`,
      `research_stage`, `research_evidence_token_budget`, `risk_stage`) as an immediate,
      no-new-code stopgap — risk: varies per knob, all already shipped and understood; risk here is
      only in choosing to *use* them, not in building anything new.
- [ ] **7. Retrieval-instead-of-injection for memory (Tier 2)** — best long-term answer to memory's
      unbounded growth, bigger architectural change — risk: medium (retrieval quality becomes a new
      failure mode; only pursue after #3 is tried and found insufficient).
- [ ] **8. Explicit compaction-step graph node (Tier 2)** — only if #1's inline approach needs
      cleaner separation — risk: low-medium (mostly a refactor of #1's own logic into a new node).
- [ ] **9. `num_ctx` / KV-cache guidance (Tier 2)** — pursue based on #148's truncation finding, not
      on this checklist's priority order — risk: none to decision quality (a capacity change, not a
      content change).
- [ ] **10. Prompt caching (Tier 2)** — lowest priority for this codebase (no support on the default
      Ollama path) — risk: none to decision quality, but doesn't reduce measured token counts either.
