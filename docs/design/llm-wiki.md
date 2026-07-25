# LLM-wiki: strategy knowledge base design

Status: confirmed design (2026-07-25), realizes the #94 spike deliverable and the
"Confirmed design" section of #100. This document is the sub-issue 1 (#101)
deliverable — the on-disk contract every later LLM-wiki sub-issue (#102–#107)
builds on. No production retrieval/consumption code lands in this issue; see
"Phased follow-up plan" below for what each later sub-issue adds.

## Problem

Decision agents (today: portfolio manager, swing trader) re-derive trading-strategy
knowledge (e.g. "what is the Piotroski F-score, and when does it work") from model
priors on every run instead of drawing on a curated, human-reviewed reference
library built from the user's own corpus of strategy papers. The LLM-wiki is that
reference library, plus a retrieval tool so agents can consult it on demand.

## Build approach: in-repo markdown builder

**Decision: a minimal, self-contained in-repo builder.** An ingestion script (#102)
reads PDFs from a configured folder, has an LLM draft a structured **article**, and
a human reviews/commits the result as a **markdown file** (YAML frontmatter + body)
under `knowledge/wiki/`. No new runtime service, no external account, no vendor
lock-in — the wiki is just files in the repo, versioned and diffable like everything
else `tradingagents/memory/` and `docs/` already do.

### Alternatives considered and rejected

- **openclaw** and **Hermes (Nous Research)** — both are hosted third-party agent
  platforms. Adopting either means taking on a network dependency, an external
  account/auth surface, and an availability/cost dependency for something that
  should be a small, deterministic, offline, git-tracked corpus of ~10s of
  documents. The integration glue code needed to shell out to either system would
  likely exceed the size of the in-repo builder itself, for a worse operational
  story (an outage or auth expiry breaks wiki *builds*, not just runs).
- **karpathy's LLM-wiki gist** and **Google Cloud `okf`** — neither is a shippable
  system; they are structuring *conventions* (a way to think about LLM-built
  knowledge bases) rather than code to depend on. There is nothing to "not build
  ourselves" by adopting them — using them still means writing the same ingestion
  script and article format this repo needs, just under someone else's naming.
  Their useful ideas (structured per-topic articles, LLM-assisted drafting,
  human-in-the-loop review) are folded into the in-repo design below instead of
  taken as a dependency.
- **Net effect**: none of the four external options remove any work the in-repo
  builder still has to do (parse a PDF, draft structured content, store it
  reviewably), while all four add an external dependency this repo doesn't
  otherwise have. Recommendation stands: minimal in-repo builder.

### Storage

- Articles live at `knowledge/wiki/<id>.md`, one file per article, `<id>` matching
  the article's own `id` frontmatter field (kebab-case).
- `knowledge/wiki/README.md` documents the layout for humans; `_TEMPLATE.md` is the
  copy-paste starting point for a new article (hand-written or ingestion-drafted).
- Default ingest folder for new source documents is `paper/` (config key
  `knowledge_ingest_dir`, default `"paper"` — see Configuration below). It holds
  the 7 seed papers driving the first wiki articles: Piotroski F-score, Mohanram
  G-score, Jegadeesh–Titman momentum, Betting-Against-Beta, Daniel–Moskowitz
  momentum crashes, Ball–Brown, Heyns–Hamman–Smit.
- Plain markdown + YAML frontmatter (not a database, not a vector index) mirrors
  the *legacy* markdown decision log's storage choice
  (`tradingagents/agents/utils/memory.py`) and keeps the corpus human-editable and
  PR-reviewable — a strategy article is exactly the kind of content a human should
  be able to read and correct in a normal code review, which a DB row or binary
  index would not allow.

## Article schema

Every article is one markdown file: a YAML frontmatter block, then six required
body sections in order. This is the schema the validation helper in this issue
(`tradingagents/knowledge/wiki_schema.py`) checks, and that later ingestion
(#102) and retrieval (#103) both build against.

### Frontmatter (required keys)

| Key            | Type                                 | Notes                                                                 |
|----------------|---------------------------------------|------------------------------------------------------------------------|
| `id`           | string, kebab-case, unique            | Matches the filename stem, e.g. `piotroski-f-score`.                  |
| `title`        | string                                | Human-readable article title.                                          |
| `tags`         | list of strings                       | Free-text topic tags (e.g. `value`, `momentum`, `quality`).           |
| `signals`      | list of strings                       | Named computable signals the article defines (e.g. `f_score`, `roa`). |
| `asset_classes`| list of strings                       | e.g. `equity`, `crypto`, `fx`.                                         |
| `horizon`      | list of strings                       | e.g. `swing`, `position`, `intraday`.                                  |
| `source`       | mapping: `authors`, `title`, `year`, `file` | `file` is a repo-relative path into `paper/` pointing at the source PDF. |

### Body (required sections, in order)

1. `## Summary` — one/two-paragraph thesis: what the strategy claims and why.
2. `## Signal — what it is` — precise definition of the signal(s) named in
   frontmatter `signals`.
3. `## How to compute` — the actual formula/procedure, in enough detail that an
   agent (or a future computation module, cf. `fundamentals_computation.py`'s
   "precompute in Python, don't ask the LLM to do arithmetic" convention from
   `LEARNINGS.md`) could implement it.
4. `## Empirical evidence` — what the source paper(s) found (sample, period,
   effect size), plus any known replications/failures.
5. `## When to apply / regime` — asset classes, horizons, and market regimes the
   signal is expected to work (or not work) in.
6. `## Caveats` — known failure modes, data requirements, look-ahead risks,
   crowding/decay concerns.

Sections are matched by exact `## ` heading text (see `REQUIRED_SECTIONS` in
`tradingagents/knowledge/wiki_schema.py`); the validator does not enforce section
*order*, only presence, so ingestion output that happens to reorder sections still
passes — order is a house-style convention documented in `_TEMPLATE.md`, not a
hard schema requirement.

See `knowledge/wiki/piotroski-f-score.md` for a complete worked example, and
`knowledge/wiki/_TEMPLATE.md` for the fill-in-the-blanks starting point.

## Consumption approach: agent-callable tool

**Decision: a new agent-callable tool, `search_strategy_wiki`, in the existing
`agents/utils/*_tools.py` + `dataflows/` vendor pattern** — not automatic prompt
injection, not MCP-server exposure.

- **Retrieval**: keyword/BM25 over article title + tags + body (#103,
  `tradingagents/dataflows/wiki_search.py`, `search_wiki(query: str, k: int) ->
  list[dict]`). Deterministic, offline, $0/run, no embedding-provider dependency.
  Routed through `data_vendors["knowledge_base"]` (default `"bm25"`) via
  `interface.route_to_vendor`, exactly like `news_data`/`web_search` are routed
  today — this keeps a vector backend a pure vendor swap later, without touching
  callers.
- **Tool**: `agents/utils/wiki_tools.py` exposes `@tool search_strategy_wiki(query:
  str, k: int = 3) -> str` (#104), mirroring `news_data_tools.get_news`'s
  "thin `@tool` wrapper over `route_to_vendor`" shape. Returns a formatted
  markdown string of the top-k articles/sections.
- **Loop conversion**: the portfolio manager and swing trader are today single-shot
  structured-output nodes (no tool loop — only analysts loop in the current
  graph). They are converted to **bounded** in-node tool-calling loops (#105,
  #106) via a shared loop helper, so both nodes can call the wiki zero-or-more
  times (config-capped round count) before emitting their existing structured
  decision. The structured-output contract, free-text fallback, and prompt
  logging both nodes already have are preserved unchanged.

### Alternatives considered and rejected

- **Automatic prompt injection** (always paste the whole wiki, or a static subset,
  into every PM/swing-trader prompt) — rejected. Token cost grows unbounded as the
  wiki grows past its current 7-article seed, with no relevance filtering: most
  runs would pay for articles irrelevant to the ticker/date being analyzed. This
  also directly contradicts the `LEARNINGS.md` convention "trim redundancy, not
  load-bearing context" — injected-but-irrelevant articles are exactly the kind of
  bloat that has previously made this system *more* unstable. An on-demand,
  query-scoped tool call keeps prompts lean and lets the agent decide when
  strategy context is actually relevant to the ticker/date at hand.
- **MCP-server exposure** (expose the wiki the way `tradingagents/memory/` is
  exposed via the networked memory MCP server) — rejected for this consumer set.
  The memory MCP server exists because memory is *shared, concurrently written*
  state across the LangGraph pipeline and the standalone `skills/` scripts
  (see CLAUDE.md "Persistence"); that justifies a network hop and a hard
  dependency on a running server. The wiki is the opposite: **read-mostly**,
  **small**, and **local** — every consumer that needs it already runs inside the
  same Python process as `tradingagents/dataflows/`. Adding a server (and a new
  hard-failure mode if it's unreachable, matching the memory MCP's
  `MemoryMCPConnectionError` behavior) buys nothing here and adds an operational
  dependency for a plain file-search lookup. A same-process tool call, exactly
  like `get_news`/`get_fundamentals`, is simpler and fits the existing
  `agents/utils/*_tools.py` pattern with zero new infrastructure.

## Extensibility

`search_strategy_wiki` and the shared loop helper are not PM/swing-trader-specific.
Any other node that binds the tool the same way the loop helper binds it for
PM/swing (#105, #106) gets wiki access — trader, researchers, and risk-team nodes
can be wired in later purely by adding the tool binding to their node, with no
change to `dataflows/wiki_search.py`, the article schema, or the vendor seam.
Analyst nodes already run tool loops, so wiring the wiki tool into an analyst is
even simpler than the PM/swing conversion this design requires.

### Wiring the wiki tool into other agents

For **single-shot structured-output nodes** (e.g., Trader, Researchers):
1. Import `run_structured_with_tools` from `tradingagents/agents/utils/structured.py`
2. Import `search_strategy_wiki` from `tradingagents/agents/utils/wiki_tools.py`
3. Gate tool availability on `knowledge_base_enabled` config key (read via `get_config()`)
4. Replace direct `structured_llm.invoke(prompt)` calls with `run_structured_with_tools(llm, messages, [search_strategy_wiki], StructuredOutputClass, max_rounds=config["knowledge_base_tool_max_rounds"])`
5. The helper returns `(structured_result, fallback_text, message_trace)` — exactly one of the first two is non-None; if both are None, a double failure (structured call failed AND free-text fallback also raised) propagates uncaught, matching the codebase's "abort, don't guess" convention (see issue #53 for the memory MCP hard-dependency precedent)

For **analyst-style tool-loop nodes** (already loop-native):
1. Add `search_strategy_wiki` to the tools list (gate on `knowledge_base_enabled`)
2. No loop helper needed; the existing analyst tool-loop pattern in `tradingagents/agents/analysts/news_analyst.py` already handles it

### Skills and future integrations

The `skills/` standalone agent reimplementation (see CLAUDE.md "What this is") will consume the same wiki retrieval tools via MCP tool registration to Claude Desktop/Code once that work resumes. Wiki searches will route through the same `tradingagents/dataflows/interface.py` vendor seam, keeping a vector backend a pure config/vendor swap (no caller changes).

The `record_agent_prompt`-based prompt logging of wiki tool interactions (e.g., in issue #105/#106 branch work) is not yet part of the `main` branch — that integration point exists as a known follow-up once the `feat/pdf-report-95` branch merges.

## Configuration

Following the existing `default_config.py` / `data_vendors` / `tool_vendors`
conventions (see CLAUDE.md "Configuration" and "Data vendors"):

- `knowledge_ingest_dir` (env: `TRADINGAGENTS_KNOWLEDGE_INGEST_DIR`, default
  `"paper"`) — default folder the ingestion script (#102) scans for new source
  documents.
- `data_vendors["knowledge_base"]` (config key, default `"bm25"`) — which vendor
  implements `search_wiki`; currently only `"bm25"` is planned, matching the
  `data_vendors["web_search"] = "tavily"` precedent of "one implemented option,
  vendor-routed so alternatives are a config change, not a rewrite."
- Bounded tool-loop round counts for the PM/swing conversions (#105, #106) get
  their own config keys at that point (e.g. `portfolio_manager_tool_loop_max_rounds`),
  not introduced here since no loop code lands in this issue.

These keys are **not** added to `default_config.py` in this issue — #101 is the
schema/design issue only; wiring config in belongs to the issues that consume it
(#103 for `data_vendors["knowledge_base"]`, #102 for `knowledge_ingest_dir`, #107
for consolidating env-override docs). Recording the intended keys and defaults
here now avoids each later sub-issue re-litigating names.

## Integration points

- `tradingagents/agents/managers/portfolio_manager.py` — gains a bounded tool loop
  (#105) that can call `search_strategy_wiki` before producing
  `final_trade_decision`.
- `tradingagents/agents/trader/swing_trader.py` — gains the same bounded tool loop
  (#106) before producing `swing_trade_decision` / `swing_structured_data`.
- `tradingagents/dataflows/interface.py` — gains a `route_to_vendor("search_wiki",
  ...)` case once #103 lands, alongside the existing `get_news`/`get_fundamentals`
  routing.
- `tradingagents/dataflows/config.py` — no schema change needed; `knowledge_base`
  is just another `data_vendors` entry, deep-merged the same way existing entries
  are.
- Memory: wiki lookups are **not** written to the SQLite memory core or the
  legacy markdown decision log — they are read-only reference lookups, not
  decisions or outcomes, so none of the `tradingagents/memory/` write paths are
  touched by this feature.

## Validation helper

`tradingagents/knowledge/wiki_schema.py` provides `validate_article(text: str) ->
ValidationResult`, a pure function (no file I/O, no network) that checks:

- All seven required frontmatter keys are present (`id`, `title`, `tags`,
  `signals`, `asset_classes`, `horizon`, `source`), `id` is kebab-case, the
  list-typed keys are actually lists, and `source` is a mapping with `authors`,
  `title`, `year`, `file`.
- All six required `## `-headed body sections are present.

`validate_article_file(path)` is a thin convenience wrapper that reads a file and
calls `validate_article`. Keeping the core check a pure string-in/result-out
function means the PDF→article ingestion pipeline (#102) and the BM25 retrieval
dataflow (#103) can both reuse it directly (e.g. ingestion rejects a drafted
article before commit; retrieval could optionally skip malformed files) without
either depending on the other or on any I/O the validator itself performs.

## Phased follow-up plan

Sequencing: 1 → (2 ∥ 3) → 4 → (5 ∥ 6) → 7. Unchanged from #100; recorded here for
convenience since this document is the reference other sub-issues link back to.

1. **#101 (this issue)** — design doc + article schema + repo layout + example
   article + schema validator. Foundational, no dependencies.
2. **#102** — PDF→article ingestion pipeline over `paper/`; generates the 7 seed
   articles. Depends on 1.
3. **#103** — BM25 retrieval dataflow + `knowledge_base` vendor + config wiring.
   Depends on 1.
4. **#104** — Agent-callable `search_strategy_wiki` tool + shared bounded
   tool-loop helper. Depends on 3.
5. **#105** — Wire the portfolio manager as a tool-loop consumer. Depends on 4.
6. **#106** — Wire the swing trader as a tool-loop consumer. Depends on 4.
7. **#107** — Config env-overrides, docs (CLAUDE.md/README), extensibility note
   for other agents. Depends on 5, 6.

## Risks / open questions

- **Corpus quality/staleness**: LLM-drafted articles need human review before
  commit; ingestion (#102) is human-in-the-loop, not auto-committing.
- **Retrieval quality**: BM25 over a small (currently 7-article) corpus can
  under-retrieve paraphrased queries; good `tags`/`signals` frontmatter mitigates
  this. The `data_vendors["knowledge_base"]` seam is the escape hatch to a vector
  backend if this proves insufficient in practice.
- **Latency/cost of loop conversion**: PM/swing tool loops must be bounded
  (config-capped rounds) to keep runs terminating and cheap; must preserve the
  structured-output final answer, free-text fallback, and prompt logging both
  nodes have today (#105, #106 concern, not this issue).
- **Determinism**: the retrieval path itself (BM25 over static files) is fully
  deterministic; only the *decision* to call the tool and what to do with results
  stays LLM-stochastic, consistent with the `LEARNINGS.md` convention of keeping
  non-determinism in the agent's reasoning, not in supporting computation.

## Out of scope (this issue and this feature's first phase)

- Web crawling / public-source ingestion — design is around the user-supplied
  `paper/` corpus; web ingestion is future work.
- Wiring the tool into agents beyond PM/swing trader (see Extensibility).
- A vector/embedding retrieval backend — the vendor seam exists, no vector backend
  is implemented.
- Any ingestion, retrieval, or tool code — this issue is schema + design + one
  hand-written example article only.
