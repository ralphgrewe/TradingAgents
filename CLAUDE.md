# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

TradingAgents is a multi-agent LLM framework that simulates a trading firm: specialized agents
(analysts → researchers → trader → risk team → portfolio manager) collaboratively analyze a
stock/crypto ticker for a given date and emit a BUY/SELL/HOLD decision. There are two parallel
implementations in this repo:

- **`tradingagents/` + `cli/`** — the primary, working system. A LangGraph state machine wired up
  by `tradingagents/graph/trading_graph.py`, driven by the `tradingagents` CLI (`cli/main.py`) or
  the `mcp_server.py` MCP tool (`analyze_stock`).
- **`skills/`** — an in-progress reimplementation of the same agents as standalone Claude
  Cowork/subagent skills (see `skills/README.md` for the mapping from LangGraph nodes to skills).
  These read/write JSON envelopes (schema in `skills/SCHEMA.md`) instead of LangGraph state, and
  share the SQLite memory core in `tradingagents/memory/` with the legacy pipeline.

## Commands

Always use the project virtualenv (`./venv`), never system Python.

```bash
# Run the full test suite
./venv/bin/pytest

# Run a single test file / test
./venv/bin/pytest tests/test_memory_store.py
./venv/bin/pytest tests/test_memory_store.py::test_store_decision_is_idempotent -v

# Run by marker (unit / integration / smoke — see pyproject.toml)
./venv/bin/pytest -m unit

# Lint (CI runs this with a strict rule set over the full repo)
./venv/bin/ruff check .

# Launch the interactive CLI
./venv/bin/tradingagents            # installed entry point
./venv/bin/python -m cli.main       # equivalent, run from source

# Run one or more tickers non-interactively from a JSON file (see README.md)
echo '[{"ticker": "AAPL", "date": "2024-01-15"}]' > stocks.json
./venv/bin/python run_trading_agents.py stocks.json --show-summary

# Run the MCP server
./venv/bin/python mcp_server.py               # stdio transport (default)
./start_server.sh                             # networked transport (streamable-http)
MCP_TRANSPORT=sse FASTMCP_HOST=0.0.0.0 FASTMCP_PORT=8001 ./venv/bin/python mcp_server.py  # SSE transport

# Structured-output smoke test against a real provider (costs API credits)
OPENAI_API_KEY=... ./venv/bin/python scripts/smoke_structured_output.py openai
```

Tests never hit real LLM APIs by default — `tests/conftest.py` autouse-fixtures placeholder
values for every provider's API-key env var, and `mock_llm_client` patches
`tradingagents.llm_clients.factory.create_llm_client` for tests that need a fake client.

## Architecture: the LangGraph pipeline

Six sequential stages (five core + one optional), defined in `tradingagents/graph/setup.py` and gated by
`tradingagents/graph/conditional_logic.py`. The research stage (II) can be configured via
`research_stage` config key (env var `TRADINGAGENTS_RESEARCH_STAGE`). The swing trader (VI) is
optional and controlled via `swing_trader_enabled` config key (env var `TRADINGAGENTS_SWING_TRADER_ENABLED`):

```
I.   ANALYST TEAM    → selected analysts run in sequence (default: market → social → news →
                       fundamentals; optional opt-in: macro_fundamentals — reviews the
                       deterministic macro indicator pack from `macro_pack.py`, see #131/#132),
                       each loops with its own tools until it writes one report
II.  RESEARCH STAGE  → configured by research_stage:
     - "researcher" (default): single Researcher node synthesizes analyst reports + live web search
       evidence (when trade_date == today, via Tavily API; historical dates degrade to
       synthesis-only with metadata "disabled (historical date)")
     - "debate": Bull vs Bear debate (alternating, `max_debate_rounds`) → Research Manager
       writes a structured verdict (`investment_plan`)
     - "none": skip research entirely, send analyst reports directly to trader
III. TRADER          → turns the research plan into a concrete trade proposal
IV.  RISK TEAM       → Aggressive → Conservative → Neutral debate (`max_risk_discuss_rounds`)
V.   PORTFOLIO MGR   → writes the final decision (`final_trade_decision`)
VI.  SWING TRADER    → (optional, when `swing_trader_enabled=True`) makes regime-gated short-term
                       (3–15 day) swing trade decisions with numeric entry/stop/target levels
```

### Research stage modes in detail

**Mode: `"researcher"` (default, recommended for analysis depth)**
- Single Researcher node plan–execute–synthesize pipeline:
  1. **Plan** (quick-thinking LLM): reads analyst envelopes + instrument context, outputs
     up to `research_search_queries_max` web search queries (≥1 bull-seeking, ≥1 bear-seeking).
  2. **Gate check** (no LLM): if `trade_date == today` AND `research_web_search=True` AND
     `TAVILY_API_KEY` is set, gate opens for search; otherwise synthesis-only. Metadata line
     in brief reflects gate outcome: "enabled" or "disabled (reason)".
  3. **Execute** (no LLM): runs queries via Tavily API, assembles evidence pack (budget
     `research_evidence_token_budget` tokens).
  4. **Synthesize** (deep-thinking LLM, structured output): combines analyst envelopes +
     evidence pack into a `ResearchBrief` (bull/bear arguments, rating, confidence,
     new_information). Each argument must be source-tagged to an envelope or web evidence ID.
- Config:
  - `research_web_search` (default True): enable/disable web search.
  - `research_search_queries_max` (default 4): max queries per plan.
  - `research_evidence_token_budget` (default 3000): token budget for evidence assembly.
  - `data_vendors["web_search"]` (default "tavily"): web search vendor. Requires
    `TAVILY_API_KEY` env var when enabled.
- Memory stored under agent key `"researcher"`.
- Output: `investment_plan` (rendered brief with metadata line), `researcher_evidence`
  (JSON dict with query plan, gate outcomes, evidence pack for full-state log).

**Mode: `"debate"`**
- Bull and Bear researchers debate the analyst reports, culminating in a Research Manager verdict.
- Config: `max_debate_rounds` (default 1) controls debate length.
- Memory stored under agent key `"research_manager"`.

**Mode: `"none"`**
- Analyst reports flow directly to the Trader (no research stage).
- Fastest path, lowest LLM cost.

Key files:
- `agent_states.py` — the shared `AgentState` TypedDict every node reads/writes. Analyst reports
  (`market_report`, `sentiment_report`, `news_report`, `fundamentals_report`, and the opt-in
  `macro_report`) are the load-bearing fields; `investment_debate_state` / `risk_debate_state`
  hold per-speaker history + a
  `judge_decision`. Message history is cleared ("Msg Clear" nodes) after each analyst so only the
  written report — not the tool-call chatter — survives into later stages.
- `analyst_execution.py` — builds the analyst execution plan (order, concurrency) from the
  `selected_analysts` list; supports running analysts with limited concurrency
  (`analyst_concurrency_limit`).
- `conditional_logic.py` — debate/risk-discussion termination conditions (round counts).
- `trading_graph.py` — `TradingAgentsGraph`, the orchestrator: builds LLM clients (deep/quick
  thinking models can be different providers/models), compiles the graph, and drives
  `propagate(ticker, date)`. Also owns checkpoint-resume recompilation and the memory-log
  resolve/write around each run (see Persistence below).
- Each agent's prompt logic lives under `tradingagents/agents/{analysts,researchers,managers,
  risk_mgmt,trader}/`; tool implementations live in `tradingagents/agents/utils/*_tools.py`.

## LLM providers

`tradingagents/llm_clients/factory.py` is the single entry point (`create_llm_client`). Provider
SDKs are imported lazily inside each branch so importing the factory never pulls in unused SDKs.
OpenAI-compatible providers (openai, xai, deepseek, qwen/qwen-cn, glm/glm-cn, minimax/minimax-cn,
openrouter, mistral) all share `OpenAIClient`; anthropic, google, azure, ollama, and perplexity
each have a dedicated client. Provider-specific "thinking" knobs (`google_thinking_level`,
`openai_reasoning_effort`, `anthropic_effort`) are applied per-provider in
`TradingAgentsGraph._get_provider_kwargs`.

**Ollama talks to the native `/api/chat` endpoint, not the OpenAI-compatible one (issue #169).**
Ollama's OpenAI-compatible endpoint (`/v1/chat/completions`) **silently drops both `num_ctx` and
`think`** — verified live against Ollama 0.32.3: the same request with `options.num_ctx` or `think`
set is honored on `/api/chat` and ignored on `/v1/chat/completions` (`finish_reason=length` at the
daemon's default 4096-token window either way). Because of this, `ollama_client.OllamaClient`
routes the `ollama` provider to `langchain_ollama.ChatOllama` (via `factory.create_llm_client`,
matched *before* the OpenAI-compatible registry) instead of through `OpenAIClient`. **Do not
"fix" this back to `extra_body`/the compat endpoint** — that is precisely the silently-broken
behavior #169 replaced. Users who deliberately want the OpenAI-compat endpoint in front of Ollama
(e.g. a proxy) can still reach it via the generic `openai_compatible` provider with an explicit
`backend_url`.
- **Base URL**: resolution precedence is `backend_url`/`TRADINGAGENTS_LLM_BACKEND_URL` >
  `OLLAMA_BASE_URL` > default `http://localhost:11434` (no `/v1` — the native endpoint isn't
  OpenAI-compat). A configured URL ending in `/v1` or `/v1/` (the old documented shape) has that
  suffix stripped, so existing `.../v1` configs keep working against the new endpoint.
- **`ollama_num_ctx`** (issue #149) and the issue #154 per-request derivation are unchanged in
  arithmetic/semantics — only the delivery mechanism moved: `OllamaClient.get_llm` sets ChatOllama's
  native `num_ctx` field for the explicit-override case, and `NormalizedChatOllama._chat_params`
  (`ollama_client.py`) — the equivalent per-request hook to `ChatOpenAI._get_request_payload` on a
  client with no such method — derives and attaches it for the unset case, both landing under the
  request's `options.num_ctx`, same as `/api/chat` expects.
- **`ollama_think`** (issue #155) is a three-state knob (`True`/`False`/`None`) forwarded to
  ChatOllama's native `reasoning` field, which controls the request's `think` field: `True` sends
  `think: true`; `False` sends `think: false` explicitly; `None` sends no `think` field at all
  (model decides). **The effective default is `False`** (`default_config.py`'s `ollama_think`) — a
  deliberate behaviour change from pre-#169: the old compat endpoint silently dropped `think`
  regardless of this setting, so a reasoning-capable model (e.g. `qwen3.5:9b`) always burned
  1–3k tokens per call on reasoning; the native endpoint actually honors `false`, reclaiming those
  tokens by default. Set to `None` to explicitly defer to the model's own default instead.
See "Oversize-prompt enforcement" under Configuration below for why `ollama_num_ctx` exists.

**Structured output method (issue #161):** By default, most providers use `function_calling` for
structured output (LangChain's default). However, Ollama's OpenAI-compatibility layer does not
honor `tool_choice` directives, meaning `function_calling` can silently return `None` when the
model writes prose instead of a tool call. The `structured_output_method` config key (env var
`TRADINGAGENTS_STRUCTURED_OUTPUT_METHOD`, default `"auto"`) allows selecting the method globally:
- `"auto"` (default): use the capability table's `preferred_structured_method` for each model,
  except Ollama defaults to `json_schema` (grammar-constrained at decode time, via ChatOllama's
  native `format` field — issue #169).
- `"function_calling"`, `"json_schema"`, `"json_mode"`: force a specific method for all models.
Legal values: `"auto"`, `"function_calling"`, `"json_schema"`, `"json_mode"`. Precedence order
(highest first): explicit `method=` argument passed by caller > config value (when not `"auto"`)
> Ollama provider override (→ `json_schema`) > capability table > default. Models with
`preferred_structured_method="none"` still raise `NotImplementedError` and fall back to free text,
regardless of config overrides.

## Data vendors

`tradingagents/dataflows/config.py` holds a process-global config (`get_config`/`set_config`,
deep-merged one level for dict-valued keys like `data_vendors`). Each data category
(`core_stock_apis`, `technical_indicators`, `fundamental_data`, `news_data`) can be routed to a
different vendor (`yfinance`, `alpha_vantage`, `perplexity`) via `data_vendors` (category default)
or `tool_vendors` (per-tool override) in config. `tradingagents/dataflows/interface.py` and the
per-vendor modules (`alpha_vantage_*.py`, `y_finance.py`, `yfinance_news.py`, etc.) implement the
actual fetches; `symbol_utils.normalize_symbol` maps broker/forex/commodity tickers (e.g.
`XAUUSD`) to vendor-specific symbols (e.g. `GC=F`) at the point of use — never at storage time.

## Persistence

Three independent, still-coexisting memory systems:

- **Legacy markdown decision log** (`tradingagents/agents/utils/memory.py`,
  `TradingMemoryLog`) — always on. Appends a pending entry at the end of every `propagate()`
  call to `~/.tradingagents/memory/trading_memory.md` (override:
  `TRADINGAGENTS_MEMORY_LOG_PATH`). On the *next* run for the same ticker,
  `TradingAgentsGraph._resolve_pending_entries` fetches realized return, generates a reflection,
  and `get_past_context` injects recent same-ticker + cross-ticker history into the Portfolio
  Manager prompt.
- **SQLite memory core** (`tradingagents/memory/{store,resolve,query}.py`) — the shared backend
  for the `skills/` reimplementation and (as of issue #51/#53) for `trading_graph.py` as well.
  Stores one row per `(agent, ticker, decision_date)` in a `decisions` table (default path
  `runs/memory/memory.db`, override `TRADINGAGENTS_MEMORY_DB_PATH`). `store.py` is
  write-only/idempotent (`INSERT OR IGNORE`); `resolve.py` fills in `forward_return`/`lesson`
  once enough trading days have elapsed (`numpy.busday_count`-based, no holiday calendar);
  `query.py` is the read/formatting path for prompt injection.

  **Named memory IDs** (issue #114): a run can be isolated into its own decision history via
  `--memory-id MEMORY_ID` (`run_trading_agents.py`) or the `memory_id` config key
  (`TRADINGAGENTS_MEMORY_ID` env var) — each ID resolves to its own DB file at
  `runs/memory/<id>/memory.db` rather than the shared `runs/memory/memory.db`, so decisions
  from different models/configurations run against the same stock list don't contaminate each
  other's history. `tradingagents/memory/store.resolve_memory_id_to_db_path` implements the
  resolution: `memory_id` argument > `TRADINGAGENTS_MEMORY_ID` env var >
  `TRADINGAGENTS_MEMORY_DB_PATH` env var > default `runs/memory/memory.db`, and validates the ID
  against `^[A-Za-z0-9][A-Za-z0-9._-]*$` (rejecting `/`, `\`, `..` anywhere in the string, and
  empty strings) since it becomes a directory name. The ID → path derivation happens
  client-side in `TradingAgentsGraph.propagate`, which resolves it once and passes the resulting
  `db_path` into the `MemoryMCPClient` constructor as a default applied across all five tool
  calls — the memory MCP server itself stays stateless and keeps receiving an explicit
  `db_path` per call, unchanged. With no `--memory-id`/`TRADINGAGENTS_MEMORY_ID` set, the
  resolved path is identical to pre-#114 behavior. `cli/main.py` and the `analyze_stock` MCP
  tool don't get a `--memory-id` flag, but can still select a memory via
  `TRADINGAGENTS_MEMORY_ID`. The legacy markdown decision log is explicitly out of scope and
  stays global/unnamespaced.
  
  **Access path**: `trading_graph.py` and `skills/memory-review/find_patterns.py` reach the
  SQLite core over the networked memory MCP server (via `MemoryMCPClient`), not by opening
  the DB in-process. The memory MCP server (`mcp_server.py`) is a **hard dependency** for
  `trading_graph.py` — if the server is unreachable or a tool call fails, the run aborts with
  a `MemoryMCPConnectionError` or `MemoryMCPToolError` and does not proceed. This is a
  deliberate behavior change from the prior in-process warn-and-continue pattern (issue #53).
  `skills/` agents access the same tools via MCP tool registration in Claude Desktop/Code
  (see README.md "To register the server with Claude agents").

  Connecting and every tool call are bounded by `memory_mcp_timeout` (default 30s, env
  `TRADINGAGENTS_MEMORY_MCP_TIMEOUT`): the MCP SDK surfaces transport-level failures only when
  its internal task group unwinds, so without that bound an error answered by something other
  than the server (e.g. an HTTP proxy) leaves `session.initialize()`/`call_tool()` waiting on a
  response forever. Relatedly, requests to a *loopback* server URL bypass
  `http_proxy`/`https_proxy` — httpx's `no_proxy` matching does not understand CIDR entries like
  `127.0.0.0/8`, so loopback traffic would otherwise be handed to the proxy (issue #108).
  
  See the module docstrings in `store.py`, `resolve.py`, and `query.py` — they carry the
  design rationale (idempotency semantics, normalization boundary, single-transaction batching)
  and are the canonical reference.
- **Checkpoint/resume** (`tradingagents/graph/checkpointer.py`) — opt-in via
  `config["checkpoint_enabled"]` / `--checkpoint`. LangGraph `SqliteSaver`, one DB per ticker at
  `<data_cache_dir>/checkpoints/<TICKER>.db`, keyed by a `thread_id` derived from
  `(ticker, trade_date)` so the same ticker+date resumes and a new date starts fresh. Cleared
  automatically on successful completion.

## Configuration

`tradingagents/default_config.py` (`DEFAULT_CONFIG`) is the single source of truth for run
config. Any key can be overridden via a `TRADINGAGENTS_*` env var listed in the `_ENV_OVERRIDES`
table at the top of that file — coercion is driven by the *existing default's* type, so add a new
override by adding one row there, not by touching CLI/entry-point code. `set_config` /`get_config`
in `tradingagents/dataflows/config.py` hold the live, process-global copy that agent tool code
reads from.

A row's value is normally a flat top-level key (e.g. `"temperature"`). For a config key that
lives inside a nested dict (e.g. `data_vendors`), use dot notation instead (e.g.
`"data_vendors.knowledge_base"` for `TRADINGAGENTS_DATA_VENDORS_KNOWLEDGE_BASE`) —
`_apply_env_overrides` detects the `.` and routes through `_get_nested`/`_set_nested` to read/write
the nested value in place, coercing against the existing nested default's type just like the flat
case. Only add a nested row when the target key is genuinely nested; flat keys should keep using
the plain top-level form.

### Research stage configuration

- **`research_stage`** (env: `TRADINGAGENTS_RESEARCH_STAGE`, default `"researcher"`): which research
  pipeline to run — `"researcher"` (analyst → researcher node with optional web search → trader),
  `"debate"` (analyst → bull/bear debate → research manager → trader), or `"none"` (analyst → trader).
- **`research_web_search`** (env: `TRADINGAGENTS_RESEARCH_WEB_SEARCH`, default `True`): enable
  live web search in researcher mode (when `trade_date == today`). Requires `TAVILY_API_KEY`.
- **`research_search_queries_max`** (env: `TRADINGAGENTS_RESEARCH_SEARCH_QUERIES_MAX`, default
  `4`): max web search queries the researcher plan can request (researcher mode only).
- **`research_evidence_token_budget`** (env: `TRADINGAGENTS_RESEARCH_EVIDENCE_TOKEN_BUDGET`,
  default `3000`): token budget for assembling the evidence pack from search results (researcher
  mode only).
- **`data_vendors["web_search"]`** (config key, default `"tavily"`): which vendor to use for web
  search. Currently only "tavily" is implemented, and it requires `TAVILY_API_KEY` to be set
  in the environment.

### Swing Trader configuration

- **`swing_trader_enabled`** (env: `TRADINGAGENTS_SWING_TRADER_ENABLED`, default `False`): enable/disable
  the optional swing trader node (runs after Portfolio Manager if enabled). When enabled, the swing trader
  makes regime-gated short-term (3–15 trading day) swing trade decisions with numeric entry/stop/target
  levels and outputs `swing_trade_decision` (rendered markdown) and `swing_structured_data` (parsed
  `SwingDecision` dict or None on fallback).
- **`swing_trader_min_risk_reward`** (env: `TRADINGAGENTS_SWING_TRADER_MIN_RISK_REWARD`, default `1.5`):
  minimum reward-to-risk ratio required for a non-HOLD swing decision.
- **`swing_trader_max_holding_days`** (env: `TRADINGAGENTS_SWING_TRADER_MAX_HOLDING_DAYS`, default `15`):
  hard cap on holding period in trading days (validator clamps declared holding_period_days to this max).
- **`swing_trader_conviction_threshold`** (env: `TRADINGAGENTS_SWING_TRADER_CONVICTION_THRESHOLD`, default
  `0.55`): minimum conviction score (0.0–1.0) required to force action (BUY/SELL) instead of HOLD.
- Memory stored under agent key `"swing_trader"` (via the SQLite core + MCP server). Past context is
  injected at run start when `swing_trader_enabled=True`; decisions are stored with `horizon_days`
  set to `holding_period_days` for later analysis and reflection.

### LLM-wiki knowledge base configuration

- **`knowledge_base_enabled`** (env: `TRADINGAGENTS_KNOWLEDGE_BASE_ENABLED`, default `True`):
  enable/disable the strategy knowledge base. When enabled, the portfolio manager and swing trader
  can consult the wiki via the `search_strategy_wiki` tool to ground decisions in established
  trading principles and regime-specific approaches.
- **`knowledge_base_dir`** (env: `TRADINGAGENTS_KNOWLEDGE_BASE_DIR`, default `"knowledge/wiki"`):
  directory where the strategy knowledge base articles live (relative to cwd). Each article is a
  markdown file with YAML frontmatter (keys: `id`, `title`, `tags`, `signals`, `asset_classes`,
  `horizon`, `source`) and six required body sections (Summary, Signal definition, Computation,
  Empirical evidence, When to apply/regime, Caveats).
- **`knowledge_ingest_dir`** (env: `TRADINGAGENTS_KNOWLEDGE_INGEST_DIR`, default `"paper"`):
  default folder the ingestion pipeline scans for new source documents (PDFs) to convert into
  wiki articles. Set via `scripts/ingest_wiki.py --ingest-dir <path>`.
- **`data_vendors["knowledge_base"]`** (env: `TRADINGAGENTS_DATA_VENDORS_KNOWLEDGE_BASE`, default
  `"bm25"`): which vendor implements keyword/BM25 retrieval over the wiki. Currently only "bm25"
  is implemented (deterministic, offline, no API keys required); the vendor seam is reserved for
  future vector-embedding backends.
- **`knowledge_base_tool_max_rounds`** (env: `TRADINGAGENTS_KNOWLEDGE_BASE_TOOL_MAX_ROUNDS`,
  default `2`): maximum number of tool-calling loop rounds for the wiki search tool (gating PM
  and swing trader tool calls to keep runs deterministic and cost-bounded).

The corpus and retrieval are designed to support **on-demand, query-scoped wiki lookups** rather
than automatic prompt injection. See `docs/design/llm-wiki.md` for the design rationale, article
schema, ingestion pipeline, and extensibility guidance for wiring the wiki into other agents
(trader, researchers, risk team).

### Structured output repair retry (issue #153)

When a structured-output call made through `run_structured_with_tools`
(`tradingagents/agents/utils/structured.py`) fails — typically malformed JSON from a weak model —
the helper retries exactly once with an explicit schema-repair instruction appended to the trace
before falling back to free text. This improves recovery rates for small local models that produce
nearly-correct JSON and correct it when told plainly what shape is required.

This covers the Portfolio Manager and Swing Trader in **both** knowledge-base configurations. Both
nodes call `run_structured_with_tools` unconditionally; `knowledge_base_enabled` selects only what
the LLM is offered (`tools=[search_strategy_wiki]` and `max_rounds=knowledge_base_tool_max_rounds`
when True, `tools=[]` and `max_rounds=0` when False). With no tools nothing is bound and the loop
body never runs, so the helper degenerates to a single structured call plus the shared
fallback/retry — which is precisely why neither node keeps its own copy of that logic any more
(before #153 the knowledge-base-off branch hand-rolled it, and the retry was dead code there). The
older `invoke_structured_or_freetext` helper, used by other agents, has no retry.

- **`structured_output_repair_retry`** (env: `TRADINGAGENTS_STRUCTURED_OUTPUT_REPAIR_RETRY`, default
  `True`): enable/disable the retry. When enabled and the first structured call fails, the system
  appends a self-contained instruction and states that the reply must be JSON only, with no prose.
  The retry attempt and its outcome (success or failure) are logged at WARNING level. When disabled
  or when the provider does not support structured output (`structured_llm is None`), no retry
  happens — the fallback to free text fires immediately on the first failure.

- The instruction is derived from `response_model.model_json_schema()`, not from raw Python
  annotations: each field is rendered with its requiredness, a readable type phrase (enum members as
  an explicit list of legal values, e.g. `one of: "Buy", "Overweight", …`; numeric bounds; array item
  shapes) and its `Field(description=...)` text, which per `agents/schemas.py` *is* the model's
  output instruction. Rendering `field_info.annotation` instead produced `<enum 'PortfolioRating'>`
  and named none of the legal values on `rating`/`action` — the fields most likely to be malformed.

- If the retry also fails, behaviour is identical to the post-#152 fallback: the *tool-loop* trace's
  last `AIMessage` content is reused if it's non-empty and non-whitespace-only, otherwise a fresh
  `llm.invoke` is made for a fallback response. The repair instruction is deliberately excluded from
  that decision (it would always force the extra invoke the #152 fix exists to avoid), but it *is*
  present in the returned `message_trace`, which records everything that was sent.

### Structured output method selection (issue #161)

The method used for structured output binding (`function_calling`, `json_schema`, `json_mode`)
affects which models can reliably emit structured responses. LangChain defaults to `function_calling`,
but Ollama's OpenAI-compatibility layer ignores `tool_choice` directives, causing `function_calling`
to silently return `None` when the model emits prose instead of a tool call.

- **`structured_output_method`** (env: `TRADINGAGENTS_STRUCTURED_OUTPUT_METHOD`, default `"auto"`):
  controls which method is used for structured-output binding. Legal values:
  - `"auto"` (default): use the capability table's `preferred_structured_method` for each model,
    except Ollama defaults to `json_schema` (grammar-constrained at decode time, cannot produce
    invalid output).
  - `"function_calling"`, `"json_schema"`, `"json_mode"`: force a specific method globally for
    all models.

  Precedence order (highest first): explicit `method=` argument in code > config value (when not
  `"auto"`) > Ollama provider override (→ `json_schema`) > capability table's
  `preferred_structured_method` > default. Models with `preferred_structured_method="none"` still
  raise `NotImplementedError` and fall back to free text, regardless of config overrides, so
  agents always degrade gracefully when a provider has no structured-output support.

### Structured output text extraction (issue #162)

The recovery ladder for structured output is now: structured call → (after #160) schema-repair retry
→ free-text fallback → **LLM-free text extraction** (new, issue #162) → abort. The final rung attempts
to build a `response_model` instance directly from the free-text fallback using deterministic text
matching (no LLM call), recovering the decision the model already produced when structured calls failed.

Text extraction tries candidates in priority order:
1. Bare JSON (the whole string parses as a JSON object)
2. Fenced ```json … ``` block
3. Bare ``` … ``` fenced block (any content)
4. First balanced `{…}` object embedded in surrounding prose (respects string literals)

Each candidate is validated against `response_model`; the first that validates wins. Extraction is
designed to be fast and deterministic — no LLM calls, no latency penalty — and handles the common case
where a structured call fails but the free-text fallback contains a perfectly good JSON response.

- **`structured_output_text_extraction`** (env: `TRADINGAGENTS_STRUCTURED_OUTPUT_TEXT_EXTRACTION`,
  default `True`): enable/disable text extraction. When disabled, behaviour is byte-for-byte
  pre-#162 (fall back to free text and abort). When enabled and extraction succeeds, the extracted
  structured result replaces the fallback so the ticker proceeds with a valid decision.
- Successful extractions are logged at WARNING level so a run that only survived via extraction is
  visible in the logs rather than looking like a clean structured success. The log message includes
  the agent name for context (e.g., "PortfolioManager recovered from free-text fallback via text
  extraction").
- No per-node changes are required; both the Portfolio Manager and Swing Trader benefit automatically
  since both go through `run_structured_with_tools`.

This rung is a complement to the schema-repair retry (#153): where the retry asks the model "try
again" at an LLM call, extraction recovers the answer the model **already gave** at zero cost. Issue
#161 (Ollama → `json_schema`) is expected to make this path rarely fire in practice; it stays worth
having as the last line of defence for other providers and for schema-validation failures.

### Oversize-prompt enforcement (issues #149, #154, #169)

`docs/analysis/prompt-truncation-diagnosis.md` (issue #148) found that a prompt exceeding a local
Ollama model's actual (VRAM-tiered auto-fit, not fixed) context window can be silently truncated with
no error and no indication in `llm_calls.jsonl` — and that the provider-reported `input_tokens` cannot
be used to detect this (it locks to a wrong constant past a size threshold). `TradingAgentsGraph`
addresses this by attaching `ContextWindowGuardHandler` (`tradingagents/llm_call_log.py`) to every LLM
call's callbacks, regardless of what callbacks the caller passed in:

- **`ollama_num_ctx`** (env: `TRADINGAGENTS_OLLAMA_NUM_CTX`, default `None`/unset): explicit,
  *fixed* context length forwarded to Ollama's native `/api/chat` endpoint on every request (issue
  #169 moved this off the OpenAI-compatible endpoint, which silently dropped it — see "LLM
  providers" above) and used as the *known* context window for `quick_think_llm`/`deep_think_llm`
  when `llm_provider` is `"ollama"`. When set, it wins outright — issue #154's per-request
  derivation below never runs, and behavior matches #149 exactly (same value on every call).
- **`ollama_think`** (env: `TRADINGAGENTS_OLLAMA_THINK`, default `False`): a three-state
  (`True`/`False`/`None`) think-mode knob for Ollama models, forwarded to `ChatOllama`'s native
  `reasoning` field (issue #169 moved this off the OpenAI-compatible endpoint, which silently
  dropped `think` entirely — see "LLM providers" above for the full True/False/None contract and
  why the effective default changed to `False`). Only meaningful for `llm_provider == "ollama"`.
  Composable with `ollama_num_ctx` and per-request derivation — both can be active at the same time.
- **`context_window_overrides`** (config-only dict, `{model_name: context_window_tokens}`, default
  `{}`): lets any provider/model opt into the same check without this codebase guessing a limit it has
  no evidence for. Takes precedence over the `ollama_num_ctx`-derived entry for the same model name.
- **`context_window_safety_margin`** (env: `TRADINGAGENTS_CONTEXT_WINDOW_SAFETY_MARGIN`, default
  `1.3`): multiplier applied to the #147 `prompt_tokens_estimated` figure before comparing it to the
  known window (or, in #154's derivation, before adding response headroom), since #148's calibration
  found that estimate under-counts the real (provider-native) tokenizer's output on the
  Ollama/Ministral corpus by roughly 1.3x–1.9x.
- **`context_window_check_enabled`** (env: `TRADINGAGENTS_CONTEXT_WINDOW_CHECK_ENABLED`, default
  `True`): the escape hatch for the *abort*. It does not disable #154's derivation/sending of
  `num_ctx` itself (see below) — only whether an oversize request aborts the run instead of being
  sent anyway.

**Issue #154 — per-request derivation.** Leaving `ollama_num_ctx` unset no longer means "unchecked and
uncontrolled": for the `ollama` provider, `num_ctx` is instead derived **per request** from that
request's own measured prompt size, so every agent's prompt — not just whichever ones happen to fit
under one static value picked for the whole run — reaches Ollama untruncated:

```
needed  = ceil(prompt_tokens_estimated * context_window_safety_margin) + ollama_num_ctx_response_headroom
num_ctx = min(needed, ollama_num_ctx_max)
```

- **`ollama_num_ctx_max`** (env: `TRADINGAGENTS_OLLAMA_NUM_CTX_MAX`, default `32768`): ceiling on the
  derived value. When a request's `needed` exceeds it, the call is not dispatched.
- **`ollama_num_ctx_response_headroom`** (env: `TRADINGAGENTS_OLLAMA_NUM_CTX_RESPONSE_HEADROOM`,
  default `2048`): tokens reserved on top of the prompt requirement so the model has room to write its
  response — Ollama's `num_ctx` bounds prompt + completion together, not just the prompt.

`TradingAgentsGraph._build_ollama_num_ctx_derivation` builds an `OllamaNumCtxDerivation` (in
`tradingagents/llm_call_log.py`) from these three values plus `quick_think_llm`/`deep_think_llm`, and
the *same* object is shared by three collaborators so their arithmetic can't drift apart:
`ContextWindowGuardHandler` (aborts before dispatch when `needed > ollama_num_ctx_max`),
`LLMCallLogHandler` (records the derived `num_ctx` on the successful call's `llm_calls.jsonl` record,
under the `ollama_num_ctx` field, so truncation can be audited after the fact), and
`NormalizedChatOllama._chat_params` (`tradingagents/llm_clients/ollama_client.py` — issue #169
re-pointed this from `OllamaChatOpenAI._get_request_payload` on the now-removed OpenAI-compatible
client to `ChatOllama`'s own per-request assembly hook, the equivalent seam since `ChatOllama` has
no `_get_request_payload`) which actually attaches the derived value to the outgoing request's
`options.num_ctx`.

A model with no known context window (fixed or derived) is never checked. When a checked prompt's
adjusted estimate exceeds the known window/ceiling, the run aborts with `PromptContextOverflowError`
(naming the agent, model, measured/adjusted prompt size, and the limit that was exceeded) before the
call is dispatched, and the failure is also written to that ticker's `llm_calls.jsonl` with the same
error-record shape a failed call gets. This follows the same hard-fail precedent as the memory MCP
server above: `run_trading_agents.py`'s per-ticker `except Exception` already treats this like any
other run-aborting error (flushes the call log, prints the error, exits) with no code change needed
for that behavior.

### Portfolio Manager structured decision requirement (issue #156)

The Portfolio Manager produces decisions via structured output (`PortfolioDecision` schema). When a
provider does not support structured output or the LLM fails to produce valid output, the node falls
back to free-text generation and attempts to parse the prose. A completely failed decision —
`portfolio_structured_data` is `None` or the `rating` is invalid — is now a hard failure, following
the same precedent as `MemoryMCPConnectionError` and `PromptContextOverflowError`.

- **`portfolio_manager_require_structured_decision`** (env: `TRADINGAGENTS_PORTFOLIO_MANAGER_REQUIRE_STRUCTURED_DECISION`,
  default `True`): when enabled, the Portfolio Manager node aborts the ticker with `PortfolioDecisionError`
  if the structured decision is missing or invalid (rating not in the 5-tier scale). When disabled, the
  node falls back to the pre-#156 behavior and attempts to extract a rating from the free-text fallback
  via `SignalProcessor` parsing.

When a ticker aborts with `PortfolioDecisionError`, the exception propagates to `run_trading_agents.py`'s
per-ticker `except Exception` handler, which treats it like any other run-aborting error: flushes the
call log, prints the error, and **exits the process** (`run_trading_agents.py:977` calls `sys.exit(1)`),
ending the entire batch rather than continuing to the next ticker. This fail-fast behaviour is
deliberate (see `docs/analysis/structured-output-failure-diagnosis.md` for the investigation that
surfaced it) and stays as-is; a ticker that never gets far enough to hit this handler simply never
appears in `portfolio_ratings` passed to `run_portfolio_mode`.

### Accessing configuration in code

Inside agents and tools, call `get_config()` from `tradingagents.dataflows.config` to read the
live config dict. Changes made to the dict after startup do not affect existing references —
calling `set_config(new_dict)` updates the global copy for subsequent `get_config()` calls.

## Conventions from prior work (see `LEARNINGS.md`)

- Non-determinism across repeated runs of the same ticker/date comes mostly from LLM
  stochasticity and from computations the LLM does inline in prose; prefer pre-computing values
  in Python and handing agents the result over asking them to compute/reason numerically.
  Structured/explicit decision+confidence output improves stability in borderline cases.
- Shortening prompts to save tokens has previously made the system *more* unstable when it cut
  context the agent actually needed — trim redundancy, not load-bearing context.
