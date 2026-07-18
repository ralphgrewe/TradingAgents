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

Five sequential stages, defined in `tradingagents/graph/setup.py` and gated by
`tradingagents/graph/conditional_logic.py`. The research stage (II) can be configured via
`research_stage` config key (env var `TRADINGAGENTS_RESEARCH_STAGE`):

```
I.   ANALYST TEAM   → selected analysts run in sequence (default: market → social → news →
                       fundamentals), each loops with its own tools until it writes one report
II.  RESEARCH STAGE → configured by research_stage:
     - "researcher" (default): single Researcher node synthesizes analyst reports + live web search
       evidence (when trade_date == today, via Tavily API; historical dates degrade to
       synthesis-only with metadata "disabled (historical date)")
     - "debate": Bull vs Bear debate (alternating, `max_debate_rounds`) → Research Manager
       writes a structured verdict (`investment_plan`)
     - "none": skip research entirely, send analyst reports directly to trader
III. TRADER         → turns the research plan into a concrete trade proposal
IV.  RISK TEAM       → Aggressive → Conservative → Neutral debate (`max_risk_discuss_rounds`)
V.   PORTFOLIO MGR   → writes the final decision (`final_trade_decision`)
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
  (`market_report`, `sentiment_report`, `news_report`, `fundamentals_report`) are the load-bearing
  fields; `investment_debate_state` / `risk_debate_state` hold per-speaker history + a
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
ollama, openrouter, mistral) all share `OpenAIClient`; anthropic, google, azure, and perplexity
each have a dedicated client. Provider-specific "thinking" knobs (`google_thinking_level`,
`openai_reasoning_effort`, `anthropic_effort`) are applied per-provider in
`TradingAgentsGraph._get_provider_kwargs`.

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
  
  **Access path**: `trading_graph.py` and `skills/memory-review/find_patterns.py` reach the
  SQLite core over the networked memory MCP server (via `MemoryMCPClient`), not by opening
  the DB in-process. The memory MCP server (`mcp_server.py`) is a **hard dependency** for
  `trading_graph.py` — if the server is unreachable or a tool call fails, the run aborts with
  a `MemoryMCPConnectionError` or `MemoryMCPToolError` and does not proceed. This is a
  deliberate behavior change from the prior in-process warn-and-continue pattern (issue #53).
  `skills/` agents access the same tools via MCP tool registration in Claude Desktop/Code
  (see README.md "To register the server with Claude agents").
  
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
