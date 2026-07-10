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

# Launch the interactive CLI
./venv/bin/tradingagents            # installed entry point
./venv/bin/python -m cli.main       # equivalent, run from source

# Run the MCP server
./venv/bin/python mcp_server.py               # stdio transport (default)
./start_server.sh                             # networked transport (streamable-http)
MCP_TRANSPORT=sse FASTMCP_HOST=0.0.0.0 FASTMCP_PORT=8000 ./venv/bin/python mcp_server.py  # SSE transport

# Structured-output smoke test against a real provider (costs API credits)
OPENAI_API_KEY=... ./venv/bin/python scripts/smoke_structured_output.py openai
```

Tests never hit real LLM APIs by default — `tests/conftest.py` autouse-fixtures placeholder
values for every provider's API-key env var, and `mock_llm_client` patches
`tradingagents.llm_clients.factory.create_llm_client` for tests that need a fake client.

## Architecture: the LangGraph pipeline

Five sequential stages, defined in `tradingagents/graph/setup.py` and gated by
`tradingagents/graph/conditional_logic.py`:

```
I.   ANALYST TEAM   → selected analysts run in sequence (default: market → social → news →
                       fundamentals), each loops with its own tools until it writes one report
II.  RESEARCH TEAM  → Bull vs Bear debate (alternating, `max_debate_rounds`) → Research Manager
                       writes a structured verdict (`investment_plan`)
III. TRADER         → turns the research plan into a concrete trade proposal
IV.  RISK TEAM       → Aggressive → Conservative → Neutral debate (`max_risk_discuss_rounds`)
V.   PORTFOLIO MGR   → writes the final decision (`final_trade_decision`)
```

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

Two independent, still-coexisting memory systems:

- **Legacy markdown decision log** (`tradingagents/agents/utils/memory.py`,
  `TradingMemoryLog`) — always on. Appends a pending entry at the end of every `propagate()`
  call to `~/.tradingagents/memory/trading_memory.md` (override:
  `TRADINGAGENTS_MEMORY_LOG_PATH`). On the *next* run for the same ticker,
  `TradingAgentsGraph._resolve_pending_entries` fetches realized return, generates a reflection,
  and `get_past_context` injects recent same-ticker + cross-ticker history into the Portfolio
  Manager prompt.
- **SQLite memory core** (`tradingagents/memory/{store,resolve,query}.py`) — the shared backend
  for the `skills/` reimplementation, storing one row per `(agent, ticker, decision_date)` in a
  `decisions` table (default path `runs/memory/memory.db`, override
  `TRADINGAGENTS_MEMORY_DB_PATH`). `store.py` is write-only/idempotent (`INSERT OR IGNORE`);
  `resolve.py` fills in `forward_return`/`lesson` once enough trading days have elapsed
  (`numpy.busday_count`-based, no holiday calendar); `query.py` is the read/formatting path for
  prompt injection. See the module docstrings in each file — they carry the design rationale
  (idempotency semantics, normalization boundary, single-transaction batching) and are the
  canonical reference, not this file.
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

## Conventions from prior work (see `LEARNINGS.md`)

- Non-determinism across repeated runs of the same ticker/date comes mostly from LLM
  stochasticity and from computations the LLM does inline in prose; prefer pre-computing values
  in Python and handing agents the result over asking them to compute/reason numerically.
  Structured/explicit decision+confidence output improves stability in borderline cases.
- Shortening prompts to save tokens has previously made the system *more* unstable when it cut
  context the agent actually needed — trim redundancy, not load-bearing context.
