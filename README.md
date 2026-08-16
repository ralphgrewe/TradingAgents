# TradingAgents-Prompts

This repository is a fork of [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents),
a multi-agent LLM framework that simulates a trading firm (analysts → researchers → trader → risk
team → portfolio manager) to analyze a stock/crypto ticker for a given date and produce a
BUY/SELL/HOLD decision.

The original upstream README — project background, paper, architecture diagram, full CLI/config
reference — has been preserved as [README-original.md](README-original.md). This README only
covers the two things you need to get going quickly in this fork: running a single ticker through
the pipeline, and starting the memory-aware MCP server.

## Setup

Always use the project virtualenv, never system Python:

```bash
python -m venv venv
./venv/bin/pip install -r requirements.txt
```

## Running a single trading agent

`run_trading_agents.py` runs the full agent pipeline for one or more tickers. It accepts either:
1. A **stock list file** (legacy): a JSON array of `{"ticker": ..., "date": ...}` objects
2. A **run config file** (new): a JSON object with pipeline parameters and a `stocks_file` reference

### Research stages

By default, the system uses `research_stage=researcher`, where a single Researcher node synthesizes
analyst reports and live web search evidence. Two alternative modes are available:

- **`research_stage=researcher`** (default): A single Researcher node synthesizes analyst reports
  and live web search evidence. When `trade_date == today`, web search runs via the Tavily API
  (requires `TAVILY_API_KEY`); historical dates degrade gracefully (no searches, metadata line
  shows "disabled (historical date)").
  - Config keys: `research_web_search` (enable/disable, default True), `research_search_queries_max`
    (default 4), `research_evidence_token_budget` (default 3000), `data_vendors["web_search"]`
    (default "tavily" — needs `TAVILY_API_KEY`).
- **`research_stage=debate`**: Bull and Bear researchers debate with a Research Manager
  judge. Set via `TRADINGAGENTS_RESEARCH_STAGE=debate`.
- **`research_stage=none`**: Skipping the research stage entirely, sending
  analyst reports directly to the trader. Set via `TRADINGAGENTS_RESEARCH_STAGE=none`.

### Swing Trader (optional)

The swing trader is an optional stage that makes regime-gated short-term (3–15 trading day) swing
trade decisions with numeric entry/stop/target levels. It runs after the portfolio manager when enabled:

- **Enable with:** `TRADINGAGENTS_SWING_TRADER_ENABLED=true`
- **Config keys:**
  - `swing_trader_min_risk_reward` (default 1.5): minimum reward-to-risk ratio
  - `swing_trader_max_holding_days` (default 15): hard cap on holding period
  - `swing_trader_conviction_threshold` (default 0.55): minimum conviction to force action

Example:

```bash
TRADINGAGENTS_SWING_TRADER_ENABLED=true ./venv/bin/python run_trading_agents.py stocks.json --show-summary
```

When enabled, the swing trader outputs appear in the CLI report under "Swing Trader Decision" and in
the full-state JSON log. Decisions are stored in the SQLite memory core under agent key `"swing_trader"`
for pattern analysis and reflection.

### LLM-wiki strategy knowledge base (optional)

The knowledge base is a curated reference library of trading strategies, signals, and risk-management
principles extracted from academic research and practitioner wisdom. It's consulted by the portfolio
manager and swing trader on demand (when enabled) to ground decisions in established approaches.

**Adding knowledge:**

1. **Drop a PDF** into the `paper/` folder (or your configured `knowledge_ingest_dir`).
2. **Run the ingestion pipeline:**
   ```bash
   ./venv/bin/python scripts/ingest_wiki.py
   ```
   This reads the PDF, has the LLM draft a structured article, and saves it under `knowledge/wiki/`.
3. **Review and commit** the generated article (check that summary, signals, computation, evidence,
   and caveats are accurate).

**Enabling/disabling the wiki:**

- **Enable (default):** by default, `knowledge_base_enabled=True` and wiki tool calls are available
  to the portfolio manager and swing trader.
- **Disable:** set `TRADINGAGENTS_KNOWLEDGE_BASE_ENABLED=false` to skip wiki lookups (faster runs,
  lower token usage).

**Configuration:**

- `knowledge_base_dir` (default `knowledge/wiki`): where articles live.
- `knowledge_ingest_dir` (default `paper`): where the ingestion pipeline looks for PDFs.
- `data_vendors["knowledge_base"]` (default `"bm25"`): retrieval backend (currently only BM25).
- `knowledge_base_tool_max_rounds` (default `2`): maximum tool-calling loop rounds to prevent
  unbounded searches.

Set any of these via `TRADINGAGENTS_*` env vars (e.g., `TRADINGAGENTS_KNOWLEDGE_BASE_ENABLED=false`).

### Using a run config file

Instead of typing long command lines, save your run parameters in a config file:

```json
{
  "stocks_file": "stocks.json",
  "llm_provider": "mistral",
  "deep_think_llm": "mistral-large",
  "quick_think_llm": "mistral-small",
  "report_dir": "./reports/mistral-aggressive",
  "show_summary": true,
  "portfolio": true,
  "style": "aggressive",
  "depot_id": "mistral-depot",
  "memory_id": "mistral-aggressive"
}
```

Then run it with:

```bash
./venv/bin/python run_trading_agents.py config.json
```

The `stocks_file` path is resolved relative to the config file's directory if relative, or used
as-is if absolute. All recognized config keys are optional except `stocks_file`; unspecified keys
use the same defaults as CLI flags.

**Config key reference:**
- `stocks_file` (required): path to stock list JSON
- `llm_provider` (default: "ollama"): LLM provider
- `deep_think_llm`, `quick_think_llm`: model names (required when llm_provider is not ollama)
- `report_dir` (default: "./reports"): output directory for reports
- `show_summary` (default: false): display formatted summary
- `use_dates_from_json` (default: false): use date field from stock list
- `portfolio` (default: false): enable portfolio mode
- `style`: "aggressive" or "conservative" (required with portfolio)
- `depot_id`: named depot (required with portfolio)
- `memory_id`: isolate decision history (optional)

**Config precedence:** CLI flags override top-level config keys, which override the nested `"config"` block,
which override environment variables, which override defaults. This full five-tier chain applies to any
`DEFAULT_CONFIG` key (those in `tradingagents/default_config.py` with a corresponding `TRADINGAGENTS_*`
env var: `research_stage`, `max_debate_rounds`, `temperature`, `swing_trader_enabled`, etc., plus
`llm_provider`, `deep_think_llm`, `quick_think_llm`, and `memory_id`). The remaining keys
(`report_dir`, `show_summary`, `use_dates_from_json`, `portfolio`, `style`, `depot_id`) are script-only
with no env var of their own, so only the CLI-flag/top-level-config/built-in-default tiers apply to them.

If a key appears in both a top-level field and in the nested `"config"` block, the top-level value wins.
For nested dict keys (like `data_vendors`), the `"config"` block does a one-level deep merge: setting
`"config": {"data_vendors": {"news_data": "alpha_vantage"}}` keeps the other `data_vendors` entries intact.

This allows tweaking a saved config ad hoc without editing the file:

```bash
./venv/bin/python run_trading_agents.py config.json --show-summary  # Override config if needed
```

#### Nested "config" block for DEFAULT_CONFIG keys

To set runtime parameters (like research strategy, temperature, or swing trader settings) without CLI
flags or env vars, use the nested `"config"` object. Any `DEFAULT_CONFIG` key is settable:

```json
{
  "stocks_file": "stocks.json",
  "llm_provider": "mistral",
  "deep_think_llm": "mistral-large",
  "quick_think_llm": "mistral-small",
  "config": {
    "research_stage": "debate",
    "max_debate_rounds": 2,
    "swing_trader_enabled": true,
    "temperature": 0.3,
    "data_vendors": {
      "news_data": "alpha_vantage"
    }
  }
}
```

All config keys are optional except `stocks_file`. Values in the `"config"` block must match the type
of the corresponding `DEFAULT_CONFIG` default (bool for booleans, int/float for numbers, string for strings,
object for nested dicts). This is a type *check*, not a coercion (unlike the `TRADINGAGENTS_*` env var
path, JSON values already carry real types). Unrecognized keys in the `"config"` block will cause an
error at run start, as will a type mismatch — with three deliberate exceptions:
- an `int` is accepted where the default is a `float` (e.g. `"swing_trader_min_risk_reward": 2` for
  its `1.5` default);
- `bool` is never treated as interchangeable with `int`/`float` in either direction, even though `bool`
  is a subclass of `int` in Python — `"max_debate_rounds": true` (bool for an int default) and
  `"swing_trader_enabled": 1` (int for a bool default) are both rejected;
- keys whose own `DEFAULT_CONFIG` default is `None` (e.g. `temperature`, `memory_id`, `backend_url`,
  `benchmark_ticker`, `simulation_server_command`/`_args`, `memory_mcp_url`, `memory_log_max_entries`,
  and the provider "thinking" knobs) have no type to check against, so any JSON type is accepted and
  passed through as-is.

### Stock list format

**Default behavior (today's date mode):** by default, every ticker is run against today's date:

```bash
echo '[{"ticker": "AAPL"}, {"ticker": "MSFT"}]' > stocks.json
./venv/bin/python run_trading_agents.py stocks.json --show-summary

# With researcher mode and live web search:
TRADINGAGENTS_RESEARCH_STAGE=researcher ./venv/bin/python run_trading_agents.py stocks.json --show-summary
```

The `"date"` field is optional and ignored in default mode. The script computes today's date once
at startup and uses it for all tickers in the batch.

**Legacy behavior (per-ticker dates from JSON):** to run each ticker against a date field in the
JSON, use the `--use-dates-from-json` flag:

```bash
echo '[{"ticker": "AAPL", "date": "2024-01-15"}, {"ticker": "MSFT", "date": "2024-01-16"}]' > stocks.json
./venv/bin/python run_trading_agents.py stocks.json --use-dates-from-json --show-summary
```

When `--use-dates-from-json` is active, every stock entry must have a non-empty `"date"` field, or
the script exits with an error.

By default this uses local Ollama models (`ministral-3:8b` / `ministral-3:3b`) — no API key
needed. To use a hosted provider instead, pass `--llm-provider` plus both model flags, and make
sure the corresponding API key env var is set:

```bash
OPENAI_API_KEY=... ./venv/bin/python run_trading_agents.py stocks.json \
  --llm-provider openai --deep-think-llm gpt-5.5 --quick-think-llm gpt-5.5-mini \
  --show-summary --report-dir ./reports
```

See the docstring at the top of `run_trading_agents.py` for the full flag reference (portfolio
mode, report directory, etc.).

### Isolating decision memory per model/configuration (optional)

By default, every run shares one SQLite decision-memory store at `runs/memory/memory.db`. If
you run the same stock list under different models or configurations, their decisions land in
the same DB and contaminate each other's history (each run injects past context from that DB
into the agent prompts). `--memory-id` gives each configuration its own isolated history:

```bash
./venv/bin/python run_trading_agents.py stocks.json --memory-id gpt5
./venv/bin/python run_trading_agents.py stocks.json --memory-id ollama-qwen
```

This writes to `runs/memory/gpt5/memory.db` and `runs/memory/ollama-qwen/memory.db`
respectively, instead of the shared default. `TRADINGAGENTS_MEMORY_ID` sets the same thing via
environment variable (useful for `cli/main.py` and the MCP server's `analyze_stock` tool, which
don't have a `--memory-id` flag). Precedence when both a flag/config value and env vars are
present: `--memory-id` > `TRADINGAGENTS_MEMORY_ID` > `TRADINGAGENTS_MEMORY_DB_PATH` > default.
Memory IDs must match `^[A-Za-z0-9][A-Za-z0-9._-]*$` (no `/`, `\`, `..`, or empty string) since
they become a directory name; an invalid ID exits with an error before the run starts. With no
`--memory-id` and no `TRADINGAGENTS_MEMORY_ID` set, behavior is unchanged from before this
option existed.

## Starting the memory MCP server

`mcp_server.py` exposes the pipeline and the shared decision-memory store as MCP tools
(`analyze_stock`, plus `memory_store_decision` / `memory_resolve_pending` /
`memory_get_past_context` / `memory_get_statistics`).

### Stdio transport (default — for Claude Desktop, Claude Code)

```bash
./venv/bin/python mcp_server.py
```

To register it with an MCP client (e.g. Claude Desktop or Claude Code), point it at this script
with the repo as the working directory, for example:

```json
{
  "mcpServers": {
    "tradingagents": {
      "command": "<absolute_path>/venv/bin/python",
      "args": ["<absolute_path>/TradingAgents-Prompts/mcp_server.py"],
      "cwd": "<absolute_path>/TradingAgents-Prompts"
    }
  }
}
```

### Networked transports (streamable-http, sse)

For running the server on a different machine or accessing it over a network, use one of the
bundled start scripts (which set up the environment and launch the server on `127.0.0.1:8001`;
port 8001 keeps the memory MCP server clear of the trading simulation server on port 8000):

**macOS / Linux:**
```bash
./start_server.sh
```

**Windows:**
```cmd
start_server.bat
```

Alternatively, set the environment variables manually:

```bash
MCP_TRANSPORT=streamable-http FASTMCP_HOST=0.0.0.0 FASTMCP_PORT=8001 ./venv/bin/python mcp_server.py
```

**To point `trading_graph.py` at a networked server:**

`trading_graph.py` connects to the memory MCP server using `MemoryMCPClient` (issue #51). When
running against a networked transport, point it at the server via:

```bash
TRADINGAGENTS_MEMORY_MCP_URL=http://<host>:8001/mcp ./venv/bin/tradingagents  # streamable-http (default path: /mcp)
TRADINGAGENTS_MEMORY_MCP_URL=http://<host>:8001/sse TRADINGAGENTS_MEMORY_MCP_TRANSPORT=sse ./venv/bin/tradingagents  # SSE (default path: /sse)
```

If the server is on `localhost` and the client is on the same machine, the default connection
parameters work automatically (both default to `http://127.0.0.1:8001` with the appropriate
path for the resolved transport).

**To register the server with Claude agents (Claude Desktop, Claude Code):**

When running the server on a networked transport, Claude agents (e.g. skills in `skills/`)
can call the `memory_*` MCP tools by registering the server. Add to your Claude Desktop or
Claude Code MCP configuration (e.g. `claude_desktop_config.json` or `.claude/settings.json`):

For **streamable-http** (recommended):
```json
{
  "mcpServers": {
    "tradingagents-memory": {
      "url": "http://localhost:8001/mcp"
    }
  }
}
```

For **SSE**:
```json
{
  "mcpServers": {
    "tradingagents-memory": {
      "url": "http://localhost:8001/sse"
    }
  }
}
```

Replace `localhost` with the actual hostname/IP if the server is on a different machine. The
`memory_*` tools (`memory_store_decision`, `memory_resolve_pending`, `memory_get_past_context`,
`memory_get_statistics`, `memory_get_decisions`) are then available to any agent that runs in
that environment.

**Environment variables:**
- `MCP_TRANSPORT`: Transport type (`"stdio"` default, or `"streamable-http"`, `"sse"`).
- `FASTMCP_HOST`: Host/interface to bind to for networked transports (`"127.0.0.1"` default).
- `FASTMCP_PORT`: Port to bind to for networked transports (`"8001"` default).
- `TRADINGAGENTS_MEMORY_MCP_URL`: Full server URL for `trading_graph.py` / `MemoryMCPClient` (e.g. `"http://127.0.0.1:8001/mcp"`). If unset, the client derives a default from the *resolved* transport's FastMCP mount path (`/mcp` or `/sse`) on `127.0.0.1:8001` — a hardcoded client-side default, independent of the server's `FASTMCP_HOST`/`FASTMCP_PORT` bind settings (only relevant if server and client happen to both run locally with defaults).
- `TRADINGAGENTS_MEMORY_MCP_TRANSPORT`: Transport type for the memory client (`"streamable-http"` default, or `"sse"`) — selects which client implementation connects (`streamable_http_client` vs `sse_client`). Resolved **independently** of `TRADINGAGENTS_MEMORY_MCP_URL`: each of URL and transport follows its own precedence (explicit `MemoryMCPClient(...)` argument > this env var > built-in default — see `_resolve_connection` in `tradingagents/memory/mcp_client.py`), so setting a URL does not disable transport resolution — you must set both together when they need to match (as in the SSE example above).
- `TRADINGAGENTS_MEMORY_MCP_TIMEOUT`: Seconds allowed for establishing the memory MCP session and for each individual tool call (`30` default). A server that accepts the connection but never answers — or an HTTP proxy answering in its place — now fails the run with a `MemoryMCPConnectionError`/`MemoryMCPToolError` after this long instead of hanging indefinitely.

> **Behind an HTTP proxy:** the client talks to `127.0.0.1` by default, and httpx's `no_proxy`
> handling only matches literal hosts, so a CIDR entry such as `no_proxy=127.0.0.0/8` does *not*
> exempt `127.0.0.1` and the request gets handed to `http_proxy` (which typically answers `503`).
> Requests to a loopback memory server therefore bypass environment proxies entirely; a remote
> `TRADINGAGENTS_MEMORY_MCP_URL` still honors them, so list that host in `no_proxy` by name if it
> should be reached directly.

## More detail

- [CLAUDE.md](CLAUDE.md) — architecture, LLM providers, data vendors, persistence, and
  configuration reference for this fork.
- [README-original.md](README-original.md) — upstream project README (paper, background, full CLI
  walkthrough).
