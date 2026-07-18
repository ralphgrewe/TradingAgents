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

`run_trading_agents.py` runs the full agent pipeline for one or more tickers from a JSON file and
prints the resulting decision.

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

## More detail

- [CLAUDE.md](CLAUDE.md) — architecture, LLM providers, data vendors, persistence, and
  configuration reference for this fork.
- [README-original.md](README-original.md) — upstream project README (paper, background, full CLI
  walkthrough).
