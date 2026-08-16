import os

_TRADINGAGENTS_HOME = os.path.join(os.path.expanduser("~"), ".tradingagents")

# Single source of truth for env-var → config-key overrides. To expose
# a new config key for environment-based override, add a row here — no
# entry-point script changes required. Coercion is driven by the type
# of the existing default, so users can keep writing plain strings in
# their .env file.
_ENV_OVERRIDES = {
    "TRADINGAGENTS_LLM_PROVIDER":         "llm_provider",
    "TRADINGAGENTS_DEEP_THINK_LLM":       "deep_think_llm",
    "TRADINGAGENTS_QUICK_THINK_LLM":      "quick_think_llm",
    "TRADINGAGENTS_LLM_BACKEND_URL":      "backend_url",
    "TRADINGAGENTS_OUTPUT_LANGUAGE":      "output_language",
    "TRADINGAGENTS_MAX_DEBATE_ROUNDS":    "max_debate_rounds",
    "TRADINGAGENTS_MAX_RISK_ROUNDS":      "max_risk_discuss_rounds",
    "TRADINGAGENTS_CHECKPOINT_ENABLED":   "checkpoint_enabled",
    "TRADINGAGENTS_BENCHMARK_TICKER":     "benchmark_ticker",
    "TRADINGAGENTS_TEMPERATURE":          "temperature",
    "TRADINGAGENTS_SIMULATION_SERVER_COMMAND": "simulation_server_command",
    "TRADINGAGENTS_SIMULATION_SERVER_ARGS": "simulation_server_args",
    "TRADINGAGENTS_MEMORY_MCP_URL":       "memory_mcp_url",
    "TRADINGAGENTS_MEMORY_MCP_TRANSPORT": "memory_mcp_transport",
    "TRADINGAGENTS_MEMORY_MCP_TIMEOUT":   "memory_mcp_timeout",
    "TRADINGAGENTS_MEMORY_ID":            "memory_id",
    "TRADINGAGENTS_RESEARCH_STAGE":       "research_stage",
    "TRADINGAGENTS_RESEARCH_WEB_SEARCH":  "research_web_search",
    "TRADINGAGENTS_RESEARCH_SEARCH_QUERIES_MAX": "research_search_queries_max",
    "TRADINGAGENTS_RESEARCH_EVIDENCE_TOKEN_BUDGET": "research_evidence_token_budget",
    "TRADINGAGENTS_LLM_TIMEOUT":          "llm_timeout",
    "TRADINGAGENTS_SWING_TRADER_ENABLED": "swing_trader_enabled",
    "TRADINGAGENTS_SWING_TRADER_MIN_RISK_REWARD": "swing_trader_min_risk_reward",
    "TRADINGAGENTS_SWING_TRADER_MAX_HOLDING_DAYS": "swing_trader_max_holding_days",
    "TRADINGAGENTS_SWING_TRADER_CONVICTION_THRESHOLD": "swing_trader_conviction_threshold",
    "TRADINGAGENTS_KNOWLEDGE_BASE_ENABLED": "knowledge_base_enabled",
    "TRADINGAGENTS_KNOWLEDGE_BASE_DIR":     "knowledge_base_dir",
    "TRADINGAGENTS_KNOWLEDGE_INGEST_DIR":   "knowledge_ingest_dir",
    "TRADINGAGENTS_KNOWLEDGE_BASE_TOOL_MAX_ROUNDS": "knowledge_base_tool_max_rounds",
    "TRADINGAGENTS_DATA_VENDORS_KNOWLEDGE_BASE": "data_vendors.knowledge_base",
    "TRADINGAGENTS_SELECTED_ANALYSTS": "selected_analysts",
}


def _coerce(value: str, reference):
    """Coerce env-var string to the type of the existing default value."""
    if isinstance(reference, bool):
        return value.strip().lower() in ("true", "1", "yes", "on")
    if isinstance(reference, int) and not isinstance(reference, bool):
        return int(value)
    if isinstance(reference, float):
        return float(value)
    if isinstance(reference, list):
        # For lists, parse as space/comma-separated values
        return [x.strip() for x in value.replace(",", " ").split() if x.strip()]
    return value


def _get_nested(d, keys):
    """Get a nested value from a dict using a list of keys (separated by dot in key_path)."""
    current = d
    for k in keys:
        if isinstance(current, dict):
            current = current.get(k)
        else:
            return None
    return current


def _set_nested(d, keys, value):
    """Set a nested value in a dict using a list of keys, creating intermediate dicts as needed."""
    current = d
    for k in keys[:-1]:
        if k not in current:
            current[k] = {}
        current = current[k]
    current[keys[-1]] = value


def _apply_env_overrides(config: dict) -> dict:
    """Apply TRADINGAGENTS_* env vars to the config dict in-place.

    Supports both top-level keys and nested keys using dot notation (e.g., "data_vendors.knowledge_base").
    """
    for env_var, key_path in _ENV_OVERRIDES.items():
        raw = os.environ.get(env_var)
        if raw is None or raw == "":
            continue

        # Handle nested keys using dot notation (e.g., "data_vendors.knowledge_base")
        if "." in key_path:
            keys = key_path.split(".")
            reference = _get_nested(config, keys)
            _set_nested(config, keys, _coerce(raw, reference))
        else:
            # Top-level key
            config[key_path] = _coerce(raw, config.get(key_path))
    return config


DEFAULT_CONFIG = _apply_env_overrides({
    "project_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
    "results_dir": os.getenv("TRADINGAGENTS_RESULTS_DIR", os.path.join(_TRADINGAGENTS_HOME, "logs")),
    "data_cache_dir": os.getenv("TRADINGAGENTS_CACHE_DIR", os.path.join(_TRADINGAGENTS_HOME, "cache")),
    "memory_log_path": os.getenv("TRADINGAGENTS_MEMORY_LOG_PATH", os.path.join(_TRADINGAGENTS_HOME, "memory", "trading_memory.md")),
    # Optional cap on the number of resolved memory log entries. When set,
    # the oldest resolved entries are pruned once this limit is exceeded.
    # Pending entries are never pruned. None disables rotation entirely.
    "memory_log_max_entries": None,
    # LLM settings
    "llm_provider": "ollama",
    "deep_think_llm": "ministral-3:8b",
    "quick_think_llm": "ministral-3:3b",
    # When None, each provider's client falls back to its own default endpoint
    # (api.openai.com for OpenAI, generativelanguage.googleapis.com for Gemini, ...).
    # The CLI overrides this per provider when the user picks one. Keeping a
    # provider-specific URL here would leak (e.g. OpenAI's /v1 was previously
    # being forwarded to Gemini, producing malformed request URLs).
    "backend_url": None,
    # Provider-specific thinking configuration
    "google_thinking_level": None,      # "high", "minimal", etc.
    "openai_reasoning_effort": None,    # "medium", "high", "low"
    "anthropic_effort": None,           # "high", "medium", "low"
    # Sampling temperature, forwarded to every provider when set. None leaves
    # each provider at its own default. Lower values reduce run-to-run
    # variation on models that honor it; reasoning models largely ignore it
    # and no setting makes LLM output bit-identical across runs.
    "temperature": None,
    # Network timeout (seconds) for LLM API calls, forwarded as ``timeout`` to
    # the underlying chat client (ChatOpenAI/ChatAnthropic/...). Without an
    # explicit timeout, langchain-openai passes ``timeout=None`` straight
    # through to httpx, which disables timeouts entirely rather than falling
    # back to the openai SDK's own 600s default — so a wedged local provider
    # (e.g. an ollama daemon that's up but never responds: mid model-pull, or
    # stuck loading) hangs the run forever with no CPU/GPU activity and no
    # error (#108). 120s is generous enough for a slow CPU-bound local model
    # to finish a normal turn while still failing fast on a truly wedged
    # endpoint; override via TRADINGAGENTS_LLM_TIMEOUT for slower hardware.
    "llm_timeout": 120,
    # Checkpoint/resume: when True, LangGraph saves state after each node
    # so a crashed run can resume from the last successful step.
    "checkpoint_enabled": False,
    # Output language for analyst reports and final decision
    # Internal agent debate stays in English for reasoning quality
    "output_language": "English",
    # Research stage mode: "researcher" (default) uses a single Researcher node with optional web search;
    # "none" bypasses research and sends analyst outputs directly to the trader;
    # "debate" keeps Bull/Bear/Research Manager debate
    "research_stage": "researcher",
    # Research web search settings
    "research_web_search": True,           # Enable/disable web search for live runs
    "research_search_queries_max": 4,      # Maximum number of search queries
    "research_evidence_token_budget": 3000, # Token budget for assembled evidence pack
    # Swing Trader settings
    "swing_trader_enabled": False,         # Enable/disable the swing trader node
    "swing_trader_min_risk_reward": 1.5,   # Minimum reward:risk for non-HOLD decisions
    "swing_trader_max_holding_days": 15,   # Hard cap on holding period (trading days)
    "swing_trader_conviction_threshold": 0.55, # Minimum conviction to force action
    # LLM-wiki strategy knowledge base (issue #100/#103): BM25 keyword retrieval
    # over knowledge/wiki/*.md articles, consulted by the portfolio manager and
    # swing trader via the search_wiki tool.
    "knowledge_base_enabled": True,       # Enable/disable knowledge-base retrieval
    "knowledge_base_dir": "knowledge/wiki", # Article directory (relative to cwd)
    "knowledge_ingest_dir": "paper",      # Default folder the ingestion pipeline scans (#102)
    "knowledge_base_tool_max_rounds": 2,  # Max tool-calling loop rounds for wiki search (issue #104)
    # Debate and discussion settings
    "max_debate_rounds": 1,
    "max_risk_discuss_rounds": 1,
    "max_recur_limit": 100,
    "analyst_concurrency_limit": 1,
    # News / data fetching parameters
    # Increase for longer lookback strategies or to broaden macro coverage;
    # decrease to reduce token usage in agent prompts.
    "news_article_limit": 20,             # max articles per ticker (ticker-news)
    "global_news_article_limit": 10,      # max articles for global/macro news
    "global_news_lookback_days": 7,       # macro news lookback window
    # Search queries used by get_global_news for macro headlines. Extend or
    # replace to broaden geographic / sector coverage.
    "global_news_queries": [
        "Federal Reserve interest rates inflation",
        "S&P 500 earnings GDP economic outlook",
        "geopolitical risk trade war sanctions",
        "ECB Bank of England BOJ central bank policy",
        "oil commodities supply chain energy",
    ],
    # Data vendor configuration
    # Category-level configuration (default for all tools in category)
    "data_vendors": {
        "core_stock_apis": "yfinance",       # Options: alpha_vantage, yfinance
        "technical_indicators": "yfinance",  # Options: alpha_vantage, yfinance
        "fundamental_data": "yfinance",      # Options: alpha_vantage, yfinance
        "news_data": "yfinance",             # Options: alpha_vantage, yfinance
        "macro_data": "fred",                # Options: fred (needs FRED_API_KEY)
        "prediction_markets": "polymarket",  # Options: polymarket (keyless)
        "web_search": "tavily",              # Options: tavily (needs TAVILY_API_KEY)
        "knowledge_base": "bm25",            # Options: bm25 (keyless, local)
    },
    # Perplexity-specific configuration
    "perplexity_model": "sonar-pro",
    "perplexity_use_agent_api": True,       # Use Agent API with web_search tools
    # Tool-level configuration (takes precedence over category-level)
    "tool_vendors": {
        # Example: "get_stock_data": "alpha_vantage",  # Override category default
    },
    # Benchmark for alpha calculation in the reflection layer.
    # ``benchmark_ticker`` (when set) overrides the suffix map for all
    # tickers; leave it None to use ``benchmark_map`` for auto-detection
    # based on the ticker's exchange suffix. SPY remains the US default
    # so the reflection label keeps reading "Alpha vs SPY" for US tickers
    # while non-US tickers get their regional index automatically.
    "benchmark_ticker": None,
    "benchmark_map": {
        ".NS":  "^NSEI",    # NSE India (Nifty 50)
        ".BO":  "^BSESN",   # BSE India (Sensex)
        ".T":   "^N225",    # Tokyo (Nikkei 225)
        ".HK":  "^HSI",     # Hong Kong (Hang Seng)
        ".L":   "^FTSE",    # London (FTSE 100)
        ".TO":  "^GSPTSE",  # Toronto (TSX Composite)
        ".AX":  "^AXJO",    # Australia (ASX 200)
        "":     "SPY",      # default for US-listed tickers (no suffix)
    },
    # McpTradingSimulation client configuration (issue #34)
    # When not set, SimulationClient auto-detects from sibling ../McpTradingSimulation checkout.
    # Override via TRADINGAGENTS_SIMULATION_SERVER_COMMAND and TRADINGAGENTS_SIMULATION_SERVER_ARGS
    # env vars if running on a different machine/checkout layout.
    "simulation_server_command": None,   # Path to Python interpreter (e.g. /path/to/venv/bin/python)
    "simulation_server_args": None,      # List of args: ["/path/to/mcp_server.py"]
    # MemoryMCPClient configuration (issue #51) — networked client for the
    # memory core's MCP server (mcp_server.py run under start_server.sh).
    # When memory_mcp_url is None, MemoryMCPClient derives a default from
    # memory_mcp_transport ("http://127.0.0.1:8001/mcp" for streamable-http,
    # "http://127.0.0.1:8001/sse" for sse) — see tradingagents/memory/mcp_client.py.
    "memory_mcp_url": None,
    "memory_mcp_transport": "streamable-http",   # Options: streamable-http, sse
    # Wall-clock bound (seconds) on every memory MCP interaction: establishing
    # the session in connect() and each individual tool call. Without it, a
    # server (or an HTTP proxy standing in front of it) that answers a request
    # with an error the MCP client never routes back to the waiting caller
    # hangs the whole run indefinitely — issue #108.
    "memory_mcp_timeout": 30.0,
    # Named memory ID for isolating SQLite decision history per run (issue #114).
    # When set, decisions are stored in runs/memory/<id>/memory.db instead of the
    # default runs/memory/memory.db. This allows different models or configurations
    # to maintain independent decision histories for the same stock list.
    # Precedence: --memory-id flag > TRADINGAGENTS_MEMORY_ID env var >
    # TRADINGAGENTS_MEMORY_DB_PATH env var > default.
    "memory_id": None,
    # List of analyst types to include in the pipeline (issue #118).
    # Options: "market", "social", "news", "fundamentals" (the keys in
    # ANALYST_NODE_SPECS, tradingagents/graph/analyst_execution.py).
    # "perplexity_news" is NOT a valid option here — it is not wired into
    # ANALYST_NODE_SPECS / the execution plan (see the NOTE in
    # tradingagents/graph/setup.py) and will fail validation before any run
    # starts.
    # Validated before run start: must be non-empty and all entries must be
    # one of the known analyst types.
    "selected_analysts": ["market", "social", "news", "fundamentals"],
})
