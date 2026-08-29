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
    "TRADINGAGENTS_RISK_STAGE":           "risk_stage",
    "TRADINGAGENTS_RESEARCH_WEB_SEARCH":  "research_web_search",
    "TRADINGAGENTS_RESEARCH_SEARCH_QUERIES_MAX": "research_search_queries_max",
    "TRADINGAGENTS_RESEARCH_EVIDENCE_TOKEN_BUDGET": "research_evidence_token_budget",
    "TRADINGAGENTS_LLM_TIMEOUT":          "llm_timeout",
    "TRADINGAGENTS_OLLAMA_NUM_CTX":       "ollama_num_ctx",
    "TRADINGAGENTS_OLLAMA_NUM_CTX_MAX":   "ollama_num_ctx_max",
    "TRADINGAGENTS_OLLAMA_NUM_CTX_RESPONSE_HEADROOM": "ollama_num_ctx_response_headroom",
    "TRADINGAGENTS_OLLAMA_THINK":         "ollama_think",
    "TRADINGAGENTS_CONTEXT_WINDOW_CHECK_ENABLED": "context_window_check_enabled",
    "TRADINGAGENTS_CONTEXT_WINDOW_SAFETY_MARGIN": "context_window_safety_margin",
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
    "TRADINGAGENTS_MACRO_NEWS_CATEGORY_CAP": "macro_news_category_cap",
    "TRADINGAGENTS_LLM_CALL_LOG_ENABLED": "llm_call_log_enabled",
    "TRADINGAGENTS_LLM_CALL_LOG_PROMPTS": "llm_call_log_prompts",
    "TRADINGAGENTS_STRUCTURED_OUTPUT_REPAIR_RETRY": "structured_output_repair_retry",
    "TRADINGAGENTS_PORTFOLIO_MANAGER_REQUIRE_STRUCTURED_DECISION": "portfolio_manager_require_structured_decision",
    "TRADINGAGENTS_STRUCTURED_OUTPUT_METHOD": "structured_output_method",
    "TRADINGAGENTS_STRUCTURED_OUTPUT_TEXT_EXTRACTION": "structured_output_text_extraction",
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
    "ollama_think": False,              # enable think mode for Ollama models
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
    # Explicit Ollama context length (issue #149, per the #148 diagnosis at
    # docs/analysis/prompt-truncation-diagnosis.md): without this, Ollama's
    # actual serving context window is an invisible, VRAM-tiered auto-fit
    # ("4k/32k/256k based on VRAM") that this codebase cannot see or reason
    # about, and #148 observed it landing as low as 4096 under real memory
    # pressure. When set, this value is forwarded to Ollama's OpenAI-compatible
    # endpoint as a non-standard top-level ``options.num_ctx`` request field
    # (see ``tradingagents.llm_clients.openai_client.OpenAIClient.get_llm``)
    # AND used as the known context window for the oversize-prompt check
    # below. None (default) means "don't set it" -- Ollama keeps auto-fitting,
    # and the oversize check has no known window to check the ollama provider
    # against (an unknown limit is never enforced -- see
    # context_window_check_enabled). Only meaningful for provider "ollama".
    # See docs/local-models.md "Context-Length Knobs" for sizing guidance.
    "ollama_num_ctx": None,
    # Derived-num_ctx ceiling (issue #154): when ollama_num_ctx above is left
    # unset (the common/recommended case), num_ctx is instead derived PER
    # REQUEST from that request's own measured prompt size --
    #   needed  = ceil(prompt_tokens_estimated * context_window_safety_margin)
    #             + ollama_num_ctx_response_headroom
    #   num_ctx = min(needed, ollama_num_ctx_max)
    # -- so every agent's prompt (not just the ones that happen to fit under a
    # single static value) reaches Ollama untruncated. This is the ceiling on
    # that derived value: when `needed` exceeds it, the run aborts with
    # PromptContextOverflowError instead of sending a prompt that would be
    # silently truncated. 32768 is a context length most locally-served
    # Ministral/Llama-class 7-8B models can serve without exceeding typical
    # consumer-GPU VRAM; raise it if your Ollama daemon has the VRAM for a
    # bigger window. Only meaningful for provider "ollama", and only when
    # ollama_num_ctx above is unset -- an explicit ollama_num_ctx always wins
    # outright and this derivation does not run at all.
    "ollama_num_ctx_max": 32768,
    # Generation headroom (issue #154): tokens reserved on top of the derived
    # prompt requirement so the model actually has room to write its response
    # -- Ollama's num_ctx bounds prompt + completion together, not just the
    # prompt, so sizing num_ctx to the prompt alone would truncate the
    # response instead. 2048 comfortably covers the longest response observed
    # in docs/analysis/prompt-truncation-diagnosis.md (a round-1 Portfolio
    # Manager reply measured at 1006 tokens) with margin for longer
    # rounds/models.
    "ollama_num_ctx_response_headroom": 2048,
    # Oversize-prompt enforcement (issue #149): before an LLM call is
    # dispatched, ContextWindowGuardHandler (tradingagents/llm_call_log.py)
    # compares the #147 tiktoken/heuristic prompt-size estimate --
    # deliberately NOT the provider-reported input_tokens, which #148 showed
    # is unreliable for this comparison on Ollama past a size threshold --
    # against the known context window for that model (currently: only
    # ollama_num_ctx above, plus any per-model context_window_overrides
    # entry). A model with no known window is never checked (a guessed limit
    # is worse than no check). This is the escape hatch: set to False to
    # disable enforcement entirely (not recommended -- the whole point of
    # #149 is to fail loudly instead of silently deciding on a truncated
    # prompt).
    "context_window_check_enabled": True,
    # Multiplier applied to the prompt-size estimate before comparing it to
    # the known context window. #148's calibration against the Ollama/
    # Ministral corpus found the #147 estimate under-counts the real
    # (provider-native) tokenizer's output by roughly 1.3x-1.9x -- so
    # comparing the raw estimate to the window at face value would miss
    # prompts that are actually oversize by the model's own tokenizer. This
    # single global margin is a simplification of #148's "provider-family-
    # aware margin would be more accurate" recommendation; override per
    # environment via the env var below if calibration for your model
    # differs.
    "context_window_safety_margin": 1.3,
    # Per-model known context windows (model name -> tokens), for providers
    # other than ollama to opt into the same oversize check without this
    # codebase guessing a limit it has no evidence for. Takes precedence over
    # the ollama_num_ctx-derived entry for the same model name. Empty by
    # default: config-only (no single env var fits a dict of model names), set
    # programmatically or in a config file.
    "context_window_overrides": {},
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
    # Risk stage mode (issue #119, mirrors research_stage/#79): "debate" (default) keeps
    # the Aggressive/Conservative/Neutral risk-debate stage before the Portfolio Manager;
    # "none" bypasses it entirely and routes the Trader's plan straight to the Portfolio
    # Manager. Validated at TradingAgentsGraph construction time (must be "debate"/"none").
    "risk_stage": "debate",
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
    # Per-call LLM call log (issue #138, part of #137): when True, a
    # LLMCallLogHandler (tradingagents/llm_call_log.py) is wired into every
    # run's callbacks and appends one JSONL record per LLM call (agent/node,
    # model, prompt size, token usage, duration) to llm_calls.jsonl alongside
    # the run's other outputs. Metadata-only and small, so it defaults on.
    "llm_call_log_enabled": True,
    # Full prompt dumps for LLM calls (issue #139, part of #137): when True
    # and llm_call_log_enabled is True, the handler also writes the complete
    # rendered prompt (all messages with roles, in order) of every LLM call
    # to disk under prompts/ subdirectory. Disabled by default since dumps
    # are large and may contain fetched data users don't always want written.
    "llm_call_log_prompts": False,
    # Structured output repair retry (issue #153): when the first structured
    # output call fails, retry exactly once with an explicit schema-repair
    # instruction appended to the trace, before falling back to free text.
    # Small models often produce nearly-correct JSON and correct it when told
    # plainly what shape is required. Set to False to disable (no extra LLM
    # call). Never retries when structured_llm is None (provider doesn't
    # support structured output).
    "structured_output_repair_retry": True,
    # Portfolio Manager structured decision requirement (issue #156): when True,
    # abort the ticker with PortfolioDecisionError if the Portfolio Manager
    # produces no usable structured decision (missing portfolio_structured_data
    # or invalid rating). Set to False to accept degraded output and fall back
    # to SignalProcessor parsing (pre-#156 behavior).
    "portfolio_manager_require_structured_decision": True,
    # Structured output method selection (issue #161): controls which method
    # is used for structured-output binding (with_structured_output). Legal
    # values: "auto" (default, use capability-table resolution), "function_calling",
    # "json_schema", "json_mode". With "auto", Ollama defaults to json_schema
    # instead of function_calling. Can be overridden via env var
    # TRADINGAGENTS_STRUCTURED_OUTPUT_METHOD.
    "structured_output_method": "auto",
    # Structured output text extraction (issue #162): when True, attempt to
    # extract and parse a structured response from the free-text fallback
    # using deterministic text matching (no LLM call). This is the final rung
    # before giving up: bare JSON → fenced block → embedded braces in prose.
    # Successful extractions are logged at WARNING level. Set to False to
    # disable extraction (pre-#162 behavior).
    "structured_output_text_extraction": True,
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
    # Per-category article cap for get_macro_news (issue #133): bounds prompt
    # size after dedup + category tagging + recency ordering, applied per
    # taxonomy category (monetary_policy, inflation_prices, labor_market,
    # growth_output, markets_volatility, geopolitical_trade, uncategorized).
    "macro_news_category_cap": 3,
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
    # Options: "market", "social", "news", "fundamentals", "macro_fundamentals",
    # "macro_news" (the keys in ANALYST_NODE_SPECS, tradingagents/graph/analyst_execution.py).
    # "macro_fundamentals" (issue #132) and "macro_news" (issue #134) are opt-in:
    # they are NOT in the default list below (decision 2 in #126) so existing runs
    # see no cost/latency change.
    # "perplexity_news" is NOT a valid option here — it is not wired into
    # ANALYST_NODE_SPECS / the execution plan (see the NOTE in
    # tradingagents/graph/setup.py) and will fail validation before any run
    # starts.
    # Validated before run start: must be non-empty and all entries must be
    # one of the known analyst types.
    "selected_analysts": ["market", "social", "news", "fundamentals"],
})
