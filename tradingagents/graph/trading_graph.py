# TradingAgents/graph/trading_graph.py

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import yfinance as yf
from langgraph.prebuilt import ToolNode

# Import the new abstract tool methods from agent_utils.
#
# Only get_news is imported here: it's the one tool still wired to a live
# ToolNode ("social" — see _create_tool_nodes below). market/news/
# fundamentals compute deterministically and never call tools (#37), so
# their former ToolNode tool lists (get_stock_data, get_indicators,
# get_verified_market_snapshot, get_global_news, get_insider_transactions,
# get_fundamentals, get_balance_sheet, get_cashflow, get_income_statement)
# were dropped along with the dead ToolNodes they backed.
from tradingagents.agents.utils.agent_utils import get_news
from tradingagents.agents.utils.memory import TradingMemoryLog
from tradingagents.agents.utils.rating import parse_rating
from tradingagents.dataflows.config import set_config
from tradingagents.dataflows.utils import safe_ticker_component
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.llm_call_log import (
    ContextWindowGuardHandler,
    LLMCallLogHandler,
    OllamaNumCtxDerivation,
    TruncationGuardHandler,
)
from tradingagents.llm_clients import create_llm_client
from tradingagents.memory.mcp_client import MemoryMCPClient
from tradingagents.reporting import write_report_tree

from .checkpointer import checkpoint_step, clear_checkpoint, get_checkpointer, thread_id
from .conditional_logic import ConditionalLogic
from .propagation import Propagator
from .reflection import Reflector
from .setup import GraphSetup
from .signal_processing import SignalProcessor

logger = logging.getLogger(__name__)


class TradingAgentsGraph:
    """Main class that orchestrates the trading agents framework."""

    def __init__(
        self,
        selected_analysts=None,
        debug=False,
        config: dict[str, Any] = None,
        callbacks: list | None = None,
    ):
        """Initialize the trading agents graph and components.

        Args:
            selected_analysts: List of analyst types to include. Options:
                - "market": Market/technical analysis
                - "social": Social media sentiment analysis
                - "news": Traditional news analysis (yfinance/alpha_vantage)
                - "fundamentals": Fundamental data analysis
                - "macro_fundamentals": Macro indicator pack review (opt-in, issue #132)
                - "perplexity_news": Perplexity Agent API-based news analysis
            debug: Whether to run in debug mode
            config: Configuration dictionary. If None, uses default config
            callbacks: Optional list of callback handlers (e.g., for tracking LLM/tool stats)
        """
        self.debug = debug
        self.config = config or DEFAULT_CONFIG
        if selected_analysts is None:
            selected_analysts = self.config.get("selected_analysts", ["market", "social", "news", "fundamentals"])

        # Validate selected_analysts before proceeding
        self._validate_selected_analysts(selected_analysts)
        self.selected_analysts = selected_analysts

        # Validate risk_stage before proceeding (issue #119, mirrors the
        # selected_analysts validation above).
        self._validate_risk_stage(self.config.get("risk_stage", "debate"))

        self.callbacks = callbacks or []

        # Oversize-prompt enforcement (issue #149): a dedicated callback
        # handler that aborts a run before a prompt exceeding a *known*
        # context window is dispatched -- see llm_call_log.py's "oversize-
        # prompt enforcement" section for the full design rationale. Added
        # for every TradingAgentsGraph regardless of what callbacks the
        # caller passed in (so every entry point -- cli/main.py,
        # run_trading_agents.py, and any future one -- gets the check without
        # having to wire it up itself), and is a safe no-op when no context
        # window is known for the configured models (context_windows empty)
        # or when context_window_check_enabled is False.
        self.callbacks = [*self.callbacks, self._build_context_window_guard()]

        # Truncated-response enforcement (issue #171): a second dedicated
        # callback handler, following the exact same "guard that must
        # actually abort" precedent as the context-window guard above -- see
        # llm_call_log.py's "truncated-response enforcement" section. Must be
        # appended *after* whichever LLMCallLogHandler the caller passed in
        # (found by type, same as _build_context_window_guard does) so that
        # handler's completed-call JSONL record is always written before this
        # one raises -- see TruncationGuardHandler's docstring.
        self.callbacks = [*self.callbacks, self._build_truncation_guard()]

        # Update the interface's config
        set_config(self.config)

        # Create necessary directories
        os.makedirs(self.config["data_cache_dir"], exist_ok=True)
        os.makedirs(self.config["results_dir"], exist_ok=True)

        # Initialize LLMs with provider-specific thinking configuration
        llm_kwargs = self._get_provider_kwargs()

        # Add callbacks to kwargs if provided (passed to LLM constructor)
        if self.callbacks:
            llm_kwargs["callbacks"] = self.callbacks

        deep_client = create_llm_client(
            provider=self.config["llm_provider"],
            model=self.config["deep_think_llm"],
            base_url=self.config.get("backend_url"),
            **llm_kwargs,
        )
        quick_client = create_llm_client(
            provider=self.config["llm_provider"],
            model=self.config["quick_think_llm"],
            base_url=self.config.get("backend_url"),
            **llm_kwargs,
        )

        self.deep_thinking_llm = deep_client.get_llm()
        self.quick_thinking_llm = quick_client.get_llm()

        self.memory_log = TradingMemoryLog(self.config)

        # Create tool nodes
        self.tool_nodes = self._create_tool_nodes()

        # Initialize components
        self.conditional_logic = ConditionalLogic(
            max_debate_rounds=self.config["max_debate_rounds"],
            max_risk_discuss_rounds=self.config["max_risk_discuss_rounds"],
        )
        self.graph_setup = GraphSetup(
            self.quick_thinking_llm,
            self.deep_thinking_llm,
            self.tool_nodes,
            self.conditional_logic,
            analyst_concurrency_limit=self.config.get("analyst_concurrency_limit", 1),
            research_stage=self.config.get("research_stage", "none"),
            risk_stage=self.config.get("risk_stage", "debate"),
            swing_trader_enabled=self.config.get("swing_trader_enabled", False),
        )

        self.propagator = Propagator(
            max_recur_limit=self.config.get("max_recur_limit", 100),
        )
        self.reflector = Reflector(self.quick_thinking_llm)
        self.signal_processor = SignalProcessor(self.quick_thinking_llm)

        # State tracking
        self.curr_state = None
        self.ticker = None
        self.log_states_dict = {}  # date to full state dict

        # Set up the graph: keep the workflow for recompilation with a checkpointer.
        self.workflow = self.graph_setup.setup_graph(selected_analysts)
        self.graph = self.workflow.compile()
        self._checkpointer_ctx = None
        # Networked memory MCP client (#53) — connected once per propagate()
        # run and torn down at the end of that run; see propagate().
        self._memory_client: MemoryMCPClient | None = None

    def _validate_selected_analysts(self, selected_analysts) -> None:
        """Validate that selected_analysts is valid before graph setup.

        Delegates the actual emptiness/membership checks to
        ``build_analyst_execution_plan`` — the same function
        ``self.graph_setup.setup_graph`` calls moments later — so the set of
        known analyst keys (``ANALYST_NODE_SPECS``) and the "non-empty list"
        invariant are enforced in exactly one place instead of being
        duplicated (and risking drift) here. This method only adds the
        type check (rejecting non-list/tuple inputs, e.g. a bare string,
        which ``build_analyst_execution_plan`` would otherwise iterate
        character-by-character) and re-raises with the constructor's error
        framing.

        Raises ValueError with clear error messages if:
        - selected_analysts is not a list/tuple
        - The list is empty
        - Any entry is not a known analyst type
        """
        from .analyst_execution import ANALYST_NODE_SPECS, build_analyst_execution_plan

        if not isinstance(selected_analysts, (list, tuple)):
            raise ValueError(f"selected_analysts must be a list, got {type(selected_analysts).__name__}")

        try:
            build_analyst_execution_plan(selected_analysts)
        except ValueError as exc:
            if not selected_analysts:
                raise ValueError("selected_analysts must be a non-empty list") from exc
            valid_analysts = ", ".join(sorted(ANALYST_NODE_SPECS.keys()))
            raise ValueError(f"{exc}. Valid options are: {valid_analysts}") from exc

    def _validate_risk_stage(self, risk_stage) -> None:
        """Validate the risk_stage config value before graph setup.

        Mirrors the research_stage bypass added in #79 (issue #119): the only
        supported values are "debate" (today's Aggressive/Conservative/Neutral
        risk-debate stage, default) and "none" (bypass it entirely, routing the
        Trader's plan straight to the Portfolio Manager). Unlike research_stage
        (which silently treats any unrecognized value as "none" in
        GraphSetup.setup_graph), risk_stage is validated explicitly so a typo'd
        config value fails fast with a clear error instead of silently
        disabling the risk-debate stage.

        Raises ValueError if risk_stage is not "debate" or "none".
        """
        valid_values = {"debate", "none"}
        if risk_stage not in valid_values:
            raise ValueError(
                f"Invalid risk_stage {risk_stage!r}. Valid options are: "
                f"{', '.join(sorted(valid_values))}"
            )

    def _build_ollama_num_ctx_derivation(self) -> OllamaNumCtxDerivation | None:
        """Build the issue #154 per-request ``num_ctx`` derivation policy.

        Applies only when the configured provider is ``"ollama"`` and no
        explicit ``ollama_num_ctx`` override is set -- that key, when set, is
        forwarded verbatim by ``_get_provider_kwargs`` (unchanged #149
        behaviour) and no derivation happens at all, per #154's explicit
        escape hatch. Returns ``None`` in every other case (a different
        provider, an explicit ``ollama_num_ctx``, or no models configured),
        which every #154 code path (``ContextWindowGuardHandler``,
        ``LLMCallLogHandler``, ``NormalizedChatOllama``) treats as "derivation
        does not apply to this run" -- exactly today's pre-#154 behaviour.
        """
        provider = self.config.get("llm_provider", "").lower()
        if provider != "ollama" or self.config.get("ollama_num_ctx"):
            return None

        models = frozenset(
            m for m in (self.config.get("quick_think_llm"), self.config.get("deep_think_llm")) if m
        )
        if not models:
            return None

        return OllamaNumCtxDerivation(
            models=models,
            num_ctx_max=int(self.config.get("ollama_num_ctx_max", 32768)),
            response_headroom=int(self.config.get("ollama_num_ctx_response_headroom", 0)),
            safety_margin=float(self.config.get("context_window_safety_margin", 1.3)),
            response_headroom_overrides=self.config.get("ollama_num_ctx_response_headroom_overrides", {}),
        )

    def _build_context_window_guard(self) -> ContextWindowGuardHandler:
        """Build the issue #149 oversize-prompt guard from ``self.config``.

        ``context_windows`` (model name -> known context-window tokens) is
        resolved once here, not per call: the ollama-derived entry comes from
        ``ollama_num_ctx`` (applied to both ``quick_think_llm`` and
        ``deep_think_llm`` -- the only two model names this run will ever use)
        when the configured provider is ``"ollama"``, and any
        ``context_window_overrides`` entries are layered on top (explicit
        overrides win over the ollama-derived value for the same model name).
        A model with no entry either way is passed through unchecked by
        ``ContextWindowGuardHandler`` -- see that class's docstring.

        Issue #154: when ``ollama_num_ctx`` is left unset, ``context_windows``
        stays empty for the ollama-served models (nothing to `.setdefault`
        above) and ``_build_ollama_num_ctx_derivation`` instead builds a
        per-request derivation policy, handed to both the guard (which aborts
        when a request's derived requirement exceeds ``ollama_num_ctx_max``)
        and, below, whichever ``LLMCallLogHandler`` is found in
        ``self.callbacks`` (so successful-call records carry the ``num_ctx``
        that was actually sent).

        The guard writes its JSONL audit record through whichever
        ``LLMCallLogHandler`` is present in ``self.callbacks`` (found by
        type, not by name, since callers construct and pass it themselves --
        see cli/main.py / run_trading_agents.py). ``None`` when no such
        handler was passed in: the guard still raises, it just has nowhere to
        write the audit record.
        """
        provider = self.config.get("llm_provider", "").lower()

        context_windows: dict[str, int] = dict(self.config.get("context_window_overrides") or {})
        if provider == "ollama":
            num_ctx = self.config.get("ollama_num_ctx")
            if num_ctx:
                num_ctx = int(num_ctx)
                context_windows.setdefault(self.config.get("quick_think_llm"), num_ctx)
                context_windows.setdefault(self.config.get("deep_think_llm"), num_ctx)

        ollama_num_ctx_derivation = self._build_ollama_num_ctx_derivation()

        log_handler = next(
            (cb for cb in self.callbacks if isinstance(cb, LLMCallLogHandler)), None
        )
        if log_handler is not None:
            log_handler.ollama_num_ctx_derivation = ollama_num_ctx_derivation

        return ContextWindowGuardHandler(
            context_windows=context_windows,
            safety_margin=float(self.config.get("context_window_safety_margin", 1.3)),
            enabled=bool(self.config.get("context_window_check_enabled", True)),
            log_handler=log_handler,
            ollama_num_ctx_derivation=ollama_num_ctx_derivation,
        )

    def _build_truncation_guard(self) -> TruncationGuardHandler:
        """Build the issue #171 truncated-response guard.

        Detection and the JSONL audit write live in whichever
        ``LLMCallLogHandler`` is present in ``self.callbacks`` (found by
        type, same lookup ``_build_context_window_guard`` uses) --
        ``TruncationGuardHandler``'s only job is reading the
        ``LLMResponseTruncatedError`` that handler stashed for a given call
        and re-raising it, so the two can't produce inconsistent records.
        ``None`` when no such handler was passed in: the guard is then a
        no-op, since there is nothing for it to consult (see
        ``TruncationGuardHandler``'s docstring).
        """
        log_handler = next(
            (cb for cb in self.callbacks if isinstance(cb, LLMCallLogHandler)), None
        )
        return TruncationGuardHandler(log_handler=log_handler)

    def _get_provider_kwargs(self) -> dict[str, Any]:
        """Get provider-specific kwargs for LLM client creation."""
        kwargs = {}
        provider = self.config.get("llm_provider", "").lower()

        if provider == "google":
            thinking_level = self.config.get("google_thinking_level")
            if thinking_level:
                kwargs["thinking_level"] = thinking_level

        elif provider == "openai":
            reasoning_effort = self.config.get("openai_reasoning_effort")
            if reasoning_effort:
                kwargs["reasoning_effort"] = reasoning_effort

        elif provider == "anthropic":
            effort = self.config.get("anthropic_effort")
            if effort:
                kwargs["effort"] = effort

        elif provider == "ollama":
            # Explicit context-length override (issue #149) -- see
            # default_config.py's "ollama_num_ctx" doc comment and
            # docs/analysis/prompt-truncation-diagnosis.md. int() tolerates a
            # string value here (e.g. from TRADINGAGENTS_OLLAMA_NUM_CTX,
            # which _coerce leaves as a raw string when the default is None
            # -- same pattern as "temperature" below). When set, it wins
            # outright and no per-request derivation happens (issue #154).
            num_ctx = self.config.get("ollama_num_ctx")
            if num_ctx:
                kwargs["num_ctx"] = int(num_ctx)
            else:
                # Per-request num_ctx derivation (issue #154): forwarded to
                # OllamaClient.get_llm (tradingagents/llm_clients/ollama_client.py
                # -- re-pointed here from OpenAIClient/OllamaChatOpenAI by issue
                # #169), which attaches it to the constructed NormalizedChatOllama
                # instance for its _chat_params hook to consult on every call.
                # None (no models configured, or this isn't actually the ollama
                # provider -- can't happen inside this branch, but
                # _build_ollama_num_ctx_derivation is the single source of truth
                # either way) forwards nothing, same as today's pre-#154 behaviour.
                derivation = self._build_ollama_num_ctx_derivation()
                if derivation is not None:
                    kwargs["ollama_num_ctx_derivation"] = derivation

            # Think mode for Ollama (issue #155, re-pointed at the native
            # /api/chat endpoint's `think` field by issue #169): a three-state
            # config value (True/False/None -- see default_config.py's
            # "ollama_think" doc comment) forwarded verbatim, not just when
            # truthy, so OllamaClient.get_llm can distinguish "unset" (key
            # absent, e.g. a caller that never passes this kwarg at all -->
            # effective default False) from an explicit False (send `think:
            # false`) from an explicit None (send no `think` field at all).
            kwargs["ollama_think"] = self.config.get("ollama_think")

        # Temperature is supported by all providers. Cast through float() so a
        # string env var (TRADINGAGENTS_TEMPERATURE) is tolerated: the config
        # default is None, so default_config.py's type-driven coercion can't
        # infer float and passes the raw string through untouched.
        temperature = self.config.get("temperature")
        if temperature is not None:
            kwargs["temperature"] = float(temperature)

        # Bound every provider's network calls so a wedged endpoint (most
        # notably a local ollama daemon that accepts the connection but never
        # responds) fails fast with a clear timeout error instead of hanging
        # the run indefinitely with no CPU/GPU activity (#108). Each client's
        # own passthrough kwargs list decides whether it forwards "timeout"
        # to its underlying SDK; clients that don't support it simply ignore
        # the extra kwarg.
        llm_timeout = self.config.get("llm_timeout")
        if llm_timeout is not None:
            kwargs["timeout"] = llm_timeout

        return kwargs

    def _create_tool_nodes(self) -> dict[str, ToolNode]:
        """Create tool nodes for different data sources using abstract methods.

        market/news/fundamentals have no entries here: those analysts compute
        deterministically and never call tools, so `GraphSetup.setup_graph()`
        wires them straight to their "Msg Clear" node with no ToolNode round
        trip (#37). Only "social" (sentiment) still makes real tool calls.
        """
        return {
            "social": ToolNode(
                [
                    # News tools for social media analysis
                    get_news,
                ]
            ),
            "perplexity_news": ToolNode(
                [
                    # Perplexity news analyst uses its own tools internally
                    # via the Agent API (web_search, fetch_url)
                    # No external tools needed here
                ]
            ),
        }

    def _resolve_benchmark(self, ticker: str) -> str:
        """Pick the benchmark ticker for alpha calculation against ``ticker``.

        ``config["benchmark_ticker"]`` overrides everything when set; otherwise
        the suffix map matches the ticker's exchange suffix (e.g. ``.T`` for
        Tokyo). US-listed tickers without a dotted suffix fall through to the
        empty-suffix entry (SPY by default). Unrecognised suffixes (including
        US tickers with dots like ``BRK.B``) also fall back to the empty-suffix
        entry, which is the right default because the alpha calculation works
        in USD.
        """
        explicit = self.config.get("benchmark_ticker")
        if explicit:
            return explicit
        benchmark_map = self.config.get("benchmark_map", {})
        ticker_upper = ticker.upper()
        for suffix, benchmark in benchmark_map.items():
            if suffix and ticker_upper.endswith(suffix.upper()):
                return benchmark
        return benchmark_map.get("", "SPY")

    def _fetch_returns(
        self, ticker: str, trade_date: str, holding_days: int = 5,
        benchmark: str = "SPY",
    ) -> tuple[float | None, float | None, int | None]:
        """Fetch raw and alpha return for ticker over holding_days from trade_date.

        ``benchmark`` is the index used as the alpha baseline (resolved by the
        caller via ``_resolve_benchmark``). Returns ``(raw_return, alpha_return,
        actual_holding_days)`` or ``(None, None, None)`` if price data is
        unavailable (too recent, delisted, or network error).
        """
        from tradingagents.dataflows.symbol_utils import normalize_symbol

        try:
            start = datetime.strptime(trade_date, "%Y-%m-%d")
            end = start + timedelta(days=holding_days + 7)  # buffer for weekends/holidays
            end_str = end.strftime("%Y-%m-%d")

            # Normalize so the realized-return lookup hits the same instrument
            # the analysis priced (e.g. XAUUSD -> GC=F) (#984). The benchmark is
            # already a canonical Yahoo symbol from ``_resolve_benchmark``.
            stock = yf.Ticker(normalize_symbol(ticker)).history(start=trade_date, end=end_str)
            bench = yf.Ticker(benchmark).history(start=trade_date, end=end_str)

            if len(stock) < 2 or len(bench) < 2:
                return None, None, None

            actual_days = min(holding_days, len(stock) - 1, len(bench) - 1)
            raw = float(
                (stock["Close"].iloc[actual_days] - stock["Close"].iloc[0])
                / stock["Close"].iloc[0]
            )
            bench_ret = float(
                (bench["Close"].iloc[actual_days] - bench["Close"].iloc[0])
                / bench["Close"].iloc[0]
            )
            alpha = raw - bench_ret
            return raw, alpha, actual_days
        except Exception as e:
            logger.warning(
                "Could not resolve outcome for %s on %s vs %s (will retry next run): %s",
                ticker, trade_date, benchmark, e,
            )
            return None, None, None

    def _resolve_pending_entries(self, ticker: str) -> None:
        """Resolve pending log entries for ticker at the start of a new run.

        Fetches returns for each same-ticker pending entry, generates reflections,
        then writes all updates in a single atomic batch write to avoid redundant I/O.
        Skips entries whose price data is not yet available (too recent or delisted).

        Trade-off: only same-ticker entries are resolved per run.  Entries for
        other tickers accumulate until that ticker is run again.
        """
        pending = [e for e in self.memory_log.get_pending_entries() if e["ticker"] == ticker]
        if not pending:
            return

        benchmark = self._resolve_benchmark(ticker)
        updates = []
        for entry in pending:
            raw, alpha, days = self._fetch_returns(
                ticker, entry["date"], benchmark=benchmark,
            )
            if raw is None:
                continue  # price not available yet — try again next run
            reflection = self.reflector.reflect_on_final_decision(
                final_decision=entry.get("decision", ""),
                raw_return=raw,
                alpha_return=alpha,
                benchmark_name=benchmark,
            )
            updates.append({
                "ticker": ticker,
                "trade_date": entry["date"],
                "raw_return": raw,
                "alpha_return": alpha,
                "holding_days": days,
                "reflection": reflection,
            })

        if updates:
            self.memory_log.batch_update_with_outcomes(updates)

    def propagate(self, company_name, trade_date, asset_type: str = "stock"):
        """Run the trading agents graph for a company on a specific date.

        ``asset_type`` selects between the stock pipeline (default) and the
        crypto pipeline (``"crypto"``) shipped in #567 — the CLI auto-detects
        from the ticker; programmatic callers pass it explicitly. When
        ``checkpoint_enabled`` is set in config, the graph is recompiled with
        a per-ticker SqliteSaver so a crashed run can resume from the last
        successful node on a subsequent invocation with the same ticker+date.
        """
        self.ticker = company_name

        # Resolve any pending memory-log entries for this ticker before the pipeline runs.
        self._resolve_pending_entries(company_name)

        # Connect to the networked memory MCP server (#53) for the SQLite memory
        # core's resolve-pending call and this run's store-decision calls. This
        # is a hard dependency: an unreachable server (MemoryMCPConnectionError)
        # or a failing tool call (MemoryMCPToolError) both propagate and fail
        # the run rather than being logged and swallowed — a deliberate
        # behavior change from the prior in-process warn-and-continue pattern.
        # Resolve the memory ID to a DB path (issue #114) — if set, creates
        # an isolated decision history for this run.
        from tradingagents.memory.store import resolve_memory_id_to_db_path

        memory_id = self.config.get("memory_id")
        try:
            resolved_db_path = resolve_memory_id_to_db_path(memory_id)
        except ValueError as e:
            raise ValueError(f"Invalid memory_id configuration: {e}") from e

        with MemoryMCPClient(db_path=str(resolved_db_path)) as memory_client:
            self._memory_client = memory_client

            # Resolve any pending decisions in the SQLite memory core.
            memory_client.resolve_pending(ticker=company_name)

            # Recompile with a checkpointer if the user opted in.
            if self.config.get("checkpoint_enabled"):
                self._checkpointer_ctx = get_checkpointer(
                    self.config["data_cache_dir"], company_name
                )
                saver = self._checkpointer_ctx.__enter__()
                self.graph = self.workflow.compile(checkpointer=saver)

                step = checkpoint_step(
                    self.config["data_cache_dir"], company_name, str(trade_date)
                )
                if step is not None:
                    logger.info(
                        "Resuming from step %d for %s on %s", step, company_name, trade_date
                    )
                else:
                    logger.info("Starting fresh for %s on %s", company_name, trade_date)

            try:
                return self._run_graph(company_name, trade_date, asset_type=asset_type)
            finally:
                if self._checkpointer_ctx is not None:
                    self._checkpointer_ctx.__exit__(None, None, None)
                    self._checkpointer_ctx = None
                    self.graph = self.workflow.compile()
                self._memory_client = None

    def save_reports(self, final_state, ticker, save_path=None) -> Path:
        """Write the report tree for a completed run, like the CLI does.

        Programmatic callers get the same on-disk reports the CLI produces,
        including the consolidated ``complete_report.pdf`` (issue #165). Pass
        an explicit ``save_path`` or let it default under ``results_dir``.
        """
        if save_path is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = (
                Path(self.config["results_dir"])
                / "reports"
                / f"{safe_ticker_component(ticker)}_{stamp}"
            )
        return write_report_tree(final_state, ticker, save_path)

    def _run_graph(self, company_name, trade_date, asset_type: str = "stock"):
        """Execute the graph and write the resulting state to disk and memory log."""
        # Initialize state — inject memory log context for PM.
        past_context = self.memory_log.get_past_context(company_name)

        # Fetch swing trader past context via MCP if enabled (hard dependency #53).
        swing_past_context = ""
        if self.config.get("swing_trader_enabled"):
            swing_past_context = self._memory_client.get_past_context(
                agent="swing_trader", ticker=company_name
            )

        # Fetch macro fundamentals past context via MCP, only when selected
        # (#132, mirrors the swing_trader pattern above).
        macro_past_context = ""
        if "macro_fundamentals" in self.selected_analysts:
            macro_past_context = self._memory_client.get_past_context(
                agent="macro_fundamentals", ticker=company_name
            )

        # Fetch macro news past context via MCP, only when selected (#134).
        macro_news_past_context = ""
        if "macro_news" in self.selected_analysts:
            macro_news_past_context = self._memory_client.get_past_context(
                agent="macro_news", ticker=company_name
            )

        init_agent_state = self.propagator.create_initial_state(
            company_name,
            trade_date,
            asset_type=asset_type,
            past_context=past_context,
            swing_past_context=swing_past_context,
            macro_past_context=macro_past_context,
            macro_news_past_context=macro_news_past_context,
        )
        args = self.propagator.get_graph_args()

        # Inject thread_id so same ticker+date resumes, different date starts fresh.
        if self.config.get("checkpoint_enabled"):
            tid = thread_id(company_name, str(trade_date))
            args.setdefault("config", {}).setdefault("configurable", {})["thread_id"] = tid

        if self.debug:
            last_printed_message = None
            # Start with the initial state in case the graph yields no "values"
            # chunk at all (e.g. an empty/no-op graph) — kept in sync with the
            # non-debug path's starting point.
            final_state = dict(init_agent_state)

            # Combine "values" and "updates" modes: "updates" chunks carry
            # {node_name: state_delta} so we can label each printed message
            # with its originating node, while "values" chunks carry the full,
            # already-reduced state after each step (LangGraph applies each
            # channel's reducer — e.g. `add_messages` for `messages` — before
            # yielding a "values" chunk). Taking the last "values" chunk as the
            # final state means we never have to hand-reimplement reducer
            # semantics: it matches graph.invoke()'s output exactly, unlike
            # last-write-wins merging of raw "updates" deltas.
            debug_args = {**args, "stream_mode": ["values", "updates"]}
            for mode, chunk in self.graph.stream(init_agent_state, **debug_args):
                if mode == "values":
                    final_state = chunk
                    continue

                # mode == "updates": chunk is {node_name: state_delta}
                for node_name, state_delta in chunk.items():
                    # Check if this node delta has messages
                    if state_delta.get("messages"):
                        current_message = state_delta["messages"][-1]

                        # Deduplicate: only print if message content or type differs
                        # Create a key from message type and content for comparison
                        message_key = (
                            type(current_message).__name__,
                            current_message.content,
                        )
                        if last_printed_message is None or last_printed_message != message_key:
                            # Print node name header
                            print(f"\n── {node_name} ──")
                            current_message.pretty_print()
                            last_printed_message = message_key
        else:
            final_state = self.graph.invoke(init_agent_state, **args)

        # Store current state for reflection.
        self.curr_state = final_state

        # Log state to disk.
        self._log_state(trade_date, final_state)

        # Store decision for deferred reflection on the next same-ticker run.
        self.memory_log.store_decision(
            ticker=company_name,
            trade_date=trade_date,
            final_trade_decision=final_state["final_trade_decision"],
        )

        # Store decisions in SQLite memory core (via the memory MCP client, #53)
        # for the decision-bearing stages that actually ran. Each stage's signal
        # is derived via parse_rating from its own output text. Connection/tool
        # failures propagate — the memory MCP server is a hard dependency for a
        # run to complete, not a best-effort side write.
        #
        # The research stage is conditional on research_stage (#79, #85): in
        # "none" mode no research node ever runs and investment_plan stays "" by design
        # (see propagation.py), so storing a decision here would fabricate a row
        # indistinguishable from a real one — parse_rating("") silently defaults to
        # "Hold", which would later surface as a nonsensical "past lesson" via query.py.
        # Guard on research_stage itself, not on "is investment_plan empty", since a
        # run could in principle produce a genuinely thin plan.
        #
        # "debate" mode stores under agent="research_manager"; "researcher" mode
        # stores under agent="researcher" (no continuity with research_manager rows).
        research_stage = self.config.get("research_stage", "none")
        if research_stage == "debate":
            research_signal = parse_rating(final_state.get("investment_plan", ""))
            self._memory_client.store_decision(
                agent="research_manager",
                ticker=company_name,
                date=trade_date,
                signal=research_signal,
                confidence=None,
                key_drivers=None,
                thesis=final_state.get("investment_plan", "")[:500],  # truncate for DB
            )
        elif research_stage == "researcher":
            research_signal = parse_rating(final_state.get("investment_plan", ""))
            self._memory_client.store_decision(
                agent="researcher",
                ticker=company_name,
                date=trade_date,
                signal=research_signal,
                confidence=None,
                key_drivers=None,
                thesis=final_state.get("investment_plan", "")[:500],  # truncate for DB
            )

        trader_signal = parse_rating(final_state.get("trader_investment_plan", ""))
        self._memory_client.store_decision(
            agent="trader",
            ticker=company_name,
            date=trade_date,
            signal=trader_signal,
            confidence=None,
            key_drivers=None,
            thesis=final_state.get("trader_investment_plan", "")[:500],  # truncate for DB
        )

        pm_signal = parse_rating(final_state.get("final_trade_decision", ""))
        self._memory_client.store_decision(
            agent="portfolio_manager",
            ticker=company_name,
            date=trade_date,
            signal=pm_signal,
            confidence=None,
            key_drivers=None,
            thesis=final_state.get("final_trade_decision", "")[:500],  # truncate for DB
        )

        # Store macro fundamentals decision, only when selected (#132). Stored
        # under the run's own ticker, not a synthetic macro ticker (decision 6
        # in #126): the analyst's signal is a macro-conditioned call on this
        # ticker, and resolve.py needs a real price series to resolve
        # forward_return against.
        if "macro_fundamentals" in self.selected_analysts:
            macro_report_raw = final_state.get("macro_report", "")
            macro_signal = None
            macro_confidence = None
            macro_drivers = None
            macro_thesis = ""
            try:
                macro_envelope = json.loads(macro_report_raw) if macro_report_raw else {}
                macro_signal = macro_envelope.get("signal")
                macro_details = macro_envelope.get("details") or {}
                conservative = macro_details.get("conservative") or {}
                risky = macro_details.get("risky") or {}
                cons_conf = conservative.get("confidence")
                risky_conf = risky.get("confidence")
                if cons_conf is not None and risky_conf is not None:
                    macro_confidence = (float(cons_conf) + float(risky_conf)) / 2.0
                macro_drivers = macro_details.get("drivers")
                macro_thesis = macro_envelope.get("summary") or ""
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
            if not macro_signal:
                # Fallback: envelope unparseable/empty — mirrors trader/PM's
                # free-text fallback via parse_rating.
                macro_signal = parse_rating(macro_report_raw)

            self._memory_client.store_decision(
                agent="macro_fundamentals",
                ticker=company_name,
                date=trade_date,
                signal=macro_signal,
                confidence=macro_confidence,
                key_drivers=macro_drivers,
                thesis=macro_thesis[:500] if macro_thesis else "",  # truncate for DB
            )

        # Store macro news decision, only when selected (#134). Mirrors the
        # macro_fundamentals pattern above.
        if "macro_news" in self.selected_analysts:
            macro_news_raw = final_state.get("macro_news_report", "")
            macro_news_signal = None
            macro_news_confidence = None
            macro_news_drivers = None
            macro_news_thesis = ""
            try:
                macro_news_envelope = json.loads(macro_news_raw) if macro_news_raw else {}
                macro_news_signal = macro_news_envelope.get("signal")
                macro_news_details = macro_news_envelope.get("details") or {}
                conservative = macro_news_details.get("conservative") or {}
                risky = macro_news_details.get("risky") or {}
                cons_conf = conservative.get("confidence")
                risky_conf = risky.get("confidence")
                if cons_conf is not None and risky_conf is not None:
                    macro_news_confidence = (float(cons_conf) + float(risky_conf)) / 2.0
                macro_news_drivers = macro_news_details.get("category_sentiments")
                macro_news_thesis = macro_news_envelope.get("summary") or ""
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
            if not macro_news_signal:
                # Fallback: envelope unparseable/empty — mirrors trader/PM's
                # free-text fallback via parse_rating.
                macro_news_signal = parse_rating(macro_news_raw)

            self._memory_client.store_decision(
                agent="macro_news",
                ticker=company_name,
                date=trade_date,
                signal=macro_news_signal,
                confidence=macro_news_confidence,
                key_drivers=macro_news_drivers,
                thesis=macro_news_thesis[:500] if macro_news_thesis else "",  # truncate for DB
            )

        # Store swing trader decision when enabled (#93).
        if self.config.get("swing_trader_enabled"):
            swing_structured = final_state.get("swing_structured_data")
            if swing_structured:
                # Extract structured fields matching the schema.
                signal = swing_structured.get("action", "Hold")
                conviction = swing_structured.get("conviction")
                thesis = swing_structured.get("thesis", "")
                holding_period_days = swing_structured.get("holding_period_days")
                entry_price = swing_structured.get("entry_price")
                stop_loss = swing_structured.get("stop_loss")
                take_profit = swing_structured.get("take_profit")
                setup_type = swing_structured.get("setup_type", "")
                regime = swing_structured.get("regime", "unknown")  # regime from precompute
                drivers = swing_structured.get("key_drivers", [])

                # Compute risk_reward from prices (same formula as swing_trader_computation).
                risk_reward = None
                if entry_price is not None and stop_loss is not None and take_profit is not None:
                    risk = abs(entry_price - stop_loss)
                    reward = abs(take_profit - entry_price)
                    if risk > 0:
                        risk_reward = round(reward / risk, 2)

                # Build key_drivers JSON with regime, setup, prices, and driver list.
                key_drivers_json = {
                    "regime": regime,
                    "setup_type": setup_type,
                    "risk_reward": risk_reward,
                    "planned_horizon_days": holding_period_days,
                    "entry_price": entry_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "drivers": drivers,
                }

                self._memory_client.store_decision(
                    agent="swing_trader",
                    ticker=company_name,
                    date=trade_date,
                    signal=signal,
                    confidence=conviction,
                    key_drivers=key_drivers_json,
                    thesis=thesis[:500] if thesis else "",  # truncate for DB
                    horizon_days=holding_period_days,
                )
            else:
                # Fallback: parse markdown when structured data unavailable (mirrors trader/PM behavior).
                swing_signal = parse_rating(final_state.get("swing_trade_decision", ""))
                self._memory_client.store_decision(
                    agent="swing_trader",
                    ticker=company_name,
                    date=trade_date,
                    signal=swing_signal,
                    confidence=None,
                    key_drivers=None,
                    thesis=final_state.get("swing_trade_decision", "")[:500],  # truncate for DB
                )

        # Clear checkpoint on successful completion to avoid stale state.
        if self.config.get("checkpoint_enabled"):
            clear_checkpoint(
                self.config["data_cache_dir"], company_name, str(trade_date)
            )

        return final_state, self.process_signal(final_state["final_trade_decision"])

    def _log_state(self, trade_date, final_state):
        """Log the final state to a JSON file."""
        log_dict = {
            "company_of_interest": final_state["company_of_interest"],
            "trade_date": final_state["trade_date"],
            "market_report": final_state.get("market_report", ""),
            "sentiment_report": final_state.get("sentiment_report", ""),
            "news_report": final_state.get("news_report", ""),
            "perplexity_news_report": final_state.get("perplexity_news_report", ""),
            "fundamentals_report": final_state.get("fundamentals_report", ""),
            "macro_report": final_state.get("macro_report", ""),
            "trader_investment_decision": final_state["trader_investment_plan"],
            "investment_plan": final_state["investment_plan"],
            "final_trade_decision": final_state["final_trade_decision"],
        }

        # Include investment_debate_state when in debate mode. Gated on research_stage,
        # not dict truthiness: the state dict is always seeded (non-empty) regardless of
        # mode, so a truthiness check here would always pass.
        if self.config.get("research_stage") == "debate":
            log_dict["investment_debate_state"] = {
                "bull_history": final_state["investment_debate_state"]["bull_history"],
                "bear_history": final_state["investment_debate_state"]["bear_history"],
                "history": final_state["investment_debate_state"]["history"],
                "current_response": final_state["investment_debate_state"][
                    "current_response"
                ],
                "judge_decision": final_state["investment_debate_state"][
                    "judge_decision"
                ],
            }

        # Include researcher_evidence when in researcher mode
        if final_state.get("researcher_evidence"):
            log_dict["researcher_evidence"] = final_state["researcher_evidence"]

        # Include risk_debate_state's history breakdown only in "debate" mode.
        # Gated on risk_stage, not dict truthiness, for the same reason as
        # investment_debate_state above: the state dict is always seeded
        # (non-empty) regardless of mode, so histories stay empty strings by
        # design in "none" mode (#119, mirrors #79) rather than being a
        # failure. The Portfolio Manager's decision itself is never lost --
        # it's already captured unconditionally in final_trade_decision above
        # (risk_debate_state["judge_decision"] is set to the same value
        # regardless of risk_stage).
        if self.config.get("risk_stage", "debate") == "debate":
            log_dict["risk_debate_state"] = {
                "aggressive_history": final_state["risk_debate_state"]["aggressive_history"],
                "conservative_history": final_state["risk_debate_state"]["conservative_history"],
                "neutral_history": final_state["risk_debate_state"]["neutral_history"],
                "history": final_state["risk_debate_state"]["history"],
                "judge_decision": final_state["risk_debate_state"]["judge_decision"],
            }

        # Include swing trader decision when enabled (#93).
        if self.config.get("swing_trader_enabled"):
            swing_decision = final_state.get("swing_trade_decision", "")
            if swing_decision:
                log_dict["swing_trade_decision"] = swing_decision
            if final_state.get("swing_structured_data"):
                log_dict["swing_structured_data"] = final_state["swing_structured_data"]

        self.log_states_dict[str(trade_date)] = log_dict

        # Save to file. Reject ticker values that would escape the
        # results directory when joined as a path component.
        safe_ticker = safe_ticker_component(self.ticker)
        directory = Path(self.config["results_dir"]) / safe_ticker / "TradingAgentsStrategy_logs"
        directory.mkdir(parents=True, exist_ok=True)

        log_path = directory / f"full_states_log_{trade_date}.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(self.log_states_dict[str(trade_date)], f, indent=4)

    def process_signal(self, full_signal):
        """Process a signal to extract the core decision."""
        return self.signal_processor.process_signal(full_signal)
