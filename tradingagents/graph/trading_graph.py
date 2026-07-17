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
                - "perplexity_news": Perplexity Agent API-based news analysis
            debug: Whether to run in debug mode
            config: Configuration dictionary. If None, uses default config
            callbacks: Optional list of callback handlers (e.g., for tracking LLM/tool stats)
        """
        if selected_analysts is None:
            selected_analysts = ["market", "social", "news", "fundamentals"]
        self.debug = debug
        self.config = config or DEFAULT_CONFIG
        self.callbacks = callbacks or []

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

        # Temperature is supported by all providers. Cast through float() so a
        # string env var (TRADINGAGENTS_TEMPERATURE) is tolerated: the config
        # default is None, so default_config.py's type-driven coercion can't
        # infer float and passes the raw string through untouched.
        temperature = self.config.get("temperature")
        if temperature is not None:
            kwargs["temperature"] = float(temperature)

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
        with MemoryMCPClient() as memory_client:
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
        """Write the markdown report tree for a completed run, like the CLI does.

        Programmatic callers get the same on-disk reports the CLI produces. Pass
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
        init_agent_state = self.propagator.create_initial_state(
            company_name, trade_date, asset_type=asset_type, past_context=past_context
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
        # The research_manager stage is conditional on research_stage (#79): in
        # "none" mode no Bull/Bear/Research Manager node ever runs and
        # investment_plan stays "" by design (see propagation.py), so storing a
        # decision here would fabricate a research_manager row indistinguishable
        # from a real one — parse_rating("") silently defaults to "Hold", which
        # would later surface as a nonsensical "past lesson" via query.py. Guard
        # on research_stage itself, not on "is investment_plan empty", since a
        # "debate"-mode run could in principle produce a genuinely thin plan.
        research_stage = self.config.get("research_stage", "none")
        if research_stage != "none":
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

        # Clear checkpoint on successful completion to avoid stale state.
        if self.config.get("checkpoint_enabled"):
            clear_checkpoint(
                self.config["data_cache_dir"], company_name, str(trade_date)
            )

        return final_state, self.process_signal(final_state["final_trade_decision"])

    def _log_state(self, trade_date, final_state):
        """Log the final state to a JSON file."""
        self.log_states_dict[str(trade_date)] = {
            "company_of_interest": final_state["company_of_interest"],
            "trade_date": final_state["trade_date"],
            "market_report": final_state.get("market_report", ""),
            "sentiment_report": final_state.get("sentiment_report", ""),
            "news_report": final_state.get("news_report", ""),
            "perplexity_news_report": final_state.get("perplexity_news_report", ""),
            "fundamentals_report": final_state.get("fundamentals_report", ""),
            "investment_debate_state": {
                "bull_history": final_state["investment_debate_state"]["bull_history"],
                "bear_history": final_state["investment_debate_state"]["bear_history"],
                "history": final_state["investment_debate_state"]["history"],
                "current_response": final_state["investment_debate_state"][
                    "current_response"
                ],
                "judge_decision": final_state["investment_debate_state"][
                    "judge_decision"
                ],
            },
            "trader_investment_decision": final_state["trader_investment_plan"],
            "risk_debate_state": {
                "aggressive_history": final_state["risk_debate_state"]["aggressive_history"],
                "conservative_history": final_state["risk_debate_state"]["conservative_history"],
                "neutral_history": final_state["risk_debate_state"]["neutral_history"],
                "history": final_state["risk_debate_state"]["history"],
                "judge_decision": final_state["risk_debate_state"]["judge_decision"],
            },
            "investment_plan": final_state["investment_plan"],
            "final_trade_decision": final_state["final_trade_decision"],
        }

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
