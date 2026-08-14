#!/usr/bin/env python3
"""
TradingAgents MCP Server

Exposes:
- analyze_stock(ticker, date) — runs a full multi-agent trading analysis and
  returns the complete report (markdown) plus structured JSON on success, or
  an error message on failure.
- memory_store_decision / memory_resolve_pending / memory_get_past_context /
  memory_get_statistics — thin pass-throughs to the shared SQLite memory
  core (`tradingagents/memory/`, see CLAUDE.md "Persistence"), so MCP
  clients (e.g. the `skills/` reimplementation) can read/write the same
  decision history the legacy LangGraph pipeline's Python callers use
  directly. No logic of their own beyond argument/error plumbing.

Usage:

    # Stdio transport (default — for Claude Desktop, Claude Code)
    python mcp_server.py

    # Networked transports (streamable-http, sse) via env vars
    MCP_TRANSPORT=streamable-http FASTMCP_HOST=0.0.0.0 FASTMCP_PORT=8001 python mcp_server.py
    # or use the provided start_server.sh script

Claude Desktop config example (stdio):
    {
      "mcpServers": {
        "tradingagents": {
          "command": "python",
          "args": ["<absolute_path>/TradingAgents/mcp_server.py"],
          "cwd": "<absolute_path>/TradingAgents"
        }
      }
    }

Environment variables:
    MCP_TRANSPORT: Transport type (default "stdio"). Options: "stdio", "streamable-http", "sse".
    FASTMCP_HOST: Host to bind to for networked transports (default "127.0.0.1").
    FASTMCP_PORT: Port to bind to for networked transports (default "8001").
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

# ── make sure the TradingAgents package is importable ───────────────────────
_HERE = Path(__file__).parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from mcp.server.fastmcp import FastMCP  # noqa: E402

# ── read transport and network configuration from environment ──────────────────
_mcp_transport = os.environ.get("MCP_TRANSPORT", "stdio")
_fastmcp_host = os.environ.get("FASTMCP_HOST", "127.0.0.1")
_fastmcp_port = os.environ.get("FASTMCP_PORT", "8001")

# Module logger, defined unconditionally — same convention used elsewhere in
# this codebase (trading_graph.py, dataflows/interface.py, memory/resolve.py,
# memory/mcp_client.py): a plain `logging.getLogger(__name__)`, with no
# `if logger:` guard at call sites. For stdio transport, stdout is the
# JSON-RPC protocol channel, so logging is suppressed process-wide via
# `logging.disable` below; every `logger.info`/`logger.error` call anywhere
# in this module then costs nothing and doesn't need special-casing.
logger = logging.getLogger(__name__)

# ── configure logging based on transport ──────────────────────────────────────
# For stdio, keep logging suppressed to avoid corrupting the JSON-RPC protocol.
# For networked transports (streamable-http, sse), enable logging for visibility.
if _mcp_transport == "stdio":
    logging.disable(logging.CRITICAL)
else:
    # force=True: guarantees the root logger is (re)configured even if a
    # handler already exists (e.g. this module reloaded within one process,
    # as the test suite does) rather than basicConfig silently no-op'ing.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        force=True,
    )

mcp = FastMCP(
    "TradingAgents",
    instructions=(
        "Use analyze_stock to run a full multi-agent trading analysis for a "
        "stock ticker and receive a detailed report with a final BUY / SELL / "
        "HOLD recommendation."
    ),
    host=_fastmcp_host,
    port=int(_fastmcp_port),
)


# ── lazy-load heavy dependencies only when the tool is actually called ───────
def _run_analysis(ticker: str, date: str) -> tuple[str, dict]:
    """
    Run TradingAgents and return (report_markdown, structured_data).
    All intermediate output is suppressed (stdout + stderr → /dev/null).
    """
    with open(os.devnull, "w") as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
        from dotenv import load_dotenv

        load_dotenv()

        from tradingagents.default_config import DEFAULT_CONFIG
        from tradingagents.graph.trading_graph import TradingAgentsGraph
        from tradingagents.report_generator import save_report_to_disk

        config = DEFAULT_CONFIG.copy()
        config["llm_provider"] = "ollama"
        config["quick_think_llm"] = "ministral-3:3b"
        config["deep_think_llm"] = "ministral-3:8b"
        config["max_debate_rounds"] = 1
        config["max_risk_discuss_rounds"] = 1
        config["debug"] = False

        ta = TradingAgentsGraph(
            selected_analysts=["market", "news", "fundamentals"],
            debug=False,
            config=config,
        )

        final_state, decision = ta.propagate(ticker, date)

    # Build the report in a temp dir (cleaned up automatically)
    with tempfile.TemporaryDirectory() as tmpdir:
        report_dir = Path(tmpdir) / f"{ticker}_{date}"
        report_file, structured_data = save_report_to_disk(
            final_state, ticker, report_dir
        )
        report_md = report_file.read_text(encoding="utf-8")

    return report_md, structured_data, decision


# ── shared start/success/error logging for tool wrappers ────────────────────
# Factored out of the 6 tool functions below (design-review follow-up on
# issue #59: the log-start/log-success/log-error control flow was previously
# hand-copy-pasted, with slightly different phrasing, in every one of them).
# Each wrapper uses this the same way:
#
#   try:
#       with _log_tool_call("tool_name", **args_worth_logging):
#           result = ...  # the call that may raise
#       return result
#   except Exception as exc:
#       return f"ERROR: {exc}"
#
# The context manager only owns logging (start / completed / failed lines,
# each naming the tool and its arguments); it re-raises so each wrapper keeps
# its own conversion of exceptions into the "ERROR: ..." string return value.
@contextmanager
def _log_tool_call(name: str, **fields: Any) -> Iterator[None]:
    arg_str = ", ".join(f"{key}={value}" for key, value in fields.items())
    logger.info("%s: starting (%s)", name, arg_str)
    try:
        yield
    except Exception as exc:  # noqa: BLE001 - logged for visibility, then re-raised
        logger.error("%s: failed (%s): %s", name, arg_str, exc)
        raise
    else:
        logger.info("%s: completed (%s)", name, arg_str)


@mcp.tool()
def analyze_stock(ticker: str, date: str) -> str:
    """
    Run a full TradingAgents analysis for a stock and return a comprehensive report.

    The analysis runs three agents (Market, News, Fundamentals), a bull/bear
    research debate, a trader, and a risk-management team, then synthesises
    everything into a final BUY / SELL / HOLD decision.

    Args:
        ticker: Stock ticker symbol, e.g. "AAPL" or "NVDA".
        date:   Reference date for the analysis in YYYY-MM-DD format,
                e.g. "2024-05-10". Data up to this date is used.

    Returns:
        On success: full analysis report as markdown + structured JSON summary.
        On error:   A string starting with "ERROR:" describing what went wrong.
    """
    ticker = ticker.strip().upper()

    try:
        with _log_tool_call("analyze_stock", ticker=ticker, date=date):
            report_md, structured_data, decision = _run_analysis(ticker, date)
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: {exc}"

    structured_block = (
        "```json\n" + json.dumps(structured_data, indent=2) + "\n```"
    )

    return "\n\n".join(
        [
            f"## Final Decision: {decision}",
            "---",
            report_md,
            "---",
            "## Structured Summary",
            structured_block,
        ]
    )


# ── memory core tools (issue #24) ────────────────────────────────────────
# Thin pass-throughs to tradingagents/memory/ — one shared SQLite store
# (see CLAUDE.md "Persistence") used by both this MCP server's callers and
# the legacy graph's Python code. No logic of their own: each tool imports
# tradingagents.memory lazily (so importing this module doesn't pull in
# resolve.py's numpy/yfinance dependency until a tool is actually invoked)
# and either returns the wrapped function's result unchanged or, on
# exception, a string starting with "ERROR:" (matching analyze_stock's
# error-handling convention above).


@mcp.tool()
def memory_store_decision(
    agent: str,
    ticker: str,
    date: str,
    signal: str,
    confidence: float | None = None,
    key_drivers: Any = None,
    thesis: str | None = None,
    db_path: str | None = None,
    horizon_days: int | None = None,
) -> bool | str:
    """
    Record a pending trading decision in the shared memory core.

    Thin pass-through to tradingagents.memory.store_decision. Idempotent on
    (agent, ticker, date): a duplicate call is a no-op that leaves the
    original row untouched.

    Args:
        agent: Free-form agent identifier (e.g. "trader" or a skill name).
        ticker: Ticker symbol, stored exactly as given (no normalization).
        date: Decision date, "YYYY-MM-DD".
        signal: Directional call, e.g. "Buy"/"Hold"/"Sell" or "BUY"/"HOLD"/"SELL".
        confidence: Optional numeric confidence score.
        key_drivers: Optional JSON-serializable list/dict of key drivers.
        thesis: Optional short free-text rationale.
        db_path: Optional override for the memory DB path (defaults to the
            TRADINGAGENTS_MEMORY_DB_PATH env var, then runs/memory/memory.db).
        horizon_days: Optional intended holding period in trading days. If
            given, stored in the row's horizon_days column; if None, the
            column is left NULL and filled by resolve_pending with the default.

    Returns:
        True if a new pending row was inserted, False if a row already
        existed for this (agent, ticker, date) key. On error, a string
        starting with "ERROR:".
    """
    try:
        with _log_tool_call(
            "memory_store_decision", agent=agent, ticker=ticker, date=date, signal=signal
        ):
            from tradingagents.memory import store_decision

            result = store_decision(
                agent=agent,
                ticker=ticker,
                date=date,
                signal=signal,
                confidence=confidence,
                key_drivers=key_drivers,
                thesis=thesis,
                db_path=db_path,
                horizon_days=horizon_days,
            )
        return result
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: {exc}"


@mcp.tool()
def memory_resolve_pending(
    agent: str | None = None,
    ticker: str | None = None,
    db_path: str | None = None,
) -> list[int] | str:
    """
    Resolve pending memory-core decisions whose horizon has elapsed.

    Thin pass-through to tradingagents.memory.resolve_pending. For every
    pending row (optionally filtered to one agent and/or ticker) whose
    decision is at least DEFAULT_HORIZON_DAYS trading days old, fetches the
    realized forward return and an LLM-generated lesson, and writes both
    back. Rows whose window hasn't elapsed yet, or whose price data isn't
    available yet, are left pending untouched (no error).

    Why this needs an LLM call at all: turning a resolved decision into a
    prose "lesson" is a synthesis step, not a lookup — see the module
    docstring above `_build_reflector`/`_generate_lesson` in
    tradingagents/memory/resolve.py for the full rationale.

    Args:
        agent: Optional exact agent id to restrict to.
        ticker: Optional exact (un-normalized) ticker to restrict to.
        db_path: Optional override for the memory DB path.

    Returns:
        A list of `decisions.id` values resolved by this call (possibly
        empty). On error, a string starting with "ERROR:".
    """
    try:
        with _log_tool_call("memory_resolve_pending", agent=agent, ticker=ticker):
            from tradingagents.memory import resolve_pending

            result = resolve_pending(agent=agent, ticker=ticker, db_path=db_path)
        return result
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: {exc}"


@mcp.tool()
def memory_get_past_context(
    agent: str,
    ticker: str,
    n_same: int = 5,
    n_cross: int = 3,
    db_path: str | None = None,
) -> str:
    """
    Render past resolved memory-core lessons for an agent/ticker as markdown.

    Thin pass-through to tradingagents.memory.get_past_context. Only
    resolved decisions (a realized forward return + lesson already written
    by memory_resolve_pending) are ever included.

    Args:
        agent: Exact agent identifier to match.
        ticker: Exact ticker to match for the same-ticker section (and to
            exclude from the cross-ticker section).
        n_same: Max number of same-ticker entries to include.
        n_cross: Max number of cross-ticker entries (same agent, other
            tickers) to include.
        db_path: Optional override for the memory DB path.

    Returns:
        A non-empty markdown string, safe to inject into a prompt directly
        (a "no prior lessons" placeholder is returned when nothing matches).
        On error, a string starting with "ERROR:".
    """
    try:
        with _log_tool_call(
            "memory_get_past_context", agent=agent, ticker=ticker, n_same=n_same, n_cross=n_cross
        ):
            from tradingagents.memory import get_past_context

            result = get_past_context(
                agent=agent, ticker=ticker, n_same=n_same, n_cross=n_cross, db_path=db_path
            )
        return result
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: {exc}"


@mcp.tool()
def memory_get_statistics(
    agent: str | None = None,
    ticker: str | None = None,
    since: str | None = None,
    db_path: str | None = None,
) -> dict[str, Any] | str:
    """
    Compute hit-rate / average-return / calibration statistics from the memory core.

    Thin pass-through to tradingagents.memory.get_statistics.

    Args:
        agent: Optional exact agent id to restrict to.
        ticker: Optional exact (un-normalized) ticker to restrict to.
        since: Optional inclusive lower bound on decision_date ("YYYY-MM-DD").
        db_path: Optional override for the memory DB path.

    Returns:
        A JSON-serializable dict with "filters", "per_agent_ticker",
        "per_agent", and "calibration" keys — see
        tradingagents.memory.stats.get_statistics for the exact shape. On
        error, a string starting with "ERROR:".
    """
    try:
        with _log_tool_call("memory_get_statistics", agent=agent, ticker=ticker, since=since):
            from tradingagents.memory import get_statistics

            result = get_statistics(agent=agent, ticker=ticker, since=since, db_path=db_path)
        return result
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: {exc}"


@mcp.tool()
def memory_get_decisions(
    agent: str,
    ticker: str,
    db_path: str | None = None,
    limit: int | None = None,
    misses_only: bool = False,
) -> list[dict[str, Any]] | str:
    """
    Fetch raw per-decision context rows for an agent/ticker pair from the memory core.

    Thin pass-through to tradingagents.memory.gather_context_rows. Returns resolved
    decision rows (forward_return + lesson already resolved) with their key_drivers,
    thesis, and lesson — the material needed for pattern analysis and reasoning about
    why a divergent performance pattern exists for a given (agent, ticker) pair.

    Only resolved decisions (``resolved_at IS NOT NULL``) are returned; pending rows
    are always excluded.

    Args:
        agent: Exact agent identifier to match (no normalization).
        ticker: Exact (un-normalized) ticker to match against the stored column
            (no normalization, consistent with resolve_pending and get_past_context).
        db_path: Optional override for the memory DB path.
        limit: Maximum number of rows to return, most recent first
            (``decision_date`` DESC, ``id`` DESC). If None, defaults to
            DEFAULT_CONTEXT_LIMIT (10).
        misses_only: If True, only rows scored "incorrect" are returned; useful
            for focusing on failures when investigating underperformance patterns.

    Returns:
        A JSON-serializable list of dicts, one per resolved decision:
        ``{"decision_date", "signal", "confidence", "key_drivers" (parsed JSON
        object/list or None), "thesis", "lesson", "forward_return", "correct"}``.
        The ``correct`` field is ``None`` if the signal doesn't normalize to
        BUY/HOLD/SELL, and is ``True``/``False`` otherwise. On error, a string
        starting with "ERROR:".
    """
    try:
        with _log_tool_call(
            "memory_get_decisions", agent=agent, ticker=ticker, limit=limit, misses_only=misses_only
        ):
            from tradingagents.memory import DEFAULT_CONTEXT_LIMIT, gather_context_rows

            effective_limit = limit if limit is not None else DEFAULT_CONTEXT_LIMIT
            result = gather_context_rows(
                agent=agent,
                ticker=ticker,
                db_path=db_path,
                limit=effective_limit,
                misses_only=misses_only,
            )
        return result
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: {exc}"


if __name__ == "__main__":
    # Validate MCP_TRANSPORT env var and dispatch to appropriate transport
    valid_transports = {"stdio", "streamable-http", "sse"}
    if _mcp_transport not in valid_transports:
        # "stdio" is itself a valid transport, so this branch is only ever
        # reached for a value other than "stdio" — logging is always enabled
        # here (see the transport branch above) and this call surfaces the
        # error via logging's default stderr handler.
        error_msg = (
            f"ERROR: Invalid MCP_TRANSPORT '{_mcp_transport}'. "
            f"Must be one of: {', '.join(sorted(valid_transports))}"
        )
        logger.error(error_msg)
        sys.exit(2)

    # Log startup for networked transports (a no-op under stdio, see above).
    logger.info(
        "TradingAgents MCP server starting on %s (host=%s, port=%s)",
        _mcp_transport,
        _fastmcp_host,
        _fastmcp_port,
    )

    if _mcp_transport == "stdio":
        mcp.run()
    else:
        mcp.run(transport=_mcp_transport)
