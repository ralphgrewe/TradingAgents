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

Usage (stdio transport, default):
    python mcp_server.py

Claude Desktop config example:
    {
      "mcpServers": {
        "tradingagents": {
          "command": "python",
          "args": ["<absolute_path>/TradingAgents/mcp_server.py"],
          "cwd": "<absolute_path>/TradingAgents"
        }
      }
    }
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

# ── silence all logging before any imports that might configure it ──────────
logging.disable(logging.CRITICAL)

# ── make sure the TradingAgents package is importable ───────────────────────
_HERE = Path(__file__).parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from mcp.server.fastmcp import FastMCP  # noqa: E402

mcp = FastMCP(
    "TradingAgents",
    instructions=(
        "Use analyze_stock to run a full multi-agent trading analysis for a "
        "stock ticker and receive a detailed report with a final BUY / SELL / "
        "HOLD recommendation."
    ),
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
) -> Any:
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

    Returns:
        True if a new pending row was inserted, False if a row already
        existed for this (agent, ticker, date) key. On error, a string
        starting with "ERROR:".
    """
    try:
        from tradingagents.memory import store_decision

        return store_decision(
            agent=agent,
            ticker=ticker,
            date=date,
            signal=signal,
            confidence=confidence,
            key_drivers=key_drivers,
            thesis=thesis,
            db_path=db_path,
        )
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: {exc}"


@mcp.tool()
def memory_resolve_pending(
    agent: str | None = None,
    ticker: str | None = None,
    db_path: str | None = None,
) -> Any:
    """
    Resolve pending memory-core decisions whose horizon has elapsed.

    Thin pass-through to tradingagents.memory.resolve_pending. For every
    pending row (optionally filtered to one agent and/or ticker) whose
    decision is at least DEFAULT_HORIZON_DAYS trading days old, fetches the
    realized forward return and an LLM-generated lesson, and writes both
    back. Rows whose window hasn't elapsed yet, or whose price data isn't
    available yet, are left pending untouched (no error).

    Args:
        agent: Optional exact agent id to restrict to.
        ticker: Optional exact (un-normalized) ticker to restrict to.
        db_path: Optional override for the memory DB path.

    Returns:
        A list of `decisions.id` values resolved by this call (possibly
        empty). On error, a string starting with "ERROR:".
    """
    try:
        from tradingagents.memory import resolve_pending

        return resolve_pending(agent=agent, ticker=ticker, db_path=db_path)
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: {exc}"


@mcp.tool()
def memory_get_past_context(
    agent: str,
    ticker: str,
    n_same: int = 5,
    n_cross: int = 3,
    db_path: str | None = None,
) -> Any:
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
        from tradingagents.memory import get_past_context

        return get_past_context(
            agent=agent, ticker=ticker, n_same=n_same, n_cross=n_cross, db_path=db_path
        )
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: {exc}"


@mcp.tool()
def memory_get_statistics(
    agent: str | None = None,
    ticker: str | None = None,
    since: str | None = None,
    db_path: str | None = None,
) -> Any:
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
        from tradingagents.memory import get_statistics

        return get_statistics(agent=agent, ticker=ticker, since=since, db_path=db_path)
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: {exc}"


if __name__ == "__main__":
    mcp.run()
