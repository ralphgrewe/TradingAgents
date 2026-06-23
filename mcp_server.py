#!/usr/bin/env python3
"""
TradingAgents MCP Server

Exposes a single tool: analyze_stock(ticker, date)
Returns the complete analysis report (markdown) plus structured JSON on success,
or an error message on failure.

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
    devnull = open(os.devnull, "w")
    try:
        with redirect_stdout(devnull), redirect_stderr(devnull):
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

    finally:
        devnull.close()


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


if __name__ == "__main__":
    mcp.run()
