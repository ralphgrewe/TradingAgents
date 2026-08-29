"""Reusable report-tree writer shared by the CLI and the programmatic API.

Writes a run's per-section files (analysts, research, trading, risk, portfolio)
plus a consolidated ``complete_report.md`` under ``save_path``. The CLI and
``TradingAgentsGraph.save_reports`` both call this, so a headless / API run
produces the same on-disk report tree a CLI run does.

Since #30-#32 and #71, ``market_report``/``fundamentals_report``/``news_report``/
``sentiment_report`` are all JSON envelope strings (``skill``/``ticker``/``date``/
``signal``/``confidence``/``summary``/``details``, see skills/SCHEMA.md) and are
saved as ``.json`` files; every other text field handled here (debate history,
trader plan, final decision) intentionally remains prose and is saved as ``.md``.
The ``format_report_*`` helpers below detect which shape a given field is and
render/preview it accordingly, so mixed-format reports flow through the same
code path without special-casing per analyst.
"""

import json
from datetime import datetime
from pathlib import Path

# Static mapping of analyst field names to their output file extensions.
# JSON envelopes (since #30-#32, #71) are saved as .json; prose fields as .md.
# Update this mapping as more analysts migrate to JSON-envelope output.
_ANALYST_REPORT_EXTENSIONS = {
    "market_report": ".json",      # JSON envelope since #30
    "sentiment_report": ".json",   # JSON envelope since #71
    "news_report": ".json",        # JSON envelope since #32
    "fundamentals_report": ".json", # JSON envelope since #31
    "macro_report": ".json",       # JSON envelope since #132
    "macro_news_report": ".json",  # JSON envelope since #134
}


def _try_parse_envelope(report: str) -> dict | None:
    """Return the parsed dict if ``report`` looks like a JSON envelope, else None."""
    if not isinstance(report, str):
        return None
    stripped = report.strip()
    if not stripped.startswith("{"):
        return None
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def backfill_portfolio_adjustments(
    envelope: dict,
    per_ticker_report_dirs: dict[str, str],
    missing_ratings: list[str] | None = None,
    missing_prices: list[str] | None = None,
) -> None:
    """Backfill per-ticker reports with portfolio adjustments and executed orders.

    Called after portfolio mode completes to update each ticker's `5_portfolio/decision.md`
    and the `## V.` section of its `complete_report.md` with:
    - Proposed adjustments table (ticker, signal, target weight, Δ shares)
    - Executed trades list (with status, fill price, message)
    - Rejected orders shown explicitly with their messages
    - Dropped tickers (no rating or no price) shown with explanatory line

    Args:
        envelope: The dict returned by run_portfolio_mode() containing:
            - details.depot_id, details.universe, details.signals, details.allocation,
              details.trades_executed, details.rejected_orders
        per_ticker_report_dirs: {ticker: path_to_report_dir} mapping from the per-ticker
            loop that already ran. Only tickers present in this dict have their reports updated.
        missing_ratings: Tickers that had no rating (pipeline failed). Their reports
            get an explanatory line instead of a table.
        missing_prices: Tickers that had no usable price (dropped inside run_portfolio_mode).
            Their reports get an explanatory line instead of a table.

    Idempotency: Running this function twice on the same report directory is safe.
    The function detects if the portfolio section is already present (by checking for
    the "### Proposed adjustments" marker) and skips that ticker if found, avoiding
    duplicate or nested appends.
    """
    missing_ratings = missing_ratings or []
    missing_prices = missing_prices or []

    details = envelope.get("details", {})
    depot_id = details.get("depot_id", "unknown")
    universe = details.get("universe", [])
    signals = details.get("signals", {})
    allocation = details.get("allocation", {})
    trades_executed = details.get("trades_executed", [])

    # Build a map of tickers to their trades (for the "Executed" section)
    ticker_trades_map = {}
    for trade in trades_executed:
        symbol = trade.get("symbol")
        if symbol:
            if symbol not in ticker_trades_map:
                ticker_trades_map[symbol] = []
            ticker_trades_map[symbol].append(trade)

    # Helper function to render proposed adjustments table for a specific ticker or full universe
    def render_table(ticker_filter=None):
        """Render the proposed adjustments table, optionally filtered to one ticker."""
        table_lines = [
            "| ticker | signal | target wt | Δ shares |",
            "|--------|--------|-----------|----------|",
        ]
        tickers_to_show = [ticker_filter] if ticker_filter else universe
        for t in tickers_to_show:
            if t not in universe:
                continue
            signal_data = signals.get(t, {})
            signal = signal_data.get("signal", "HOLD")
            alloc = allocation.get(t, {})
            target_wt = alloc.get("target_weight", 0.0)
            delta = alloc.get("delta", 0)
            table_lines.append(f"| {t} | {signal} | {target_wt:.2f} | {delta} |")
        return "\n".join(table_lines)

    # Helper function to render executed trades for a specific ticker or full universe
    def render_executed(ticker_filter=None):
        """Render the executed trades section, optionally filtered to one ticker."""
        executed_lines = []
        tickers_to_show = [ticker_filter] if ticker_filter else universe
        for t in tickers_to_show:
            if t not in universe:
                continue
            trades = ticker_trades_map.get(t, [])
            if not trades:
                executed_lines.append(f"- {t}: (no trade)")
                continue

            for trade in trades:
                side = trade.get("side", "unknown").upper()
                quantity = trade.get("quantity", 0)
                status = trade.get("status", "unknown")
                fill_price = trade.get("fill_price")
                message = trade.get("message", "")

                if status == "rejected":
                    executed_lines.append(
                        f"- {t}: {side} {quantity} shares — REJECTED: {message}"
                    )
                else:
                    price_str = f" @ ${fill_price:.2f}" if fill_price else ""
                    executed_lines.append(
                        f"- {t}: {side} {quantity} shares{price_str} — {status}"
                    )

        return "\n".join(executed_lines) if executed_lines else "(none — HOLD)"

    # For dropped tickers, collect explanations per ticker
    dropped_explanations = {}
    for ticker in missing_ratings:
        if ticker in per_ticker_report_dirs:
            dropped_explanations[ticker] = "no rating (pipeline run failed) — dropped from portfolio run"
    for ticker in missing_prices:
        if ticker in per_ticker_report_dirs:
            dropped_explanations[ticker] = "no usable price (no quote or recorded price) — dropped from portfolio run"

    # Update each ticker's report files
    for ticker, report_dir in per_ticker_report_dirs.items():
        report_path = Path(report_dir)

        # Check if this ticker was dropped
        is_dropped = ticker in dropped_explanations

        # Build the per-ticker backfill text once, shared by decision.md and
        # complete_report.md so both files get equivalent, correctly-scoped
        # content for this ticker (see issue #157 escalation: complete_report.md
        # previously reused an unfiltered, whole-universe rendering here).
        if is_dropped:
            # Dropped ticker gets an explanatory line instead of a table.
            backfill_text = (
                f"\n\n### Dropped\n"
                f"- {ticker}: {dropped_explanations[ticker]}"
            )
        else:
            # Render table and executed list filtered to just this ticker.
            ticker_table = render_table(ticker_filter=ticker)
            ticker_executed = render_executed(ticker_filter=ticker)

            backfill_text = (
                f"\n\n### Proposed adjustments (depot {depot_id})\n"
                f"{ticker_table}\n\n"
                f"### Executed\n"
                f"{ticker_executed}"
            )

        # Update decision.md (5_portfolio/decision.md)
        decision_file = report_path / "5_portfolio" / "decision.md"
        if decision_file.exists():
            decision_text = decision_file.read_text(encoding="utf-8")
            # Check if already backfilled (idempotency)
            already_backfilled = (
                "### Proposed adjustments" in decision_text or "### Dropped" in decision_text
            )
            if not already_backfilled:
                decision_file.write_text(decision_text + backfill_text, encoding="utf-8")

        # Update complete_report.md
        complete_file = report_path / "complete_report.md"
        if complete_file.exists():
            complete_text = complete_file.read_text(encoding="utf-8")
            # Check if already backfilled (idempotency)
            if "### Proposed adjustments" in complete_text or "### Dropped" in complete_text:
                continue

            # Find and replace the "## V. Portfolio Manager Decision" section
            # The current section is just "## V. Portfolio Manager Decision\n\n### Portfolio Manager\n{judge_decision}"
            # We need to extend it with the proposed adjustments and executed sections

            # Use a marker to find where to insert the portfolio adjustments
            v_section_marker = "## V. Portfolio Manager Decision"
            if v_section_marker not in complete_text:
                continue

            # Find the PM decision section start
            v_idx = complete_text.find(v_section_marker)
            if v_idx == -1:
                continue

            # The section contains "### Portfolio Manager" followed by the judge_decision
            # We want to find the end of that section and insert our portfolio adjustments
            # Look for the next "## " (next section) or end of file
            after_v = complete_text[v_idx + len(v_section_marker):]
            next_section = after_v.find("\n## ")
            if next_section == -1:
                # This is the last section
                section_end = len(complete_text)
            else:
                section_end = v_idx + len(v_section_marker) + next_section

            # Extract the PM decision section
            pm_section = complete_text[v_idx:section_end]

            # Insert this ticker's (filtered) backfill text at the end of its
            # own "## V." section — same text used for decision.md above.
            updated_pm_section = pm_section + backfill_text

            # Replace the old section with the updated one
            complete_text = complete_text[:v_idx] + updated_pm_section + complete_text[section_end:]
            complete_file.write_text(complete_text, encoding="utf-8")


def render_rebalance_summary(
    envelope: dict,
    depot_id: str,
    style: str,
    missing_ratings: list[str] | None = None,
    missing_prices: list[str] | None = None,
) -> str:
    """Render the run-level markdown summary written next to the per-ticker reports.

    ``envelope`` is the dict returned by ``run_portfolio_mode`` (same shape
    consumed by :func:`backfill_portfolio_adjustments`). Written to
    ``portfolio-rebalance-<depot_id>.md`` by the CLI after portfolio mode
    completes.
    """
    # Local import: tradingagents.portfolio.runner pulls in tradingagents.graph
    # (for SignalProcessor), which itself imports this module at package-init
    # time — a module-level import here would be circular.
    from tradingagents.portfolio.runner import count_buys_and_sells

    missing_ratings = missing_ratings or []
    missing_prices = missing_prices or []

    details = envelope.get("details", {})
    pre_snapshot = details.get("pre_snapshot", {})
    post_snapshot = details.get("post_snapshot", {})
    equity_change = details.get("equity_change", 0.0)
    trades_executed = details.get("trades_executed", [])
    rejected_orders = details.get("rejected_orders", [])

    n_buys, n_sells = count_buys_and_sells(trades_executed)
    n_rejected = len(rejected_orders)

    summary_md = f"""# Portfolio Rebalance Summary

**Depot**: {depot_id}
**Style**: {style}
**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Pre-Rebalance Snapshot
- **Total Equity**: ${pre_snapshot.get("equity", 0):,.2f}
- **Cash**: ${pre_snapshot.get("cash", 0):,.2f}
- **Number of Positions**: {len(pre_snapshot.get("positions", {}))}

## Post-Rebalance Snapshot
- **Total Equity**: ${post_snapshot.get("equity", 0):,.2f}
- **Cash**: ${post_snapshot.get("cash", 0):,.2f}
- **Number of Positions**: {len(post_snapshot.get("positions", {}))}

## Execution Summary
- **Equity Change**: ${equity_change:,.2f}
- **Buys**: {n_buys}
- **Sells**: {n_sells}
- **Rejected Orders**: {n_rejected}

## Universe
{len(details.get("universe", []))} tickers rebalanced.
"""

    if missing_ratings:
        summary_md += f"\n**Note**: {len(missing_ratings)} ticker(s) had no rating and were excluded from the rebalance.\n"
    if missing_prices:
        summary_md += f"\n**Note**: {len(missing_prices)} ticker(s) had no usable price and were excluded from the rebalance.\n"

    return summary_md


def format_report_preview(report: str, max_len: int = 150) -> str:
    """Return a short, human-readable one-line preview of an analyst report field.

    For JSON envelopes, prefer the envelope's own ``signal``/``confidence``/
    ``summary`` over a truncated raw-JSON snippet (which would just show
    ``{\\n  "skill": ...``). Prose fields (debate history, trader plan,
    final decision) are truncated unchanged.
    """
    text = report or ""
    envelope = _try_parse_envelope(text)
    if envelope is not None and "summary" in envelope:
        signal = envelope.get("signal")
        confidence = envelope.get("confidence")
        tag = f"[{signal}/{confidence}] " if signal else ""
        text = f"{tag}{envelope['summary']}"
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text


def format_report_markdown(report: str) -> str:
    """Render an analyst report field for embedding in a markdown document.

    JSON envelopes are rendered as a one-line signal/confidence header
    followed by the (already pretty-printed) envelope in a fenced ```json
    code block, so markdown renderers don't mangle the raw braces/quotes.
    Prose fields pass through unchanged.
    """
    envelope = _try_parse_envelope(report)
    if envelope is None:
        return report
    signal = envelope.get("signal")
    confidence = envelope.get("confidence")
    summary = envelope.get("summary") or ""
    header = f"**Signal:** {signal} (confidence: {confidence})\n\n{summary}" if signal else summary
    return f"{header}\n\n```json\n{report}\n```"


def write_report_tree(final_state: dict, ticker: str, save_path) -> Path:
    """Save a completed run's reports to ``save_path``; return the complete-report path."""
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    sections = []

    # 1. Analysts
    analysts_dir = save_path / "1_analysts"
    analyst_parts = []
    if final_state.get("market_report"):
        analysts_dir.mkdir(exist_ok=True)
        ext = _ANALYST_REPORT_EXTENSIONS["market_report"]
        (analysts_dir / f"market{ext}").write_text(final_state["market_report"], encoding="utf-8")
        analyst_parts.append(("Market Analyst", final_state["market_report"]))
    if final_state.get("sentiment_report"):
        analysts_dir.mkdir(exist_ok=True)
        ext = _ANALYST_REPORT_EXTENSIONS["sentiment_report"]
        (analysts_dir / f"sentiment{ext}").write_text(final_state["sentiment_report"], encoding="utf-8")
        analyst_parts.append(("Sentiment Analyst", final_state["sentiment_report"]))
    if final_state.get("news_report"):
        analysts_dir.mkdir(exist_ok=True)
        ext = _ANALYST_REPORT_EXTENSIONS["news_report"]
        (analysts_dir / f"news{ext}").write_text(final_state["news_report"], encoding="utf-8")
        analyst_parts.append(("News Analyst", final_state["news_report"]))
    if final_state.get("fundamentals_report"):
        analysts_dir.mkdir(exist_ok=True)
        ext = _ANALYST_REPORT_EXTENSIONS["fundamentals_report"]
        (analysts_dir / f"fundamentals{ext}").write_text(final_state["fundamentals_report"], encoding="utf-8")
        analyst_parts.append(("Fundamentals Analyst", final_state["fundamentals_report"]))
    if final_state.get("macro_report"):
        analysts_dir.mkdir(exist_ok=True)
        ext = _ANALYST_REPORT_EXTENSIONS["macro_report"]
        (analysts_dir / f"macro{ext}").write_text(final_state["macro_report"], encoding="utf-8")
        analyst_parts.append(("Macro Fundamentals Analyst", final_state["macro_report"]))
    if final_state.get("macro_news_report"):
        analysts_dir.mkdir(exist_ok=True)
        ext = _ANALYST_REPORT_EXTENSIONS["macro_news_report"]
        (analysts_dir / f"macro_news{ext}").write_text(final_state["macro_news_report"], encoding="utf-8")
        analyst_parts.append(("Macro News Analyst", final_state["macro_news_report"]))
    if analyst_parts:
        # Individual per-analyst .md files above keep the raw field (JSON
        # envelope or prose) so they stay machine-parseable; the consolidated
        # report below renders JSON envelopes fenced for readability.
        content = "\n\n".join(
            f"### {name}\n{format_report_markdown(text)}" for name, text in analyst_parts
        )
        sections.append(f"## I. Analyst Team Reports\n\n{content}")

    # 2. Research
    if final_state.get("investment_debate_state"):
        research_dir = save_path / "2_research"
        debate = final_state["investment_debate_state"]
        research_parts = []
        if debate.get("bull_history"):
            research_dir.mkdir(exist_ok=True)
            (research_dir / "bull.md").write_text(debate["bull_history"], encoding="utf-8")
            research_parts.append(("Bull Researcher", debate["bull_history"]))
        if debate.get("bear_history"):
            research_dir.mkdir(exist_ok=True)
            (research_dir / "bear.md").write_text(debate["bear_history"], encoding="utf-8")
            research_parts.append(("Bear Researcher", debate["bear_history"]))
        if debate.get("judge_decision"):
            research_dir.mkdir(exist_ok=True)
            (research_dir / "manager.md").write_text(debate["judge_decision"], encoding="utf-8")
            research_parts.append(("Research Manager", debate["judge_decision"]))
        if research_parts:
            content = "\n\n".join(f"### {name}\n{text}" for name, text in research_parts)
            sections.append(f"## II. Research Team Decision\n\n{content}")

    # 3. Trading
    if final_state.get("trader_investment_plan"):
        trading_dir = save_path / "3_trading"
        trading_dir.mkdir(exist_ok=True)
        (trading_dir / "trader.md").write_text(final_state["trader_investment_plan"], encoding="utf-8")
        sections.append(f"## III. Trading Team Plan\n\n### Trader\n{final_state['trader_investment_plan']}")

    # 4. Risk Management
    if final_state.get("risk_debate_state"):
        risk_dir = save_path / "4_risk"
        risk = final_state["risk_debate_state"]
        risk_parts = []
        if risk.get("aggressive_history"):
            risk_dir.mkdir(exist_ok=True)
            (risk_dir / "aggressive.md").write_text(risk["aggressive_history"], encoding="utf-8")
            risk_parts.append(("Aggressive Analyst", risk["aggressive_history"]))
        if risk.get("conservative_history"):
            risk_dir.mkdir(exist_ok=True)
            (risk_dir / "conservative.md").write_text(risk["conservative_history"], encoding="utf-8")
            risk_parts.append(("Conservative Analyst", risk["conservative_history"]))
        if risk.get("neutral_history"):
            risk_dir.mkdir(exist_ok=True)
            (risk_dir / "neutral.md").write_text(risk["neutral_history"], encoding="utf-8")
            risk_parts.append(("Neutral Analyst", risk["neutral_history"]))
        if risk_parts:
            content = "\n\n".join(f"### {name}\n{text}" for name, text in risk_parts)
            sections.append(f"## IV. Risk Management Team Decision\n\n{content}")

        # 5. Portfolio Manager
        if risk.get("judge_decision"):
            portfolio_dir = save_path / "5_portfolio"
            portfolio_dir.mkdir(exist_ok=True)
            (portfolio_dir / "decision.md").write_text(risk["judge_decision"], encoding="utf-8")
            sections.append(f"## V. Portfolio Manager Decision\n\n### Portfolio Manager\n{risk['judge_decision']}")

    # Write consolidated report
    header = f"# Trading Analysis Report: {ticker}\n\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    (save_path / "complete_report.md").write_text(header + "\n\n".join(sections), encoding="utf-8")
    return save_path / "complete_report.md"
