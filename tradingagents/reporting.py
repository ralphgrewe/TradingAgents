"""Reusable report-tree writer shared by the CLI and the programmatic API.

Writes a run's per-section markdown (analysts, research, trading, risk,
portfolio) plus a consolidated ``complete_report.md`` under ``save_path``. The
CLI and ``TradingAgentsGraph.save_reports`` both call this, so a headless / API
run produces the same on-disk report tree a CLI run does.

Since #30-#32, ``market_report``/``fundamentals_report``/``news_report`` are
JSON envelope strings (``skill``/``ticker``/``date``/``signal``/``confidence``/
``summary``/``details``, see skills/SCHEMA.md) rather than markdown;
``sentiment_report`` (social analyst) and every other text field handled here
(debate history, trader plan, final decision) intentionally remain prose. The
``format_report_*`` helpers below detect which shape a given field is and
render/preview it accordingly, so mixed-format reports flow through the same
code path without special-casing per analyst.
"""

import json
from datetime import datetime
from pathlib import Path


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


def format_report_preview(report: str, max_len: int = 150) -> str:
    """Return a short, human-readable one-line preview of an analyst report field.

    For JSON envelopes, prefer the envelope's own ``signal``/``confidence``/
    ``summary`` over a truncated raw-JSON snippet (which would just show
    ``{\\n  "skill": ...``). Prose fields (``sentiment_report``, debate
    history, trader plan, final decision) are truncated unchanged.
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
        (analysts_dir / "market.md").write_text(final_state["market_report"], encoding="utf-8")
        analyst_parts.append(("Market Analyst", final_state["market_report"]))
    if final_state.get("sentiment_report"):
        analysts_dir.mkdir(exist_ok=True)
        (analysts_dir / "sentiment.md").write_text(final_state["sentiment_report"], encoding="utf-8")
        analyst_parts.append(("Sentiment Analyst", final_state["sentiment_report"]))
    if final_state.get("news_report"):
        analysts_dir.mkdir(exist_ok=True)
        (analysts_dir / "news.md").write_text(final_state["news_report"], encoding="utf-8")
        analyst_parts.append(("News Analyst", final_state["news_report"]))
    if final_state.get("fundamentals_report"):
        analysts_dir.mkdir(exist_ok=True)
        (analysts_dir / "fundamentals.md").write_text(final_state["fundamentals_report"], encoding="utf-8")
        analyst_parts.append(("Fundamentals Analyst", final_state["fundamentals_report"]))
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
