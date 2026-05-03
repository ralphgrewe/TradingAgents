#!/usr/bin/env python3
"""
Report generation module for TradingAgents.

This module provides functionality to save analysis reports to disk
in markdown format, similar to the CLI application.
"""

import datetime
from pathlib import Path
from typing import Dict, Any


def save_report_to_disk(final_state: Dict[str, Any], ticker: str, save_path: Path) -> tuple[Path, dict]:
    """
    Save complete analysis report to disk with organized subfolders.
    
    Args:
        final_state: The final state dictionary from the trading graph
        ticker: Stock ticker symbol
        save_path: Base directory to save reports
        
    Returns:
        Tuple of (Path to the complete report file, structured_data dict)
    """
    save_path.mkdir(parents=True, exist_ok=True)
    sections = []
    
    # Create summary.txt and JSON from structured data if available
    if final_state.get("portfolio_structured_data"):
        # Use structured data from Portfolio Manager
        pm_data = final_state["portfolio_structured_data"]
        
        # Create structured JSON for backtesting
        structured_data = {
            "ticker": ticker,
            "rating": pm_data.get("rating", "N/A"),
            "executive_summary": pm_data.get("executive_summary", "N/A"),
            "investment_thesis": pm_data.get("investment_thesis", "N/A"),
            "price_target": pm_data.get("price_target"),
            "time_horizon": pm_data.get("time_horizon"),
        }
        
        # Add trader data if available
        if final_state.get("trader_structured_data"):
            trader_data = final_state["trader_structured_data"]
            structured_data.update({
                "action": trader_data.get("action", "N/A"),
                "reasoning": trader_data.get("reasoning", "N/A"),
                "entry_price": trader_data.get("entry_price"),
                "stop_loss": trader_data.get("stop_loss"),
                "position_sizing": trader_data.get("position_sizing"),
            })
        
        # Save structured JSON for backtesting
        import json
        json_file = save_path / "trading_recommendation.json"
        json_file.write_text(json.dumps(structured_data, indent=2), encoding="utf-8")
        
        # Create summary.txt with key trading information
        summary_content = f"Trading Recommendation Summary for {ticker}\n{'='*50}\n"
        summary_content += f"Rating: {pm_data.get('rating', 'N/A')}\n"
        summary_content += f"Executive Summary: {pm_data.get('executive_summary', 'N/A')}\n\n"
        summary_content += f"Investment Thesis:\n{pm_data.get('investment_thesis', 'N/A')}\n"
        
        if final_state.get("trader_structured_data"):
            trader_data = final_state["trader_structured_data"]
            summary_content += f"\n{'='*50}\n"
            summary_content += f"Trader Action: {trader_data.get('action', 'N/A')}\n"
            if trader_data.get("entry_price"):
                summary_content += f"Entry Price: {trader_data['entry_price']}\n"
            if trader_data.get("stop_loss"):
                summary_content += f"Stop Loss: {trader_data['stop_loss']}\n"
            if trader_data.get("position_sizing"):
                summary_content += f"Position Sizing: {trader_data['position_sizing']}\n"
            summary_content += f"\nReasoning:\n{trader_data.get('reasoning', 'N/A')}\n"
        
        summary_file = save_path / "summary.txt"
        summary_file.write_text(summary_content, encoding="utf-8")
    else:
        # Fallback: Try to parse from markdown if structured data not available
        trade_decision = final_state.get("final_trade_decision", "")
        trader_plan = final_state.get("trader_investment_plan", "")
        
        # Simple parsing for key fields
        import re
        
        def extract_field(text, pattern):
            match = re.search(pattern, text, re.DOTALL)
            return match.group(1).strip() if match else None
        
        # Try to extract from trader plan first (more likely to have action/stop loss)
        action = extract_field(trader_plan, r'\*\*Action\*\*:\s*(.+?)(?=\n\*\*|$)')
        reasoning = extract_field(trader_plan, r'\*\*Reasoning\*\*:\s*(.+?)(?=\n\*\*|$)')
        stop_loss = extract_field(trader_plan, r'\*\*Stop Loss\*\*:\s*([\d.]+)')
        position_sizing = extract_field(trader_plan, r'\*\*Position Sizing\*\*:\s*(.+?)(?=\n\*\*|$)')
        entry_price = extract_field(trader_plan, r'\*\*Entry Price\*\*:\s*([\d.]+)')
        
        # Try to extract from final trade decision
        rating = extract_field(trade_decision, r'\*\*Rating\*\*:\s*(.+?)(?=\n\*\*|$)')
        executive_summary = extract_field(trade_decision, r'\*\*Executive Summary\*\*:\s*(.+?)(?=\n\*\*|$)')
        investment_thesis = extract_field(trade_decision, r'\*\*Investment Thesis\*\*:\s*(.+?)(?=\n\*\*|$)')
        
        # Create structured data from parsed fields
        structured_data = {
            "ticker": ticker,
            "rating": rating or "N/A",
            "executive_summary": executive_summary or "N/A",
            "investment_thesis": investment_thesis or "N/A",
            "price_target": None,
            "time_horizon": None,
            "action": action or "N/A",
            "reasoning": reasoning or "N/A",
            "entry_price": float(entry_price) if entry_price else None,
            "stop_loss": float(stop_loss) if stop_loss else None,
            "position_sizing": position_sizing or None,
        }
        
        # Save structured JSON for backtesting
        import json
        json_file = save_path / "trading_recommendation.json"
        json_file.write_text(json.dumps(structured_data, indent=2), encoding="utf-8")
        
        # Create summary.txt with key trading information
        summary_content = f"Trading Recommendation Summary for {ticker}\n{'='*50}\n"
        if rating:
            summary_content += f"Rating: {rating}\n"
        if executive_summary:
            summary_content += f"Executive Summary: {executive_summary}\n\n"
        if investment_thesis:
            summary_content += f"Investment Thesis:\n{investment_thesis}\n"
        
        if action:
            summary_content += f"\n{'='*50}\n"
            summary_content += f"Trader Action: {action}\n"
            if entry_price:
                summary_content += f"Entry Price: {entry_price}\n"
            if stop_loss:
                summary_content += f"Stop Loss: {stop_loss}\n"
            if position_sizing:
                summary_content += f"Position Sizing: {position_sizing}\n"
            if reasoning:
                summary_content += f"\nReasoning:\n{reasoning}\n"
        
        summary_file = save_path / "summary.txt"
        summary_file.write_text(summary_content, encoding="utf-8")

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
        analyst_parts.append(("Social Analyst", final_state["sentiment_report"]))
    if final_state.get("news_report"):
        analysts_dir.mkdir(exist_ok=True)
        (analysts_dir / "news.md").write_text(final_state["news_report"], encoding="utf-8")
        analyst_parts.append(("News Analyst", final_state["news_report"]))
    if final_state.get("fundamentals_report"):
        analysts_dir.mkdir(exist_ok=True)
        (analysts_dir / "fundamentals.md").write_text(final_state["fundamentals_report"], encoding="utf-8")
        analyst_parts.append(("Fundamentals Analyst", final_state["fundamentals_report"]))
    if analyst_parts:
        content = "\n\n".join(f"### {name}\n{text}" for name, text in analyst_parts)
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
    header = f"# Trading Analysis Report: {ticker}\n\nGenerated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    (save_path / "complete_report.md").write_text(header + "\n\n".join(sections), encoding="utf-8")
    return save_path / "complete_report.md", structured_data