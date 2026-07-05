#!/usr/bin/env python3
"""
Report generation module for TradingAgents.

This module layers a structured ``trading_recommendation.json`` export (plus a
``summary.txt``, with regex-based fallback parsing when structured agent
output isn't available) on top of the shared markdown report tree written by
``tradingagents.reporting.write_report_tree``. Used by entry points that need
the richer JSON/summary export (``mcp_server.py``, ``run_trading_agents.py``);
``cli/main.py`` has its own, simpler structured-export step.
"""

from pathlib import Path
from typing import Dict, Any

from tradingagents.reporting import write_report_tree


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

    # Markdown report tree (1_analysts/ … 5_portfolio/ plus complete_report.md)
    # is shared with the CLI and the programmatic API via write_report_tree.
    report_file = write_report_tree(final_state, ticker, save_path)
    return report_file, structured_data