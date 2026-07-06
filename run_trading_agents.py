#!/usr/bin/env python3
"""
Script to run Trading Agents with Ollama provider for a list of stocks from JSON file.

Usage:
    python run_trading_agents.py stocks.json [--report-dir REPORT_DIR] [--show-summary]
    python run_trading_agents.py stocks.json --portfolio --style aggressive --depot-id my-depot

Expected JSON format:
    [
        {"ticker": "AAPL", "date": "2024-01-15"},
        {"ticker": "MSFT", "date": "2024-01-15"}
    ]

Optional arguments:
    --report-dir REPORT_DIR  Directory to save analysis reports (default: ./reports)
    --show-summary           Display formatted analysis summary for each stock
    --portfolio               Opt-in portfolio mode (see below). Requires --style and --depot-id.
    --style {aggressive,conservative}  Portfolio style-table to use.
    --depot-id DEPOT_ID       Named simulated depot to get-or-create and rebalance.

The script now produces the same quality output as the CLI application:
- Uses explicit analyst selection (market, social, news, fundamentals)
- Configures proper research depth settings
- Generates comprehensive reports with --report-dir
- Displays formatted summaries with --show-summary

Portfolio mode (--portfolio): without it, behavior is unchanged (per-ticker
reports only). With it, the stock-list JSON is treated as the investment
universe: the full pipeline still runs for every ticker, but afterwards the
final 5-tier rating from each run is mapped to a style-table signal
(tradingagents.portfolio.rebalance), a target allocation is computed, and
rebalancing trades are executed against the simulator (tradingagents.simulation
.SimulationClient) in the named depot. A JSON run report is written to
--report-dir alongside the per-ticker reports.
"""

import argparse
import datetime
import json
import sys
from pathlib import Path

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.portfolio.runner import extract_rating, run_portfolio_mode
from tradingagents.report_generator import save_report_to_disk
from tradingagents.reporting import format_report_preview
from tradingagents.simulation import SimulationClientError


def display_summary(final_state, ticker):
    """Display a formatted summary of the analysis similar to CLI output."""
    print(f"\n{'='*60}")
    print(f"ANALYSIS SUMMARY FOR {ticker}")
    print(f"{'='*60}")

    # Analyst Reports. market/news/fundamentals are JSON envelopes (#30-#32);
    # format_report_preview prefers the envelope's signal/confidence/summary
    # over a truncated raw-JSON snippet. sentiment_report stays prose.
    if any(final_state.get(f"{analyst}_report") for analyst in ["market", "sentiment", "news", "fundamentals"]):
        print("\n📊 ANALYST TEAM REPORTS:")
        if final_state.get("market_report"):
            print(f"• Market Analyst: {format_report_preview(final_state['market_report'])}")
        if final_state.get("sentiment_report"):
            print(f"• Social Analyst: {format_report_preview(final_state['sentiment_report'])}")
        if final_state.get("news_report"):
            print(f"• News Analyst: {format_report_preview(final_state['news_report'])}")
        if final_state.get("fundamentals_report"):
            print(f"• Fundamentals Analyst: {format_report_preview(final_state['fundamentals_report'])}")

    # Research Team
    if final_state.get("investment_debate_state"):
        debate = final_state["investment_debate_state"]
        print("\n🔍 RESEARCH TEAM DECISION:")
        if debate.get("judge_decision"):
            print(f"• Research Manager: {debate['judge_decision'][:200]}...")

    # Trading Team
    if final_state.get("trader_investment_plan"):
        print("\n💼 TRADING TEAM PLAN:")
        print(f"• Trader: {final_state['trader_investment_plan'][:200]}...")

    # Risk Management
    if final_state.get("risk_debate_state"):
        risk = final_state["risk_debate_state"]
        print("\n🛡️  RISK MANAGEMENT DECISION:")
        if risk.get("judge_decision"):
            print(f"• Portfolio Manager: {risk['judge_decision'][:200]}...")

    print(f"\n{'='*60}")
    print(f"FINAL DECISION: {final_state.get('final_trade_decision', 'N/A')}")
    print(f"{'='*60}\n")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run Trading Agents for stocks from JSON file')
    parser.add_argument('json_file', help='Path to JSON file containing stock list')
    parser.add_argument('--report-dir', help='Directory to save analysis reports', default='./reports')
    parser.add_argument('--show-summary', action='store_true', help='Display formatted analysis summary')
    parser.add_argument('--portfolio', action='store_true',
                         help='Opt-in portfolio mode: after running the full pipeline for every '
                              'ticker, compute a style-table allocation and execute rebalancing '
                              'trades in a simulated depot. The stock-list JSON is the universe.')
    parser.add_argument('--style', choices=['aggressive', 'conservative'],
                         help='Portfolio style (required with --portfolio).')
    parser.add_argument('--depot-id', help='Named simulated depot to get-or-create and trade in '
                                            '(required with --portfolio).')
    args = parser.parse_args()

    if args.portfolio and (not args.style or not args.depot_id):
        print("Error: --portfolio requires both --style and --depot-id")
        sys.exit(1)

    # Read stock list from JSON file
    try:
        with open(args.json_file) as f:
            stocks = json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{args.json_file}' not found")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in '{args.json_file}'")
        sys.exit(1)

    # Validate stock list format
    if not isinstance(stocks, list):
        print("Error: JSON should contain an array of stock objects")
        sys.exit(1)

    for stock in stocks:
        if not isinstance(stock, dict) or 'ticker' not in stock or 'date' not in stock:
            print("Error: Each stock should have 'ticker' and 'date' fields")
            sys.exit(1)

    # Configure Trading Agents with Ollama provider (similar to CLI defaults)
    config = DEFAULT_CONFIG.copy()
    # Ensure we're using Ollama provider
    config["llm_provider"] = "ollama"
    # Use ministral-3:3b model (already default for quick_think_llm)
    config["quick_think_llm"] = "ministral-3:3b"
    config["deep_think_llm"] = "ministral-3:8b"
    # Use same research depth as CLI default, Medium research depth
    config["max_debate_rounds"] = 3
    config["max_risk_discuss_rounds"] = 3

    # Initialize Trading Agents with explicit analyst selection (same as CLI)
    # These are the same analysts that CLI uses by default
    selected_analysts = ["market", "social", "news", "fundamentals"]
    ta = TradingAgentsGraph(selected_analysts, debug=True, config=config)

    # Initialize list to collect structured data for consolidated summary
    all_structured_data = []
    # Ratings collected for portfolio mode (only tickers whose run succeeded)
    portfolio_ratings = {}

    # Process each stock
    for stock in stocks:
        ticker = stock['ticker']
        date = stock['date']
        print(f"\nProcessing {ticker} for date {date}...")

        try:
            final_state, decision = ta.propagate(ticker, date)
            print(f"Decision for {ticker}: {decision}")

            if args.portfolio:
                portfolio_ratings[ticker] = extract_rating(final_state)

            # Display formatted summary if requested
            if args.show_summary:
                display_summary(final_state, ticker)

            # Generate and save report if report directory is specified
            if args.report_dir:
                report_dir = Path(args.report_dir)
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                stock_report_dir = report_dir / f"{ticker}_{date}_{timestamp}"

                try:
                    report_file, structured_data = save_report_to_disk(final_state, ticker, stock_report_dir)
                    print(f"Report saved to: {report_file}")

                    # Collect structured data for consolidated summary
                    all_structured_data.append({
                        "ticker": ticker,
                        "date": date,
                        "data": structured_data
                    })
                except Exception as e:
                    print(f"Warning: Failed to save report for {ticker}: {str(e)}")
        except Exception as e:
            print(f"Error processing {ticker}: {str(e)}")

    # Create consolidated trading summary if we have data
    if args.report_dir and all_structured_data:
        try:
            report_dir = Path(args.report_dir)
            consolidated_summary = {
                "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "stocks": all_structured_data,
                "summary": {
                    "total_stocks_analyzed": len(all_structured_data),
                    "stocks": [{
                        "ticker": item["ticker"],
                        "date": item["date"],
                        "rating": item["data"].get("rating", "N/A"),
                        "action": item["data"].get("action", "N/A"),
                        "entry_price": item["data"].get("entry_price"),
                        "stop_loss": item["data"].get("stop_loss")
                    } for item in all_structured_data]
                }
            }

            summary_file = report_dir / "trading_summary.json"
            summary_file.write_text(json.dumps(consolidated_summary, indent=2), encoding="utf-8")
            print(f"\nConsolidated trading summary saved to: {summary_file}")
        except Exception as e:
            print(f"Warning: Failed to create consolidated summary: {str(e)}")

    # Portfolio mode: turn per-ticker ratings into a style-table allocation and
    # execute simulated rebalancing trades in the named depot.
    if args.portfolio:
        universe = [stock['ticker'] for stock in stocks]
        print(f"\nRunning portfolio mode ({args.style}, depot '{args.depot_id}')...")
        try:
            envelope, report_file = run_portfolio_mode(
                universe=universe,
                ratings=portfolio_ratings,
                style=args.style,
                depot_id=args.depot_id,
                report_dir=Path(args.report_dir),
            )
            print(envelope["summary"])
            print(f"Portfolio report saved to: {report_file}")
        except SimulationClientError as e:
            print(f"Error: portfolio mode failed to reach the simulator: {str(e)}")
        except Exception as e:
            print(f"Error: portfolio mode failed: {str(e)}")

if __name__ == "__main__":
    main()
