#!/usr/bin/env python3
"""
Script to run Trading Agents with Ollama provider for a list of stocks from JSON file.

Usage:
    python run_trading_agents.py stocks.json

Expected JSON format:
    [
        {"ticker": "AAPL", "date": "2024-01-15"},
        {"ticker": "MSFT", "date": "2024-01-15"}
    ]
"""

import argparse
import json
import sys
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run Trading Agents for stocks from JSON file')
    parser.add_argument('json_file', help='Path to JSON file containing stock list')
    args = parser.parse_args()
    
    # Read stock list from JSON file
    try:
        with open(args.json_file, 'r') as f:
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
    
    # Configure Trading Agents with Ollama provider (already default)
    config = DEFAULT_CONFIG.copy()
    # Ensure we're using Ollama provider
    config["llm_provider"] = "ollama"
    # Use ministral-3:3b model (already default for quick_think_llm)
    config["quick_think_llm"] = "ministral-3:3b"
    config["deep_think_llm"] = "ministral-3:8b"
    
    # Initialize Trading Agents
    ta = TradingAgentsGraph(debug=True, config=config)
    
    # Process each stock
    for stock in stocks:
        ticker = stock['ticker']
        date = stock['date']
        print(f"\nProcessing {ticker} for date {date}...")
        
        try:
            _, decision = ta.propagate(ticker, date)
            print(f"Decision for {ticker}: {decision}")
        except Exception as e:
            print(f"Error processing {ticker}: {str(e)}")

if __name__ == "__main__":
    main()