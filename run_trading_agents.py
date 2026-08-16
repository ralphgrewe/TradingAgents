#!/usr/bin/env python3
"""
Script to run Trading Agents for a list of stocks from JSON file or a run config file.

Usage (stock list — legacy behavior, unchanged):
    python run_trading_agents.py stocks.json [--report-dir REPORT_DIR] [--show-summary]
    python run_trading_agents.py stocks.json --use-dates-from-json  # to use dates from JSON
    python run_trading_agents.py stocks.json --portfolio --style aggressive --depot-id my-depot
    python run_trading_agents.py stocks.json --llm-provider mistral --deep-think-llm mistral-large --quick-think-llm mistral-small

Usage (run config file — new in issue #116):
    python run_trading_agents.py config.json

Expected JSON format for stock lists (date field is optional by default):
    [
        {"ticker": "AAPL"},  # date defaults to today when script runs
        {"ticker": "MSFT"}   # date defaults to today when script runs
    ]

With --use-dates-from-json, date field is required:
    [
        {"ticker": "AAPL", "date": "2024-01-15"},
        {"ticker": "MSFT", "date": "2024-01-15"}
    ]

Run config file format (top-level JSON object, new feature):
    {
        "stocks_file": "stocks.json",
        "llm_provider": "mistral",
        "deep_think_llm": "mistral-large",
        "quick_think_llm": "mistral-small",
        "report_dir": "./reports/mistral-aggressive",
        "show_summary": true,
        "use_dates_from_json": false,
        "portfolio": true,
        "style": "aggressive",
        "depot_id": "mistral-depot",
        "memory_id": "mistral-aggressive",
        "config": {
            "research_stage": "debate",
            "max_debate_rounds": 2,
            "swing_trader_enabled": true,
            "temperature": 0.3,
            "data_vendors": {"news_data": "alpha_vantage"}
        }
    }

Recognized top-level config keys (exactly the argparse dest names in snake_case, plus stocks_file):
- stocks_file (string, required): path to the stock list JSON, resolved relative to the
  config file's directory (if relative) or used as-is (if absolute)
- llm_provider (string, default "ollama")
- deep_think_llm (string)
- quick_think_llm (string)
- report_dir (string, default "./reports")
- show_summary (bool, default false)
- use_dates_from_json (bool, default false)
- portfolio (bool, default false)
- style (string, "aggressive" | "conservative")
- depot_id (string)
- memory_id (string)
- config (object, optional): a nested object containing DEFAULT_CONFIG keys and values (issue #117).
  Any key from DEFAULT_CONFIG (including nested keys like data_vendors.news_data) can be set here.
  Nested dict values deep-merge one level only (e.g., setting data_vendors.news_data keeps
  other data_vendors entries intact). Type of values must match the DEFAULT_CONFIG default's type.
  Top-level keys (above) take precedence if set in both places.

Precedence (highest to lowest):
1. CLI flag (if provided on command line)
2. Top-level config file key (if the positional argument is a config object)
3. Config file's nested "config" block (issue #117)
4. TRADINGAGENTS_* environment variable
5. DEFAULT_CONFIG

The full five-tier chain applies to keys that are backed by a DEFAULT_CONFIG
entry (those in _ENV_OVERRIDES: llm_provider, deep_think_llm,
quick_think_llm, memory_id, research_stage, swing_trader_enabled, etc.).
When neither the CLI flag nor a top-level config key sets one of these, the
nested "config" block is consulted. If neither that nor any TRADINGAGENTS_*
env var is set, the script leaves it unset so DEFAULT_CONFIG.copy()'s own
value — already resolved against its TRADINGAGENTS_* env var at import time
— takes effect untouched, rather than a hardcoded literal silently
overwriting it. Top-level keys (tier 2) take precedence over the nested
"config" block (tier 3) for any key that appears in both places.

The remaining recognized keys (report_dir, show_summary, use_dates_from_json,
portfolio, style, depot_id) are script-only settings with no DEFAULT_CONFIG
entry or TRADINGAGENTS_* env var of their own; for those, "default" in tier 5
above is just this script's built-in literal (e.g. report_dir → "./reports"),
so effectively only tiers 1, 2, and 5 apply.

The set of recognized config-file keys and the CLI/config-file merge live in
one place in the source (_CONFIG_FILE_FLAG_KEYS in run_trading_agents.py) so
they cannot drift out of sync with each other.

Optional arguments (CLI-only):
    --llm-provider PROVIDER  LLM provider to use (default: ollama). Supported providers:
                            openai, anthropic, google, mistral, ollama, etc.
    --deep-think-llm MODEL  Model name for deep thinking tasks (required if --llm-provider is not ollama).
    --quick-think-llm MODEL Model name for quick thinking tasks (required if --llm-provider is not ollama).
    --report-dir REPORT_DIR  Directory to save analysis reports (default: ./reports)
    --show-summary           Display formatted analysis summary for each stock
    --use-dates-from-json    Use date field from JSON file for each ticker (default: off, use today's date).
    --portfolio               Opt-in portfolio mode (see below). Requires --style and --depot-id.
    --style {aggressive,conservative}  Portfolio style-table to use.
    --depot-id DEPOT_ID       Named simulated depot to get-or-create and rebalance.
    --memory-id MEMORY_ID     Named memory ID to isolate SQLite decision history for this run
                              (default: off, uses runs/memory/memory.db). Each ID creates a
                              separate DB at runs/memory/<id>/memory.db, allowing different
                              models or configurations to maintain independent histories.

Date handling:
- Default behavior (no --use-dates-from-json flag): runs every ticker against today's date
  (datetime.date.today()), computed once when the script starts and reused for all tickers.
  The "date" field in JSON is ignored if present and is optional.
- With --use-dates-from-json: each ticker is run against its own "date" field from the JSON.
  The "date" field becomes required; if missing, the script exits with a clear error.
  This matches the original behavior prior to issue #72.

LLM Provider Configuration:
- Default behavior (no --llm-provider flag): uses ollama with ministral-3:8b (deep) and
  ministral-3:3b (quick) — byte-for-byte unchanged from before.
- When --llm-provider is not ollama: both --deep-think-llm and --quick-think-llm are required.
- API key requirements: before processing any tickers, the script checks that the required
  API key environment variable for the provider is set (e.g., MISTRAL_API_KEY for mistral).
  If missing, the script exits immediately with a clear error.

The script produces the same quality output as the CLI application:
- Uses explicit analyst selection (market, social, news, fundamentals)
- Configures proper research depth settings
- Generates comprehensive reports with --report-dir
- Displays formatted summaries with --show-summary

Portfolio mode (--portfolio or config file): without it, behavior is unchanged (per-ticker
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
import os
import sys
from pathlib import Path

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.llm_clients.api_key_env import get_api_key_env
from tradingagents.portfolio.runner import extract_rating, run_portfolio_mode
from tradingagents.report_generator import save_report_to_disk
from tradingagents.reporting import format_report_preview
from tradingagents.simulation import SimulationClientError

# Single source of truth: the config-file keys that mirror an existing CLI
# flag, in snake_case argparse-dest form. Both the config-file validation
# (recognized vs. unrecognized top-level keys) and the CLI/config-file merge
# loop in main() derive from this one tuple, so a new flag added later can't
# update one and forget the other (this is the same pattern used by
# tradingagents/default_config.py's _ENV_OVERRIDES table). `stocks_file` is
# handled separately because it is required and has no corresponding CLI
# flag. Kept as its own tuple (rather than folded into argparse itself) so
# issue #117's nested "config" block can extend recognition later without
# touching this set.
_CONFIG_FILE_FLAG_KEYS = (
    "llm_provider", "deep_think_llm", "quick_think_llm", "report_dir",
    "show_summary", "use_dates_from_json", "portfolio", "style",
    "depot_id", "memory_id",
)
_CONFIG_FILE_RECOGNIZED_KEYS = frozenset({"stocks_file", "config", *_CONFIG_FILE_FLAG_KEYS})


def _validate_config_block(config_block: dict, config_path: str) -> None:
    """Validate a 'config' block inside a run config file.

    The config block contains DEFAULT_CONFIG keys and their values. This
    function validates that all keys are recognized DEFAULT_CONFIG keys and
    that all values have compatible types with the DEFAULT_CONFIG defaults.

    Args:
        config_block: The parsed "config" dict (must be a dict or None).
        config_path: Path to the config file, used only in error messages.

    Raises:
        SystemExit on any validation error.
    """
    if config_block is None:
        return

    if not isinstance(config_block, dict):
        print(f"Error: 'config' block in '{config_path}' must be a JSON object (dict), not {type(config_block).__name__}")
        sys.exit(1)

    # Helper to recursively check if a key exists in DEFAULT_CONFIG (supporting dot notation)
    def _key_exists_in_default_config(key_path: str) -> bool:
        """Check if a dot-notation key path exists in DEFAULT_CONFIG."""
        from tradingagents.default_config import _get_nested
        if "." in key_path:
            keys = key_path.split(".")
            return _get_nested(DEFAULT_CONFIG, keys) is not None
        return key_path in DEFAULT_CONFIG

    # Helper to check type compatibility
    def _type_is_compatible(value, reference) -> bool:
        """Check if a JSON value's type is compatible with the reference type."""
        # None is always compatible
        if value is None:
            return True
        # For int/float compatibility: accept int where float is expected
        if isinstance(reference, float) and isinstance(value, int):
            return True
        # Otherwise, types must match exactly
        return type(value) == type(reference)

    # Validate each key and type
    from tradingagents.default_config import _get_nested

    for key, value in config_block.items():
        # Check if key exists in DEFAULT_CONFIG
        if not _key_exists_in_default_config(key):
            print(f"Error: Unrecognized config block key: '{key}'")
            sorted_keys = sorted(DEFAULT_CONFIG.keys())
            print(f"Recognized config keys: {', '.join(sorted_keys[:10])}... (and more)")
            sys.exit(1)

        # Get the default value to check type compatibility
        if "." in key:
            keys = key.split(".")
            default_value = _get_nested(DEFAULT_CONFIG, keys)
        else:
            default_value = DEFAULT_CONFIG.get(key)

        # Type check: JSON type must match DEFAULT_CONFIG type
        if default_value is not None:
            if isinstance(default_value, dict) and isinstance(value, dict):
                # For dict values, we don't deeply validate nested keys —
                # they'll be validated by set_config's one-level merge
                pass
            elif not _type_is_compatible(value, default_value):
                print(f"Error: Config block key '{key}' has type {type(value).__name__}, "
                      f"expected {type(default_value).__name__}")
                sys.exit(1)


def display_summary(final_state, ticker):
    """Display a formatted summary of the analysis similar to CLI output."""
    print(f"\n{'='*60}")
    print(f"ANALYSIS SUMMARY FOR {ticker}")
    print(f"{'='*60}")

    # Analyst Reports. market/news/fundamentals/sentiment are all JSON
    # envelopes (#30-#32, #71); format_report_preview prefers the envelope's
    # signal/confidence/summary over a truncated raw-JSON snippet.
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

    # Swing Trader (when enabled)
    if final_state.get("swing_trade_decision"):
        print("\n⚡ SWING TRADER DECISION:")
        print(f"• Swing Trader: {final_state['swing_trade_decision'][:200]}...")

    # Risk Management
    if final_state.get("risk_debate_state"):
        risk = final_state["risk_debate_state"]
        print("\n🛡️  RISK MANAGEMENT DECISION:")
        if risk.get("judge_decision"):
            print(f"• Portfolio Manager: {risk['judge_decision'][:200]}...")

    print(f"\n{'='*60}")
    print(f"FINAL DECISION: {final_state.get('final_trade_decision', 'N/A')}")
    print(f"{'='*60}\n")

def _validate_config_file(config_data: dict, config_path: str) -> None:
    """Validate an already-parsed run config file's shape.

    The caller (main()) is responsible for opening and json.load()-ing the
    file exactly once (it already has to, to disambiguate array-vs-object);
    this function only validates the resulting dict, so the file is never
    read or parsed a second time here.

    Args:
        config_data: The parsed config dict (top-level JSON object).
        config_path: Path to the config file, used only in error messages.

    Raises:
        SystemExit on any validation error.
    """
    # Check for required stocks_file key
    if "stocks_file" not in config_data:
        print(f"Error: Config file '{config_path}' missing required key 'stocks_file'")
        sys.exit(1)

    # Check for unrecognized keys
    unrecognized = set(config_data.keys()) - _CONFIG_FILE_RECOGNIZED_KEYS
    if unrecognized:
        sorted_keys = sorted(_CONFIG_FILE_RECOGNIZED_KEYS)
        print(f"Error: Unrecognized config key(s): {', '.join(sorted(unrecognized))}")
        print(f"Recognized keys: {', '.join(sorted_keys)}")
        sys.exit(1)

    # If 'config' block is present, validate it
    if "config" in config_data:
        _validate_config_block(config_data["config"], config_path)


def _resolve_stocks_file_path(stocks_file: str, config_path: str | None) -> str:
    """Resolve a stocks_file path, handling relative paths.

    If stocks_file is relative and config_path is provided, resolve relative to
    the config file's directory. Otherwise, use as-is.

    Args:
        stocks_file: The stocks_file path from config
        config_path: The path to the config file (None if from CLI)

    Returns:
        The resolved stocks_file path
    """
    stocks_path = Path(stocks_file)
    if not stocks_path.is_absolute() and config_path:
        # Resolve relative to the config file's directory
        config_dir = Path(config_path).parent
        stocks_path = config_dir / stocks_file
    return str(stocks_path)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run Trading Agents for stocks from JSON file or run config file')
    parser.add_argument('json_file', help='Path to JSON file (stock list array or run config object)')
    parser.add_argument('--llm-provider', dest='llm_provider', default=None,
                        help='LLM provider to use (default: ollama). Examples: openai, anthropic, mistral, google, etc.')
    parser.add_argument('--deep-think-llm', dest='deep_think_llm', default=None,
                        help='Model name for deep thinking tasks (required unless --llm-provider is ollama).')
    parser.add_argument('--quick-think-llm', dest='quick_think_llm', default=None,
                        help='Model name for quick thinking tasks (required unless --llm-provider is ollama).')
    parser.add_argument('--report-dir', dest='report_dir', default=None,
                        help='Directory to save analysis reports (default: ./reports)')
    parser.add_argument('--show-summary', dest='show_summary', action='store_true', default=None,
                        help='Display formatted analysis summary')
    parser.add_argument('--use-dates-from-json', dest='use_dates_from_json', action='store_true', default=None,
                         help='Use date field from JSON file for each ticker (default: off, use today\'s date).')
    parser.add_argument('--portfolio', dest='portfolio', action='store_true', default=None,
                         help='Opt-in portfolio mode: after running the full pipeline for every '
                              'ticker, compute a style-table allocation and execute rebalancing '
                              'trades in a simulated depot. The stock-list JSON is the universe.')
    parser.add_argument('--style', dest='style', choices=['aggressive', 'conservative'], default=None,
                         help='Portfolio style (required with --portfolio).')
    parser.add_argument('--depot-id', dest='depot_id', default=None,
                        help='Named simulated depot to get-or-create and trade in '
                             '(required with --portfolio).')
    parser.add_argument('--memory-id', dest='memory_id', default=None,
                        help='Named memory ID to isolate SQLite decision history '
                             '(default: off, uses runs/memory/memory.db).')
    args = parser.parse_args()

    # Load and parse json_file exactly once. The parsed value's top-level
    # JSON type disambiguates a stock list (array) from a run config file
    # (object) — see _validate_config_file, which validates this same
    # already-parsed dict rather than re-opening/re-parsing the file.
    config_data = None
    config_path = None
    try:
        with open(args.json_file) as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{args.json_file}' not found")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in '{args.json_file}'")
        sys.exit(1)

    # Disambiguate: array → stock list, object → config file
    if isinstance(data, dict):
        # It's a config file
        _validate_config_file(data, args.json_file)
        config_data = data
        config_path = args.json_file
    elif isinstance(data, list):
        # It's a stock list (legacy)
        stocks = data
    else:
        print("Error: JSON should contain either an array (stock list) or an object (run config)")
        sys.exit(1)

    # If config file was loaded, extract stocks_file and merge config into args
    if config_data:
        stocks_file = config_data.get("stocks_file")
        stocks_file = _resolve_stocks_file_path(stocks_file, config_path)

        # Load the stock list from the resolved path
        try:
            with open(stocks_file) as f:
                stocks = json.load(f)
        except FileNotFoundError:
            print(f"Error: stocks_file '{stocks_file}' (from config) not found")
            sys.exit(1)
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in stocks_file '{stocks_file}' (from config)")
            sys.exit(1)

        if not isinstance(stocks, list):
            print(f"Error: stocks_file '{stocks_file}' should contain an array of stock objects")
            sys.exit(1)

        # Merge config file values into args, respecting precedence: CLI > config > defaults
        # For each key, use: CLI value (if provided) > config value > existing arg default (None)
        for key in _CONFIG_FILE_FLAG_KEYS:
            cli_value = getattr(args, key)
            config_value = config_data.get(key)

            # CLI flag wins if explicitly provided (not None)
            if cli_value is not None:
                # Keep the CLI value
                pass
            elif config_value is not None:
                # Use config value
                setattr(args, key, config_value)

    # Apply defaults for any remaining None values. These are script-only
    # settings with no DEFAULT_CONFIG/TRADINGAGENTS_* env-var backing, so a
    # plain hardcoded fallback is correct for them (unlike llm_provider,
    # deep_think_llm, quick_think_llm, and memory_id below, which are backed
    # by DEFAULT_CONFIG entries and must stay None here so an unset value
    # defers to DEFAULT_CONFIG's own — possibly env-resolved — default
    # instead of a literal baked in at this point).
    if args.report_dir is None:
        args.report_dir = './reports'
    if args.show_summary is None:
        args.show_summary = False
    if args.use_dates_from_json is None:
        args.use_dates_from_json = False
    if args.portfolio is None:
        args.portfolio = False

    if args.portfolio and (not args.style or not args.depot_id):
        print("Error: --portfolio requires both --style and --depot-id")
        sys.exit(1)

    # Validate style is one of the allowed values (from config file, it's not enforced by argparse)
    if args.style and args.style not in ['aggressive', 'conservative']:
        print(f"Error: --style must be 'aggressive' or 'conservative', got '{args.style}'")
        sys.exit(1)

    # Validate memory_id if provided (issue #114)
    if args.memory_id:
        from tradingagents.memory.store import resolve_memory_id_to_db_path
        try:
            resolve_memory_id_to_db_path(args.memory_id)
        except ValueError as e:
            print(f"Error: Invalid --memory-id: {e}")
            sys.exit(1)

    # Configure Trading Agents. Built here, BEFORE the provider-dependent
    # validation below, so that validation and the API-key check can use the
    # fully resolved effective provider (config["llm_provider"]) rather than
    # the raw args.llm_provider — which is None whenever neither the
    # --llm-provider flag nor the config file set it, deliberately, so that
    # DEFAULT_CONFIG.copy()'s own env-override-aware value (honoring
    # TRADINGAGENTS_LLM_PROVIDER) is used instead of being silently
    # overwritten by a hardcoded 'ollama' literal here.
    config = DEFAULT_CONFIG.copy()

    # Apply config file's nested "config" block (issue #117) to DEFAULT_CONFIG
    # BEFORE the CLI-flag layer overwrites it. This maintains the precedence:
    # CLI flag > config file > TRADINGAGENTS_* env var > DEFAULT_CONFIG.
    # One-level deep merge for dict values, matching set_config semantics
    # (tradingagents/dataflows/config.py).
    if config_data and "config" in config_data:
        config_block = config_data["config"]
        for key, value in config_block.items():
            if isinstance(value, dict) and isinstance(config.get(key), dict):
                # One-level deep merge for dict values
                config[key].update(value)
            else:
                config[key] = value

    # CLI flags override config file settings
    if args.llm_provider:
        config["llm_provider"] = args.llm_provider

    # Set model names: only override the DEFAULT_CONFIG.copy() values when the
    # corresponding CLI flag / config-file key was actually provided, so the
    # ollama defaults stay sourced from DEFAULT_CONFIG (single source of
    # truth) instead of being duplicated here.
    if args.deep_think_llm:
        config["deep_think_llm"] = args.deep_think_llm

    if args.quick_think_llm:
        config["quick_think_llm"] = args.quick_think_llm

    # Set memory_id if provided (issue #114)
    if args.memory_id:
        config["memory_id"] = args.memory_id

    effective_provider = config["llm_provider"]

    # Validate provider and model requirements. deep_think_llm/quick_think_llm
    # are checked at the args level (not config's DEFAULT_CONFIG-backed
    # value): they are required to be explicitly set via CLI flag or config
    # file whenever the effective provider isn't ollama, regardless of any
    # TRADINGAGENTS_DEEP_THINK_LLM/TRADINGAGENTS_QUICK_THINK_LLM env default.
    if effective_provider.lower() != 'ollama' and (not args.deep_think_llm or not args.quick_think_llm):
        parser.error(f"--llm-provider {effective_provider} requires both --deep-think-llm and --quick-think-llm")

    # Check API key environment variable for the provider before processing tickers
    api_key_env = get_api_key_env(effective_provider)
    if api_key_env is not None and api_key_env not in os.environ:
        print(f"Error: LLM provider '{effective_provider}' requires environment variable '{api_key_env}' to be set")
        sys.exit(1)

    # Determine the date to use for all tickers
    # In default mode: use today's date (computed once, reused for all tickers)
    # In --use-dates-from-json mode: validate that each stock has a date field
    if args.use_dates_from_json:
        # Validate that all stocks have required 'date' field in from-JSON mode
        for i, stock in enumerate(stocks):
            if not isinstance(stock, dict) or 'ticker' not in stock or not stock.get('date'):
                print(f"Error: Stock at index {i} must have 'ticker' and non-empty 'date' fields (--use-dates-from-json mode active)")
                sys.exit(1)
    else:
        # In default mode, just validate 'ticker' is present
        for stock in stocks:
            if not isinstance(stock, dict) or 'ticker' not in stock:
                print("Error: Each stock should have a 'ticker' field")
                sys.exit(1)

    # Compute the run date once, before the per-ticker loop. Resolved as an
    # ISO string (not a datetime.date) to match the convention used
    # everywhere else dates flow through this codebase (cli/main.py,
    # tradingagents/graph/propagation.py, tradingagents/agents/utils/memory.py,
    # tradingagents/memory/mcp_client.py all treat dates as "YYYY-MM-DD"
    # strings). A raw datetime.date here would fail to JSON-serialize in the
    # consolidated trading_summary.json and would break the memory MCP
    # client's JSON-RPC store_decision call.
    run_date = datetime.date.today().isoformat() if not args.use_dates_from_json else None

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
        # Resolve the date: use run_date (today) in default mode, or the stock's date in from-JSON mode
        date = run_date if not args.use_dates_from_json else stock['date']
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
            exc_type = type(e).__name__
            print(f"\nFatal error processing {ticker}: [{exc_type}] {str(e)}")
            sys.exit(1)

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
            exc_type = type(e).__name__
            print(f"\nFatal error: portfolio mode failed to reach the simulator: [{exc_type}] {str(e)}")
            sys.exit(1)
        except Exception as e:
            exc_type = type(e).__name__
            print(f"\nFatal error: portfolio mode failed: [{exc_type}] {str(e)}")
            sys.exit(1)

if __name__ == "__main__":
    main()
