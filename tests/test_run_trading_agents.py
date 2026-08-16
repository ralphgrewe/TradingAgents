"""Tests for run_trading_agents.py error handling, fail-fast behavior, and LLM provider configuration.

These are integration-level tests that verify the error handling by running
the script with mocked dependencies and checking the exit behavior.
"""

import importlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import tradingagents.default_config as default_config_module


class _TempStockFileTestCase(unittest.TestCase):
    """Shared fixture: a temp dir with a JSON stock-list file for run_trading_agents.py.

    Subclasses may override STOCKS to control the file's contents; the temp
    dir and file are created in setUp and cleaned up in tearDown either way.
    """

    STOCKS = [
        {"ticker": "AAPL", "date": "2024-01-15"},
        {"ticker": "MSFT", "date": "2024-01-15"},
    ]

    def setUp(self):
        """Create temporary JSON file with test stocks."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.stock_list_file = Path(self.temp_dir.name) / "test_stocks.json"
        self.stock_list_file.write_text(json.dumps(self.STOCKS))

    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()


@pytest.mark.unit
class RunTradingAgentsErrorHandlingTests(_TempStockFileTestCase):
    """Test fail-fast error handling in run_trading_agents.py."""

    def test_propagate_failure_prints_exception_type_and_message(self):
        """Test that exception diagnostics include both type and message.

        The issue requirement states: "include the exception type/class name and
        message (not just str(e)), so e.g. an LLM-provider connection failure
        is distinguishable from a bad ticker/date or a data-vendor error."
        """
        import io
        from contextlib import redirect_stderr, redirect_stdout

        import run_trading_agents

        # Mock the TradingAgentsGraph to raise a clear exception
        with patch('run_trading_agents.TradingAgentsGraph') as mock_graph:
            mock_instance = MagicMock()
            mock_graph.return_value = mock_instance
            mock_instance.propagate.side_effect = ConnectionError("Failed to connect to Ollama at localhost:11434")

            # Capture output
            output = io.StringIO()
            with (
                patch('sys.argv', ['run_trading_agents.py', str(self.stock_list_file)]),
                redirect_stdout(output),
                redirect_stderr(output),
                self.assertRaises(SystemExit) as cm,
            ):
                run_trading_agents.main()

            # Verify exit code
            self.assertEqual(cm.exception.code, 1)

            # Verify diagnostic output includes exception type and message
            output_str = output.getvalue()
            self.assertIn('Fatal error processing', output_str)
            self.assertIn('ConnectionError', output_str)
            self.assertIn('Failed to connect to Ollama', output_str)

    def test_propagate_stops_processing_on_first_failure(self):
        """Test that the script fails fast and doesn't continue to next ticker."""
        import run_trading_agents

        # Mock the TradingAgentsGraph to fail on first call
        with patch('run_trading_agents.TradingAgentsGraph') as mock_graph:
            mock_instance = MagicMock()
            mock_graph.return_value = mock_instance
            mock_instance.propagate.side_effect = RuntimeError("Test error on first ticker")

            with patch('sys.argv', ['run_trading_agents.py', str(self.stock_list_file)]):
                with self.assertRaises(SystemExit) as cm:
                    run_trading_agents.main()

                # Verify exit code
                self.assertEqual(cm.exception.code, 1)
                # Verify propagate was only called once (for AAPL, not MSFT)
                self.assertEqual(mock_instance.propagate.call_count, 1)

    def test_portfolio_mode_failure_exits_with_diagnostic(self):
        """Test that portfolio mode failures also exit with code 1 and print diagnostics."""
        import io
        from contextlib import redirect_stderr, redirect_stdout

        import run_trading_agents
        from tradingagents.simulation import SimulationClientError

        # Mock both TradingAgentsGraph and run_portfolio_mode
        with patch('run_trading_agents.TradingAgentsGraph') as mock_graph, \
             patch('run_trading_agents.run_portfolio_mode') as mock_portfolio:

            # Setup successful propagate
            mock_instance = MagicMock()
            mock_graph.return_value = mock_instance
            mock_instance.propagate.return_value = (
                {"final_trade_decision": "BUY", "portfolio_structured_data": {"rating": "STRONG_BUY"}},
                "BUY"
            )

            # Setup failing portfolio mode with SimulationClientError
            mock_portfolio.side_effect = SimulationClientError("Simulator not running at localhost:8000")

            # Capture output
            output = io.StringIO()
            with (
                patch('sys.argv', ['run_trading_agents.py', str(self.stock_list_file),
                                   '--portfolio', '--style', 'aggressive', '--depot-id', 'test-depot']),
                redirect_stdout(output),
                redirect_stderr(output),
                self.assertRaises(SystemExit) as cm,
            ):
                run_trading_agents.main()

            # Verify exit code
            self.assertEqual(cm.exception.code, 1)

            # Verify diagnostic output
            output_str = output.getvalue()
            self.assertIn('Fatal error', output_str)
            self.assertIn('SimulationClientError', output_str)

    def test_generic_portfolio_mode_exception_exits_with_diagnostic(self):
        """Test that generic exceptions in portfolio mode also exit properly."""
        import io
        from contextlib import redirect_stderr, redirect_stdout

        import run_trading_agents

        with patch('run_trading_agents.TradingAgentsGraph') as mock_graph, \
             patch('run_trading_agents.run_portfolio_mode') as mock_portfolio:

            # Setup successful propagate
            mock_instance = MagicMock()
            mock_graph.return_value = mock_instance
            mock_instance.propagate.return_value = (
                {"final_trade_decision": "BUY"},
                "BUY"
            )

            # Setup failing portfolio mode with generic exception
            mock_portfolio.side_effect = ValueError("Invalid style parameter: 'moderate'")

            # Capture output
            output = io.StringIO()
            with (
                patch('sys.argv', ['run_trading_agents.py', str(self.stock_list_file),
                                   '--portfolio', '--style', 'aggressive', '--depot-id', 'test-depot']),
                redirect_stdout(output),
                redirect_stderr(output),
                self.assertRaises(SystemExit) as cm,
            ):
                run_trading_agents.main()

            # Verify exit code
            self.assertEqual(cm.exception.code, 1)

            # Verify exception type is in output
            output_str = output.getvalue()
            self.assertIn('ValueError', output_str)

    def test_report_saving_failure_does_not_exit(self):
        """Test that failures in report saving are non-fatal (don't exit).

        Per issue requirement: "No change to the report-saving try/except around
        save_report_to_disk... those are non-essential/cosmetic failures that
        already print a clear warning and continue; leave that behavior as-is."
        """
        import run_trading_agents

        with patch('run_trading_agents.TradingAgentsGraph') as mock_graph, \
             patch('run_trading_agents.save_report_to_disk') as mock_save_report:

            # Setup successful propagate
            mock_instance = MagicMock()
            mock_graph.return_value = mock_instance
            mock_instance.propagate.return_value = (
                {"final_trade_decision": "BUY"},
                "BUY"
            )

            # Setup failing report saving
            mock_save_report.side_effect = OSError("Cannot write to disk")

            with patch('sys.argv', ['run_trading_agents.py', str(self.stock_list_file),
                                   '--report-dir', './reports']):
                # Should NOT raise SystemExit for report-saving failures
                run_trading_agents.main()

                # Verify propagate was called for both tickers (not stopped on report error)
                self.assertEqual(mock_instance.propagate.call_count, 2)


@pytest.mark.unit
class RunTradingAgentDateHandlingTests(_TempStockFileTestCase):
    """Test date handling: default mode (today's date) and --use-dates-from-json mode."""

    def test_default_mode_uses_todays_date_when_date_field_omitted(self):
        """Test that in default mode, stocks without 'date' field use today's date."""
        import datetime

        import run_trading_agents

        # Stocks with no 'date' field
        self.STOCKS = [{"ticker": "AAPL"}, {"ticker": "MSFT"}]
        self.setUp()  # Recreate the JSON file with updated STOCKS

        with patch('run_trading_agents.TradingAgentsGraph') as mock_graph:
            mock_instance = MagicMock()
            mock_graph.return_value = mock_instance
            mock_instance.propagate.return_value = (
                {"final_trade_decision": "BUY"},
                "BUY"
            )

            with patch('sys.argv', ['run_trading_agents.py', str(self.stock_list_file)]):
                run_trading_agents.main()

                # Verify propagate was called twice
                self.assertEqual(mock_instance.propagate.call_count, 2)

                # Verify both calls used today's date, resolved as an ISO string
                # (not a datetime.date object) — matching the convention used
                # everywhere else dates flow through this codebase.
                today = datetime.date.today().isoformat()
                calls = mock_instance.propagate.call_args_list
                self.assertEqual(calls[0][0], ("AAPL", today))
                self.assertEqual(calls[1][0], ("MSFT", today))
                self.assertIsInstance(calls[0][0][1], str)
                self.assertIsInstance(calls[1][0][1], str)

    def test_default_mode_ignores_date_field_if_present(self):
        """Test that in default mode, 'date' field in JSON is ignored."""
        import datetime

        import run_trading_agents

        # Stocks with 'date' field that should be ignored
        self.STOCKS = [
            {"ticker": "AAPL", "date": "2024-01-15"},
            {"ticker": "MSFT", "date": "2024-02-20"}
        ]
        self.setUp()  # Recreate the JSON file with updated STOCKS

        with patch('run_trading_agents.TradingAgentsGraph') as mock_graph:
            mock_instance = MagicMock()
            mock_graph.return_value = mock_instance
            mock_instance.propagate.return_value = (
                {"final_trade_decision": "BUY"},
                "BUY"
            )

            with patch('sys.argv', ['run_trading_agents.py', str(self.stock_list_file)]):
                run_trading_agents.main()

                # Verify both calls used today's date (as an ISO string), not
                # the dates from JSON
                today = datetime.date.today().isoformat()
                calls = mock_instance.propagate.call_args_list
                self.assertEqual(calls[0][0], ("AAPL", today))
                self.assertEqual(calls[1][0], ("MSFT", today))

    def test_use_dates_from_json_mode_uses_stock_dates(self):
        """Test that --use-dates-from-json mode uses each stock's 'date' field."""

        import run_trading_agents

        self.STOCKS = [
            {"ticker": "AAPL", "date": "2024-01-15"},
            {"ticker": "MSFT", "date": "2024-02-20"}
        ]
        self.setUp()  # Recreate the JSON file

        with patch('run_trading_agents.TradingAgentsGraph') as mock_graph:
            mock_instance = MagicMock()
            mock_graph.return_value = mock_instance
            mock_instance.propagate.return_value = (
                {"final_trade_decision": "BUY"},
                "BUY"
            )

            with patch('sys.argv', ['run_trading_agents.py', str(self.stock_list_file),
                                   '--use-dates-from-json']):
                run_trading_agents.main()

                # Verify calls used dates from JSON
                calls = mock_instance.propagate.call_args_list
                self.assertEqual(calls[0][0], ("AAPL", "2024-01-15"))
                self.assertEqual(calls[1][0], ("MSFT", "2024-02-20"))

    def test_use_dates_from_json_mode_requires_date_field(self):
        """Test that --use-dates-from-json mode fails if 'date' field is missing."""
        import io
        from contextlib import redirect_stderr, redirect_stdout

        import run_trading_agents

        # Stock without 'date' field
        self.STOCKS = [{"ticker": "AAPL"}]
        self.setUp()  # Recreate the JSON file

        output = io.StringIO()
        with (
            patch('sys.argv', ['run_trading_agents.py', str(self.stock_list_file),
                               '--use-dates-from-json']),
            redirect_stdout(output),
            redirect_stderr(output),
            self.assertRaises(SystemExit) as cm,
        ):
            run_trading_agents.main()

        # Should exit with code 1
        self.assertEqual(cm.exception.code, 1)

        # Verify error message indicates the problem
        output_str = output.getvalue()
        self.assertIn('Error', output_str)
        self.assertIn('date', output_str.lower())
        self.assertIn('use-dates-from-json', output_str.lower() or 'from_json' in output_str.lower())

    def test_default_mode_writes_consolidated_summary_with_serializable_date(self):
        """Test that the default-mode date survives a real json.dumps of the
        consolidated trading_summary.json.

        Regression coverage for issue #72: a prior fix resolved `run_date` as
        a bare `datetime.date` object, which `json.dumps` cannot serialize.
        That `TypeError` was silently swallowed by the broad `except
        Exception` around the consolidated-summary write, so
        `trading_summary.json` was never written and no test caught it
        (earlier tests only asserted on the mocked `propagate` call args, not
        on `all_structured_data`/the summary file). This test drives the real
        `json.dumps` call over `all_structured_data` populated in default
        mode and asserts the summary file is actually written, with a plain
        string date field.
        """
        import datetime

        import run_trading_agents

        self.STOCKS = [{"ticker": "AAPL"}]
        self.setUp()  # Recreate the JSON file with updated STOCKS

        with tempfile.TemporaryDirectory() as report_dir, \
             patch('run_trading_agents.TradingAgentsGraph') as mock_graph, \
             patch('run_trading_agents.save_report_to_disk') as mock_save_report:
            mock_instance = MagicMock()
            mock_graph.return_value = mock_instance
            mock_instance.propagate.return_value = (
                {"final_trade_decision": "BUY"},
                "BUY"
            )
            mock_save_report.return_value = (
                "report.md",
                {"rating": "BUY", "action": "BUY", "entry_price": 100.0, "stop_loss": 90.0},
            )

            with patch('sys.argv', ['run_trading_agents.py', str(self.stock_list_file),
                                   '--report-dir', report_dir]):
                run_trading_agents.main()

            summary_file = Path(report_dir) / "trading_summary.json"
            self.assertTrue(summary_file.exists(),
                             "trading_summary.json was not written — a non-serializable "
                             "date in all_structured_data would raise TypeError inside "
                             "json.dumps, silently swallowed by the surrounding except.")

            summary = json.loads(summary_file.read_text())
            resolved_date = summary["stocks"][0]["date"]
            self.assertIsInstance(resolved_date, str)
            self.assertEqual(resolved_date, datetime.date.today().isoformat())

    def test_use_dates_from_json_mode_validates_all_stocks_upfront(self):
        """Test that --use-dates-from-json mode validates all stocks before processing any."""
        import io
        from contextlib import redirect_stderr, redirect_stdout

        import run_trading_agents

        # First stock is valid, second is missing 'date'
        self.STOCKS = [
            {"ticker": "AAPL", "date": "2024-01-15"},
            {"ticker": "MSFT"}  # Missing date
        ]
        self.setUp()  # Recreate the JSON file

        with patch('run_trading_agents.TradingAgentsGraph') as mock_graph:
            mock_instance = MagicMock()
            mock_graph.return_value = mock_instance

            output = io.StringIO()
            with (
                patch('sys.argv', ['run_trading_agents.py', str(self.stock_list_file),
                                   '--use-dates-from-json']),
                redirect_stdout(output),
                redirect_stderr(output),
                self.assertRaises(SystemExit) as cm,
            ):
                run_trading_agents.main()

            # Should exit with code 1
            self.assertEqual(cm.exception.code, 1)

            # Verify propagate was never called (validation happened upfront)
            self.assertEqual(mock_instance.propagate.call_count, 0)


@pytest.mark.unit
class RunTradingAgentsHelpTests(_TempStockFileTestCase):
    """Test no-arguments help behavior (new in issue #124)."""

    STOCKS = [
        {"ticker": "AAPL", "date": "2024-01-15"},
        {"ticker": "MSFT", "date": "2024-01-15"},
    ]

    def test_no_arguments_prints_help_and_exits_0(self):
        """With no command-line arguments, script prints help to stdout and exits 0 (issue #124)."""
        import io
        from contextlib import redirect_stderr, redirect_stdout

        import run_trading_agents

        output = io.StringIO()
        with (
            patch('sys.argv', ['run_trading_agents.py']),
            redirect_stdout(output),
            redirect_stderr(output),
            self.assertRaises(SystemExit) as cm,
        ):
            run_trading_agents.main()

        # Should exit with code 0 (not 2, not 1)
        self.assertEqual(cm.exception.code, 0)

        # Should print help to stdout (not usage error)
        output_str = output.getvalue()
        # Help output should contain usage and description
        self.assertIn('usage:', output_str.lower())
        self.assertIn('positional arguments', output_str.lower())
        # "optional arguments" in older argparse, "options:" in newer versions
        self.assertTrue('optional arguments' in output_str.lower() or 'options:' in output_str.lower())


@pytest.mark.unit
class RunTradingAgentsConfigFileTests(_TempStockFileTestCase):
    """Test run config file functionality."""

    STOCKS = [
        {"ticker": "AAPL", "date": "2024-01-15"},
        {"ticker": "MSFT", "date": "2024-01-15"},
    ]

    def test_config_file_plus_string_flag_rejected(self):
        """Config file + string flag (e.g. --llm-provider) is rejected (issue #124)."""
        import io
        from contextlib import redirect_stderr, redirect_stdout

        import run_trading_agents

        config_file = Path(self.temp_dir.name) / "config.json"
        config_file.write_text(json.dumps({
            "stocks_file": str(self.stock_list_file),
            "llm_provider": "ollama",
        }))

        # Config file + string flag should be rejected before doing work
        output = io.StringIO()
        with (
            patch('sys.argv', ['run_trading_agents.py', str(config_file),
                               '--llm-provider', 'mistral']),
            redirect_stdout(output),
            redirect_stderr(output),
            self.assertRaises(SystemExit) as cm,
        ):
            run_trading_agents.main()

        # Should exit with code 1
        self.assertEqual(cm.exception.code, 1)

        # Verify error message names the flag
        output_str = output.getvalue()
        self.assertIn('Error', output_str)
        self.assertIn('--llm-provider', output_str)

    def test_config_file_basic_loading(self):
        """Test that a basic config file is loaded and applied correctly."""
        import run_trading_agents

        # Create a config file
        config_file = Path(self.temp_dir.name) / "config.json"
        config_file.write_text(json.dumps({
            "stocks_file": str(self.stock_list_file),
            "llm_provider": "mistral",
            "deep_think_llm": "mistral-large",
            "quick_think_llm": "mistral-small",
            "show_summary": True,
        }))

        with patch('run_trading_agents.TradingAgentsGraph') as mock_graph, \
             patch.dict(os.environ, {'MISTRAL_API_KEY': 'test-key'}):

            mock_instance = MagicMock()
            mock_graph.return_value = mock_instance
            mock_instance.propagate.return_value = (
                {"final_trade_decision": "BUY"},
                "BUY"
            )

            with patch('sys.argv', ['run_trading_agents.py', str(config_file)]):
                run_trading_agents.main()

                # Verify config was applied
                call_args = mock_graph.call_args
                config = call_args[1]['config']
                self.assertEqual(config["llm_provider"], "mistral")
                self.assertEqual(config["deep_think_llm"], "mistral-large")
                self.assertEqual(config["quick_think_llm"], "mistral-small")

    def test_config_file_cli_mixing_rejected(self):
        """Config file + CLI flags is rejected (strict exclusive mode per issue #124)."""
        import io
        from contextlib import redirect_stderr, redirect_stdout

        import run_trading_agents

        # Create a config file
        config_file = Path(self.temp_dir.name) / "config.json"
        config_file.write_text(json.dumps({
            "stocks_file": str(self.stock_list_file),
            "llm_provider": "mistral",
            "deep_think_llm": "mistral-large",
            "quick_think_llm": "mistral-small",
        }))

        # Config file + CLI flags should be rejected before doing any work
        output = io.StringIO()
        with (
            patch('sys.argv', ['run_trading_agents.py', str(config_file),
                               '--llm-provider', 'openai',
                               '--deep-think-llm', 'gpt-4-turbo',
                               '--quick-think-llm', 'gpt-4o']),
            redirect_stdout(output),
            redirect_stderr(output),
            self.assertRaises(SystemExit) as cm,
        ):
            run_trading_agents.main()

        # Should exit with code 1
        self.assertEqual(cm.exception.code, 1)

        # Verify error message names the offending flags
        output_str = output.getvalue()
        self.assertIn('Error', output_str)
        self.assertIn('--llm-provider', output_str)
        self.assertIn('--deep-think-llm', output_str)
        self.assertIn('--quick-think-llm', output_str)
        self.assertIn('exclusive', output_str.lower())

    def test_config_file_relative_stocks_file_path(self):
        """Test that relative stocks_file paths resolve against config directory."""
        import run_trading_agents

        # Create config file in temp dir, stocks file relative to it
        config_file = Path(self.temp_dir.name) / "config.json"
        stocks_file_name = "my_stocks.json"
        stocks_file = Path(self.temp_dir.name) / stocks_file_name
        stocks_file.write_text(json.dumps(self.STOCKS))

        config_file.write_text(json.dumps({
            "stocks_file": stocks_file_name,  # Relative path
            "llm_provider": "ollama",
        }))

        with patch('run_trading_agents.TradingAgentsGraph') as mock_graph:
            mock_instance = MagicMock()
            mock_graph.return_value = mock_instance
            mock_instance.propagate.return_value = (
                {"final_trade_decision": "BUY"},
                "BUY"
            )

            with patch('sys.argv', ['run_trading_agents.py', str(config_file)]):
                run_trading_agents.main()

                # Verify propagate was called twice (for both stocks)
                self.assertEqual(mock_instance.propagate.call_count, 2)

    def test_config_file_unrecognized_key_error(self):
        """Test that unrecognized config keys cause an error."""
        import io
        from contextlib import redirect_stderr, redirect_stdout

        import run_trading_agents

        # Create a config file with unrecognized key
        config_file = Path(self.temp_dir.name) / "config.json"
        config_file.write_text(json.dumps({
            "stocks_file": str(self.stock_list_file),
            "llm_provider": "ollama",
            "invalid_key": "some_value",
        }))

        output = io.StringIO()
        with (
            patch('sys.argv', ['run_trading_agents.py', str(config_file)]),
            redirect_stdout(output),
            redirect_stderr(output),
            self.assertRaises(SystemExit) as cm,
        ):
            run_trading_agents.main()

        # Should exit with code 1
        self.assertEqual(cm.exception.code, 1)

        # Verify error message names the unrecognized key
        output_str = output.getvalue()
        self.assertIn('Unrecognized', output_str)
        self.assertIn('invalid_key', output_str)
        self.assertIn('Recognized keys', output_str)

    def test_config_file_missing_stocks_file_key(self):
        """Test that missing stocks_file key causes an error."""
        import io
        from contextlib import redirect_stderr, redirect_stdout

        import run_trading_agents

        # Create a config file without stocks_file
        config_file = Path(self.temp_dir.name) / "config.json"
        config_file.write_text(json.dumps({
            "llm_provider": "ollama",
        }))

        output = io.StringIO()
        with (
            patch('sys.argv', ['run_trading_agents.py', str(config_file)]),
            redirect_stdout(output),
            redirect_stderr(output),
            self.assertRaises(SystemExit) as cm,
        ):
            run_trading_agents.main()

        # Should exit with code 1
        self.assertEqual(cm.exception.code, 1)

        # Verify error message
        output_str = output.getvalue()
        self.assertIn('stocks_file', output_str)

    def test_config_file_missing_stocks_file_error(self):
        """Test that missing stocks_file itself causes an error."""
        import io
        from contextlib import redirect_stderr, redirect_stdout

        import run_trading_agents

        # Create a config file pointing to non-existent stocks file
        config_file = Path(self.temp_dir.name) / "config.json"
        config_file.write_text(json.dumps({
            "stocks_file": "/nonexistent/stocks.json",
            "llm_provider": "ollama",
        }))

        output = io.StringIO()
        with (
            patch('sys.argv', ['run_trading_agents.py', str(config_file)]),
            redirect_stdout(output),
            redirect_stderr(output),
            self.assertRaises(SystemExit) as cm,
        ):
            run_trading_agents.main()

        # Should exit with code 1
        self.assertEqual(cm.exception.code, 1)

        # Verify error message
        output_str = output.getvalue()
        self.assertIn('stocks_file', output_str)
        self.assertIn('not found', output_str)

    def test_config_file_with_portfolio_mode(self):
        """Test config file with portfolio mode enabled."""
        import run_trading_agents

        # Create a config file with portfolio mode
        config_file = Path(self.temp_dir.name) / "config.json"
        config_file.write_text(json.dumps({
            "stocks_file": str(self.stock_list_file),
            "llm_provider": "ollama",
            "portfolio": True,
            "style": "aggressive",
            "depot_id": "test-depot",
        }))

        with patch('run_trading_agents.TradingAgentsGraph') as mock_graph, \
             patch('run_trading_agents.run_portfolio_mode') as mock_portfolio:

            mock_instance = MagicMock()
            mock_graph.return_value = mock_instance
            mock_instance.propagate.return_value = (
                {"final_trade_decision": "BUY"},
                "BUY"
            )
            mock_portfolio.return_value = (
                {"summary": "Portfolio rebalancing summary"},
                "portfolio_report.json"
            )

            with patch('sys.argv', ['run_trading_agents.py', str(config_file)]):
                run_trading_agents.main()

                # Verify portfolio mode was called
                self.assertTrue(mock_portfolio.called)
                call_args = mock_portfolio.call_args
                self.assertEqual(call_args[1]['style'], 'aggressive')
                self.assertEqual(call_args[1]['depot_id'], 'test-depot')

    def test_config_file_with_memory_id(self):
        """Test config file with memory_id specified."""
        import run_trading_agents

        # Create a config file with memory_id
        config_file = Path(self.temp_dir.name) / "config.json"
        config_file.write_text(json.dumps({
            "stocks_file": str(self.stock_list_file),
            "llm_provider": "ollama",
            "memory_id": "test-memory-1",
        }))

        with patch('run_trading_agents.TradingAgentsGraph') as mock_graph:
            mock_instance = MagicMock()
            mock_graph.return_value = mock_instance
            mock_instance.propagate.return_value = (
                {"final_trade_decision": "BUY"},
                "BUY"
            )

            with patch('sys.argv', ['run_trading_agents.py', str(config_file)]):
                run_trading_agents.main()

                # Verify memory_id was applied to config
                call_args = mock_graph.call_args
                config = call_args[1]['config']
                self.assertEqual(config["memory_id"], "test-memory-1")

    def test_config_file_llm_provider_absent_falls_back_to_ollama_default(self):
        """Precedence 'neither' case for the string key llm_provider.

        Neither the CLI flag nor the config file sets llm_provider, and no
        TRADINGAGENTS_LLM_PROVIDER env var is set in the test environment, so
        the resolved config must fall back to DEFAULT_CONFIG's built-in
        "ollama" default.
        """
        import run_trading_agents

        config_file = Path(self.temp_dir.name) / "config.json"
        config_file.write_text(json.dumps({
            "stocks_file": str(self.stock_list_file),
            # llm_provider intentionally omitted
        }))

        with (
            patch('run_trading_agents.TradingAgentsGraph') as mock_graph,
            patch.dict(os.environ, {}, clear=False),
        ):
            os.environ.pop('TRADINGAGENTS_LLM_PROVIDER', None)
            mock_instance = MagicMock()
            mock_graph.return_value = mock_instance
            mock_instance.propagate.return_value = (
                {"final_trade_decision": "BUY"},
                "BUY"
            )

            with patch('sys.argv', ['run_trading_agents.py', str(config_file)]):
                run_trading_agents.main()

                call_args = mock_graph.call_args
                config = call_args[1]['config']
                self.assertEqual(config["llm_provider"], "ollama")

    def test_config_file_store_true_flag_mixing_rejected(self):
        """Config file + store_true flag (e.g. --show-summary) is rejected (issue #124)."""
        import io
        from contextlib import redirect_stderr, redirect_stdout

        import run_trading_agents

        config_file = Path(self.temp_dir.name) / "config.json"
        config_file.write_text(json.dumps({
            "stocks_file": str(self.stock_list_file),
            "show_summary": False,
        }))

        # Config file + store_true flag should be rejected before doing work
        output = io.StringIO()
        with (
            patch('sys.argv', ['run_trading_agents.py', str(config_file), '--show-summary']),
            redirect_stdout(output),
            redirect_stderr(output),
            self.assertRaises(SystemExit) as cm,
        ):
            run_trading_agents.main()

        # Should exit with code 1
        self.assertEqual(cm.exception.code, 1)

        # Verify error message names the flag
        output_str = output.getvalue()
        self.assertIn('Error', output_str)
        self.assertIn('--show-summary', output_str)

    def test_config_file_boolean_precedence_config_wins_when_flag_absent(self):
        """store_true key precedence: config file value wins when the flag is absent."""
        import run_trading_agents

        config_file = Path(self.temp_dir.name) / "config.json"
        config_file.write_text(json.dumps({
            "stocks_file": str(self.stock_list_file),
            "show_summary": True,
        }))

        with (
            patch('run_trading_agents.TradingAgentsGraph') as mock_graph,
            patch('run_trading_agents.display_summary') as mock_display,
        ):
            mock_instance = MagicMock()
            mock_graph.return_value = mock_instance
            mock_instance.propagate.return_value = (
                {"final_trade_decision": "BUY"},
                "BUY"
            )

            with patch('sys.argv', ['run_trading_agents.py', str(config_file)]):
                run_trading_agents.main()

                self.assertTrue(mock_display.called)

    def test_config_file_boolean_precedence_default_when_neither(self):
        """store_true key precedence: falls back to False when neither flag nor config set it.

        This is the subtle case flagged by design review: with
        action='store_true', default=None, "flag absent" must be
        distinguishable from "flag given" so the config-file value isn't
        clobbered by a stray truthy default.
        """
        import run_trading_agents

        config_file = Path(self.temp_dir.name) / "config.json"
        config_file.write_text(json.dumps({
            "stocks_file": str(self.stock_list_file),
            # show_summary intentionally omitted
        }))

        with (
            patch('run_trading_agents.TradingAgentsGraph') as mock_graph,
            patch('run_trading_agents.display_summary') as mock_display,
        ):
            mock_instance = MagicMock()
            mock_graph.return_value = mock_instance
            mock_instance.propagate.return_value = (
                {"final_trade_decision": "BUY"},
                "BUY"
            )

            with patch('sys.argv', ['run_trading_agents.py', str(config_file)]):
                run_trading_agents.main()

                self.assertFalse(mock_display.called)

    def test_config_file_stocks_file_not_array_object(self):
        """A stocks_file that parses but is a JSON object (not an array) is rejected."""
        import io
        from contextlib import redirect_stderr, redirect_stdout

        import run_trading_agents

        bad_stocks_file = Path(self.temp_dir.name) / "bad_stocks.json"
        bad_stocks_file.write_text(json.dumps({"ticker": "AAPL"}))

        config_file = Path(self.temp_dir.name) / "config.json"
        config_file.write_text(json.dumps({"stocks_file": str(bad_stocks_file)}))

        output = io.StringIO()
        with (
            patch('sys.argv', ['run_trading_agents.py', str(config_file)]),
            redirect_stdout(output),
            redirect_stderr(output),
            self.assertRaises(SystemExit) as cm,
        ):
            run_trading_agents.main()

        self.assertEqual(cm.exception.code, 1)
        self.assertIn('array', output.getvalue().lower())

    def test_config_file_stocks_file_not_array_scalar(self):
        """A stocks_file that parses but is a scalar (not an array) is rejected."""
        import io
        from contextlib import redirect_stderr, redirect_stdout

        import run_trading_agents

        bad_stocks_file = Path(self.temp_dir.name) / "bad_stocks.json"
        bad_stocks_file.write_text(json.dumps("not a list"))

        config_file = Path(self.temp_dir.name) / "config.json"
        config_file.write_text(json.dumps({"stocks_file": str(bad_stocks_file)}))

        output = io.StringIO()
        with (
            patch('sys.argv', ['run_trading_agents.py', str(config_file)]),
            redirect_stdout(output),
            redirect_stderr(output),
            self.assertRaises(SystemExit) as cm,
        ):
            run_trading_agents.main()

        self.assertEqual(cm.exception.code, 1)
        self.assertIn('array', output.getvalue().lower())

    def test_config_file_stocks_file_malformed_json(self):
        """A stocks_file containing malformed JSON is rejected with a clear error."""
        import io
        from contextlib import redirect_stderr, redirect_stdout

        import run_trading_agents

        bad_stocks_file = Path(self.temp_dir.name) / "bad_stocks.json"
        bad_stocks_file.write_text("{not valid json")

        config_file = Path(self.temp_dir.name) / "config.json"
        config_file.write_text(json.dumps({"stocks_file": str(bad_stocks_file)}))

        output = io.StringIO()
        with (
            patch('sys.argv', ['run_trading_agents.py', str(config_file)]),
            redirect_stdout(output),
            redirect_stderr(output),
            self.assertRaises(SystemExit) as cm,
        ):
            run_trading_agents.main()

        self.assertEqual(cm.exception.code, 1)
        output_str = output.getvalue()
        self.assertIn('Invalid JSON', output_str)
        self.assertIn('stocks_file', output_str)

    def test_config_file_invalid_style_value_rejected(self):
        """An invalid style value coming from the config file is rejected.

        argparse's `choices=` only validates --style on the command line, so
        this path (config-file-supplied style) needs its own check.
        """
        import io
        from contextlib import redirect_stderr, redirect_stdout

        import run_trading_agents

        config_file = Path(self.temp_dir.name) / "config.json"
        config_file.write_text(json.dumps({
            "stocks_file": str(self.stock_list_file),
            "portfolio": True,
            "style": "moderate",
            "depot_id": "test-depot",
        }))

        output = io.StringIO()
        with (
            patch('sys.argv', ['run_trading_agents.py', str(config_file)]),
            redirect_stdout(output),
            redirect_stderr(output),
            self.assertRaises(SystemExit) as cm,
        ):
            run_trading_agents.main()

        self.assertEqual(cm.exception.code, 1)
        output_str = output.getvalue()
        self.assertIn('style', output_str.lower())
        self.assertIn('moderate', output_str)

    def test_config_file_nested_config_block_basic_loading(self):
        """A nested 'config' block can set DEFAULT_CONFIG keys (issue #117)."""
        import run_trading_agents

        config_file = Path(self.temp_dir.name) / "config.json"
        config_file.write_text(json.dumps({
            "stocks_file": str(self.stock_list_file),
            "config": {
                "research_stage": "debate",
                "max_debate_rounds": 2,
                "temperature": 0.3,
            },
        }))

        with patch('run_trading_agents.TradingAgentsGraph') as mock_graph:
            mock_instance = MagicMock()
            mock_graph.return_value = mock_instance
            mock_instance.propagate.return_value = (
                {"final_trade_decision": "BUY"},
                "BUY"
            )

            with patch('sys.argv', ['run_trading_agents.py', str(config_file)]):
                run_trading_agents.main()

                call_args = mock_graph.call_args
                config = call_args[1]['config']
                self.assertEqual(config["research_stage"], "debate")
                self.assertEqual(config["max_debate_rounds"], 2)
                self.assertEqual(config["temperature"], 0.3)

    def test_config_file_nested_config_block_with_nested_keys(self):
        """Config block can set nested DEFAULT_CONFIG keys like data_vendors.news_data."""
        import run_trading_agents

        config_file = Path(self.temp_dir.name) / "config.json"
        config_file.write_text(json.dumps({
            "stocks_file": str(self.stock_list_file),
            "config": {
                "data_vendors": {
                    "news_data": "alpha_vantage",
                },
            },
        }))

        with patch('run_trading_agents.TradingAgentsGraph') as mock_graph:
            mock_instance = MagicMock()
            mock_graph.return_value = mock_instance
            mock_instance.propagate.return_value = (
                {"final_trade_decision": "BUY"},
                "BUY"
            )

            with patch('sys.argv', ['run_trading_agents.py', str(config_file)]):
                run_trading_agents.main()

                call_args = mock_graph.call_args
                config = call_args[1]['config']
                self.assertEqual(config["data_vendors"]["news_data"], "alpha_vantage")

    def test_config_file_nested_config_block_deep_merge(self):
        """Config block's nested dict values deep-merge one level (sibling keys intact)."""
        import run_trading_agents

        config_file = Path(self.temp_dir.name) / "config.json"
        config_file.write_text(json.dumps({
            "stocks_file": str(self.stock_list_file),
            "config": {
                "data_vendors": {
                    "news_data": "alpha_vantage",
                },
            },
        }))

        with patch('run_trading_agents.TradingAgentsGraph') as mock_graph:
            mock_instance = MagicMock()
            mock_graph.return_value = mock_instance
            mock_instance.propagate.return_value = (
                {"final_trade_decision": "BUY"},
                "BUY"
            )

            with patch('sys.argv', ['run_trading_agents.py', str(config_file)]):
                run_trading_agents.main()

                call_args = mock_graph.call_args
                config = call_args[1]['config']
                # Verify that news_data was set via config block
                self.assertEqual(config["data_vendors"]["news_data"], "alpha_vantage")
                # Verify other data_vendors keys are still present (deep merge, not replace)
                self.assertIn("core_stock_apis", config["data_vendors"])
                self.assertIn("fundamental_data", config["data_vendors"])
                self.assertIn("technical_indicators", config["data_vendors"])
                self.assertIn("knowledge_base", config["data_vendors"])

    def test_config_block_does_not_mutate_global_default_config(self):
        """Applying a 'config' block must leave the process-global DEFAULT_CONFIG untouched.

        Regression test: ``DEFAULT_CONFIG.copy()`` is a *shallow* copy, so
        ``config["data_vendors"]`` IS ``DEFAULT_CONFIG["data_vendors"]`` and
        the one-level deep merge's ``config[key].update(value)`` used to
        rewrite the global's own nested dict in place — permanently
        corrupting DEFAULT_CONFIG for every later consumer in the process
        (e.g. ``initialize_config()`` in tradingagents/dataflows/config.py,
        which deepcopy()s it). The fix is to deepcopy up front, matching
        ``initialize_config``'s precondition for the same merge.
        """
        import copy as copy_module

        import run_trading_agents

        default_config = run_trading_agents.DEFAULT_CONFIG
        before = copy_module.deepcopy(default_config)
        nested_ids_before = {
            key: id(value) for key, value in default_config.items() if isinstance(value, dict)
        }

        config_file = Path(self.temp_dir.name) / "config.json"
        config_file.write_text(json.dumps({
            "stocks_file": str(self.stock_list_file),
            "config": {
                "data_vendors": {"news_data": "alpha_vantage"},
                "tool_vendors": {"get_stock_data": "alpha_vantage"},
                "research_stage": "debate",
            },
        }))

        with patch('run_trading_agents.TradingAgentsGraph') as mock_graph:
            mock_instance = MagicMock()
            mock_graph.return_value = mock_instance
            mock_instance.propagate.return_value = (
                {"final_trade_decision": "BUY"},
                "BUY"
            )

            with patch('sys.argv', ['run_trading_agents.py', str(config_file)]):
                run_trading_agents.main()

            config = mock_graph.call_args[1]['config']

        # The run's own config really did receive the overrides ...
        self.assertEqual(config["data_vendors"]["news_data"], "alpha_vantage")
        self.assertEqual(config["tool_vendors"]["get_stock_data"], "alpha_vantage")
        self.assertEqual(config["research_stage"], "debate")

        # ... while the global DEFAULT_CONFIG is unchanged, value-wise ...
        self.assertEqual(
            default_config["data_vendors"]["news_data"],
            before["data_vendors"]["news_data"],
        )
        self.assertEqual(default_config["tool_vendors"], before["tool_vendors"])
        self.assertEqual(default_config["research_stage"], before["research_stage"])
        self.assertEqual(default_config, before)

        # ... and the run config shares no nested dict object with it, so a
        # later in-place merge can't reach the global either.
        self.assertIsNot(config["data_vendors"], default_config["data_vendors"])
        self.assertIsNot(config["tool_vendors"], default_config["tool_vendors"])
        self.assertEqual(
            {key: id(value) for key, value in default_config.items() if isinstance(value, dict)},
            nested_ids_before,
        )

    def test_config_block_dot_notation_key_rejected(self):
        """Dot-notation keys are rejected, pointing at the nested-object form."""
        import io
        from contextlib import redirect_stderr, redirect_stdout

        import run_trading_agents

        config_file = Path(self.temp_dir.name) / "config.json"
        config_file.write_text(json.dumps({
            "stocks_file": str(self.stock_list_file),
            "config": {
                "data_vendors.news_data": "alpha_vantage",
            },
        }))

        output = io.StringIO()
        with (
            patch('sys.argv', ['run_trading_agents.py', str(config_file)]),
            redirect_stdout(output),
            redirect_stderr(output),
            self.assertRaises(SystemExit) as cm,
        ):
            run_trading_agents.main()

        self.assertEqual(cm.exception.code, 1)
        output_str = output.getvalue()
        self.assertIn('data_vendors.news_data', output_str)
        self.assertIn('dot notation', output_str)
        # The error names the supported replacement spelling.
        self.assertIn('"data_vendors": {"news_data": ...}', output_str)

    def test_config_file_nested_config_block_invalid_key(self):
        """Config block with unrecognized key exits with clear error."""
        import io
        from contextlib import redirect_stderr, redirect_stdout

        import run_trading_agents

        config_file = Path(self.temp_dir.name) / "config.json"
        config_file.write_text(json.dumps({
            "stocks_file": str(self.stock_list_file),
            "config": {
                "invalid_config_key": "some_value",
            },
        }))

        output = io.StringIO()
        with (
            patch('sys.argv', ['run_trading_agents.py', str(config_file)]),
            redirect_stdout(output),
            redirect_stderr(output),
            self.assertRaises(SystemExit) as cm,
        ):
            run_trading_agents.main()

        self.assertEqual(cm.exception.code, 1)
        output_str = output.getvalue()
        self.assertIn('Unrecognized', output_str)
        self.assertIn('invalid_config_key', output_str)

    def test_config_file_nested_config_block_type_mismatch(self):
        """Config block with wrong type for a value exits with clear error."""
        import io
        from contextlib import redirect_stderr, redirect_stdout

        import run_trading_agents

        config_file = Path(self.temp_dir.name) / "config.json"
        config_file.write_text(json.dumps({
            "stocks_file": str(self.stock_list_file),
            "config": {
                "max_debate_rounds": "not_a_number",  # Should be int, not string
            },
        }))

        output = io.StringIO()
        with (
            patch('sys.argv', ['run_trading_agents.py', str(config_file)]),
            redirect_stdout(output),
            redirect_stderr(output),
            self.assertRaises(SystemExit) as cm,
        ):
            run_trading_agents.main()

        self.assertEqual(cm.exception.code, 1)
        output_str = output.getvalue()
        self.assertIn('type', output_str.lower())
        self.assertIn('max_debate_rounds', output_str)

    def test_config_file_nested_config_block_bool_rejected_for_int_default(self):
        """A JSON bool is rejected for an int-default key, despite bool being an int subclass.

        max_debate_rounds defaults to int 1. A naive isinstance(value, type(reference))
        type check would wrongly accept True/False here since bool is a subclass of int
        in Python; this must be rejected instead.
        """
        import io
        from contextlib import redirect_stderr, redirect_stdout

        import run_trading_agents

        config_file = Path(self.temp_dir.name) / "config.json"
        config_file.write_text(json.dumps({
            "stocks_file": str(self.stock_list_file),
            "config": {
                "max_debate_rounds": True,
            },
        }))

        output = io.StringIO()
        with (
            patch('sys.argv', ['run_trading_agents.py', str(config_file)]),
            redirect_stdout(output),
            redirect_stderr(output),
            self.assertRaises(SystemExit) as cm,
        ):
            run_trading_agents.main()

        self.assertEqual(cm.exception.code, 1)
        output_str = output.getvalue()
        self.assertIn('type', output_str.lower())
        self.assertIn('max_debate_rounds', output_str)

    def test_config_file_nested_config_block_int_rejected_for_bool_default(self):
        """A JSON int is rejected for a bool-default key, despite bool being an int subclass.

        swing_trader_enabled defaults to bool False. isinstance(1, int) is True, so a naive
        type check keyed off isinstance(reference, int) would wrongly accept 1/0 here; this
        must be rejected instead.
        """
        import io
        from contextlib import redirect_stderr, redirect_stdout

        import run_trading_agents

        config_file = Path(self.temp_dir.name) / "config.json"
        config_file.write_text(json.dumps({
            "stocks_file": str(self.stock_list_file),
            "config": {
                "swing_trader_enabled": 1,
            },
        }))

        output = io.StringIO()
        with (
            patch('sys.argv', ['run_trading_agents.py', str(config_file)]),
            redirect_stdout(output),
            redirect_stderr(output),
            self.assertRaises(SystemExit) as cm,
        ):
            run_trading_agents.main()

        self.assertEqual(cm.exception.code, 1)
        output_str = output.getvalue()
        self.assertIn('type', output_str.lower())
        self.assertIn('swing_trader_enabled', output_str)

    def test_config_file_nested_config_block_bool_accepted_for_bool_default(self):
        """An actual JSON bool is accepted for a bool-default key (sanity check)."""
        import run_trading_agents

        config_file = Path(self.temp_dir.name) / "config.json"
        config_file.write_text(json.dumps({
            "stocks_file": str(self.stock_list_file),
            "config": {
                "swing_trader_enabled": True,
            },
        }))

        with patch('run_trading_agents.TradingAgentsGraph') as mock_graph:
            mock_instance = MagicMock()
            mock_graph.return_value = mock_instance
            mock_instance.propagate.return_value = (
                {"final_trade_decision": "BUY"},
                "BUY"
            )

            with patch('sys.argv', ['run_trading_agents.py', str(config_file)]):
                run_trading_agents.main()

                call_args = mock_graph.call_args
                config = call_args[1]['config']
                self.assertIs(config["swing_trader_enabled"], True)

    def test_config_file_nested_config_block_int_accepted_for_float_default(self):
        """A JSON int is accepted for a float-default key.

        swing_trader_min_risk_reward defaults to float 1.5 (unlike temperature, whose own
        DEFAULT_CONFIG default is None — see the none_default_key test below — this key has
        a genuine float default, so it actually exercises the type-widening rule). This is
        the one explicit type-widening the issue requires: JSON has no separate whole-number-
        float literal, so an int must be accepted where DEFAULT_CONFIG's default is a float.
        """
        import run_trading_agents

        config_file = Path(self.temp_dir.name) / "config.json"
        config_file.write_text(json.dumps({
            "stocks_file": str(self.stock_list_file),
            "config": {
                "swing_trader_min_risk_reward": 2,
            },
        }))

        with patch('run_trading_agents.TradingAgentsGraph') as mock_graph:
            mock_instance = MagicMock()
            mock_graph.return_value = mock_instance
            mock_instance.propagate.return_value = (
                {"final_trade_decision": "BUY"},
                "BUY"
            )

            with patch('sys.argv', ['run_trading_agents.py', str(config_file)]):
                run_trading_agents.main()

                call_args = mock_graph.call_args
                config = call_args[1]['config']
                self.assertEqual(config["swing_trader_min_risk_reward"], 2)

    def test_config_file_nested_config_block_none_default_key_accepts_any_type(self):
        """A config-block key whose DEFAULT_CONFIG default is None skips the type check.

        benchmark_ticker defaults to None, so there is no type to check the supplied value
        against; the value is accepted and passed through untouched regardless of its JSON
        type (matching the TRADINGAGENTS_* env var path's behavior for these same keys,
        which has nothing to coerce against either). This is a deliberate design choice,
        not an oversight — see _validate_config_block's docstring.
        """
        import run_trading_agents

        config_file = Path(self.temp_dir.name) / "config.json"
        config_file.write_text(json.dumps({
            "stocks_file": str(self.stock_list_file),
            "config": {
                # benchmark_ticker is normally a string, but its DEFAULT_CONFIG
                # default is None, so an int must still be accepted here.
                "benchmark_ticker": 42,
            },
        }))

        with patch('run_trading_agents.TradingAgentsGraph') as mock_graph:
            mock_instance = MagicMock()
            mock_graph.return_value = mock_instance
            mock_instance.propagate.return_value = (
                {"final_trade_decision": "BUY"},
                "BUY"
            )

            with patch('sys.argv', ['run_trading_agents.py', str(config_file)]):
                run_trading_agents.main()

                call_args = mock_graph.call_args
                config = call_args[1]['config']
                self.assertEqual(config["benchmark_ticker"], 42)

    def test_config_file_top_level_key_beats_nested_config_block(self):
        """Top-level key (tier 2) beats config block key (tier 3) when both present."""
        import run_trading_agents

        config_file = Path(self.temp_dir.name) / "config.json"
        config_file.write_text(json.dumps({
            "stocks_file": str(self.stock_list_file),
            "memory_id": "top-level-id",  # Top-level
            "config": {
                "memory_id": "config-block-id",  # Nested (loses)
            },
        }))

        with patch('run_trading_agents.TradingAgentsGraph') as mock_graph:
            mock_instance = MagicMock()
            mock_graph.return_value = mock_instance
            mock_instance.propagate.return_value = (
                {"final_trade_decision": "BUY"},
                "BUY"
            )

            with patch('sys.argv', ['run_trading_agents.py', str(config_file)]):
                run_trading_agents.main()

                call_args = mock_graph.call_args
                config = call_args[1]['config']
                self.assertEqual(config["memory_id"], "top-level-id")

    def test_config_file_cli_flag_beats_nested_config_block(self):
        """Config file + actual CLI flag (not top-level key) is rejected (issue #124).

        This test verifies that a CLI flag like --memory-id cannot be mixed with a config file.
        The test name is preserved from the original for backwards compatibility with test discovery.
        """
        import io
        from contextlib import redirect_stderr, redirect_stdout

        import run_trading_agents

        config_file = Path(self.temp_dir.name) / "config.json"
        config_file.write_text(json.dumps({
            "stocks_file": str(self.stock_list_file),
            "config": {
                "temperature": 0.3,
            },
        }))

        # Config file + CLI flag --memory-id should be rejected
        output = io.StringIO()
        with (
            patch('sys.argv', ['run_trading_agents.py', str(config_file),
                               '--memory-id', 'cli-memory']),
            redirect_stdout(output),
            redirect_stderr(output),
            self.assertRaises(SystemExit) as cm,
        ):
            run_trading_agents.main()

        # Should exit with code 1
        self.assertEqual(cm.exception.code, 1)

        # Verify error message names the flag
        output_str = output.getvalue()
        self.assertIn('Error', output_str)
        self.assertIn('--memory-id', output_str)

    def test_config_file_cli_flag_beats_all_tiers(self):
        """Config file + multiple CLI flags is rejected, message names all flags (issue #124).

        This test verifies that multiple CLI flags cannot be mixed with a config file,
        and the error message names all offending flags.
        """
        import io
        from contextlib import redirect_stderr, redirect_stdout

        import run_trading_agents

        config_file = Path(self.temp_dir.name) / "config.json"
        config_file.write_text(json.dumps({
            "stocks_file": str(self.stock_list_file),
            "config": {
                "temperature": 0.3,
            },
        }))

        # Config file + multiple CLI flags should be rejected
        output = io.StringIO()
        with (
            patch('sys.argv', ['run_trading_agents.py', str(config_file),
                               '--memory-id', 'cli-memory',
                               '--show-summary']),
            redirect_stdout(output),
            redirect_stderr(output),
            self.assertRaises(SystemExit) as cm,
        ):
            run_trading_agents.main()

        # Should exit with code 1
        self.assertEqual(cm.exception.code, 1)

        # Verify error message names both flags
        output_str = output.getvalue()
        self.assertIn('Error', output_str)
        self.assertIn('--memory-id', output_str)
        self.assertIn('--show-summary', output_str)

    def test_config_file_alone_still_works(self):
        """Regression: config file alone (no CLI flags) still works unchanged (issue #124)."""
        import run_trading_agents

        config_file = Path(self.temp_dir.name) / "config.json"
        config_file.write_text(json.dumps({
            "stocks_file": str(self.stock_list_file),
            "llm_provider": "ollama",
        }))

        with patch('run_trading_agents.TradingAgentsGraph') as mock_graph:
            mock_instance = MagicMock()
            mock_graph.return_value = mock_instance
            mock_instance.propagate.return_value = (
                {"final_trade_decision": "BUY"},
                "BUY"
            )

            with patch('sys.argv', ['run_trading_agents.py', str(config_file)]):
                run_trading_agents.main()

                # Verify propagate was called twice (for both stocks in config file)
                self.assertEqual(mock_instance.propagate.call_count, 2)

    def test_stock_list_with_cli_flags_still_works(self):
        """Regression: stock-list array + CLI flags still works (stock-list mode unchanged)."""
        import run_trading_agents

        with patch('run_trading_agents.TradingAgentsGraph') as mock_graph:
            mock_instance = MagicMock()
            mock_graph.return_value = mock_instance
            mock_instance.propagate.return_value = (
                {"final_trade_decision": "BUY"},
                "BUY"
            )

            # Stock list (array) + CLI flags should work fine (stock-list mode is unchanged)
            with patch('sys.argv', ['run_trading_agents.py', str(self.stock_list_file),
                                   '--show-summary', '--report-dir', './my-reports']):
                run_trading_agents.main()

                # Verify propagate was called and flags were respected
                self.assertEqual(mock_instance.propagate.call_count, 2)

@pytest.mark.unit
class RunTradingAgentsTickersAndStocksFileTests(_TempStockFileTestCase):
    """Test new --tickers and --stocks-file flags (issue #125)."""

    STOCKS = [
        {"ticker": "AAPL", "date": "2024-01-15"},
        {"ticker": "MSFT", "date": "2024-01-15"},
    ]

    def test_tickers_flag_basic(self):
        """Test --tickers with single and multiple tickers."""
        import datetime

        import run_trading_agents

        with patch('run_trading_agents.TradingAgentsGraph') as mock_graph:
            mock_instance = MagicMock()
            mock_graph.return_value = mock_instance
            mock_instance.propagate.return_value = (
                {"final_trade_decision": "BUY"},
                "BUY"
            )

            # Test single ticker
            with patch('sys.argv', ['run_trading_agents.py', '--tickers', 'AAPL']):
                run_trading_agents.main()

                # Verify propagate was called once with today's date
                self.assertEqual(mock_instance.propagate.call_count, 1)
                call = mock_instance.propagate.call_args_list[0]
                self.assertEqual(call[0][0], "AAPL")
                self.assertEqual(call[0][1], datetime.date.today().isoformat())

            # Reset mock
            mock_instance.reset_mock()

            # Test multiple tickers
            with patch('sys.argv', ['run_trading_agents.py', '--tickers', 'AAPL,MSFT']):
                run_trading_agents.main()

                self.assertEqual(mock_instance.propagate.call_count, 2)
                calls = mock_instance.propagate.call_args_list
                today = datetime.date.today().isoformat()
                self.assertEqual(calls[0][0], ("AAPL", today))
                self.assertEqual(calls[1][0], ("MSFT", today))

    def test_tickers_flag_with_whitespace(self):
        """Test --tickers with surrounding whitespace per entry."""
        import run_trading_agents

        with patch('run_trading_agents.TradingAgentsGraph') as mock_graph:
            mock_instance = MagicMock()
            mock_graph.return_value = mock_instance
            mock_instance.propagate.return_value = (
                {"final_trade_decision": "BUY"},
                "BUY"
            )

            # Test whitespace stripping
            with patch('sys.argv', ['run_trading_agents.py', '--tickers', 'AAPL, MSFT, GOOGL']):
                run_trading_agents.main()

                self.assertEqual(mock_instance.propagate.call_count, 3)
                calls = mock_instance.propagate.call_args_list
                self.assertEqual(calls[0][0][0], "AAPL")
                self.assertEqual(calls[1][0][0], "MSFT")
                self.assertEqual(calls[2][0][0], "GOOGL")

    def test_tickers_flag_empty_entry_fails(self):
        """Test that --tickers with empty entries exits with error."""
        import io
        from contextlib import redirect_stderr, redirect_stdout

        import run_trading_agents

        # Test empty after stripping (e.g., trailing comma)
        output = io.StringIO()
        with (
            patch('sys.argv', ['run_trading_agents.py', '--tickers', 'AAPL,,MSFT']),
            redirect_stdout(output),
            redirect_stderr(output),
            self.assertRaises(SystemExit) as cm,
        ):
            run_trading_agents.main()

        self.assertEqual(cm.exception.code, 1)
        output_str = output.getvalue()
        self.assertIn('Error', output_str)
        self.assertIn('empty', output_str.lower())

        # Test just a comma
        output = io.StringIO()
        with (
            patch('sys.argv', ['run_trading_agents.py', '--tickers', ',']),
            redirect_stdout(output),
            redirect_stderr(output),
            self.assertRaises(SystemExit) as cm,
        ):
            run_trading_agents.main()

        self.assertEqual(cm.exception.code, 1)

    def test_tickers_flag_with_use_dates_from_json_fails(self):
        """Test that --tickers + --use-dates-from-json is rejected."""
        import io
        from contextlib import redirect_stderr, redirect_stdout

        import run_trading_agents

        output = io.StringIO()
        with (
            patch('sys.argv', ['run_trading_agents.py', '--tickers', 'AAPL',
                              '--use-dates-from-json']),
            redirect_stdout(output),
            redirect_stderr(output),
            self.assertRaises(SystemExit) as cm,
        ):
            run_trading_agents.main()

        self.assertEqual(cm.exception.code, 1)
        output_str = output.getvalue()
        self.assertIn('Error', output_str)
        self.assertIn('--tickers', output_str)
        self.assertIn('--use-dates-from-json', output_str)

    def test_stocks_file_flag_basic(self):
        """Test --stocks-file loads a stock list correctly."""
        import run_trading_agents

        with patch('run_trading_agents.TradingAgentsGraph') as mock_graph:
            mock_instance = MagicMock()
            mock_graph.return_value = mock_instance
            mock_instance.propagate.return_value = (
                {"final_trade_decision": "BUY"},
                "BUY"
            )

            with patch('sys.argv', ['run_trading_agents.py', '--stocks-file', str(self.stock_list_file)]):
                run_trading_agents.main()

                # Verify propagate was called twice
                self.assertEqual(mock_instance.propagate.call_count, 2)

    def test_stocks_file_flag_missing_file_fails(self):
        """Test that --stocks-file with missing file exits with error."""
        import io
        from contextlib import redirect_stderr, redirect_stdout

        import run_trading_agents

        output = io.StringIO()
        with (
            patch('sys.argv', ['run_trading_agents.py', '--stocks-file', '/nonexistent/stocks.json']),
            redirect_stdout(output),
            redirect_stderr(output),
            self.assertRaises(SystemExit) as cm,
        ):
            run_trading_agents.main()

        self.assertEqual(cm.exception.code, 1)
        output_str = output.getvalue()
        self.assertIn('not found', output_str)

    def test_stocks_file_flag_malformed_json_fails(self):
        """Test that --stocks-file with malformed JSON exits with error."""
        import io
        from contextlib import redirect_stderr, redirect_stdout

        import run_trading_agents

        with tempfile.TemporaryDirectory() as temp_dir:
            bad_json = Path(temp_dir) / "bad.json"
            bad_json.write_text("{invalid json")

            output = io.StringIO()
            with (
                patch('sys.argv', ['run_trading_agents.py', '--stocks-file', str(bad_json)]),
                redirect_stdout(output),
                redirect_stderr(output),
                self.assertRaises(SystemExit) as cm,
            ):
                run_trading_agents.main()

            self.assertEqual(cm.exception.code, 1)
            output_str = output.getvalue()
            self.assertIn('Invalid JSON', output_str)

    def test_stocks_file_flag_non_array_fails(self):
        """Test that --stocks-file with non-array JSON exits with error."""
        import io
        from contextlib import redirect_stderr, redirect_stdout

        import run_trading_agents

        with tempfile.TemporaryDirectory() as temp_dir:
            not_array = Path(temp_dir) / "not_array.json"
            not_array.write_text('{"ticker": "AAPL"}')

            output = io.StringIO()
            with (
                patch('sys.argv', ['run_trading_agents.py', '--stocks-file', str(not_array)]),
                redirect_stdout(output),
                redirect_stderr(output),
                self.assertRaises(SystemExit) as cm,
            ):
                run_trading_agents.main()

            self.assertEqual(cm.exception.code, 1)
            output_str = output.getvalue()
            self.assertIn('should contain an array', output_str)

    def test_stocks_file_flag_with_use_dates_from_json(self):
        """Test that --stocks-file works with --use-dates-from-json."""
        import run_trading_agents

        with patch('run_trading_agents.TradingAgentsGraph') as mock_graph:
            mock_instance = MagicMock()
            mock_graph.return_value = mock_instance
            mock_instance.propagate.return_value = (
                {"final_trade_decision": "BUY"},
                "BUY"
            )

            with patch('sys.argv', ['run_trading_agents.py', '--stocks-file', str(self.stock_list_file),
                                   '--use-dates-from-json']):
                run_trading_agents.main()

                # Verify dates from JSON were used
                calls = mock_instance.propagate.call_args_list
                self.assertEqual(calls[0][0], ("AAPL", "2024-01-15"))
                self.assertEqual(calls[1][0], ("MSFT", "2024-01-15"))

    def test_positional_and_stocks_file_mutually_exclusive(self):
        """Test that positional JSON + --stocks-file is rejected."""
        import io
        from contextlib import redirect_stderr, redirect_stdout

        import run_trading_agents

        with tempfile.TemporaryDirectory() as temp_dir:
            stocks_file = Path(temp_dir) / "stocks.json"
            stocks_file.write_text(json.dumps([{"ticker": "AAPL"}]))

            output = io.StringIO()
            with (
                patch('sys.argv', ['run_trading_agents.py', str(self.stock_list_file),
                                  '--stocks-file', str(stocks_file)]),
                redirect_stdout(output),
                redirect_stderr(output),
                self.assertRaises(SystemExit) as cm,
            ):
                run_trading_agents.main()

            self.assertEqual(cm.exception.code, 1)
            output_str = output.getvalue()
            self.assertIn('Cannot combine multiple stocks sources', output_str)

    def test_stocks_file_and_tickers_mutually_exclusive(self):
        """Test that --stocks-file + --tickers is rejected."""
        import io
        from contextlib import redirect_stderr, redirect_stdout

        import run_trading_agents

        output = io.StringIO()
        with (
            patch('sys.argv', ['run_trading_agents.py', '--stocks-file', str(self.stock_list_file),
                              '--tickers', 'AAPL']),
            redirect_stdout(output),
            redirect_stderr(output),
            self.assertRaises(SystemExit) as cm,
        ):
            run_trading_agents.main()

        self.assertEqual(cm.exception.code, 1)
        output_str = output.getvalue()
        self.assertIn('Cannot combine multiple stocks sources', output_str)

    def test_config_file_plus_tickers_rejected(self):
        """Test that config file + --tickers is rejected via #124 mixing check."""
        import io
        from contextlib import redirect_stderr, redirect_stdout

        import run_trading_agents

        config_file = Path(self.temp_dir.name) / "config.json"
        config_file.write_text(json.dumps({
            "stocks_file": str(self.stock_list_file),
            "llm_provider": "ollama",
        }))

        output = io.StringIO()
        with (
            patch('sys.argv', ['run_trading_agents.py', str(config_file), '--tickers', 'AAPL']),
            redirect_stdout(output),
            redirect_stderr(output),
            self.assertRaises(SystemExit) as cm,
        ):
            run_trading_agents.main()

        self.assertEqual(cm.exception.code, 1)
        output_str = output.getvalue()
        self.assertIn('exclusive', output_str.lower())
        self.assertIn('--tickers', output_str)

    def test_legacy_stock_list_still_works(self):
        """Test that legacy stock list arrays still work (backward compatibility)."""
        import run_trading_agents

        with patch('run_trading_agents.TradingAgentsGraph') as mock_graph:
            mock_instance = MagicMock()
            mock_graph.return_value = mock_instance
            mock_instance.propagate.return_value = (
                {"final_trade_decision": "BUY"},
                "BUY"
            )

            with patch('sys.argv', ['run_trading_agents.py', str(self.stock_list_file)]):
                run_trading_agents.main()

                # Verify propagate was called twice (for both stocks)
                self.assertEqual(mock_instance.propagate.call_count, 2)

    def test_positional_and_tickers_mutually_exclusive(self):
        """Test that positional JSON + --tickers is rejected."""
        import io
        from contextlib import redirect_stderr, redirect_stdout

        import run_trading_agents

        output = io.StringIO()
        with (
            patch('sys.argv', ['run_trading_agents.py', str(self.stock_list_file), '--tickers', 'AAPL']),
            redirect_stdout(output),
            redirect_stderr(output),
            self.assertRaises(SystemExit) as cm,
        ):
            run_trading_agents.main()

        self.assertEqual(cm.exception.code, 1)
        output_str = output.getvalue()
        self.assertIn('Cannot combine multiple stocks sources', output_str)


@pytest.mark.unit
class RunTradingAgentsEnvVarPrecedenceTests(_TempStockFileTestCase):
    """Test the TRADINGAGENTS_* env var tier of the precedence chain.

    Design review finding #4: llm_provider must fall back to
    DEFAULT_CONFIG's env-override-aware value (TRADINGAGENTS_LLM_PROVIDER)
    when neither the CLI flag nor the config file sets it, instead of a
    hardcoded 'ollama' literal silently overwriting it. These tests reload
    tradingagents.default_config with the env var set (mirroring the
    established pattern in tests/test_env_overrides.py) and point
    run_trading_agents.DEFAULT_CONFIG at the reloaded object, since
    run_trading_agents imported the DEFAULT_CONFIG binding once at module
    load time.
    """

    STOCKS = [{"ticker": "AAPL", "date": "2024-01-15"}]

    def _reload_default_config_with_env(self, **env_overrides):
        """Reload default_config with only the given env overrides set.

        Returns the reloaded module's DEFAULT_CONFIG dict. Callers are
        responsible for restoring process-global state in tearDown (done by
        this class's tearDown below), mirroring test_env_overrides.py's
        reload-then-restore convention.
        """
        for key in list(default_config_module._ENV_OVERRIDES):
            self._env_backup.setdefault(key, os.environ.get(key))
            os.environ.pop(key, None)
        for key, val in env_overrides.items():
            os.environ[key] = val
        reloaded = importlib.reload(default_config_module)
        return reloaded.DEFAULT_CONFIG

    def setUp(self):
        super().setUp()
        self._env_backup = {}

    def tearDown(self):
        super().tearDown()
        # Restore whatever env vars were present before this test touched
        # them, then reload default_config once more so the process-global
        # DEFAULT_CONFIG singleton doesn't leak into later tests in this
        # session (same restoration step test_env_overrides.py takes after
        # a reload).
        for key, val in self._env_backup.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val
        importlib.reload(default_config_module)

    def test_llm_provider_env_var_honored_when_absent_from_flag_and_config(self):
        """TRADINGAGENTS_LLM_PROVIDER is honoured when neither flag nor config file sets it."""
        import run_trading_agents

        reloaded_default_config = self._reload_default_config_with_env(
            TRADINGAGENTS_LLM_PROVIDER="mistral",
        )

        config_file = Path(self.temp_dir.name) / "config.json"
        config_file.write_text(json.dumps({
            "stocks_file": str(self.stock_list_file),
            # llm_provider intentionally omitted; deep/quick think models are
            # still supplied via the config file since the non-ollama model
            # requirement is checked independently of the provider's source.
            "deep_think_llm": "mistral-large",
            "quick_think_llm": "mistral-small",
        }))

        with (
            patch.object(run_trading_agents, 'DEFAULT_CONFIG', reloaded_default_config),
            patch('run_trading_agents.TradingAgentsGraph') as mock_graph,
            patch.dict(os.environ, {'MISTRAL_API_KEY': 'test-key'}),
        ):
            mock_instance = MagicMock()
            mock_graph.return_value = mock_instance
            mock_instance.propagate.return_value = (
                {"final_trade_decision": "BUY"},
                "BUY"
            )

            with patch('sys.argv', ['run_trading_agents.py', str(config_file)]):
                run_trading_agents.main()

                call_args = mock_graph.call_args
                config = call_args[1]['config']
                self.assertEqual(config["llm_provider"], "mistral")

    def test_config_file_llm_provider_beats_env_var(self):
        """A config-file llm_provider value still beats the TRADINGAGENTS_LLM_PROVIDER env var."""
        import run_trading_agents

        reloaded_default_config = self._reload_default_config_with_env(
            TRADINGAGENTS_LLM_PROVIDER="openai",
        )

        config_file = Path(self.temp_dir.name) / "config.json"
        config_file.write_text(json.dumps({
            "stocks_file": str(self.stock_list_file),
            "llm_provider": "mistral",
            "deep_think_llm": "mistral-large",
            "quick_think_llm": "mistral-small",
        }))

        with (
            patch.object(run_trading_agents, 'DEFAULT_CONFIG', reloaded_default_config),
            patch('run_trading_agents.TradingAgentsGraph') as mock_graph,
            patch.dict(os.environ, {'MISTRAL_API_KEY': 'test-key'}),
        ):
            mock_instance = MagicMock()
            mock_graph.return_value = mock_instance
            mock_instance.propagate.return_value = (
                {"final_trade_decision": "BUY"},
                "BUY"
            )

            with patch('sys.argv', ['run_trading_agents.py', str(config_file)]):
                run_trading_agents.main()

                call_args = mock_graph.call_args
                config = call_args[1]['config']
                self.assertEqual(config["llm_provider"], "mistral")

    def test_config_file_nested_config_block_beats_env_var(self):
        """Config block's nested key beats TRADINGAGENTS_* env var (issue #117)."""
        import run_trading_agents

        reloaded_default_config = self._reload_default_config_with_env(
            TRADINGAGENTS_MAX_DEBATE_ROUNDS="3",
        )

        config_file = Path(self.temp_dir.name) / "config.json"
        config_file.write_text(json.dumps({
            "stocks_file": str(self.stock_list_file),
            "config": {
                "max_debate_rounds": 2,
            },
        }))

        with (
            patch.object(run_trading_agents, 'DEFAULT_CONFIG', reloaded_default_config),
            patch('run_trading_agents.TradingAgentsGraph') as mock_graph,
        ):
            mock_instance = MagicMock()
            mock_graph.return_value = mock_instance
            mock_instance.propagate.return_value = (
                {"final_trade_decision": "BUY"},
                "BUY"
            )

            with patch('sys.argv', ['run_trading_agents.py', str(config_file)]):
                run_trading_agents.main()

                call_args = mock_graph.call_args
                config = call_args[1]['config']
                self.assertEqual(config["max_debate_rounds"], 2)


@pytest.mark.unit
class RunTradingAgentsLLMProviderTests(_TempStockFileTestCase):
    """Test LLM provider configuration via CLI flags."""

    STOCKS = [{"ticker": "AAPL", "date": "2024-01-15"}]

    def test_default_provider_is_ollama_unchanged(self):
        """Test that default behavior (no flags) uses ollama with default models."""
        import run_trading_agents

        with patch('run_trading_agents.TradingAgentsGraph') as mock_graph:
            mock_instance = MagicMock()
            mock_graph.return_value = mock_instance
            mock_instance.propagate.return_value = (
                {"final_trade_decision": "BUY"},
                "BUY"
            )

            with patch('sys.argv', ['run_trading_agents.py', str(self.stock_list_file)]):
                run_trading_agents.main()

                # Verify TradingAgentsGraph was called with ollama provider
                call_args = mock_graph.call_args
                config = call_args[1]['config']
                self.assertEqual(config["llm_provider"], "ollama")
                self.assertEqual(config["deep_think_llm"], "ministral-3:8b")
                self.assertEqual(config["quick_think_llm"], "ministral-3:3b")

    def test_mistral_provider_with_models_and_api_key(self):
        """Test that --llm-provider mistral with models and MISTRAL_API_KEY set works."""
        import run_trading_agents

        with patch('run_trading_agents.TradingAgentsGraph') as mock_graph, \
             patch.dict(os.environ, {'MISTRAL_API_KEY': 'test-key'}):

            mock_instance = MagicMock()
            mock_graph.return_value = mock_instance
            mock_instance.propagate.return_value = (
                {"final_trade_decision": "BUY"},
                "BUY"
            )

            with patch('sys.argv', ['run_trading_agents.py', str(self.stock_list_file),
                                   '--llm-provider', 'mistral',
                                   '--deep-think-llm', 'mistral-large',
                                   '--quick-think-llm', 'mistral-small']):
                run_trading_agents.main()

                # Verify TradingAgentsGraph was called with mistral provider and specified models
                call_args = mock_graph.call_args
                config = call_args[1]['config']
                self.assertEqual(config["llm_provider"], "mistral")
                self.assertEqual(config["deep_think_llm"], "mistral-large")
                self.assertEqual(config["quick_think_llm"], "mistral-small")

    def test_non_ollama_provider_missing_deep_think_model_fails(self):
        """Test that non-ollama provider without --deep-think-llm fails with argparse error."""
        import io
        from contextlib import redirect_stderr

        import run_trading_agents

        with (
            patch.dict(os.environ, {'MISTRAL_API_KEY': 'test-key'}),
            patch('sys.argv', ['run_trading_agents.py', str(self.stock_list_file),
                               '--llm-provider', 'mistral',
                               '--quick-think-llm', 'mistral-small']),
        ):
            output = io.StringIO()
            with (
                redirect_stderr(output),
                self.assertRaises(SystemExit) as cm,
            ):
                run_trading_agents.main()

            # argparse.error() calls sys.exit(2)
            self.assertEqual(cm.exception.code, 2)
            error_output = output.getvalue()
            self.assertIn('--deep-think-llm', error_output)

    def test_non_ollama_provider_missing_quick_think_model_fails(self):
        """Test that non-ollama provider without --quick-think-llm fails with argparse error."""
        import io
        from contextlib import redirect_stderr

        import run_trading_agents

        with (
            patch.dict(os.environ, {'MISTRAL_API_KEY': 'test-key'}),
            patch('sys.argv', ['run_trading_agents.py', str(self.stock_list_file),
                               '--llm-provider', 'mistral',
                               '--deep-think-llm', 'mistral-large']),
        ):
            output = io.StringIO()
            with (
                redirect_stderr(output),
                self.assertRaises(SystemExit) as cm,
            ):
                run_trading_agents.main()

            # argparse.error() calls sys.exit(2)
            self.assertEqual(cm.exception.code, 2)
            error_output = output.getvalue()
            self.assertIn('--quick-think-llm', error_output)

    def test_missing_api_key_for_mistral_exits_before_processing(self):
        """Test that missing MISTRAL_API_KEY exits with code 1 before processing tickers."""
        import io
        from contextlib import redirect_stderr, redirect_stdout

        import run_trading_agents

        # Remove MISTRAL_API_KEY from environment if it exists
        env_copy = os.environ.copy()
        env_copy.pop('MISTRAL_API_KEY', None)

        with (
            patch.dict(os.environ, env_copy, clear=True),
            patch('sys.argv', ['run_trading_agents.py', str(self.stock_list_file),
                               '--llm-provider', 'mistral',
                               '--deep-think-llm', 'mistral-large',
                               '--quick-think-llm', 'mistral-small']),
        ):
            output = io.StringIO()
            with (
                redirect_stdout(output),
                redirect_stderr(output),
                self.assertRaises(SystemExit) as cm,
            ):
                run_trading_agents.main()

            # Should exit with code 1
            self.assertEqual(cm.exception.code, 1)

            # Verify error message names the provider and the env var
            output_str = output.getvalue()
            self.assertIn('Error', output_str)
            self.assertIn('mistral', output_str)
            self.assertIn('MISTRAL_API_KEY', output_str)

    def test_ollama_provider_does_not_require_api_key(self):
        """Test that ollama provider (which requires no API key) works without MISTRAL_API_KEY."""
        import run_trading_agents

        # Ensure MISTRAL_API_KEY is NOT in environment
        env_copy = os.environ.copy()
        env_copy.pop('MISTRAL_API_KEY', None)

        with patch.dict(os.environ, env_copy, clear=True), \
             patch('run_trading_agents.TradingAgentsGraph') as mock_graph:

            mock_instance = MagicMock()
            mock_graph.return_value = mock_instance
            mock_instance.propagate.return_value = (
                {"final_trade_decision": "BUY"},
                "BUY"
            )

            with patch('sys.argv', ['run_trading_agents.py', str(self.stock_list_file)]):
                # Should not raise SystemExit for missing API key
                run_trading_agents.main()

                # Verify propagate was called (i.e., we got past the API key check)
                self.assertEqual(mock_instance.propagate.call_count, 1)

    def test_openai_provider_with_models_and_api_key(self):
        """Test that --llm-provider openai with models and OPENAI_API_KEY set works."""
        import run_trading_agents

        with patch('run_trading_agents.TradingAgentsGraph') as mock_graph, \
             patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):

            mock_instance = MagicMock()
            mock_graph.return_value = mock_instance
            mock_instance.propagate.return_value = (
                {"final_trade_decision": "BUY"},
                "BUY"
            )

            with patch('sys.argv', ['run_trading_agents.py', str(self.stock_list_file),
                                   '--llm-provider', 'openai',
                                   '--deep-think-llm', 'gpt-4-turbo',
                                   '--quick-think-llm', 'gpt-4o']):
                run_trading_agents.main()

                # Verify TradingAgentsGraph was called with openai provider and specified models
                call_args = mock_graph.call_args
                config = call_args[1]['config']
                self.assertEqual(config["llm_provider"], "openai")
                self.assertEqual(config["deep_think_llm"], "gpt-4-turbo")
                self.assertEqual(config["quick_think_llm"], "gpt-4o")
