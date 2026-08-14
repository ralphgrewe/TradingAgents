"""Tests for run_trading_agents.py error handling, fail-fast behavior, and LLM provider configuration.

These are integration-level tests that verify the error handling by running
the script with mocked dependencies and checking the exit behavior.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


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
            with patch('sys.argv', ['run_trading_agents.py', str(self.stock_list_file)]):
                with redirect_stdout(output), redirect_stderr(output):
                    with self.assertRaises(SystemExit) as cm:
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
            with patch('sys.argv', ['run_trading_agents.py', str(self.stock_list_file),
                                   '--portfolio', '--style', 'aggressive', '--depot-id', 'test-depot']):
                with redirect_stdout(output), redirect_stderr(output):
                    with self.assertRaises(SystemExit) as cm:
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
            with patch('sys.argv', ['run_trading_agents.py', str(self.stock_list_file),
                                   '--portfolio', '--style', 'aggressive', '--depot-id', 'test-depot']):
                with redirect_stdout(output), redirect_stderr(output):
                    with self.assertRaises(SystemExit) as cm:
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
        with patch('sys.argv', ['run_trading_agents.py', str(self.stock_list_file),
                               '--use-dates-from-json']):
            with redirect_stdout(output), redirect_stderr(output):
                with self.assertRaises(SystemExit) as cm:
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
            with patch('sys.argv', ['run_trading_agents.py', str(self.stock_list_file),
                                   '--use-dates-from-json']):
                with redirect_stdout(output), redirect_stderr(output):
                    with self.assertRaises(SystemExit) as cm:
                        run_trading_agents.main()

                    # Should exit with code 1
                    self.assertEqual(cm.exception.code, 1)

                    # Verify propagate was never called (validation happened upfront)
                    self.assertEqual(mock_instance.propagate.call_count, 0)


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

        with patch.dict(os.environ, {'MISTRAL_API_KEY': 'test-key'}):
            with patch('sys.argv', ['run_trading_agents.py', str(self.stock_list_file),
                                   '--llm-provider', 'mistral',
                                   '--quick-think-llm', 'mistral-small']):
                output = io.StringIO()
                with redirect_stderr(output):
                    with self.assertRaises(SystemExit) as cm:
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

        with patch.dict(os.environ, {'MISTRAL_API_KEY': 'test-key'}):
            with patch('sys.argv', ['run_trading_agents.py', str(self.stock_list_file),
                                   '--llm-provider', 'mistral',
                                   '--deep-think-llm', 'mistral-large']):
                output = io.StringIO()
                with redirect_stderr(output):
                    with self.assertRaises(SystemExit) as cm:
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

        with patch.dict(os.environ, env_copy, clear=True):
            with patch('sys.argv', ['run_trading_agents.py', str(self.stock_list_file),
                                   '--llm-provider', 'mistral',
                                   '--deep-think-llm', 'mistral-large',
                                   '--quick-think-llm', 'mistral-small']):
                output = io.StringIO()
                with redirect_stdout(output), redirect_stderr(output):
                    with self.assertRaises(SystemExit) as cm:
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
