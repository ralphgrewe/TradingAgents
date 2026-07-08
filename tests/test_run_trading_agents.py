"""Tests for run_trading_agents.py error handling and fail-fast behavior.

These are integration-level tests that verify the error handling by running
the script with mocked dependencies and checking the exit behavior.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.unit
class RunTradingAgentsErrorHandlingTests(unittest.TestCase):
    """Test fail-fast error handling in run_trading_agents.py."""

    def setUp(self):
        """Create temporary JSON file with test stocks."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.stock_list_file = Path(self.temp_dir.name) / "test_stocks.json"
        self.stock_list_file.write_text(json.dumps([
            {"ticker": "AAPL", "date": "2024-01-15"},
            {"ticker": "MSFT", "date": "2024-01-15"},
        ]))

    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()

    def test_propagate_failure_prints_exception_type_and_message(self):
        """Test that exception diagnostics include both type and message.

        The issue requirement states: "include the exception type/class name and
        message (not just str(e)), so e.g. an LLM-provider connection failure
        is distinguishable from a bad ticker/date or a data-vendor error."
        """
        import run_trading_agents
        import io
        from contextlib import redirect_stdout, redirect_stderr

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
        import run_trading_agents
        import io
        from contextlib import redirect_stdout, redirect_stderr
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
        import run_trading_agents
        import io
        from contextlib import redirect_stdout, redirect_stderr

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
            mock_save_report.side_effect = IOError("Cannot write to disk")

            with patch('sys.argv', ['run_trading_agents.py', str(self.stock_list_file),
                                   '--report-dir', './reports']):
                # Should NOT raise SystemExit for report-saving failures
                run_trading_agents.main()

                # Verify propagate was called for both tickers (not stopped on report error)
                self.assertEqual(mock_instance.propagate.call_count, 2)
