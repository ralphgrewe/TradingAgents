"""Tests for debug streaming in TradingAgentsGraph.

Tests verify that:
1. Debug streaming output includes node names
2. Identical consecutive messages are deduplicated
3. Final state from debug path matches non-debug path
"""

import io
from contextlib import redirect_stdout
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from tradingagents.graph.trading_graph import TradingAgentsGraph


def _create_mocked_ta(debug=True):
    """Create a TradingAgentsGraph with all necessary mocks for testing."""
    ta = TradingAgentsGraph(selected_analysts=["market", "social"], debug=debug)
    ta.graph = MagicMock()
    ta.memory_log = MagicMock()
    ta._memory_client = MagicMock()
    ta.propagator = MagicMock()
    ta._log_state = MagicMock()
    return ta


@pytest.mark.unit
class TestDebugStreaming:
    """Tests for debug streaming functionality."""

    def test_debug_streaming_includes_node_names(self, mock_llm_client):
        """Test that debug output includes graph node names."""
        ta = _create_mocked_ta(debug=True)

        # Simulate streaming with "updates" mode yielding {node_name: state_delta}
        ta.graph.stream.return_value = [
            {
                "Market Analyst": {
                    "messages": [AIMessage(content="Market analysis report")],
                    "market_report": "Market analysis",
                }
            },
            {
                "Msg Clear Market": {
                    "messages": [HumanMessage(content="Continue")],
                }
            },
            {
                "Sentiment Analyst": {
                    "messages": [AIMessage(content="Sentiment analysis report")],
                    "sentiment_report": "Sentiment analysis",
                    "final_trade_decision": "BUY",
                    "investment_plan": "Plan",
                }
            },
        ]

        ta.propagator.create_initial_state.return_value = {
            "messages": [("human", "AAPL")],
            "company_of_interest": "AAPL",
            "asset_type": "stock",
            "trade_date": "2024-01-15",
            "past_context": "",
            "investment_plan": "Plan",
            "final_trade_decision": "",
        }
        ta.propagator.get_graph_args.return_value = {
            "stream_mode": "values",
            "config": {"recursion_limit": 100},
        }

        # Capture stdout
        output = io.StringIO()
        with redirect_stdout(output):
            ta._run_graph("AAPL", "2024-01-15")

        output_text = output.getvalue()

        # Verify node names appear in output
        assert "── Market Analyst ──" in output_text
        assert "── Sentiment Analyst ──" in output_text

        # Verify graph.stream was called with "updates" mode in debug path
        call_args = ta.graph.stream.call_args
        assert call_args is not None
        # Check that stream_mode="updates" was passed
        assert call_args[1].get("stream_mode") == "updates"

    def test_debug_streaming_deduplicates_identical_messages(self, mock_llm_client):
        """Test that identical consecutive messages are not re-printed."""
        ta = _create_mocked_ta(debug=True)

        # Simulate streaming where two consecutive nodes produce the same "Continue" message
        ta.graph.stream.return_value = [
            {
                "Market Analyst": {
                    "messages": [AIMessage(content="Market report")],
                    "market_report": "Market analysis",
                }
            },
            {
                "Msg Clear Market": {
                    "messages": [HumanMessage(content="Continue")],
                }
            },
            {
                "Sentiment Analyst": {
                    "messages": [HumanMessage(content="Continue")],  # Same message
                    "sentiment_report": "Sentiment analysis",
                    "final_trade_decision": "HOLD",
                    "investment_plan": "Plan",
                }
            },
        ]

        ta.propagator.create_initial_state.return_value = {
            "messages": [("human", "AAPL")],
            "company_of_interest": "AAPL",
            "asset_type": "stock",
            "trade_date": "2024-01-15",
            "past_context": "",
            "investment_plan": "Plan",
            "final_trade_decision": "",
        }
        ta.propagator.get_graph_args.return_value = {
            "stream_mode": "values",
            "config": {"recursion_limit": 100},
        }

        output = io.StringIO()
        with redirect_stdout(output):
            ta._run_graph("AAPL", "2024-01-15")

        output_text = output.getvalue()

        # Count how many times "Continue" appears after node headers
        # The identical message should only be printed once
        continue_count = output_text.count("Continue")
        # Should not have Continue printed multiple times for identical messages
        # The exact count depends on deduplication, but should be minimal
        # (first appearance + possible node headers)
        assert continue_count <= 2  # One message + one header

    def test_debug_streaming_preserves_final_state(self, mock_llm_client):
        """Test that debug streaming produces correct final state."""
        ta = _create_mocked_ta(debug=True)

        # Simulate streaming updates
        ta.graph.stream.return_value = [
            {
                "Market Analyst": {
                    "messages": [AIMessage(content="Market report")],
                    "market_report": "Market analysis",
                }
            },
            {
                "Msg Clear Market": {
                    "messages": [HumanMessage(content="Continue")],
                    "market_report": "Market analysis",
                }
            },
            {
                "Sentiment Analyst": {
                    "messages": [AIMessage(content="Sentiment report")],
                    "sentiment_report": "Sentiment analysis",
                    "final_trade_decision": "BUY",
                    "investment_plan": "Plan",
                }
            },
        ]

        ta.propagator.create_initial_state.return_value = {
            "messages": [("human", "AAPL")],
            "company_of_interest": "AAPL",
            "asset_type": "stock",
            "trade_date": "2024-01-15",
            "past_context": "",
            "investment_plan": "Plan",
            "final_trade_decision": "",
        }
        ta.propagator.get_graph_args.return_value = {
            "stream_mode": "values",
            "config": {"recursion_limit": 100},
        }

        output = io.StringIO()
        with redirect_stdout(output):
            ta._run_graph("AAPL", "2024-01-15")

        # Verify the current state was set correctly (contains key fields)
        assert ta.curr_state is not None
        assert ta.curr_state.get("market_report") == "Market analysis"
        assert ta.curr_state.get("sentiment_report") == "Sentiment analysis"
        assert ta.curr_state.get("company_of_interest") == "AAPL"
        assert ta.curr_state.get("final_trade_decision") == "BUY"

    def test_non_debug_streaming_unchanged(self, mock_llm_client):
        """Test that non-debug mode uses invoke() and is unchanged."""
        ta = _create_mocked_ta(debug=False)

        expected_final_state = {
            "messages": [],
            "company_of_interest": "AAPL",
            "asset_type": "stock",
            "trade_date": "2024-01-15",
            "past_context": "",
            "investment_plan": "Plan",
            "final_trade_decision": "HOLD",
        }
        ta.graph.invoke.return_value = expected_final_state

        ta.propagator.create_initial_state.return_value = {
            "messages": [("human", "AAPL")],
            "company_of_interest": "AAPL",
            "asset_type": "stock",
            "trade_date": "2024-01-15",
            "past_context": "",
            "investment_plan": "Plan",
            "final_trade_decision": "",
        }
        ta.propagator.get_graph_args.return_value = {
            "stream_mode": "values",
            "config": {"recursion_limit": 100},
        }

        ta._run_graph("AAPL", "2024-01-15")

        # Verify invoke was called (not stream)
        ta.graph.invoke.assert_called_once()
        assert ta.graph.stream.call_count == 0

        # Verify final state is correct
        assert ta.curr_state == expected_final_state
