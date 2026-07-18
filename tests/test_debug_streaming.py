"""Tests for debug streaming in TradingAgentsGraph.

Tests verify that:
1. Debug streaming output includes node names
2. Identical consecutive messages are deduplicated
3. Final state from the debug path is identical to what ``graph.invoke()``
   yields — including reducer-managed channels like ``messages``
   (``add_messages``), which last-write-wins delta merging would corrupt.
   This is covered both with a mocked graph (chunk plumbing) and with a
   real compiled LangGraph graph (genuine merge equivalence).
"""

import io
from contextlib import redirect_stdout
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, MessagesState, StateGraph

from tradingagents.graph.trading_graph import TradingAgentsGraph

_INITIAL_STATE = {
    "messages": [("human", "AAPL")],
    "company_of_interest": "AAPL",
    "asset_type": "stock",
    "trade_date": "2024-01-15",
    "past_context": "",
    "investment_plan": "Plan",
    "final_trade_decision": "",
}

_GRAPH_ARGS = {
    "stream_mode": "values",
    "config": {"recursion_limit": 100},
}


def _create_mocked_ta(debug=True):
    """Create a TradingAgentsGraph with all necessary mocks for testing."""
    ta = TradingAgentsGraph(selected_analysts=["market", "social"], debug=debug)
    ta.graph = MagicMock()
    ta.memory_log = MagicMock()
    ta._memory_client = MagicMock()
    ta.propagator = MagicMock()
    ta.propagator.create_initial_state.return_value = dict(_INITIAL_STATE)
    ta.propagator.get_graph_args.return_value = dict(_GRAPH_ARGS)
    ta._log_state = MagicMock()
    return ta


@pytest.mark.unit
class TestDebugStreaming:
    """Tests for debug streaming functionality (mocked graph)."""

    def test_debug_streaming_includes_node_names(self, mock_llm_client):
        """Test that debug output includes graph node names."""
        ta = _create_mocked_ta(debug=True)

        # Simulate combined-mode streaming: (mode, chunk) tuples where
        # "updates" chunks are {node_name: state_delta} and "values" chunks
        # are the full reduced state after each step.
        ta.graph.stream.return_value = [
            ("values", dict(_INITIAL_STATE)),
            (
                "updates",
                {
                    "Market Analyst": {
                        "messages": [AIMessage(content="Market analysis report")],
                        "market_report": "Market analysis",
                    }
                },
            ),
            (
                "updates",
                {
                    "Msg Clear Market": {
                        "messages": [HumanMessage(content="Continue")],
                    }
                },
            ),
            (
                "updates",
                {
                    "Sentiment Analyst": {
                        "messages": [AIMessage(content="Sentiment analysis report")],
                        "sentiment_report": "Sentiment analysis",
                    }
                },
            ),
            (
                "values",
                {
                    **_INITIAL_STATE,
                    "market_report": "Market analysis",
                    "sentiment_report": "Sentiment analysis",
                    "final_trade_decision": "BUY",
                },
            ),
        ]

        # Capture stdout
        output = io.StringIO()
        with redirect_stdout(output):
            ta._run_graph("AAPL", "2024-01-15")

        output_text = output.getvalue()

        # Verify node names appear in output
        assert "── Market Analyst ──" in output_text
        assert "── Sentiment Analyst ──" in output_text

        # Verify graph.stream was called with combined modes so node names
        # ("updates") and reducer-correct final state ("values") both arrive.
        call_args = ta.graph.stream.call_args
        assert call_args is not None
        assert call_args[1].get("stream_mode") == ["values", "updates"]

    def test_debug_streaming_deduplicates_identical_messages(self, mock_llm_client):
        """Test that identical consecutive messages are not re-printed."""
        ta = _create_mocked_ta(debug=True)

        # Two consecutive nodes produce the same "Continue" message.
        ta.graph.stream.return_value = [
            (
                "updates",
                {
                    "Market Analyst": {
                        "messages": [AIMessage(content="Market report")],
                        "market_report": "Market analysis",
                    }
                },
            ),
            (
                "updates",
                {
                    "Msg Clear Market": {
                        "messages": [HumanMessage(content="Continue")],
                    }
                },
            ),
            (
                "updates",
                {
                    "Msg Clear Social": {
                        "messages": [HumanMessage(content="Continue")],  # Same message
                    }
                },
            ),
            (
                "values",
                {
                    **_INITIAL_STATE,
                    "market_report": "Market analysis",
                    "final_trade_decision": "HOLD",
                },
            ),
        ]

        output = io.StringIO()
        with redirect_stdout(output):
            ta._run_graph("AAPL", "2024-01-15")

        output_text = output.getvalue()

        # The identical "Continue" message should only be printed once
        # (deduplicated), so exactly one header and one message body.
        assert output_text.count("── Msg Clear Market ──") == 1
        assert output_text.count("── Msg Clear Social ──") == 0
        assert output_text.count("Continue") == 1

    def test_debug_final_state_comes_from_values_chunk(self, mock_llm_client):
        """The debug path must take its final state from the last "values"
        chunk (already reducer-merged by LangGraph), not from last-write-wins
        merging of "updates" deltas."""
        ta = _create_mocked_ta(debug=True)

        final_values = {
            **_INITIAL_STATE,
            # Reduced messages list: accumulated, not just the last delta.
            "messages": [
                HumanMessage(content="AAPL"),
                AIMessage(content="Market report"),
                HumanMessage(content="Continue"),
            ],
            "market_report": "Market analysis",
            "sentiment_report": "Sentiment analysis",
            "final_trade_decision": "BUY",
        }
        ta.graph.stream.return_value = [
            ("values", dict(_INITIAL_STATE)),
            (
                "updates",
                {
                    "Market Analyst": {
                        "messages": [AIMessage(content="Market report")],
                        "market_report": "Market analysis",
                    }
                },
            ),
            (
                "updates",
                {
                    "Msg Clear Market": {
                        "messages": [HumanMessage(content="Continue")],
                    }
                },
            ),
            ("values", final_values),
        ]

        output = io.StringIO()
        with redirect_stdout(output):
            ta._run_graph("AAPL", "2024-01-15")

        # The final state must be exactly the last "values" chunk — including
        # the full accumulated messages list, which a naive delta merge would
        # collapse to just [HumanMessage("Continue")].
        assert ta.curr_state == final_values
        assert [m.content for m in ta.curr_state["messages"]] == [
            "AAPL",
            "Market report",
            "Continue",
        ]

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

        ta._run_graph("AAPL", "2024-01-15")

        # Verify invoke was called (not stream)
        ta.graph.invoke.assert_called_once()
        assert ta.graph.stream.call_count == 0

        # Verify the non-debug invoke args are untouched by the debug-path
        # stream_mode override (debug_args is a local copy).
        invoke_kwargs = ta.graph.invoke.call_args[1]
        assert invoke_kwargs.get("stream_mode") == "values"

        # Verify final state is correct
        assert ta.curr_state == expected_final_state


class _MiniState(MessagesState):
    """Minimal schema mirroring AgentState's structure: a reducer-managed
    ``messages`` channel (``add_messages``, inherited from MessagesState)
    plus plain last-write-wins scalar fields."""

    market_report: str
    final_trade_decision: str


_MINI_INIT_STATE = {
    "messages": [HumanMessage(content="AAPL")],
    "market_report": "",
    "final_trade_decision": "",
}


def _build_mini_graph():
    """A small real graph: two message-appending nodes plus a scalar writer,
    exercising both the ``add_messages`` reducer and last-write-wins fields."""

    def market_analyst(state):
        return {
            "messages": [AIMessage(content="Market report")],
            "market_report": "Market analysis",
        }

    def msg_clear(state):
        return {"messages": [HumanMessage(content="Continue")]}

    def portfolio_manager(state):
        return {
            "messages": [AIMessage(content="Final decision")],
            "final_trade_decision": "BUY",
        }

    builder = StateGraph(_MiniState)
    builder.add_node("Market Analyst", market_analyst)
    builder.add_node("Msg Clear Market", msg_clear)
    builder.add_node("Portfolio Manager", portfolio_manager)
    builder.add_edge(START, "Market Analyst")
    builder.add_edge("Market Analyst", "Msg Clear Market")
    builder.add_edge("Msg Clear Market", "Portfolio Manager")
    builder.add_edge("Portfolio Manager", END)
    return builder.compile()


@pytest.mark.unit
class TestDebugStreamingRealGraph:
    """Merge-equivalence tests against a real compiled LangGraph graph.

    These run the actual debug code path in ``_run_graph`` (no hand-mocked
    stream output), so they fail if the debug path's final-state assembly
    diverges from LangGraph's reducer semantics — the bug in the first
    implementation of #64, where last-write-wins merging of "updates"
    deltas collapsed the ``add_messages``-reduced ``messages`` channel.
    """

    def _make_ta_with_real_graph(self):
        ta = TradingAgentsGraph(selected_analysts=["market", "social"], debug=True)
        ta.graph = _build_mini_graph()
        ta.memory_log = MagicMock()
        ta.memory_log.get_past_context.return_value = ""
        ta._memory_client = MagicMock()
        ta.propagator = MagicMock()
        ta.propagator.create_initial_state.return_value = dict(_MINI_INIT_STATE)
        ta.propagator.get_graph_args.return_value = {
            "stream_mode": "values",
            "config": {"recursion_limit": 100},
        }
        ta._log_state = MagicMock()
        ta.process_signal = MagicMock(return_value="BUY")
        return ta

    def test_debug_final_state_equals_invoke(self, mock_llm_client):
        """The debug path's final state must be identical to graph.invoke()'s
        output for the same graph and initial state."""
        ta = self._make_ta_with_real_graph()

        expected = _build_mini_graph().invoke(dict(_MINI_INIT_STATE))

        output = io.StringIO()
        with redirect_stdout(output):
            final_state, _ = ta._run_graph("AAPL", "2024-01-15")

        # Scalar (last-write-wins) fields.
        assert final_state["market_report"] == expected["market_report"]
        assert final_state["final_trade_decision"] == expected["final_trade_decision"]

        # Reducer-managed messages channel: same accumulated sequence as
        # invoke() (compare type + content; message ids differ across runs).
        assert [
            (type(m).__name__, m.content) for m in final_state["messages"]
        ] == [(type(m).__name__, m.content) for m in expected["messages"]]
        # Guard against the original bug explicitly: invoke() accumulates
        # all four messages; a last-write-wins merge would keep only one.
        assert len(expected["messages"]) == 4
        assert len(final_state["messages"]) == 4

        # Same key set overall.
        assert set(final_state.keys()) == set(expected.keys())

    def test_debug_output_labels_nodes_on_real_graph(self, mock_llm_client):
        """Node-name headers appear for each message-producing node when
        streaming a real graph."""
        ta = self._make_ta_with_real_graph()

        output = io.StringIO()
        with redirect_stdout(output):
            ta._run_graph("AAPL", "2024-01-15")

        output_text = output.getvalue()
        assert "── Market Analyst ──" in output_text
        assert "── Msg Clear Market ──" in output_text
        assert "── Portfolio Manager ──" in output_text
