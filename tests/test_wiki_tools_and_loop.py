"""Tests for wiki_tools.search_strategy_wiki and run_structured_with_tools helper.

Part of issue #104 (LLM-wiki agent-callable search tool + shared tool-loop helper).
"""

import unittest
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel, Field

from tradingagents.agents.utils import wiki_tools
from tradingagents.agents.utils.structured import run_structured_with_tools


class MockDecision(BaseModel):
    """Mock structured response for testing."""

    action: str = Field(description="BUY, HOLD, or SELL")
    confidence: float = Field(description="Confidence score 0-1")
    rationale: str = Field(description="Brief rationale")


@pytest.mark.unit
class WikiToolsTests(unittest.TestCase):
    """Tests for search_strategy_wiki tool."""

    def test_search_returns_formatted_markdown(self):
        """Tool returns correctly formatted markdown string."""
        with patch("tradingagents.agents.utils.wiki_tools.route_to_vendor") as mock_vendor:
            mock_vendor.return_value = [
                {
                    "id": "momentum-factor",
                    "title": "Momentum Factor",
                    "tags": ["momentum", "technical", "trend-following"],
                    "section": None,
                    "source": {
                        "authors": "Jegadeesh, Titman",
                        "title": "Returns to Buying Winners and Selling Losers",
                        "year": 1993,
                    },
                    "score": 0.85,
                }
            ]

            result = wiki_tools.search_strategy_wiki.invoke({"query": "momentum trend", "k": 1})

            # Check that result is markdown string
            self.assertIsInstance(result, str)
            self.assertIn("Strategy Knowledge Base Results", result)
            self.assertIn("momentum-factor", result)
            self.assertIn("Momentum Factor", result)
            self.assertIn("momentum, technical, trend-following", result)
            self.assertIn("Jegadeesh, Titman", result)
            self.assertIn("Returns to Buying Winners and Selling Losers", result)
            self.assertIn("1993", result)
            self.assertIn("0.850", result)  # Score formatted to 3 decimals

    def test_search_with_empty_results(self):
        """Tool returns appropriate message when no results found."""
        with patch("tradingagents.agents.utils.wiki_tools.route_to_vendor") as mock_vendor:
            mock_vendor.return_value = []

            result = wiki_tools.search_strategy_wiki.invoke({"query": "nonexistent query", "k": 3})

            self.assertIsInstance(result, str)
            self.assertIn("No matching", result)

    def test_search_calls_route_to_vendor_correctly(self):
        """Tool calls route_to_vendor with correct arguments."""
        with patch("tradingagents.agents.utils.wiki_tools.route_to_vendor") as mock_vendor:
            mock_vendor.return_value = []

            wiki_tools.search_strategy_wiki.invoke({"query": "test query", "k": 5})

            mock_vendor.assert_called_once_with("search_wiki", "test query", 5)

    def test_search_handles_source_as_dict(self):
        """Tool formats source dict correctly (authors, title, year)."""
        with patch("tradingagents.agents.utils.wiki_tools.route_to_vendor") as mock_vendor:
            mock_vendor.return_value = [
                {
                    "id": "test-id",
                    "title": "Test Article",
                    "tags": ["test"],
                    "section": None,
                    "source": {"authors": "Author A", "title": "Title", "year": 2020},
                    "score": 0.5,
                }
            ]

            result = wiki_tools.search_strategy_wiki.invoke({"query": "test", "k": 1})

            self.assertIn("Author A", result)
            self.assertIn("Title", result)
            self.assertIn("2020", result)

    def test_search_handles_missing_source(self):
        """Tool handles missing source gracefully."""
        with patch("tradingagents.agents.utils.wiki_tools.route_to_vendor") as mock_vendor:
            mock_vendor.return_value = [
                {
                    "id": "test-id",
                    "title": "Test Article",
                    "tags": ["test"],
                    "section": None,
                    "source": None,
                    "score": 0.5,
                }
            ]

            result = wiki_tools.search_strategy_wiki.invoke({"query": "test", "k": 1})

            # Should not crash, and result should still contain article info
            self.assertIsInstance(result, str)
            self.assertIn("Test Article", result)

    def test_search_handles_missing_tags(self):
        """Tool handles empty tags list gracefully."""
        with patch("tradingagents.agents.utils.wiki_tools.route_to_vendor") as mock_vendor:
            mock_vendor.return_value = [
                {
                    "id": "test-id",
                    "title": "Test Article",
                    "tags": [],
                    "section": None,
                    "source": None,
                    "score": 0.5,
                }
            ]

            result = wiki_tools.search_strategy_wiki.invoke({"query": "test", "k": 1})

            # Should not crash, tags line should be skipped
            self.assertIsInstance(result, str)
            self.assertIn("Test Article", result)

    def test_search_default_k_is_3(self):
        """Tool uses k=3 as default."""
        with patch("tradingagents.agents.utils.wiki_tools.route_to_vendor") as mock_vendor:
            mock_vendor.return_value = []

            wiki_tools.search_strategy_wiki.invoke({"query": "test"})

            # Check that k=3 was passed to route_to_vendor
            mock_vendor.assert_called_once_with("search_wiki", "test", 3)


@pytest.mark.unit
class RunStructuredWithToolsTests(unittest.TestCase):
    """Tests for run_structured_with_tools helper."""

    def test_tool_loop_with_single_tool_call_then_structured(self):
        """Loop executes one tool call then returns structured output."""
        # Mock LLM that first calls a tool, then returns response
        mock_llm = MagicMock()
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.invoke = MagicMock(return_value="tool result")

        # First invocation returns message with tool_calls
        tool_call_response = AIMessage(
            content="Calling tool",
            tool_calls=[{"name": "test_tool", "args": {"input": "test"}, "id": "call_1"}],
        )

        # Second invocation returns final message (no more tool calls)
        final_response = AIMessage(content="Final response")

        # Bind tools returns an LLM that handles tool calls
        mock_llm_with_tools = MagicMock()
        mock_llm_with_tools.invoke = MagicMock(
            side_effect=[tool_call_response, final_response]
        )
        mock_llm.bind_tools = MagicMock(return_value=mock_llm_with_tools)

        # Structured LLM mock
        mock_structured_llm = MagicMock()
        mock_decision = MockDecision(action="BUY", confidence=0.8, rationale="Test")
        mock_structured_llm.invoke = MagicMock(return_value=mock_decision)

        with patch(
            "tradingagents.agents.utils.structured.bind_structured",
            return_value=mock_structured_llm,
        ):
            initial_messages = [HumanMessage(content="Make a decision")]
            result, trace = run_structured_with_tools(
                mock_llm,
                initial_messages,
                [mock_tool],
                MockDecision,
                max_rounds=2,
                agent_name="TestAgent",
            )

        # Check result
        self.assertIsNotNone(result)
        self.assertEqual(result.action, "BUY")
        self.assertEqual(result.confidence, 0.8)

        # Check trace includes initial message, tool call, tool result, and final response
        self.assertEqual(len(trace), 4)  # initial + tool_call + tool_msg + final
        self.assertIsInstance(trace[0], HumanMessage)
        self.assertIsInstance(trace[1], AIMessage)  # tool call response
        self.assertIsInstance(trace[2], ToolMessage)  # tool result
        self.assertIsInstance(trace[3], AIMessage)  # final response

    def test_tool_loop_terminates_at_max_rounds(self):
        """Loop terminates when max_rounds is reached, even if LLM keeps requesting tools."""
        mock_llm = MagicMock()
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.invoke = MagicMock(return_value="tool result")

        # Both invocations return messages with tool_calls (LLM keeps requesting)
        tool_call_response_1 = AIMessage(
            content="Calling tool round 1",
            tool_calls=[{"name": "test_tool", "args": {"input": "test1"}, "id": "call_1"}],
        )
        tool_call_response_2 = AIMessage(
            content="Calling tool round 2",
            tool_calls=[{"name": "test_tool", "args": {"input": "test2"}, "id": "call_2"}],
        )

        mock_llm_with_tools = MagicMock()
        mock_llm_with_tools.invoke = MagicMock(
            side_effect=[tool_call_response_1, tool_call_response_2]
        )
        mock_llm.bind_tools = MagicMock(return_value=mock_llm_with_tools)

        mock_structured_llm = MagicMock()
        mock_decision = MockDecision(action="HOLD", confidence=0.5, rationale="Test")
        mock_structured_llm.invoke = MagicMock(return_value=mock_decision)

        with patch(
            "tradingagents.agents.utils.structured.bind_structured",
            return_value=mock_structured_llm,
        ):
            initial_messages = [HumanMessage(content="Make a decision")]
            result, trace = run_structured_with_tools(
                mock_llm,
                initial_messages,
                [mock_tool],
                MockDecision,
                max_rounds=2,
                agent_name="TestAgent",
            )

        # Should have executed exactly 2 rounds, not more
        self.assertEqual(mock_llm_with_tools.invoke.call_count, 2)

        # Check trace (initial + round1_response + tool_result + round2_response + tool_result)
        # Note: at max_rounds, we exit the loop, so we have initial + 2*invocations + 2*tool_results
        self.assertEqual(len(trace), 5)  # initial + response1 + tool1 + response2 + tool2_result

    def test_tool_loop_no_tools_requested(self):
        """Loop exits immediately if LLM doesn't request any tools."""
        mock_llm = MagicMock()
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"

        # LLM returns response with no tool calls
        final_response = AIMessage(content="No tools needed")
        final_response.tool_calls = []

        mock_llm_with_tools = MagicMock()
        mock_llm_with_tools.invoke = MagicMock(return_value=final_response)
        mock_llm.bind_tools = MagicMock(return_value=mock_llm_with_tools)

        mock_structured_llm = MagicMock()
        mock_decision = MockDecision(action="SELL", confidence=0.3, rationale="Test")
        mock_structured_llm.invoke = MagicMock(return_value=mock_decision)

        with patch(
            "tradingagents.agents.utils.structured.bind_structured",
            return_value=mock_structured_llm,
        ):
            initial_messages = [HumanMessage(content="Make a decision")]
            result, trace = run_structured_with_tools(
                mock_llm,
                initial_messages,
                [mock_tool],
                MockDecision,
                max_rounds=2,
                agent_name="TestAgent",
            )

        # Should only invoke once (no tool calls means exit loop)
        self.assertEqual(mock_llm_with_tools.invoke.call_count, 1)
        self.assertEqual(result.action, "SELL")

    def test_tool_loop_structured_output_fallback_on_failure(self):
        """When structured output fails, returns None and logs warning."""
        mock_llm = MagicMock()
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.invoke = MagicMock(return_value="tool result")

        # LLM first returns response with no tool calls
        final_response = AIMessage(content="Final response")
        final_response.tool_calls = []

        mock_llm_with_tools = MagicMock()
        mock_llm_with_tools.invoke = MagicMock(return_value=final_response)
        mock_llm.bind_tools = MagicMock(return_value=mock_llm_with_tools)

        # Structured LLM fails
        mock_structured_llm = MagicMock()
        mock_structured_llm.invoke = MagicMock(side_effect=Exception("JSON parsing failed"))

        with patch(
            "tradingagents.agents.utils.structured.bind_structured",
            return_value=mock_structured_llm,
        ):
            initial_messages = [HumanMessage(content="Make a decision")]
            with self.assertLogs("tradingagents.agents.utils.structured", level="WARNING"):
                result, trace = run_structured_with_tools(
                    mock_llm,
                    initial_messages,
                    [mock_tool],
                    MockDecision,
                    max_rounds=2,
                    agent_name="TestAgent",
                )

        # Structured output should be None due to failure
        self.assertIsNone(result)
        # Trace should still contain the messages
        self.assertGreater(len(trace), 0)

    def test_tool_loop_with_structured_output_unsupported(self):
        """When structured output is unsupported (None), returns None gracefully."""
        mock_llm = MagicMock()
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"

        final_response = AIMessage(content="Final response")
        final_response.tool_calls = []

        mock_llm_with_tools = MagicMock()
        mock_llm_with_tools.invoke = MagicMock(return_value=final_response)
        mock_llm.bind_tools = MagicMock(return_value=mock_llm_with_tools)

        # Simulate unsupported structured output
        with patch(
            "tradingagents.agents.utils.structured.bind_structured", return_value=None
        ):
            initial_messages = [HumanMessage(content="Make a decision")]
            result, trace = run_structured_with_tools(
                mock_llm,
                initial_messages,
                [mock_tool],
                MockDecision,
                max_rounds=2,
                agent_name="TestAgent",
            )

        # Structured output should be None
        self.assertIsNone(result)
        # But trace should still be returned
        self.assertGreater(len(trace), 0)

    def test_tool_execution_with_dict_input(self):
        """Tool execution handles dict-based input correctly."""
        mock_llm = MagicMock()
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.invoke = MagicMock(return_value="dict input result")

        tool_call_response = AIMessage(
            content="Calling tool",
            tool_calls=[
                {"name": "test_tool", "args": {"query": "test", "k": 5}, "id": "call_1"}
            ],
        )

        final_response = AIMessage(content="Done")
        final_response.tool_calls = []

        mock_llm_with_tools = MagicMock()
        mock_llm_with_tools.invoke = MagicMock(side_effect=[tool_call_response, final_response])
        mock_llm.bind_tools = MagicMock(return_value=mock_llm_with_tools)

        mock_structured_llm = MagicMock()
        mock_structured_llm.invoke = MagicMock(
            return_value=MockDecision(action="BUY", confidence=0.9, rationale="Test")
        )

        with patch(
            "tradingagents.agents.utils.structured.bind_structured",
            return_value=mock_structured_llm,
        ):
            initial_messages = [HumanMessage(content="Make a decision")]
            result, trace = run_structured_with_tools(
                mock_llm,
                initial_messages,
                [mock_tool],
                MockDecision,
                max_rounds=2,
                agent_name="TestAgent",
            )

        # Check that tool was invoked with the dict input
        mock_tool.invoke.assert_called_once_with({"query": "test", "k": 5})

    def test_tool_execution_failure_continues_loop(self):
        """Tool execution failure doesn't crash loop, continues to structured output."""
        mock_llm = MagicMock()
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.invoke = MagicMock(side_effect=Exception("Tool failed"))

        tool_call_response = AIMessage(
            content="Calling tool",
            tool_calls=[{"name": "test_tool", "args": {"input": "test"}, "id": "call_1"}],
        )

        final_response = AIMessage(content="Done despite tool failure")
        final_response.tool_calls = []

        mock_llm_with_tools = MagicMock()
        mock_llm_with_tools.invoke = MagicMock(side_effect=[tool_call_response, final_response])
        mock_llm.bind_tools = MagicMock(return_value=mock_llm_with_tools)

        mock_structured_llm = MagicMock()
        mock_structured_llm.invoke = MagicMock(
            return_value=MockDecision(action="HOLD", confidence=0.5, rationale="Test")
        )

        with patch(
            "tradingagents.agents.utils.structured.bind_structured",
            return_value=mock_structured_llm,
        ):
            initial_messages = [HumanMessage(content="Make a decision")]
            with self.assertLogs("tradingagents.agents.utils.structured", level="WARNING"):
                result, trace = run_structured_with_tools(
                    mock_llm,
                    initial_messages,
                    [mock_tool],
                    MockDecision,
                    max_rounds=2,
                    agent_name="TestAgent",
                )

        # Loop should continue and produce structured output despite tool failure
        self.assertIsNotNone(result)
        self.assertEqual(result.action, "HOLD")


if __name__ == "__main__":
    unittest.main()
