"""Tests for run_structured_with_tools fallback behavior (issue #152).

Tests the free-text fallback when structured output fails, ensuring that:
1. The model's real answer from the tool loop is reused when available
2. List-shaped content is normalized to strings
3. Fresh invoke happens only when needed
4. The documented invariant holds: exactly one of structured_result/fallback_text is non-None
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from tradingagents.agents.schemas import PortfolioDecision, SwingDecision
from tradingagents.agents.utils.structured import (
    _generate_repair_instruction,
    _normalize_content,
    run_structured_with_tools,
)

pytestmark = pytest.mark.unit


class SimpleResponse(BaseModel):
    """Simple schema for testing structured output."""

    decision: str
    confidence: float


class TestNormalizeContent:
    """Tests for _normalize_content helper."""

    def test_string_content_returned_as_is(self):
        """Plain string content is returned unchanged."""
        assert _normalize_content("Hello, world!") == "Hello, world!"

    def test_empty_string_returned_as_is(self):
        """Empty string is returned as empty."""
        assert _normalize_content("") == ""

    def test_list_with_strings_concatenated(self):
        """List of strings is joined together."""
        content = ["Hello", " ", "world"]
        assert _normalize_content(content) == "Hello world"

    def test_list_with_dict_blocks_extracts_text(self):
        """List with dict blocks extracts 'text' field."""
        content = [
            {"type": "text", "text": "Hello"},
            {"type": "text", "text": " "},
            {"type": "text", "text": "world"},
        ]
        assert _normalize_content(content) == "Hello world"

    def test_list_with_mixed_strings_and_dicts(self):
        """List with mixed strings and dicts extracts text from dicts."""
        content = [
            "Hello",
            {"type": "text", "text": " "},
            "world",
            {"type": "text", "text": "!"},
        ]
        assert _normalize_content(content) == "Hello world!"

    def test_list_with_blocks_without_text_field_skipped(self):
        """Dict blocks without 'text' field are skipped."""
        content = [
            "Start",
            {"type": "image", "url": "..."},
            {"type": "text", "text": "End"},
        ]
        assert _normalize_content(content) == "StartEnd"

    def test_list_with_empty_text_blocks_skipped(self):
        """Dict blocks with non-string 'text' field are skipped."""
        content = [
            "A",
            {"type": "text", "text": None},
            {"type": "text", "text": 123},
            "B",
        ]
        assert _normalize_content(content) == "AB"

    def test_non_string_non_list_content_converted_to_string(self):
        """Non-string, non-list content is converted via str()."""
        assert _normalize_content(12345) == "12345"
        assert _normalize_content(None) == "None"


class TestRunStructuredWithToolsFallbackPriority1:
    """Tests for priority 1 fallback: reusing AIMessage from trace."""

    def test_aimessage_with_string_content_reused(self, caplog):
        """AIMessage with string content is reused without extra invoke."""
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm

        # Structured output will fail
        def _with_structured_output(schema):
            structured = MagicMock()
            structured.invoke.side_effect = ValueError("Parsing failed")
            return structured

        mock_llm.with_structured_output = _with_structured_output

        invoke_calls = []

        def mock_invoke(msg_list):
            invoke_calls.append(msg_list)
            # Return AIMessage with content on first invoke (the loop)
            return AIMessage(content="My final analysis: Buy.", tool_calls=[])

        mock_llm.invoke = mock_invoke

        messages = [HumanMessage(content="Test")]
        with caplog.at_level(logging.DEBUG):
            result, fallback_text, trace = run_structured_with_tools(
                mock_llm,
                messages,
                [],
                SimpleResponse,
                max_rounds=1,
                agent_name="TestAgent",
            )

        # Structured failed, so fallback should be set
        assert result is None
        # Fallback should be the AIMessage's content
        assert fallback_text == "My final analysis: Buy."
        # invoke should only be called once (the loop), not for fallback
        assert len(invoke_calls) == 1
        # Check log message
        assert "reusing model's final AIMessage" in caplog.text

    def test_aimessage_with_list_content_normalized_and_reused(self, caplog):
        """AIMessage with list-shaped content is normalized and reused."""
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm

        # Structured output will fail
        def _with_structured_output(schema):
            structured = MagicMock()
            structured.invoke.side_effect = ValueError("Parsing failed")
            return structured

        mock_llm.with_structured_output = _with_structured_output

        invoke_calls = []

        def mock_invoke(msg_list):
            invoke_calls.append(msg_list)
            # Return AIMessage with list content on first invoke (the loop)
            list_content = [
                {"type": "text", "text": "Analysis: "},
                {"type": "text", "text": "Buy at 100."},
            ]
            return AIMessage(content=list_content, tool_calls=[])

        mock_llm.invoke = mock_invoke

        messages = [HumanMessage(content="Test")]
        with caplog.at_level(logging.DEBUG):
            result, fallback_text, trace = run_structured_with_tools(
                mock_llm,
                messages,
                [],
                SimpleResponse,
                max_rounds=1,
                agent_name="TestAgent",
            )

        # Structured failed, so fallback should be set
        assert result is None
        # Fallback should be the normalized list content
        assert fallback_text == "Analysis: Buy at 100."
        # invoke should only be called once (the loop), not for fallback
        assert len(invoke_calls) == 1
        # Check log message
        assert "reusing model's final AIMessage" in caplog.text


class TestRunStructuredWithToolsFallbackPriority2:
    """Tests for priority 2 fallback: invoking LLM when needed."""

    def test_trace_ending_in_toolmessage_invokes_fallback(self, caplog):
        """When trace ends in ToolMessage, invoke LLM for fallback."""
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm

        # Structured output will fail
        def _with_structured_output(schema):
            structured = MagicMock()
            structured.invoke.side_effect = ValueError("Parsing failed")
            return structured

        mock_llm.with_structured_output = _with_structured_output

        # Track invoke calls
        invoke_calls = []

        def mock_invoke(msg_list):
            invoke_calls.append(msg_list)
            # First call: return AIMessage with tool call (to continue the loop)
            if len(invoke_calls) == 1:
                return AIMessage(
                    content="Checking data...",
                    tool_calls=[{"id": "1", "name": "search", "args": {"q": "data"}}],
                )
            # Second call: this is the fallback (should happen)
            else:
                return MagicMock(content="Fallback response.")

        mock_llm.invoke = mock_invoke

        # Create tools to make the loop work
        mock_tool = MagicMock()
        mock_tool.name = "search"
        mock_tool.invoke.return_value = "Search result"

        messages = [HumanMessage(content="Test")]
        with caplog.at_level(logging.DEBUG):
            result, fallback_text, trace = run_structured_with_tools(
                mock_llm,
                messages,
                [mock_tool],
                SimpleResponse,
                max_rounds=1,
                agent_name="TestAgent",
            )

        # Check invariant
        assert (result is None) != (fallback_text is None)
        assert result is None
        # Since trace ends in ToolMessage, fallback invoke should have been called
        assert fallback_text == "Fallback response."
        # Should have two invoke calls: one in loop, one for fallback
        assert len(invoke_calls) == 2
        # Check log for fallback path
        assert "fallback invoked fresh LLM response" in caplog.text

    def test_trace_ending_in_empty_aimessage_invokes_fallback(self, caplog):
        """When trace ends in empty AIMessage, invoke LLM for fallback."""
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm

        # Structured output will fail
        def _with_structured_output(schema):
            structured = MagicMock()
            structured.invoke.side_effect = ValueError("Parsing failed")
            return structured

        mock_llm.with_structured_output = _with_structured_output

        invoke_calls = []

        def mock_invoke(msg_list):
            invoke_calls.append(msg_list)
            if len(invoke_calls) == 1:
                # Loop returns empty AIMessage
                return AIMessage(content="", tool_calls=[])
            else:
                # Fallback invoke
                return MagicMock(content="Fallback content.")

        mock_llm.invoke = mock_invoke

        messages = [HumanMessage(content="Test")]
        with caplog.at_level(logging.DEBUG):
            result, fallback_text, trace = run_structured_with_tools(
                mock_llm,
                messages,
                [],
                SimpleResponse,
                max_rounds=1,
                agent_name="TestAgent",
            )

        assert result is None
        assert fallback_text == "Fallback content."
        # Two invokes: loop and fallback
        assert len(invoke_calls) == 2
        assert "fallback invoked fresh LLM response" in caplog.text

    def test_trace_ending_in_whitespace_only_aimessage_invokes_fallback(self, caplog):
        """Whitespace-only AIMessage content must not be treated as reusable.

        Plain Python truthiness on ``.content`` treats a string like " " or
        "\\n" as truthy, which would incorrectly reuse it as the "real"
        answer. This must fall through to the fresh llm.invoke fallback,
        exactly like fully-empty content does.
        """
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm

        # Structured output will fail
        def _with_structured_output(schema):
            structured = MagicMock()
            structured.invoke.side_effect = ValueError("Parsing failed")
            return structured

        mock_llm.with_structured_output = _with_structured_output

        invoke_calls = []

        def mock_invoke(msg_list):
            invoke_calls.append(msg_list)
            if len(invoke_calls) == 1:
                # Loop returns whitespace-only AIMessage
                return AIMessage(content="   \n", tool_calls=[])
            else:
                # Fallback invoke
                return MagicMock(content="Fallback content.")

        mock_llm.invoke = mock_invoke

        messages = [HumanMessage(content="Test")]
        with caplog.at_level(logging.DEBUG):
            result, fallback_text, trace = run_structured_with_tools(
                mock_llm,
                messages,
                [],
                SimpleResponse,
                max_rounds=1,
                agent_name="TestAgent",
            )

        assert result is None
        assert fallback_text == "Fallback content."
        # Two invokes: loop and fallback -- the whitespace-only content must
        # NOT be reused directly.
        assert len(invoke_calls) == 2
        assert "fallback invoked fresh LLM response" in caplog.text

    def test_max_rounds_zero_still_uses_fallback_invoke(self):
        """max_rounds=0 (no loop) still gets fallback invoke, not trace[-1]."""
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm

        # Structured output will fail
        def _with_structured_output(schema):
            structured = MagicMock()
            structured.invoke.side_effect = ValueError("Parsing failed")
            return structured

        mock_llm.with_structured_output = _with_structured_output

        invoke_calls = []

        def mock_invoke(msg_list):
            invoke_calls.append(msg_list)
            # With max_rounds=0, no loop iteration, so first invoke is the fallback
            return MagicMock(content="Fallback response.")

        mock_llm.invoke = mock_invoke

        messages = [HumanMessage(content="Test")]
        result, fallback_text, trace = run_structured_with_tools(
            mock_llm,
            messages,
            [],
            SimpleResponse,
            max_rounds=0,
            agent_name="TestAgent",
        )

        assert result is None
        assert fallback_text == "Fallback response."
        # One invoke: the fallback (no loop iterations)
        assert len(invoke_calls) == 1
        # The loop contributed nothing, so the trace is the initial
        # HumanMessage plus the one schema-repair instruction the retry sent
        # (issue #153) -- the returned trace records everything that was sent.
        assert len(trace) == 2
        assert isinstance(trace[0], HumanMessage)
        assert trace[0].content == "Test"
        assert "Reply with ONLY valid JSON" in trace[1].content
        # The fallback itself ran on the pre-repair trace (see
        # run_structured_with_tools: appending the repair instruction must not
        # disturb the #152 "reuse the final AIMessage" decision).
        assert invoke_calls[0] == trace[:-1]


class TestRunStructuredWithToolsContentNormalization:
    """Tests for proper normalization of content shapes."""

    def test_fallback_normalizes_string_content(self):
        """Fallback response with string content is normalized."""
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm

        def _with_structured_output(schema):
            structured = MagicMock()
            structured.invoke.side_effect = ValueError("Parsing failed")
            return structured

        mock_llm.with_structured_output = _with_structured_output

        def mock_invoke(msg_list):
            return MagicMock(content="String response.")

        mock_llm.invoke = mock_invoke

        messages = [HumanMessage(content="Test")]
        result, fallback_text, trace = run_structured_with_tools(
            mock_llm,
            messages,
            [],
            SimpleResponse,
            max_rounds=1,
            agent_name="TestAgent",
        )

        assert fallback_text == "String response."

    def test_fallback_normalizes_list_content(self):
        """Fallback response with list content is normalized."""
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm

        def _with_structured_output(schema):
            structured = MagicMock()
            structured.invoke.side_effect = ValueError("Parsing failed")
            return structured

        mock_llm.with_structured_output = _with_structured_output

        def mock_invoke(msg_list):
            list_content = [
                {"type": "text", "text": "Part 1 "},
                {"type": "text", "text": "Part 2"},
            ]
            return MagicMock(content=list_content)

        mock_llm.invoke = mock_invoke

        messages = [HumanMessage(content="Test")]
        result, fallback_text, trace = run_structured_with_tools(
            mock_llm,
            messages,
            [],
            SimpleResponse,
            max_rounds=1,
            agent_name="TestAgent",
        )

        assert fallback_text == "Part 1 Part 2"


class TestRunStructuredWithToolsInvariant:
    """Tests for the documented invariant: exactly one of result/fallback is non-None."""

    def test_structured_success_leaves_fallback_none(self):
        """When structured succeeds, fallback is None."""
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm

        def _with_structured_output(schema):
            structured = MagicMock()
            structured.invoke.return_value = SimpleResponse(
                decision="Buy",
                confidence=0.95,
            )
            return structured

        mock_llm.with_structured_output = _with_structured_output

        messages = [HumanMessage(content="Test")]
        result, fallback_text, trace = run_structured_with_tools(
            mock_llm,
            messages,
            [],
            SimpleResponse,
            max_rounds=1,
            agent_name="TestAgent",
        )

        assert result is not None
        assert fallback_text is None

    def test_structured_failure_gives_fallback(self):
        """When structured fails, fallback is non-None."""
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm

        def _with_structured_output(schema):
            structured = MagicMock()
            structured.invoke.side_effect = ValueError("Failed")
            return structured

        mock_llm.with_structured_output = _with_structured_output

        def mock_invoke(msg_list):
            return MagicMock(content="Fallback.")

        mock_llm.invoke = mock_invoke

        messages = [HumanMessage(content="Test")]
        result, fallback_text, trace = run_structured_with_tools(
            mock_llm,
            messages,
            [],
            SimpleResponse,
            max_rounds=1,
            agent_name="TestAgent",
        )

        assert result is None
        assert fallback_text is not None

    def test_unsupported_structured_output_gives_fallback(self):
        """When structured output not supported, fallback is non-None."""
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_llm.with_structured_output.side_effect = NotImplementedError("Not supported")

        def mock_invoke(msg_list):
            return MagicMock(content="Fallback.")

        mock_llm.invoke = mock_invoke

        messages = [HumanMessage(content="Test")]
        result, fallback_text, trace = run_structured_with_tools(
            mock_llm,
            messages,
            [],
            SimpleResponse,
            max_rounds=1,
            agent_name="TestAgent",
        )

        assert result is None
        assert fallback_text is not None


class TestStructuredOutputRepairRetry:
    """Tests for structured output repair retry (issue #153)."""

    def test_first_attempt_success_no_retry(self, caplog):
        """When first attempt succeeds, no retry happens."""
        from tradingagents.dataflows.config import get_config, set_config
        config = get_config().copy()
        config["structured_output_repair_retry"] = True
        set_config(config)

        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm

        structured_calls = []

        def _with_structured_output(schema):
            structured = MagicMock()

            def mock_invoke(msg_list):
                structured_calls.append(msg_list)
                return SimpleResponse(decision="buy", confidence=0.9)

            structured.invoke = mock_invoke
            return structured

        mock_llm.with_structured_output = _with_structured_output

        messages = [HumanMessage(content="Test")]
        with caplog.at_level(logging.DEBUG):
            result, fallback_text, trace = run_structured_with_tools(
                mock_llm,
                messages,
                [],
                SimpleResponse,
                max_rounds=1,
                agent_name="TestAgent",
            )

        assert result is not None
        assert result.decision == "buy"
        assert fallback_text is None
        assert len(structured_calls) == 1

    def test_first_fails_retry_succeeds(self, caplog):
        """When first fails and retry succeeds, return the retry result."""
        from tradingagents.dataflows.config import get_config, set_config
        config = get_config().copy()
        config["structured_output_repair_retry"] = True
        set_config(config)

        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm

        structured_calls = []

        def _with_structured_output(schema):
            structured = MagicMock()

            def mock_invoke(msg_list):
                structured_calls.append(msg_list)
                if len(structured_calls) == 1:
                    raise ValueError("Invalid JSON in structured output")
                else:
                    return SimpleResponse(decision="hold", confidence=0.7)

            structured.invoke = mock_invoke
            return structured

        mock_llm.with_structured_output = _with_structured_output

        messages = [HumanMessage(content="Test")]
        with caplog.at_level(logging.WARNING):
            result, fallback_text, trace = run_structured_with_tools(
                mock_llm,
                messages,
                [],
                SimpleResponse,
                max_rounds=1,
                agent_name="TestAgent",
            )

        assert result is not None
        assert result.decision == "hold"
        assert fallback_text is None
        assert len(structured_calls) == 2
        assert "retrying once with schema-repair instruction" in caplog.text
        assert "structured output retry succeeded" in caplog.text

    def test_both_attempts_fail_fallback_used(self, caplog):
        """When both attempts fail, fallback to free text."""
        from tradingagents.dataflows.config import get_config, set_config
        config = get_config().copy()
        config["structured_output_repair_retry"] = True
        set_config(config)

        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm

        structured_calls = []

        def _with_structured_output(schema):
            structured = MagicMock()

            def mock_invoke(msg_list):
                structured_calls.append(msg_list)
                raise ValueError("Invalid JSON in structured output")

            structured.invoke = mock_invoke
            return structured

        mock_llm.with_structured_output = _with_structured_output

        def mock_fallback_invoke(msg_list):
            return MagicMock(content="Fallback response after both retry attempts.")

        mock_llm.invoke = mock_fallback_invoke

        messages = [HumanMessage(content="Test")]
        with caplog.at_level(logging.WARNING):
            result, fallback_text, trace = run_structured_with_tools(
                mock_llm,
                messages,
                [],
                SimpleResponse,
                max_rounds=1,
                agent_name="TestAgent",
            )

        assert result is None
        assert fallback_text == "Fallback response after both retry attempts."
        assert len(structured_calls) == 2
        assert "retrying once with schema-repair instruction" in caplog.text
        assert "structured output retry also failed" in caplog.text

    def test_retry_disabled_by_config(self, caplog):
        """When retry is disabled by config, no retry happens."""
        from tradingagents.dataflows.config import get_config, set_config
        config = get_config().copy()
        config["structured_output_repair_retry"] = False
        set_config(config)

        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm

        structured_calls = []

        def _with_structured_output(schema):
            structured = MagicMock()

            def mock_invoke(msg_list):
                structured_calls.append(msg_list)
                raise ValueError("Invalid JSON in structured output")

            structured.invoke = mock_invoke
            return structured

        mock_llm.with_structured_output = _with_structured_output

        def mock_fallback_invoke(msg_list):
            return MagicMock(content="Fallback response (no retry).")

        mock_llm.invoke = mock_fallback_invoke

        messages = [HumanMessage(content="Test")]
        with caplog.at_level(logging.WARNING):
            result, fallback_text, trace = run_structured_with_tools(
                mock_llm,
                messages,
                [],
                SimpleResponse,
                max_rounds=1,
                agent_name="TestAgent",
            )

        assert result is None
        assert fallback_text == "Fallback response (no retry)."
        assert len(structured_calls) == 1
        assert "retry disabled" in caplog.text

    def test_structured_llm_none_skips_retry(self, caplog):
        """When structured_llm is None, no retry is attempted."""
        from tradingagents.dataflows.config import get_config, set_config
        config = get_config().copy()
        config["structured_output_repair_retry"] = True
        set_config(config)

        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = mock_llm
        mock_llm.with_structured_output = MagicMock(return_value=None)

        def mock_fallback_invoke(msg_list):
            return MagicMock(content="Fallback (no structured support).")

        mock_llm.invoke = mock_fallback_invoke

        messages = [HumanMessage(content="Test")]
        with caplog.at_level(logging.DEBUG):
            result, fallback_text, trace = run_structured_with_tools(
                mock_llm,
                messages,
                [],
                SimpleResponse,
                max_rounds=1,
                agent_name="TestAgent",
            )

        assert result is None
        assert fallback_text == "Fallback (no structured support)."
        assert "structured output not supported" in caplog.text


class TestRepairInstructionRendering:
    """Tests for _generate_repair_instruction's rendered content (issue #153).

    The instruction is sent verbatim to a model that just failed to produce
    parseable structured output, so it has to read as instructions -- above all
    on the enum fields (``rating`` / ``action``), which are both the most
    consequential and the most likely to be malformed. The first implementation
    stringified ``field_info.annotation``, which rendered "<enum
    'PortfolioRating'>" and named none of the legal values.
    """

    def test_portfolio_decision_lists_legal_rating_values(self):
        """Every PortfolioRating member is named, so the model can pick one."""
        instruction = _generate_repair_instruction(PortfolioDecision)

        rating_line = next(
            line for line in instruction.splitlines() if line.startswith("- rating ")
        )
        for legal_value in ("Buy", "Overweight", "Hold", "Underweight", "Sell"):
            assert f'"{legal_value}"' in rating_line

    def test_swing_decision_lists_legal_action_values(self):
        """Every SwingAction member is named, so the model can pick one."""
        instruction = _generate_repair_instruction(SwingDecision)

        action_line = next(
            line for line in instruction.splitlines() if line.startswith("- action ")
        )
        for legal_value in ("Buy", "Hold", "Sell"):
            assert f'"{legal_value}"' in action_line

    @pytest.mark.parametrize("model", [PortfolioDecision, SwingDecision])
    def test_no_raw_python_reprs(self, model):
        """No "<class 'float'>" / "<enum 'SwingAction'>" leaks into the prompt."""
        instruction = _generate_repair_instruction(model)

        assert "<class" not in instruction
        assert "<enum" not in instruction
        assert "typing." not in instruction

    @pytest.mark.parametrize("model", [PortfolioDecision, SwingDecision])
    def test_field_descriptions_are_carried_through(self, model):
        """Field descriptions -- which *are* the output instructions per
        agents/schemas.py -- appear in the repair instruction."""
        instruction = _generate_repair_instruction(model)

        for field_name, field_info in model.model_fields.items():
            description = field_info.description
            if not description:
                continue
            # Descriptions are whitespace-collapsed onto one line.
            assert " ".join(description.split()) in instruction, field_name

    def test_required_and_optional_fields_are_distinguished(self):
        """Optional fields are marked optional so the model doesn't invent values."""
        instruction = _generate_repair_instruction(PortfolioDecision)

        assert "- executive_summary (required, string)" in instruction
        assert "- price_target (optional, number or null)" in instruction

    def test_numeric_bounds_are_rendered(self):
        """Constrained numbers carry their bounds (SwingDecision.conviction)."""
        instruction = _generate_repair_instruction(SwingDecision)

        assert "- conviction (required, number, >= 0.0, <= 1.0)" in instruction

    def test_instruction_demands_json_only(self):
        """The instruction still ends with the JSON-only demand."""
        instruction = _generate_repair_instruction(PortfolioDecision)

        assert instruction.rstrip().endswith(
            "Reply with ONLY valid JSON, no prose or explanation."
        )
