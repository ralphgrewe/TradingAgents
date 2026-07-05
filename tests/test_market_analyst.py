"""Tests for the market analyst node (issue #18): the fork's structured-JSON
indicator output must survive the merge with upstream's instrument-identity /
verified-market-snapshot anti-hallucination grounding.
"""

from unittest.mock import MagicMock, patch

import pytest

from tradingagents.agents.analysts.market_analyst import create_market_analyst


def _make_state(**overrides):
    state = {
        "trade_date": "2026-05-13",
        "company_of_interest": "AAPL",
        "messages": [],
    }
    state.update(overrides)
    return state


def _invoke_node_and_capture_prompt(state, response=None):
    """Run market_analyst_node with a fully-mocked LLM and capture the
    rendered ChatPromptValue passed to the (mocked) tool-bound LLM, plus the
    node's return value.
    """
    if response is None:
        response = MagicMock(tool_calls=[], content="report text")

    captured = {}

    def fake_bound_call(prompt_value):
        captured["prompt_value"] = prompt_value
        return response

    bound = MagicMock(side_effect=fake_bound_call)
    llm = MagicMock()
    llm.bind_tools.return_value = bound

    node = create_market_analyst(llm)
    result = node(state)
    return result, captured["prompt_value"], llm


@pytest.mark.unit
class TestMarketAnalystStructuredJsonContract:
    """The fork's downstream stages parse a strict JSON array of indicators
    out of the report — that contract must be untouched by the merge."""

    def test_prompt_still_requires_pure_json_indicator_array(self):
        _, prompt_value, _ = _invoke_node_and_capture_prompt(_make_state())
        text = prompt_value.to_string()
        assert "pure JSON array" in text
        assert '"indicator": "<tool_name>"' in text
        assert '"trend": "Rising" | "Falling" | "Flat"' in text
        assert '"signal": "Bullish" | "Bearish" | "Neutral"' in text

    def test_report_passthrough_when_no_tool_calls(self):
        response = MagicMock(tool_calls=[], content="## 1. Market Context\n...")
        result, _, _ = _invoke_node_and_capture_prompt(_make_state(), response=response)
        assert result["market_report"] == "## 1. Market Context\n..."

    def test_report_empty_when_tool_calls_pending(self):
        response = MagicMock(tool_calls=[{"name": "get_stock_data", "args": {}, "id": "1"}])
        result, _, _ = _invoke_node_and_capture_prompt(_make_state(), response=response)
        assert result["market_report"] == ""

    def test_prompt_renders_without_format_collision(self):
        """Regression guard: the JSON example's literal `{`/`}` braces used to
        break the old str.format()-based substitution (KeyError on the first
        JSON key) whenever the node actually ran."""
        result, prompt_value, _ = _invoke_node_and_capture_prompt(_make_state())
        assert result["market_report"] == "report text"
        assert "AAPL" in prompt_value.to_string()


@pytest.mark.unit
class TestMarketAnalystGrounding:
    """Anti-hallucination grounding grafted in from upstream d7b40a2/47cbb32."""

    def test_verified_snapshot_tool_is_bound(self):
        _, _, llm = _invoke_node_and_capture_prompt(_make_state())
        bound_tools = llm.bind_tools.call_args[0][0]
        tool_names = {t.name for t in bound_tools}
        assert "get_verified_market_snapshot" in tool_names
        assert {"get_stock_data", "get_indicators"} <= tool_names

    def test_prompt_instructs_verified_snapshot_as_source_of_truth(self):
        _, prompt_value, _ = _invoke_node_and_capture_prompt(_make_state())
        text = prompt_value.to_string()
        assert "get_verified_market_snapshot" in text
        assert "source of truth" in text
        assert "flag the discrepancy" in text

    def test_prompt_forbids_unsupported_historical_claims(self):
        _, prompt_value, _ = _invoke_node_and_capture_prompt(_make_state())
        text = prompt_value.to_string()
        assert "historical validation" in text
        assert "Never substitute a different company" in text

    def test_instrument_context_uses_resolved_identity(self):
        with patch("tradingagents.agents.utils.agent_utils.yf.Ticker") as mock_ticker:
            mock_ticker.return_value.info = {
                "longName": "Apple Inc.",
                "sector": "Technology",
            }
            _, prompt_value, _ = _invoke_node_and_capture_prompt(_make_state())
        text = prompt_value.to_string()
        assert "Apple Inc." in text
        assert "Do not substitute a different company" in text

    def test_instrument_context_falls_back_gracefully_offline(self):
        # yfinance failing must never break the analyst (fail-open).
        with patch(
            "tradingagents.agents.utils.agent_utils.yf.Ticker",
            side_effect=RuntimeError("offline"),
        ):
            result, prompt_value, _ = _invoke_node_and_capture_prompt(_make_state())
        assert result["market_report"] == "report text"
        assert "AAPL" in prompt_value.to_string()

    def test_prefers_precomputed_instrument_context(self):
        with patch("tradingagents.agents.utils.agent_utils.yf.Ticker") as mock_ticker:
            _, prompt_value, _ = _invoke_node_and_capture_prompt(
                _make_state(instrument_context="PRECOMPUTED CONTEXT MARKER")
            )
        mock_ticker.assert_not_called()
        assert "PRECOMPUTED CONTEXT MARKER" in prompt_value.to_string()
