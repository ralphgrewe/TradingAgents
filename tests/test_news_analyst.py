"""Tests for the news analyst node (issue #32): aggregated per-article scoring +
JSON envelope report.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from tradingagents.agents.analysts.news_analyst import create_news_analyst


def _make_state(**overrides):
    state = {
        "trade_date": "2026-07-06",
        "company_of_interest": "AAPL",
        "messages": [],
    }
    state.update(overrides)
    return state


def _make_sample_news_output():
    """Create sample news analyst LLM output matching the expected JSON schema."""
    return {
        "articles_analyzed": 14,
        "window_days": 30,
        "top_positive_fundamentals": [
            "Q1 revenue beat estimates by 8%",
            "New AI partnership announced with major cloud vendor",
        ],
        "top_negative_fundamentals": [
            "Tariff overhang on hardware margins",
        ],
        "top_positive_sentiment": [
            "Analyst upgrades to Overweight on AI momentum",
        ],
        "top_negative_sentiment": [
            "Short report questions channel inventory",
        ],
        "fundamentals_counts": {
            "positive": {"count": 6, "avg_confidence": 0.72},
            "neutral": {"count": 5, "avg_confidence": 0.5},
            "negative": {"count": 3, "avg_confidence": 0.6},
        },
        "sentiment_counts": {
            "positive": {"count": 7, "avg_confidence": 0.68},
            "neutral": {"count": 4, "avg_confidence": 0.5},
            "negative": {"count": 3, "avg_confidence": 0.55},
        },
        "conservative": {"rating": "BUY", "confidence": 0.65},
        "risky": {"rating": "BUY", "confidence": 0.8},
    }


@pytest.mark.unit
class TestNewsAnalystJsonEnvelope:
    """Tests for JSON envelope output per issue #32."""

    def test_news_report_is_json_string(self):
        """news_report should contain JSON envelope, not markdown."""
        llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps(_make_sample_news_output())
        mock_llm = MagicMock(return_value=mock_response)
        llm.bind_tools.return_value = mock_llm

        with patch(
            "tradingagents.agents.analysts.news_analyst.get_instrument_context_from_state",
            return_value="Test context",
        ):
            node = create_news_analyst(llm)
            result = node(_make_state())

        assert "news_report" in result
        report_str = result["news_report"]

        # Should be valid JSON
        envelope = json.loads(report_str)
        assert isinstance(envelope, dict)

    def test_envelope_has_required_fields(self):
        """JSON envelope must have skill, ticker, date, signal, confidence, summary, details."""
        llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps(_make_sample_news_output())
        mock_llm = MagicMock(return_value=mock_response)
        llm.bind_tools.return_value = mock_llm

        with patch(
            "tradingagents.agents.analysts.news_analyst.get_instrument_context_from_state",
            return_value="Test context",
        ):
            node = create_news_analyst(llm)
            result = node(_make_state())

        envelope = json.loads(result["news_report"])
        assert envelope["skill"] == "financial-news-analyst"
        assert envelope["ticker"] == "AAPL"
        assert envelope["date"] == "2026-07-06"
        assert "signal" in envelope
        assert "confidence" in envelope
        assert "summary" in envelope
        assert "details" in envelope

    def test_envelope_details_has_required_structure(self):
        """details should contain articles_analyzed, window_days, counts, top items, ratings."""
        llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps(_make_sample_news_output())
        mock_llm = MagicMock(return_value=mock_response)
        llm.bind_tools.return_value = mock_llm

        with patch(
            "tradingagents.agents.analysts.news_analyst.get_instrument_context_from_state",
            return_value="Test context",
        ):
            node = create_news_analyst(llm)
            result = node(_make_state())

        envelope = json.loads(result["news_report"])
        details = envelope["details"]

        # Check required fields
        assert "articles_analyzed" in details
        assert "window_days" in details
        assert "top_positive_fundamentals" in details
        assert "top_negative_fundamentals" in details
        assert "top_positive_sentiment" in details
        assert "top_negative_sentiment" in details
        assert "fundamentals_counts" in details
        assert "sentiment_counts" in details
        assert "conservative" in details
        assert "risky" in details

        # Check structure of counts
        assert "positive" in details["fundamentals_counts"]
        assert "count" in details["fundamentals_counts"]["positive"]
        assert "avg_confidence" in details["fundamentals_counts"]["positive"]

    def test_signal_derived_from_conservative_rating(self):
        """Signal should be derived from conservative rating."""
        news_output = _make_sample_news_output()
        news_output["conservative"]["rating"] = "HOLD"
        news_output["risky"]["rating"] = "BUY"

        llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps(news_output)
        mock_llm = MagicMock(return_value=mock_response)
        llm.bind_tools.return_value = mock_llm

        with patch(
            "tradingagents.agents.analysts.news_analyst.get_instrument_context_from_state",
            return_value="Test context",
        ):
            node = create_news_analyst(llm)
            result = node(_make_state())

        envelope = json.loads(result["news_report"])
        # Signal should be from conservative, even though risky is BUY
        assert envelope["signal"] == "HOLD"

    def test_confidence_derived_from_mean_of_ratings(self):
        """Confidence mapping: HIGH if mean > 0.7, MEDIUM if > 0.4, else LOW."""
        # Test case: mean = 0.8 (both 0.8) → HIGH
        news_output = _make_sample_news_output()
        news_output["conservative"]["confidence"] = 0.8
        news_output["risky"]["confidence"] = 0.8

        llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps(news_output)
        mock_llm = MagicMock(return_value=mock_response)
        llm.bind_tools.return_value = mock_llm

        with patch(
            "tradingagents.agents.analysts.news_analyst.get_instrument_context_from_state",
            return_value="Test context",
        ):
            node = create_news_analyst(llm)
            result = node(_make_state())

        envelope = json.loads(result["news_report"])
        assert envelope["confidence"] == "HIGH"

    def test_confidence_medium_when_mean_between_0_4_and_0_7(self):
        """Confidence should be MEDIUM when 0.4 < mean ≤ 0.7."""
        news_output = _make_sample_news_output()
        news_output["conservative"]["confidence"] = 0.5
        news_output["risky"]["confidence"] = 0.6

        llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps(news_output)
        mock_llm = MagicMock(return_value=mock_response)
        llm.bind_tools.return_value = mock_llm

        with patch(
            "tradingagents.agents.analysts.news_analyst.get_instrument_context_from_state",
            return_value="Test context",
        ):
            node = create_news_analyst(llm)
            result = node(_make_state())

        envelope = json.loads(result["news_report"])
        assert envelope["confidence"] == "MEDIUM"

    def test_confidence_low_when_mean_below_0_4(self):
        """Confidence should be LOW when mean ≤ 0.4."""
        news_output = _make_sample_news_output()
        news_output["conservative"]["confidence"] = 0.3
        news_output["risky"]["confidence"] = 0.35

        llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps(news_output)
        mock_llm = MagicMock(return_value=mock_response)
        llm.bind_tools.return_value = mock_llm

        with patch(
            "tradingagents.agents.analysts.news_analyst.get_instrument_context_from_state",
            return_value="Test context",
        ):
            node = create_news_analyst(llm)
            result = node(_make_state())

        envelope = json.loads(result["news_report"])
        assert envelope["confidence"] == "LOW"

    def test_graceful_degradation_on_llm_json_parse_failure(self):
        """Should degrade gracefully when LLM returns invalid JSON."""
        llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "This is not valid JSON at all"
        mock_llm = MagicMock(return_value=mock_response)
        llm.bind_tools.return_value = mock_llm

        with patch(
            "tradingagents.agents.analysts.news_analyst.get_instrument_context_from_state",
            return_value="Test context",
        ):
            node = create_news_analyst(llm)
            result = node(_make_state())

        envelope = json.loads(result["news_report"])
        # Should still have valid envelope structure with defaults
        assert envelope["signal"] == "HOLD"
        assert envelope["confidence"] == "MEDIUM"
        assert "details" in envelope
        assert envelope["details"]["articles_analyzed"] == 0

    def test_top_items_are_lists_of_strings(self):
        """Top items should be lists of strings (headlines, no URLs/scores)."""
        llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps(_make_sample_news_output())
        mock_llm = MagicMock(return_value=mock_response)
        llm.bind_tools.return_value = mock_llm

        with patch(
            "tradingagents.agents.analysts.news_analyst.get_instrument_context_from_state",
            return_value="Test context",
        ):
            node = create_news_analyst(llm)
            result = node(_make_state())

        envelope = json.loads(result["news_report"])
        details = envelope["details"]

        # Verify all top items are lists
        for key in ["top_positive_fundamentals", "top_negative_fundamentals",
                    "top_positive_sentiment", "top_negative_sentiment"]:
            assert isinstance(details[key], list)
            for item in details[key]:
                assert isinstance(item, str)
                assert len(item) <= 120

    def test_messages_cleared_after_processing(self):
        """Node should return empty messages list after processing."""
        llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps(_make_sample_news_output())
        mock_llm = MagicMock(return_value=mock_response)
        llm.bind_tools.return_value = mock_llm

        with patch(
            "tradingagents.agents.analysts.news_analyst.get_instrument_context_from_state",
            return_value="Test context",
        ):
            node = create_news_analyst(llm)
            result = node(_make_state())

        # Messages should be empty per analyst pattern
        assert result["messages"] == []

    def test_articles_analyzed_preserved(self):
        """articles_analyzed count should be preserved from LLM output."""
        news_output = _make_sample_news_output()
        news_output["articles_analyzed"] = 23

        llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps(news_output)
        mock_llm = MagicMock(return_value=mock_response)
        llm.bind_tools.return_value = mock_llm

        with patch(
            "tradingagents.agents.analysts.news_analyst.get_instrument_context_from_state",
            return_value="Test context",
        ):
            node = create_news_analyst(llm)
            result = node(_make_state())

        envelope = json.loads(result["news_report"])
        assert envelope["details"]["articles_analyzed"] == 23

    def test_confidence_mean_boundary_at_0_7(self):
        """Test boundary condition at 0.7 (should be HIGH, not MEDIUM)."""
        news_output = _make_sample_news_output()
        news_output["conservative"]["confidence"] = 0.65
        news_output["risky"]["confidence"] = 0.75
        # Mean = 0.7, which is not > 0.7, so should be MEDIUM

        llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps(news_output)
        mock_llm = MagicMock(return_value=mock_response)
        llm.bind_tools.return_value = mock_llm

        with patch(
            "tradingagents.agents.analysts.news_analyst.get_instrument_context_from_state",
            return_value="Test context",
        ):
            node = create_news_analyst(llm)
            result = node(_make_state())

        envelope = json.loads(result["news_report"])
        # Mean = 0.7, which is NOT > 0.7, so should be MEDIUM
        assert envelope["confidence"] == "MEDIUM"

    def test_confidence_mean_boundary_at_0_4(self):
        """Test boundary condition at 0.4 (should be MEDIUM, not LOW)."""
        news_output = _make_sample_news_output()
        news_output["conservative"]["confidence"] = 0.35
        news_output["risky"]["confidence"] = 0.45
        # Mean = 0.4, which is not > 0.4, so should be LOW

        llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps(news_output)
        mock_llm = MagicMock(return_value=mock_response)
        llm.bind_tools.return_value = mock_llm

        with patch(
            "tradingagents.agents.analysts.news_analyst.get_instrument_context_from_state",
            return_value="Test context",
        ):
            node = create_news_analyst(llm)
            result = node(_make_state())

        envelope = json.loads(result["news_report"])
        # Mean = 0.4, which is NOT > 0.4, so should be LOW
        assert envelope["confidence"] == "LOW"

    def test_news_analyst_binds_exactly_one_tool(self):
        """News analyst should bind exactly one tool (get_news for ticker-specific news).

        This test ensures that the news analyst no longer uses get_global_news,
        per issue #130.
        """
        llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps(_make_sample_news_output())
        mock_llm = MagicMock(return_value=mock_response)
        llm.bind_tools.return_value = mock_llm

        with patch(
            "tradingagents.agents.analysts.news_analyst.get_instrument_context_from_state",
            return_value="Test context",
        ):
            node = create_news_analyst(llm)
            node(_make_state())

        # Verify bind_tools was called exactly once
        assert llm.bind_tools.call_count == 1

        # Verify that bind_tools was called with a list containing exactly one tool
        call_args = llm.bind_tools.call_args
        tools_list = call_args[0][0]  # First positional argument
        assert len(tools_list) == 1, f"Expected 1 tool, got {len(tools_list)}: {[t.name for t in tools_list]}"

        # Verify the single tool is get_news
        tool = tools_list[0]
        assert tool.name == "get_news", f"Expected 'get_news' tool, got '{tool.name}'"
