"""Tests for the macro news analyst node (issue #134): the LLM reads the
deterministic macro news pack (#133) and emits a JSON envelope, following the
news_analyst.py / news_computation.py pattern and mirroring
test_macro_fundamentals_analyst.py's coverage for the sibling analyst (#132).

Escalation fix (issue #134 comment): covers the three gaps a design review
found in the original haiku implementation —
1. sentiment_score must be Python-computed from bullish/bearish/neutral
   counts, never trusted from the LLM's own claimed value.
2. When the macro news pack reports no articles / gate not enabled, the LLM
   must not be called at all — a neutral envelope is built directly in
   Python (mirrors researcher.py's gate-then-maybe-call structure).
3. Node-level tests for this analyst didn't exist before this file.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from tradingagents.agents.analysts.macro_news_analyst import (
    create_macro_news_analyst,
)


def _make_state(**overrides):
    state = {
        "trade_date": "2026-07-06",
        "company_of_interest": "AAPL",
        "messages": [],
    }
    state.update(overrides)
    return state


def _make_sample_pack(gate_enabled=True, categories=None):
    """A minimal-but-realistic macro news pack, matching macro_news_pack.py's shape."""
    if categories is None:
        categories = {
            "monetary_policy": [
                {
                    "title": "Fed holds rates steady",
                    "summary": "The Federal Reserve kept rates unchanged.",
                    "publisher": "Reuters",
                    "link": "https://example.com/fed",
                    "pub_date": "2026-07-06T10:00:00",
                },
            ],
            "inflation_prices": [
                {
                    "title": "CPI comes in hot",
                    "summary": "Inflation accelerated more than expected.",
                    "publisher": "Bloomberg",
                    "link": "https://example.com/cpi",
                    "pub_date": "2026-07-06T09:00:00",
                },
            ],
        }
    article_count = sum(len(v) for v in categories.values())
    return {
        "curr_date": "2026-07-06",
        "gate": {
            "date_is_today": gate_enabled,
            "outcome": "enabled" if gate_enabled else "disabled (historical date)",
        },
        "vendor": "yfinance" if gate_enabled else None,
        "look_back_days": 3,
        "category_cap": 3,
        "article_count": article_count if gate_enabled else 0,
        "categories": categories if gate_enabled else {},
    }


def _make_sample_llm_output(**overrides):
    """Sample macro news analyst LLM output matching the expected JSON schema
    (post-escalation-fix: the LLM supplies counts only, no sentiment_score)."""
    output = {
        "articles_analyzed": 2,
        "categories_with_articles": ["monetary_policy", "inflation_prices"],
        "category_sentiments": [
            {
                "category": "monetary_policy",
                "bullish_count": 1,
                "bearish_count": 0,
                "neutral_count": 0,
                "top_articles": ["Fed holds rates steady"],
            },
            {
                "category": "inflation_prices",
                "bullish_count": 0,
                "bearish_count": 1,
                "neutral_count": 0,
                "top_articles": ["CPI comes in hot"],
            },
        ],
        "conservative": {"rating": "HOLD", "confidence": 0.55},
        "risky": {"rating": "BUY", "confidence": 0.7},
    }
    output.update(overrides)
    return output


def _run_node(llm_content, pack=None, state_overrides=None):
    """Build the node with a mocked llm + mocked build_macro_news_pack and invoke it."""
    llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = llm_content
    llm.return_value = mock_response

    pack_value = pack if pack is not None else _make_sample_pack()

    with patch(
        "tradingagents.agents.analysts.macro_news_analyst.build_macro_news_pack",
        return_value=pack_value,
    ) as mock_build:
        node = create_macro_news_analyst(llm)
        result = node(_make_state(**(state_overrides or {})))
    return result, mock_build


@pytest.mark.unit
class TestMacroNewsAnalystJsonEnvelope:
    def test_macro_news_report_is_json_string(self):
        result, _ = _run_node(json.dumps(_make_sample_llm_output()))
        assert "macro_news_report" in result
        envelope = json.loads(result["macro_news_report"])
        assert isinstance(envelope, dict)

    def test_envelope_has_required_fields(self):
        result, _ = _run_node(json.dumps(_make_sample_llm_output()))
        envelope = json.loads(result["macro_news_report"])
        assert envelope["skill"] == "macro-news-analyst"
        assert envelope["ticker"] == "AAPL"
        assert envelope["date"] == "2026-07-06"
        assert "signal" in envelope
        assert "confidence" in envelope
        assert "summary" in envelope
        assert "details" in envelope

    def test_full_pack_never_attached_to_envelope(self):
        """The envelope carries only the standard analyst keys — the full
        macro news pack is never duplicated into it (it stays available on
        demand via get_macro_news), so report strings injected verbatim into
        downstream prompts stay compact. This is the mistake #132 had to be
        escalation-fixed for; #134's original implementation already avoided
        it, and this test locks that in at the node level."""
        result, _ = _run_node(json.dumps(_make_sample_llm_output()))
        envelope = json.loads(result["macro_news_report"])
        assert "pack" not in envelope
        assert "categories" not in envelope
        assert set(envelope) == {
            "skill", "ticker", "date", "signal", "confidence", "summary", "details",
        }

    def test_build_macro_news_pack_called_with_trade_date(self):
        _, mock_build = _run_node(json.dumps(_make_sample_llm_output()))
        mock_build.assert_called_once_with("2026-07-06")

    def test_signal_derived_from_conservative_rating(self):
        output = _make_sample_llm_output()
        output["conservative"]["rating"] = "SELL"
        output["risky"]["rating"] = "BUY"
        result, _ = _run_node(json.dumps(output))
        envelope = json.loads(result["macro_news_report"])
        assert envelope["signal"] == "SELL"

    def test_messages_cleared_after_processing(self):
        result, _ = _run_node(json.dumps(_make_sample_llm_output()))
        assert result["messages"] == []

    def test_past_context_included_in_prompt(self):
        """macro_news_past_context on state should reach the LLM prompt."""
        captured = {}

        llm = MagicMock()

        def fake_call(formatted_prompt):
            captured["text"] = str(formatted_prompt)
            response = MagicMock()
            response.content = json.dumps(_make_sample_llm_output())
            return response

        llm.side_effect = fake_call

        with patch(
            "tradingagents.agents.analysts.macro_news_analyst.build_macro_news_pack",
            return_value=_make_sample_pack(),
        ):
            node = create_macro_news_analyst(llm)
            node(_make_state(macro_news_past_context="Prior macro news call: SELL, was wrong."))

        assert "Prior macro news call: SELL, was wrong." in captured["text"]


@pytest.mark.unit
class TestMacroNewsAnalystGracefulFallback:
    def test_graceful_degradation_on_llm_json_parse_failure(self):
        """Should degrade gracefully (neutral default) on unparseable LLM output,
        matching news_analyst.py:119-142's pattern."""
        result, _ = _run_node("This is not valid JSON at all")
        envelope = json.loads(result["macro_news_report"])
        assert envelope["signal"] == "HOLD"
        assert envelope["confidence"] == "MEDIUM"
        assert envelope["details"]["category_sentiments"] == []

    def test_graceful_degradation_on_schema_violation(self):
        """Should degrade gracefully when JSON parses but fails pydantic validation."""
        bad_output = {"articles_analyzed": "not-a-number"}  # invalid type
        result, _ = _run_node(json.dumps(bad_output))
        envelope = json.loads(result["macro_news_report"])
        assert envelope["signal"] == "HOLD"
        assert envelope["details"]["category_sentiments"] == []


@pytest.mark.unit
class TestMacroNewsAnalystNoArticlesGate:
    """Escalation fix #2: when the pack reports no articles / gate not
    enabled, the LLM call must be skipped entirely — a neutral envelope is
    built directly in Python, mirroring researcher.py's gate-then-maybe-call
    structure."""

    def test_historical_date_gate_closed_skips_llm_call(self):
        """Gate outcome != 'enabled' (e.g. historical date): no LLM call at all."""
        llm = MagicMock()
        pack = _make_sample_pack(gate_enabled=False)

        with patch(
            "tradingagents.agents.analysts.macro_news_analyst.build_macro_news_pack",
            return_value=pack,
        ):
            node = create_macro_news_analyst(llm)
            result = node(_make_state())

        llm.assert_not_called()
        envelope = json.loads(result["macro_news_report"])
        assert "disabled (historical date)" in envelope["summary"]

    def test_zero_articles_available_skips_llm_call(self):
        """Gate enabled but zero articles returned: no LLM call either."""
        llm = MagicMock()
        pack = _make_sample_pack(gate_enabled=True, categories={})

        with patch(
            "tradingagents.agents.analysts.macro_news_analyst.build_macro_news_pack",
            return_value=pack,
        ):
            node = create_macro_news_analyst(llm)
            result = node(_make_state())

        llm.assert_not_called()
        envelope = json.loads(result["macro_news_report"])
        assert "No macro news available" in envelope["summary"]

    def test_unexpected_pack_result_type_skips_llm_call(self):
        """A non-dict pack result (unexpected failure) also skips the LLM call."""
        llm = MagicMock()

        with patch(
            "tradingagents.agents.analysts.macro_news_analyst.build_macro_news_pack",
            return_value="not-a-dict",
        ):
            node = create_macro_news_analyst(llm)
            result = node(_make_state())

        llm.assert_not_called()
        envelope = json.loads(result["macro_news_report"])
        assert "Macro news unavailable" in envelope["summary"]

    def test_no_articles_envelope_is_neutral_and_well_formed(self):
        """The neutral envelope built in the no-articles path still carries
        all the standard fields and does not fabricate a signal."""
        llm = MagicMock()
        pack = _make_sample_pack(gate_enabled=False)

        with patch(
            "tradingagents.agents.analysts.macro_news_analyst.build_macro_news_pack",
            return_value=pack,
        ):
            node = create_macro_news_analyst(llm)
            result = node(_make_state())

        envelope = json.loads(result["macro_news_report"])
        assert set(envelope) == {
            "skill", "ticker", "date", "signal", "confidence", "summary", "details",
        }
        assert "pack" not in envelope
        # HOLD/HOLD default ratings -> signal="HOLD", mean confidence 0.5 -> MEDIUM
        assert envelope["signal"] == "HOLD"
        assert envelope["confidence"] == "MEDIUM"

    def test_articles_available_does_call_llm(self):
        """Sanity check: when the gate is open and articles exist, the LLM IS called."""
        result, mock_build = _run_node(json.dumps(_make_sample_llm_output()))
        mock_build.assert_called_once()
        assert result["macro_news_report"] is not None


@pytest.mark.unit
class TestMacroNewsAnalystSentimentScoreComputedInPython:
    """Escalation fix #1: sentiment_score must be Python-computed from
    bullish/bearish/neutral counts, never trusted from the LLM's own value."""

    def test_sentiment_score_computed_from_counts_not_llm_claim(self):
        """Feed a mock LLM response whose counts imply one sentiment_score,
        while also (non-compliantly) claiming a wildly different one, and
        verify the envelope reflects the Python-computed value, not the LLM's."""
        output = _make_sample_llm_output()
        # monetary_policy: 3 bullish, 1 bearish, 0 neutral -> (3-1)/4 = 0.5
        output["category_sentiments"][0]["bullish_count"] = 3
        output["category_sentiments"][0]["bearish_count"] = 1
        output["category_sentiments"][0]["neutral_count"] = 0
        # Non-compliant: LLM claims a sentiment_score that doesn't match its
        # own counts at all (this field is no longer even part of the
        # pydantic schema, so pydantic silently drops it — Python must
        # recompute regardless of whether it's present).
        output["category_sentiments"][0]["sentiment_score"] = -0.99

        result, _ = _run_node(json.dumps(output))
        envelope = json.loads(result["macro_news_report"])
        details = envelope["details"]
        monetary = next(
            c for c in details["category_sentiments"] if c["category"] == "monetary_policy"
        )
        assert monetary["sentiment_score"] == pytest.approx(0.5)

    def test_sentiment_score_all_bearish_category(self):
        output = _make_sample_llm_output()
        output["category_sentiments"][1]["bullish_count"] = 0
        output["category_sentiments"][1]["bearish_count"] = 4
        output["category_sentiments"][1]["neutral_count"] = 0

        result, _ = _run_node(json.dumps(output))
        envelope = json.loads(result["macro_news_report"])
        details = envelope["details"]
        inflation = next(
            c for c in details["category_sentiments"] if c["category"] == "inflation_prices"
        )
        assert inflation["sentiment_score"] == pytest.approx(-1.0)

    def test_sentiment_score_balanced_category(self):
        output = _make_sample_llm_output()
        output["category_sentiments"][0]["bullish_count"] = 2
        output["category_sentiments"][0]["bearish_count"] = 2
        output["category_sentiments"][0]["neutral_count"] = 0

        result, _ = _run_node(json.dumps(output))
        envelope = json.loads(result["macro_news_report"])
        details = envelope["details"]
        monetary = next(
            c for c in details["category_sentiments"] if c["category"] == "monetary_policy"
        )
        assert monetary["sentiment_score"] == pytest.approx(0.0)

    def test_llm_not_asked_for_sentiment_score_in_prompt(self):
        """The prompt should instruct the LLM to supply counts only, not to
        compute an aggregate sentiment_score itself."""
        captured = {}

        llm = MagicMock()

        def fake_call(formatted_prompt):
            captured["text"] = str(formatted_prompt)
            response = MagicMock()
            response.content = json.dumps(_make_sample_llm_output())
            return response

        llm.side_effect = fake_call

        with patch(
            "tradingagents.agents.analysts.macro_news_analyst.build_macro_news_pack",
            return_value=_make_sample_pack(),
        ):
            node = create_macro_news_analyst(llm)
            node(_make_state())

        assert "Do not compute an aggregate sentiment score yourself" in captured["text"]
