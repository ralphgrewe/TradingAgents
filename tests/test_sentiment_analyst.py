"""Tests for the sentiment analyst node (issue #71): JSON envelope output,
Python-derived signal/confidence, and graceful degradation.
"""

import json
import warnings
from unittest.mock import MagicMock, patch

import pytest

from tradingagents.agents.analysts.sentiment_analyst import (
    create_sentiment_analyst,
    create_social_media_analyst,
)

pytestmark = pytest.mark.unit


def _make_state(**overrides):
    state = {
        "trade_date": "2026-07-08",
        "company_of_interest": "AAPL",
        "messages": [],
    }
    state.update(overrides)
    return state


def _sample_llm_payload(**overrides):
    payload = {
        "news": {"direction": "NEUTRAL", "confidence": 0.4, "key_items": ["Q2 beat estimates"]},
        "stocktwits": {"direction": "POSITIVE", "confidence": 0.6, "key_items": ["retail buying the dip"]},
        "reddit": {"direction": "POSITIVE", "confidence": 0.3, "key_items": ["earnings thread trending"]},
        "overall_direction": "BULLISH",
        "divergences": [],
        "narratives": ["AI momentum"],
        "catalysts": ["earnings next week"],
        "risks": ["macro headwinds"],
    }
    payload.update(overrides)
    return payload


_DEFAULT_NEWS_BLOCK = "### Q2 beat estimates (source: Reuters)\nStrong quarter\n\n"
_DEFAULT_STOCKTWITS_BLOCK = "Bullish: 14 (70%) · Bearish: 4 (20%) · Unlabeled: 2 · Total: 20 most-recent messages"
_DEFAULT_REDDIT_BLOCK = "r/stocks — 1 recent posts mentioning AAPL:\n  [2026-07-08 ·  100↑ ·   20c] Great quarter\n"


def _patch_fetchers(news_block=None, stocktwits_block=None, reddit_block=None):
    """Patch the three data fetchers. Pass an explicit `""` to simulate an
    empty/unavailable block — only omitted (`None`) args get the default."""
    return (
        patch(
            "tradingagents.agents.analysts.sentiment_analyst.get_news.func",
            return_value=_DEFAULT_NEWS_BLOCK if news_block is None else news_block,
        ),
        patch(
            "tradingagents.agents.analysts.sentiment_analyst.fetch_stocktwits_messages",
            return_value=_DEFAULT_STOCKTWITS_BLOCK if stocktwits_block is None else stocktwits_block,
        ),
        patch(
            "tradingagents.agents.analysts.sentiment_analyst.fetch_reddit_posts",
            return_value=_DEFAULT_REDDIT_BLOCK if reddit_block is None else reddit_block,
        ),
    )


def _make_llm(main_content, summary_content="Bullish retail chatter and steady news coverage"):
    """A MagicMock LLM usable in `prompt | llm` chains (LangChain coerces a
    bare callable via RunnableLambda, invoking it as `llm(input)` — so the
    response sequence is driven by `side_effect`, not `.invoke`).

    Yields `main_content` on the first chain invocation (the structured
    analysis call) and `summary_content` on the second (the one-line
    summary call), matching sentiment_analyst_node's two-LLM-call pattern.
    """
    llm = MagicMock()
    llm.side_effect = [MagicMock(content=main_content), MagicMock(content=summary_content)]
    return llm


class TestSentimentAnalystJsonEnvelope:
    def test_sentiment_report_is_json_string(self):
        llm = _make_llm(json.dumps(_sample_llm_payload()))
        p1, p2, p3 = _patch_fetchers()
        with p1, p2, p3:
            node = create_sentiment_analyst(llm)
            result = node(_make_state())

        assert "sentiment_report" in result
        envelope = json.loads(result["sentiment_report"])
        assert isinstance(envelope, dict)

    def test_envelope_has_required_fields(self):
        llm = _make_llm(json.dumps(_sample_llm_payload()))
        p1, p2, p3 = _patch_fetchers()
        with p1, p2, p3:
            node = create_sentiment_analyst(llm)
            result = node(_make_state())

        envelope = json.loads(result["sentiment_report"])
        assert envelope["skill"] == "sentiment-analyst"
        assert envelope["ticker"] == "AAPL"
        assert envelope["date"] == "2026-07-08"
        assert "signal" in envelope
        assert "confidence" in envelope
        assert "summary" in envelope
        assert "details" in envelope

    def test_details_sources_have_python_computed_counts(self):
        llm = _make_llm(json.dumps(_sample_llm_payload()))
        p1, p2, p3 = _patch_fetchers()
        with p1, p2, p3:
            node = create_sentiment_analyst(llm)
            result = node(_make_state())

        details = json.loads(result["sentiment_report"])["details"]
        assert details["sources"]["news"]["headline_count"] == 1
        assert details["sources"]["stocktwits"]["message_count"] == 20
        assert details["sources"]["stocktwits"]["bullish"] == 14
        assert details["sources"]["reddit"]["post_count"] == 1
        assert details["sources"]["reddit"]["top_engagement"] == {"score": 100, "comments": 20}

    def test_signal_and_confidence_derived_from_llm_directions(self):
        llm = _make_llm(json.dumps(_sample_llm_payload()))
        p1, p2, p3 = _patch_fetchers()
        with p1, p2, p3:
            node = create_sentiment_analyst(llm)
            result = node(_make_state())

        envelope = json.loads(result["sentiment_report"])
        # news NEUTRAL(0.4) + stocktwits POSITIVE(0.6) + reddit POSITIVE(0.3):
        # s = (0*0.4 + 1*0.6 + 1*0.3) / (0.4+0.6+0.3) = 0.9/1.3 ~ 0.69 -> BUY
        assert envelope["signal"] == "BUY"
        assert envelope["confidence"] in ("MEDIUM", "HIGH", "LOW")

    def test_messages_cleared_after_processing(self):
        llm = _make_llm(json.dumps(_sample_llm_payload()))
        p1, p2, p3 = _patch_fetchers()
        with p1, p2, p3:
            node = create_sentiment_analyst(llm)
            result = node(_make_state())

        assert result["messages"] == []

    def test_summary_comes_from_second_llm_call(self):
        llm = _make_llm(
            json.dumps(_sample_llm_payload()), summary_content="Custom one-line summary"
        )
        p1, p2, p3 = _patch_fetchers()
        with p1, p2, p3:
            node = create_sentiment_analyst(llm)
            result = node(_make_state())

        envelope = json.loads(result["sentiment_report"])
        assert envelope["summary"] == "Custom one-line summary"


class TestSentimentAnalystGracefulDegradation:
    def test_llm_json_parse_failure_falls_back_without_crashing(self):
        llm = _make_llm("This is not valid JSON at all")
        p1, p2, p3 = _patch_fetchers()
        with p1, p2, p3:
            node = create_sentiment_analyst(llm)
            result = node(_make_state())

        envelope = json.loads(result["sentiment_report"])
        # Python-computed counts still present, directions null -> no signal.
        assert envelope["signal"] is None
        assert envelope["confidence"] is None
        details = envelope["details"]
        assert details["sources"]["news"]["headline_count"] == 1
        assert details["sources"]["news"]["direction"] is None
        assert details["sources"]["stocktwits"]["direction"] is None
        assert details["sources"]["reddit"]["direction"] is None

    def test_llm_schema_validation_failure_falls_back(self):
        """Valid JSON, but the wrong shape (e.g. bad enum) fails pydantic
        validation and should degrade gracefully, not crash."""
        bad_payload = _sample_llm_payload()
        bad_payload["overall_direction"] = "SUPER_BULLISH"  # not a valid enum
        llm = _make_llm(json.dumps(bad_payload))
        p1, p2, p3 = _patch_fetchers()
        with p1, p2, p3:
            node = create_sentiment_analyst(llm)
            result = node(_make_state())

        envelope = json.loads(result["sentiment_report"])
        assert envelope["signal"] is None
        assert envelope["confidence"] is None

    def test_all_sources_unavailable_still_writes_envelope(self):
        """Edge case: all three sources unavailable -> envelope still
        written, signal/confidence null, caveats populated."""
        llm = _make_llm(json.dumps(_sample_llm_payload()))
        p1, p2, p3 = _patch_fetchers(
            news_block="Error fetching news for AAPL: timeout",
            stocktwits_block="<stocktwits unavailable: HTTPError>",
            reddit_block="",
        )
        with p1, p2, p3:
            node = create_sentiment_analyst(llm)
            result = node(_make_state())

        envelope = json.loads(result["sentiment_report"])
        assert envelope["signal"] is None
        assert envelope["confidence"] is None
        details = envelope["details"]
        assert details["data_quality"]["sources_available"] == 0
        assert len(details["data_quality"]["caveats"]) == 3
        # LLM-provided directions for unavailable sources must not leak through.
        assert details["sources"]["news"]["direction"] is None
        assert details["sources"]["stocktwits"]["direction"] is None
        assert details["sources"]["reddit"]["direction"] is None


class TestSocialMediaAnalystDeprecationShim:
    def test_shim_still_works_and_warns(self):
        llm = _make_llm(json.dumps(_sample_llm_payload()))
        p1, p2, p3 = _patch_fetchers()
        with p1, p2, p3, warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            node = create_social_media_analyst(llm)
            result = node(_make_state())

        assert any(issubclass(w.category, DeprecationWarning) for w in caught)
        envelope = json.loads(result["sentiment_report"])
        assert envelope["skill"] == "sentiment-analyst"
