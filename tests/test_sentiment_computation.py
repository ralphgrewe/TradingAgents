"""Unit tests for sentiment_computation.py (issue #71).

Covers the Python-side count/availability parsing of the pre-fetched
news/StockTwits/Reddit blocks, the signal/confidence derivation rules, and
the details-payload merge (including the parse-failure fallback shape).
"""

import json

import pytest
from pydantic import ValidationError

from tradingagents.agents.analysts.sentiment_computation import (
    SentimentAnalystOutput,
    SourceAssessment,
    build_details,
    build_json_envelope,
    build_sources_skeleton,
    compute_news_counts,
    compute_reddit_counts,
    compute_stocktwits_counts,
    derive_signal_and_confidence,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# compute_news_counts
# ---------------------------------------------------------------------------

class TestComputeNewsCounts:
    def test_counts_yfinance_headlines(self):
        block = (
            "## AAPL News, from 2026-07-01 to 2026-07-08:\n\n"
            "### Q2 beat estimates (source: Reuters)\nSummary text\nLink: http://x\n\n"
            "### New product launch (source: Bloomberg)\nSummary text\nLink: http://y\n\n"
        )
        result = compute_news_counts(block)
        assert result == {"available": True, "headline_count": 2, "reason": None}

    def test_no_news_found_is_available_with_zero_count(self):
        result = compute_news_counts("No news found for AAPL between 2026-07-01 and 2026-07-08")
        assert result["available"] is True
        assert result["headline_count"] == 0

    def test_error_fetching_is_unavailable(self):
        result = compute_news_counts("Error fetching news for AAPL: connection reset")
        assert result["available"] is False
        assert result["headline_count"] == 0
        assert "Error fetching news" in result["reason"]

    def test_empty_block_is_unavailable(self):
        result = compute_news_counts("")
        assert result["available"] is False
        assert result["headline_count"] == 0

    def test_none_block_is_unavailable(self):
        result = compute_news_counts(None)
        assert result["available"] is False


# ---------------------------------------------------------------------------
# compute_stocktwits_counts
# ---------------------------------------------------------------------------

class TestComputeStocktwitsCounts:
    def test_parses_summary_line(self):
        block = (
            "Bullish: 14 (70%) · Bearish: 4 (20%) · Unlabeled: 2 · Total: 20 most-recent messages\n\n"
            "[2026-07-08 · @trader1 · Bullish] Loading up on calls"
        )
        result = compute_stocktwits_counts(block)
        assert result == {
            "available": True,
            "message_count": 20,
            "bullish": 14,
            "bearish": 4,
            "unlabeled": 2,
            "reason": None,
        }

    def test_unavailable_marker_surfaces_reason(self):
        result = compute_stocktwits_counts("<stocktwits unavailable: HTTPError>")
        assert result["available"] is False
        assert result["message_count"] == 0
        assert result["reason"] == "HTTPError"

    def test_no_messages_found_is_available_zero(self):
        result = compute_stocktwits_counts("<no StockTwits messages found for $AAPL>")
        assert result["available"] is True
        assert result["message_count"] == 0
        assert result["bullish"] == 0

    def test_unparseable_block_is_unavailable(self):
        result = compute_stocktwits_counts("some unexpected garbage text")
        assert result["available"] is False
        assert result["reason"] == "unparseable response"

    def test_unlabeled_messages_not_counted_as_neutral_direction(self):
        """Edge case: unlabeled StockTwits messages are their own bucket, not
        folded into bullish/bearish (i.e. not treated as neutral-direction
        evidence)."""
        block = "Bullish: 0 (0%) · Bearish: 0 (0%) · Unlabeled: 5 · Total: 5 most-recent messages"
        result = compute_stocktwits_counts(block)
        assert result["unlabeled"] == 5
        assert result["bullish"] == 0
        assert result["bearish"] == 0


# ---------------------------------------------------------------------------
# compute_reddit_counts
# ---------------------------------------------------------------------------

class TestComputeRedditCounts:
    def test_counts_posts_across_subreddits(self):
        block = (
            "r/wallstreetbets — 2 recent posts mentioning AAPL:\n"
            "  [2026-07-08 ·  412↑ ·  208c] Huge AAPL rally incoming\n"
            "    body excerpt: yolo\n"
            "  [2026-07-07 ·   10↑ ·    2c] Small post\n\n"
            "r/stocks — 1 recent posts mentioning AAPL:\n"
            "  [2026-07-06 ·   50↑ ·   10c] Measured take\n"
        )
        result = compute_reddit_counts(block)
        assert result["available"] is True
        assert result["post_count"] == 3
        assert result["top_engagement"] == {"score": 412, "comments": 208}

    def test_no_posts_found_is_available_zero(self):
        result = compute_reddit_counts(
            "<no Reddit posts found mentioning AAPL across r/wallstreetbets, r/stocks, r/investing in the past 7 days>"
        )
        assert result["available"] is True
        assert result["post_count"] == 0
        assert result["top_engagement"] is None

    def test_rss_posts_without_engagement_metrics(self):
        block = "r/stocks — 1 recent posts mentioning AAPL (via RSS feed; scores/comments unavailable):\n  [2026-07-08] Some title\n"
        result = compute_reddit_counts(block)
        assert result["post_count"] == 1
        assert result["top_engagement"] is None

    def test_empty_block_is_unavailable(self):
        result = compute_reddit_counts("")
        assert result["available"] is False


# ---------------------------------------------------------------------------
# build_sources_skeleton
# ---------------------------------------------------------------------------

class TestBuildSourcesSkeleton:
    def test_skeleton_has_null_directions_pending_llm(self):
        skeleton = build_sources_skeleton(
            "No news found for AAPL",
            "<no StockTwits messages found for $AAPL>",
            "<no Reddit posts found mentioning AAPL across r/wallstreetbets, r/stocks, r/investing in the past 7 days>",
        )
        for name in ("news", "stocktwits", "reddit"):
            assert skeleton[name]["direction"] is None
            assert skeleton[name]["confidence"] is None
            assert skeleton[name]["key_items"] == []

    def test_all_sources_unavailable(self):
        skeleton = build_sources_skeleton(
            "Error fetching news for AAPL: timeout",
            "<stocktwits unavailable: TimeoutError>",
            "",
        )
        assert skeleton["news"]["available"] is False
        assert skeleton["stocktwits"]["available"] is False
        assert skeleton["reddit"]["available"] is False


# ---------------------------------------------------------------------------
# build_details (merge + graceful degradation)
# ---------------------------------------------------------------------------

class TestBuildDetails:
    def _skeleton(self):
        return build_sources_skeleton(
            "### Headline one (source: Reuters)\nSummary\n\n### Headline two (source: AP)\nSummary\n\n",
            "Bullish: 10 (80%) · Bearish: 2 (16%) · Unlabeled: 1 · Total: 13 most-recent messages",
            "r/stocks — 1 recent posts mentioning AAPL:\n  [2026-07-08 ·  100↑ ·   20c] Post title\n",
        )

    def test_fallback_details_has_python_counts_and_null_directions(self):
        """AC4: LLM parse/validation failure -> Python-computed counts, null directions."""
        details = build_details("2026-07-01", "2026-07-08", self._skeleton(), llm_output=None)

        assert details["sources"]["news"]["headline_count"] == 2
        assert details["sources"]["news"]["direction"] is None
        assert details["sources"]["stocktwits"]["message_count"] == 13
        assert details["sources"]["stocktwits"]["direction"] is None
        assert details["sources"]["reddit"]["post_count"] == 1
        assert details["overall_direction"] is None
        assert details["divergences"] == []
        assert details["data_quality"]["sources_available"] == 3
        assert details["data_quality"]["caveats"] == []

    def test_merges_llm_output_onto_skeleton(self):
        llm_output = SentimentAnalystOutput(
            news=SourceAssessment(direction="NEUTRAL", confidence=0.4, key_items=["a"]),
            stocktwits=SourceAssessment(direction="POSITIVE", confidence=0.6, key_items=["b"]),
            reddit=SourceAssessment(direction="POSITIVE", confidence=0.3, key_items=["c"]),
            overall_direction="BULLISH",
            divergences=["news lags retail"],
            narratives=["AI momentum"],
            catalysts=["earnings next week"],
            risks=["macro headwinds"],
        )
        details = build_details("2026-07-01", "2026-07-08", self._skeleton(), llm_output=llm_output)

        assert details["sources"]["news"]["direction"] == "NEUTRAL"
        assert details["sources"]["stocktwits"]["direction"] == "POSITIVE"
        assert details["sources"]["reddit"]["direction"] == "POSITIVE"
        assert details["overall_direction"] == "BULLISH"
        assert details["divergences"] == ["news lags retail"]

    def test_unavailable_source_keeps_null_direction_even_with_llm_output(self):
        """An unavailable source's direction/confidence stay null regardless
        of what the LLM says — Python owns availability."""
        skeleton = build_sources_skeleton(
            "Error fetching news for AAPL: timeout",
            "Bullish: 10 (80%) · Bearish: 2 (16%) · Unlabeled: 1 · Total: 13 most-recent messages",
            "<no Reddit posts found mentioning AAPL across r/wallstreetbets, r/stocks, r/investing in the past 7 days>",
        )
        llm_output = SentimentAnalystOutput(
            news=SourceAssessment(direction="POSITIVE", confidence=0.9),  # LLM ignoring the unavailable flag
            stocktwits=SourceAssessment(direction="POSITIVE", confidence=0.6),
            reddit=SourceAssessment(direction=None, confidence=None),
            overall_direction="NEUTRAL",
        )
        details = build_details("2026-07-01", "2026-07-08", skeleton, llm_output=llm_output)

        assert details["sources"]["news"]["available"] is False
        assert details["sources"]["news"]["direction"] is None
        assert details["sources"]["news"]["confidence"] is None

    def test_all_three_sources_unavailable_populates_caveats(self):
        """Edge case: all three sources unavailable -> signal/confidence null,
        caveats populated, envelope still produced."""
        skeleton = build_sources_skeleton(
            "Error fetching news for AAPL: timeout",
            "<stocktwits unavailable: HTTPError>",
            "",
        )
        details = build_details("2026-07-01", "2026-07-08", skeleton, llm_output=None)

        assert details["data_quality"]["sources_available"] == 0
        assert len(details["data_quality"]["caveats"]) == 3
        assert any("News" in c for c in details["data_quality"]["caveats"])
        assert any("StockTwits" in c for c in details["data_quality"]["caveats"])
        assert any("Reddit" in c for c in details["data_quality"]["caveats"])

        signal, confidence = derive_signal_and_confidence(details)
        assert signal is None
        assert confidence is None


# ---------------------------------------------------------------------------
# derive_signal_and_confidence
# ---------------------------------------------------------------------------

def _details_with_sources(**sources):
    base = {
        "news": {"available": True, "direction": None, "confidence": None},
        "stocktwits": {"available": True, "direction": None, "confidence": None},
        "reddit": {"available": True, "direction": None, "confidence": None},
    }
    base.update(sources)
    return {"sources": base}


class TestDeriveSignalAndConfidence:
    def test_no_directions_returns_none_none(self):
        details = _details_with_sources()
        assert derive_signal_and_confidence(details) == (None, None)

    def test_all_positive_high_confidence_is_buy_high(self):
        details = _details_with_sources(
            news={"available": True, "direction": "POSITIVE", "confidence": 0.8},
            stocktwits={"available": True, "direction": "POSITIVE", "confidence": 0.9},
            reddit={"available": True, "direction": "POSITIVE", "confidence": 0.8},
        )
        signal, confidence = derive_signal_and_confidence(details)
        assert signal == "BUY"
        assert confidence == "HIGH"

    def test_all_negative_is_sell(self):
        details = _details_with_sources(
            news={"available": True, "direction": "NEGATIVE", "confidence": 0.8},
            stocktwits={"available": True, "direction": "NEGATIVE", "confidence": 0.7},
        )
        signal, confidence = derive_signal_and_confidence(details)
        assert signal == "SELL"

    def test_mixed_near_zero_is_hold(self):
        details = _details_with_sources(
            news={"available": True, "direction": "POSITIVE", "confidence": 0.5},
            stocktwits={"available": True, "direction": "NEGATIVE", "confidence": 0.5},
        )
        signal, confidence = derive_signal_and_confidence(details)
        assert signal == "HOLD"

    def test_single_source_caps_confidence_at_low(self):
        """Confidence caps at LOW when fewer than 2 sources contributed,
        even if that one source's own confidence is high."""
        details = _details_with_sources(
            news={"available": True, "direction": "POSITIVE", "confidence": 0.95},
        )
        signal, confidence = derive_signal_and_confidence(details)
        assert signal == "BUY"
        assert confidence == "LOW"

    def test_two_sources_low_confidence_mean_is_low(self):
        details = _details_with_sources(
            news={"available": True, "direction": "POSITIVE", "confidence": 0.2},
            stocktwits={"available": True, "direction": "POSITIVE", "confidence": 0.3},
        )
        signal, confidence = derive_signal_and_confidence(details)
        assert confidence == "LOW"

    def test_two_sources_medium_confidence_mean_is_medium(self):
        details = _details_with_sources(
            news={"available": True, "direction": "POSITIVE", "confidence": 0.5},
            stocktwits={"available": True, "direction": "POSITIVE", "confidence": 0.6},
        )
        signal, confidence = derive_signal_and_confidence(details)
        assert confidence == "MEDIUM"

    def test_neutral_direction_scores_zero(self):
        details = _details_with_sources(
            news={"available": True, "direction": "NEUTRAL", "confidence": 0.9},
            stocktwits={"available": True, "direction": "NEUTRAL", "confidence": 0.9},
        )
        signal, confidence = derive_signal_and_confidence(details)
        assert signal == "HOLD"

    def test_weighted_mean_just_above_0_33_threshold_is_buy(self):
        # s = 1.0 / (1.0 + 2.0) = 0.3333... > 0.33 -> BUY
        details = _details_with_sources(
            news={"available": True, "direction": "POSITIVE", "confidence": 1.0},
            stocktwits={"available": True, "direction": "NEUTRAL", "confidence": 2.0},
        )
        signal, _ = derive_signal_and_confidence(details)
        assert signal == "BUY"

    def test_weighted_mean_below_0_33_threshold_is_hold(self):
        # s = 0.32 / 1.0 = 0.32, not > 0.33 -> HOLD
        details = _details_with_sources(
            news={"available": True, "direction": "POSITIVE", "confidence": 0.32},
            stocktwits={"available": True, "direction": "NEUTRAL", "confidence": 0.68},
        )
        signal, _ = derive_signal_and_confidence(details)
        assert signal == "HOLD"


# ---------------------------------------------------------------------------
# build_json_envelope
# ---------------------------------------------------------------------------

class TestBuildJsonEnvelope:
    def test_envelope_shape(self):
        envelope_str = build_json_envelope(
            signal="BUY",
            confidence="HIGH",
            summary="Bullish retail chatter",
            details={"window": {"start": "2026-07-01", "end": "2026-07-08"}},
            ticker="AAPL",
            date="2026-07-08",
        )
        envelope = json.loads(envelope_str)
        assert envelope["skill"] == "sentiment-analyst"
        assert envelope["ticker"] == "AAPL"
        assert envelope["date"] == "2026-07-08"
        assert envelope["signal"] == "BUY"
        assert envelope["confidence"] == "HIGH"
        assert envelope["summary"] == "Bullish retail chatter"

    def test_null_signal_and_confidence(self):
        envelope_str = build_json_envelope(
            signal=None, confidence=None, summary="No data", details={}, ticker="AAPL", date="2026-07-08",
        )
        envelope = json.loads(envelope_str)
        assert envelope["signal"] is None
        assert envelope["confidence"] is None


# ---------------------------------------------------------------------------
# SentimentAnalystOutput validation
# ---------------------------------------------------------------------------

class TestSentimentAnalystOutputValidation:
    def _valid_payload(self):
        return {
            "news": {"direction": "POSITIVE", "confidence": 0.5, "key_items": []},
            "stocktwits": {"direction": None, "confidence": None, "key_items": []},
            "reddit": {"direction": "NEGATIVE", "confidence": 0.2, "key_items": ["x"]},
            "overall_direction": "MIXED",
            "divergences": [],
            "narratives": [],
            "catalysts": [],
            "risks": [],
        }

    def test_valid_payload_parses(self):
        output = SentimentAnalystOutput(**self._valid_payload())
        assert output.overall_direction == "MIXED"
        assert output.stocktwits.direction is None

    def test_invalid_direction_raises(self):
        payload = self._valid_payload()
        payload["news"]["direction"] = "SUPER_BULLISH"
        with pytest.raises(ValidationError):
            SentimentAnalystOutput(**payload)

    def test_invalid_overall_direction_raises(self):
        payload = self._valid_payload()
        payload["overall_direction"] = "SIDEWAYS"
        with pytest.raises(ValidationError):
            SentimentAnalystOutput(**payload)

    def test_too_many_key_items_raises(self):
        payload = self._valid_payload()
        payload["news"]["key_items"] = ["a", "b", "c", "d"]
        with pytest.raises(ValidationError):
            SentimentAnalystOutput(**payload)
