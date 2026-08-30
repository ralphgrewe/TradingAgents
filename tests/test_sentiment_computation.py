"""Unit tests for sentiment_computation.py (issue #71).

Covers the Python-side count/availability parsing of the pre-fetched
news/StockTwits/Reddit/ApeWisdom blocks, the signal/confidence derivation rules, and
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
    compute_apewisdom_counts,
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

    def test_alpha_vantage_dict_payload_is_unavailable_not_silent_zero(self):
        # get_news is vendor-routed; alpha_vantage returns a dict, not the
        # yfinance "### " prose. It must surface as unavailable + a caveat
        # reason, never as a confirmed-empty available source (issue #71).
        block = {"feed": [{"title": "Q2 beat"}], "items": "1"}
        result = compute_news_counts(block)
        assert result["available"] is False
        assert result["headline_count"] == 0
        assert result["reason"] == "unrecognized news format"

    def test_alpha_vantage_json_string_payload_is_unavailable(self):
        # A JSON string with real headlines but no "### " markers must not be
        # miscounted as zero-but-available — it is an unrecognized format.
        block = '{"feed": [{"title": "Q2 beat estimates"}, {"title": "New launch"}]}'
        result = compute_news_counts(block)
        assert result["available"] is False
        assert result["headline_count"] == 0
        assert result["reason"] == "unrecognized news format"

    def test_no_data_available_sentinel_is_unavailable(self):
        # route_to_vendor's cross-vendor NO_DATA sentinel must degrade to
        # unavailable, carrying its reason into the caveat.
        block = "NO_DATA_AVAILABLE: No usable market data for 'AAPL' from any configured vendor."
        result = compute_news_counts(block)
        assert result["available"] is False
        assert result["headline_count"] == 0
        assert "NO_DATA_AVAILABLE" in result["reason"]

    def test_unrecognized_prose_is_unavailable(self):
        # Non-empty text that is neither a known sentinel nor "### " headlines.
        result = compute_news_counts("Some unexpected vendor blurb with no headlines.")
        assert result["available"] is False
        assert result["headline_count"] == 0
        assert result["reason"] == "unrecognized news format"


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
# compute_apewisdom_counts (issue #167)
# ---------------------------------------------------------------------------

class TestComputeApewisdomCounts:
    def test_parses_mentions_and_upvotes(self):
        block = "ApeWisdom (Reddit + 4chan /biz aggregate): 245 mentions, 1842 upvotes, rank 24h ago: #12"
        result = compute_apewisdom_counts(block)
        assert result["available"] is True
        assert result["mention_count"] == 245
        assert result["upvote_count"] == 1842
        assert result["rank_24h_ago"] == 12
        assert result["reason"] is None

    def test_parses_mentions_only(self):
        block = "ApeWisdom (Reddit + 4chan /biz aggregate): 100 mentions"
        result = compute_apewisdom_counts(block)
        assert result["available"] is True
        assert result["mention_count"] == 100
        assert result["upvote_count"] == 0
        assert result["rank_24h_ago"] is None

    def test_zero_mentions_is_available_with_zero(self):
        result = compute_apewisdom_counts("<no ApeWisdom mentions found for $AAPL>")
        assert result["available"] is True
        assert result["mention_count"] == 0
        assert result["upvote_count"] == 0
        assert result["rank_24h_ago"] is None
        assert result["reason"] is None

    def test_non_us_listing_unavailable(self):
        block = "<apewisdom unavailable: non-US listing (exchange suffix detected)>"
        result = compute_apewisdom_counts(block)
        assert result["available"] is False
        assert result["mention_count"] == 0
        assert result["reason"] == "non-US listing (exchange suffix detected)"

    def test_network_error_unavailable(self):
        block = "<apewisdom unavailable: TimeoutError>"
        result = compute_apewisdom_counts(block)
        assert result["available"] is False
        assert result["reason"] == "TimeoutError"

    def test_empty_block_is_unavailable(self):
        result = compute_apewisdom_counts("")
        assert result["available"] is False
        assert result["reason"] == "empty response"


# ---------------------------------------------------------------------------
# build_sources_skeleton
# ---------------------------------------------------------------------------

class TestBuildSourcesSkeleton:
    def test_skeleton_has_null_directions_pending_llm(self):
        skeleton = build_sources_skeleton(
            "No news found for AAPL",
            "<no StockTwits messages found for $AAPL>",
            "<no Reddit posts found mentioning AAPL across r/wallstreetbets, r/stocks, r/investing in the past 7 days>",
            "<no ApeWisdom mentions found for $AAPL>",
        )
        for name in ("news", "stocktwits", "reddit", "apewisdom"):
            assert skeleton[name]["direction"] is None
            assert skeleton[name]["confidence"] is None
            assert skeleton[name]["key_items"] == []

    def test_all_sources_unavailable(self):
        skeleton = build_sources_skeleton(
            "Error fetching news for AAPL: timeout",
            "<stocktwits unavailable: TimeoutError>",
            "",
            "<apewisdom unavailable: non-US listing (exchange suffix detected)>",
        )
        assert skeleton["news"]["available"] is False
        assert skeleton["stocktwits"]["available"] is False
        assert skeleton["reddit"]["available"] is False
        assert skeleton["apewisdom"]["available"] is False

    def test_non_yfinance_news_block_surfaces_caveat_in_details(self):
        """AC4 end-to-end: an alpha_vantage-style news payload must produce a
        news caveat, not a silent zero-headline available source (issue #71)."""
        skeleton = build_sources_skeleton(
            {"feed": [{"title": "Q2 beat"}]},  # alpha_vantage dict payload
            "<no StockTwits messages found for $AAPL>",
            "<no Reddit posts found mentioning AAPL in the past 7 days>",
            "ApeWisdom (Reddit + 4chan /biz aggregate): 50 mentions, 320 upvotes",
        )
        assert skeleton["news"]["available"] is False
        details = build_details("2026-07-01", "2026-07-08", skeleton, llm_output=None)
        caveats = details["data_quality"]["caveats"]
        assert any("News unavailable" in c for c in caveats)
        assert details["sources"]["news"]["available"] is False

    def test_skeleton_includes_apewisdom_counts(self):
        """Apewisdom counts are parsed and included in the skeleton."""
        skeleton = build_sources_skeleton(
            "No news found for AAPL",
            "<no StockTwits messages found for $AAPL>",
            "<no Reddit posts found mentioning AAPL across r/wallstreetbets, r/stocks, r/investing in the past 7 days>",
            "ApeWisdom (Reddit + 4chan /biz aggregate): 245 mentions, 1842 upvotes, rank 24h ago: #12",
        )
        assert skeleton["apewisdom"]["available"] is True
        assert skeleton["apewisdom"]["mention_count"] == 245
        assert skeleton["apewisdom"]["upvote_count"] == 1842
        assert skeleton["apewisdom"]["rank_24h_ago"] == 12


# ---------------------------------------------------------------------------
# build_details (merge + graceful degradation)
# ---------------------------------------------------------------------------

class TestBuildDetails:
    def _skeleton(self):
        return build_sources_skeleton(
            "### Headline one (source: Reuters)\nSummary\n\n### Headline two (source: AP)\nSummary\n\n",
            "Bullish: 10 (80%) · Bearish: 2 (16%) · Unlabeled: 1 · Total: 13 most-recent messages",
            "r/stocks — 1 recent posts mentioning AAPL:\n  [2026-07-08 ·  100↑ ·   20c] Post title\n",
            "ApeWisdom (Reddit + 4chan /biz aggregate): 245 mentions, 1842 upvotes, rank 24h ago: #12",
        )

    def test_fallback_details_has_python_counts_and_null_directions(self):
        """AC4: LLM parse/validation failure -> Python-computed counts, null directions."""
        details = build_details("2026-07-01", "2026-07-08", self._skeleton(), llm_output=None)

        assert details["sources"]["news"]["headline_count"] == 2
        assert details["sources"]["news"]["direction"] is None
        assert details["sources"]["stocktwits"]["message_count"] == 13
        assert details["sources"]["stocktwits"]["direction"] is None
        assert details["sources"]["reddit"]["post_count"] == 1
        assert details["sources"]["apewisdom"]["mention_count"] == 245
        assert details["overall_direction"] is None
        assert details["divergences"] == []
        assert details["data_quality"]["sources_available"] == 4
        assert details["data_quality"]["caveats"] == []

    def test_merges_llm_output_onto_skeleton(self):
        llm_output = SentimentAnalystOutput(
            news=SourceAssessment(direction="NEUTRAL", confidence=0.4, key_items=["a"]),
            stocktwits=SourceAssessment(direction="POSITIVE", confidence=0.6, key_items=["b"]),
            reddit=SourceAssessment(direction="POSITIVE", confidence=0.3, key_items=["c"]),
            apewisdom=SourceAssessment(direction="POSITIVE", confidence=0.7, key_items=["d"]),
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
        assert details["sources"]["apewisdom"]["direction"] == "POSITIVE"
        assert details["overall_direction"] == "BULLISH"
        assert details["divergences"] == ["news lags retail"]

    def test_unavailable_source_keeps_null_direction_even_with_llm_output(self):
        """An unavailable source's direction/confidence stay null regardless
        of what the LLM says — Python owns availability."""
        skeleton = build_sources_skeleton(
            "Error fetching news for AAPL: timeout",
            "Bullish: 10 (80%) · Bearish: 2 (16%) · Unlabeled: 1 · Total: 13 most-recent messages",
            "<no Reddit posts found mentioning AAPL across r/wallstreetbets, r/stocks, r/investing in the past 7 days>",
            "ApeWisdom (Reddit + 4chan /biz aggregate): 245 mentions, 1842 upvotes",
        )
        llm_output = SentimentAnalystOutput(
            news=SourceAssessment(direction="POSITIVE", confidence=0.9),  # LLM ignoring the unavailable flag
            stocktwits=SourceAssessment(direction="POSITIVE", confidence=0.6),
            reddit=SourceAssessment(direction=None, confidence=None),
            apewisdom=SourceAssessment(direction="POSITIVE", confidence=0.7),
            overall_direction="NEUTRAL",
        )
        details = build_details("2026-07-01", "2026-07-08", skeleton, llm_output=llm_output)

        assert details["sources"]["news"]["available"] is False
        assert details["sources"]["news"]["direction"] is None
        assert details["sources"]["news"]["confidence"] is None

    def test_all_four_sources_unavailable_populates_caveats(self):
        """Edge case: all four sources unavailable -> signal/confidence null,
        caveats populated, envelope still produced."""
        skeleton = build_sources_skeleton(
            "Error fetching news for AAPL: timeout",
            "<stocktwits unavailable: HTTPError>",
            "",
            "<apewisdom unavailable: non-US listing (exchange suffix detected)>",
        )
        details = build_details("2026-07-01", "2026-07-08", skeleton, llm_output=None)

        assert details["data_quality"]["sources_available"] == 0
        assert len(details["data_quality"]["caveats"]) == 4
        assert any("News" in c for c in details["data_quality"]["caveats"])
        assert any("StockTwits" in c for c in details["data_quality"]["caveats"])
        assert any("Reddit" in c for c in details["data_quality"]["caveats"])
        assert any("ApeWisdom" in c for c in details["data_quality"]["caveats"])

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
            "apewisdom": {"direction": "POSITIVE", "confidence": 0.6, "key_items": []},
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
        assert output.apewisdom.direction == "POSITIVE"

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
