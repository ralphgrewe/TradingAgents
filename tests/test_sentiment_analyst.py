"""Tests for the sentiment analyst node (issue #71): JSON envelope output,
Python-derived signal/confidence, and graceful degradation.

After issue #166, the sentiment analyst uses run_structured_with_tools for
the shared recovery ladder (schema-repair retry #153, text extraction #162).
Tests in this file cover the new structured-output path, including fallback
and extraction scenarios.
"""

import json
import logging
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
    """A MagicMock LLM suitable for run_structured_with_tools + free-text fallback.

    Since sentiment_analyst_node now uses run_structured_with_tools, we need to:
    1. Make with_structured_output() return a mock that can parse JSON strings into
       SentimentAnalystOutput objects (or return None to trigger fallback).
    2. Make bind_tools() return self for chaining.
    3. Make the LLM callable (as a __call__ side_effect) to work with LangChain chains.

    The call sequence depends on whether the structured call succeeds:
    - Success path: no llm() call in run_structured_with_tools, then summary call
    - Failure path: fallback llm() call in run_structured_with_tools, then summary call

    We use a callable side_effect that tracks state to handle both paths correctly.
    """
    from tradingagents.agents.analysts.sentiment_computation import SentimentAnalystOutput

    llm = MagicMock()

    # Mock with_structured_output() to return a structured LLM
    structured_llm = MagicMock()

    def structured_invoke_impl(messages):
        """Try to parse main_content as JSON into SentimentAnalystOutput."""
        try:
            parsed = json.loads(main_content)
            return SentimentAnalystOutput(**parsed)
        except (json.JSONDecodeError, ValueError, TypeError):
            # If parsing fails, return None to trigger fallback
            return None

    structured_llm.invoke = MagicMock(side_effect=structured_invoke_impl)
    llm.with_structured_output = MagicMock(return_value=structured_llm)

    # Track whether structured call succeeded to know which invoke() is next
    # In success path: first call is for summary
    # In failure path: first call is for fallback, second is for summary
    call_count = [0]

    # Create a simple response class that doesn't auto-create attributes like MagicMock
    class Response:
        def __init__(self, content):
            self.content = content

    def call_impl(*args, **kwargs):
        """Handle __call__ (via side_effect) and .invoke() calls via all paths."""
        call_count[0] += 1
        # Determine which call this is:
        # If structured call succeeded, there's no fallback call, so first call is summary
        # If structured call failed, first call is fallback, second is summary
        try:
            json.loads(main_content)
            # Structured call succeeded, so this call is for summary
            return Response(summary_content)
        except (json.JSONDecodeError, ValueError, TypeError):
            # Structured call failed, so route based on call count
            if call_count[0] == 1:
                # First call is fallback
                return Response(main_content)
            else:
                # Second call is summary
                return Response(summary_content)

    # Set up the mock to be callable (LangChain uses __call__ for chain.invoke)
    llm.side_effect = call_impl
    # Also set up .invoke for when run_structured_with_tools calls it directly
    llm.invoke.side_effect = call_impl

    # Mock bind_tools for run_structured_with_tools (returns self for chaining)
    llm.bind_tools = MagicMock(return_value=llm)

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

    def test_valid_structured_response_populates_directions(self):
        """AC1: a valid structured response populates per-source directions
        and yields a non-null signal via the recovery ladder."""
        payload = _sample_llm_payload()
        llm = _make_llm(json.dumps(payload))
        p1, p2, p3 = _patch_fetchers()
        with p1, p2, p3:
            node = create_sentiment_analyst(llm)
            result = node(_make_state())

        envelope = json.loads(result["sentiment_report"])
        # All three sources should have directions from the LLM
        details = envelope["details"]
        assert details["sources"]["news"]["direction"] == "NEUTRAL"
        assert details["sources"]["stocktwits"]["direction"] == "POSITIVE"
        assert details["sources"]["reddit"]["direction"] == "POSITIVE"
        assert details["overall_direction"] == "BULLISH"
        # Signal should be non-null since we have directions
        assert envelope["signal"] == "BUY"
        assert envelope["confidence"] is not None

    def test_text_extraction_recovery_logs_warning(self, caplog):
        """AC3: when JSON is embedded in prose (e.g. model emitted markdown),
        text extraction recovers the structured output and logs a WARNING.
        The payload is wrapped in ```json fences to simulate text extraction."""
        payload = _sample_llm_payload()
        # Simulate a response that wraps the JSON in markdown code fence
        llm_response = f"""Here's the sentiment analysis:

```json
{json.dumps(payload)}
```

This covers the key drivers."""
        llm = _make_llm(llm_response)
        p1, p2, p3 = _patch_fetchers()

        with p1, p2, p3, caplog.at_level(logging.WARNING):
            node = create_sentiment_analyst(llm)
            result = node(_make_state())

        envelope = json.loads(result["sentiment_report"])
        # Text extraction should have recovered the structured output
        assert envelope["signal"] == "BUY"
        assert envelope["confidence"] is not None
        # Check that the extraction was logged
        assert any(
            "recovered from free-text fallback via text extraction" in record.message
            for record in caplog.records
            if record.levelname == "WARNING"
        ), f"Expected extraction warning in logs, got: {[r.message for r in caplog.records]}"

    def test_malformed_response_logs_warning_and_degrades(self, caplog):
        """AC2: a malformed response (bad JSON, schema mismatch) is handled by
        run_structured_with_tools and logged. The node degrades gracefully to
        the Python-only skeleton (null directions)."""
        # Completely invalid JSON that can't be recovered
        llm_response = "This is random prose that doesn't contain any JSON at all"
        llm = _make_llm(llm_response)
        p1, p2, p3 = _patch_fetchers()

        with p1, p2, p3, caplog.at_level(logging.WARNING):
            node = create_sentiment_analyst(llm)
            result = node(_make_state())

        envelope = json.loads(result["sentiment_report"])
        # Fallback to Python-only skeleton: counts present, directions null
        details = envelope["details"]
        assert details["sources"]["news"]["headline_count"] == 1
        assert details["sources"]["news"]["direction"] is None
        assert details["sources"]["stocktwits"]["direction"] is None
        assert details["sources"]["reddit"]["direction"] is None
        # Signal should be None since no directions were found
        assert envelope["signal"] is None
        assert envelope["confidence"] is None
        # Verify that a failure was logged
        assert any(
            "structured output failed" in record.message.lower()
            for record in caplog.records
            if record.levelname == "WARNING"
        ), f"Expected structured output failure warning in logs, got: {[r.message for r in caplog.records]}"

    def test_double_failure_in_recovery_ladder_does_not_abort_ticker(self, caplog):
        """Design-review follow-up to 569979c: run_structured_with_tools's own
        docstring documents a "true double failure" mode -- the structured call
        fails/is unsupported *and* the free-text fallback llm.invoke also raises
        (e.g. a provider outage) -- which propagates uncaught out of the helper.
        The node must catch that, log it at WARNING, and degrade to the
        Python-only skeleton instead of letting the exception propagate (which
        would otherwise abort the whole batch run via run_trading_agents.py's
        per-ticker sys.exit(1) handler)."""
        llm = MagicMock()

        # Structured output unsupported: with_structured_output itself raises,
        # so bind_structured() catches it internally and returns None -- the
        # ladder falls straight to the free-text fallback path.
        llm.with_structured_output = MagicMock(
            side_effect=NotImplementedError("provider does not support structured output")
        )
        llm.bind_tools = MagicMock(return_value=llm)

        # The free-text fallback (llm.invoke on the final trace) also raises,
        # simulating a provider outage hitting both rungs of the ladder.
        llm.invoke = MagicMock(side_effect=RuntimeError("provider outage"))

        p1, p2, p3 = _patch_fetchers()
        with p1, p2, p3, caplog.at_level(logging.WARNING):
            node = create_sentiment_analyst(llm)
            # Must not raise -- the node has to swallow the double failure and
            # still return a valid envelope with null directions.
            result = node(_make_state())

        assert "sentiment_report" in result
        envelope = json.loads(result["sentiment_report"])
        assert envelope["signal"] is None
        assert envelope["confidence"] is None
        details = envelope["details"]
        assert details["sources"]["news"]["direction"] is None
        assert details["sources"]["stocktwits"]["direction"] is None
        assert details["sources"]["reddit"]["direction"] is None
        # Python-computed counts are still present even though the LLM path
        # failed entirely.
        assert details["sources"]["news"]["headline_count"] == 1

        assert any(
            "structured-output ladder raised an uncaught exception" in record.message
            for record in caplog.records
            if record.levelname == "WARNING"
        ), f"Expected double-failure warning in logs, got: {[r.message for r in caplog.records]}"


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
