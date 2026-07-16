"""Sentiment analyst computation module for multi-source sentiment scoring +
JSON envelope (issue #71).

This module mirrors ``news_computation.py``'s split of responsibilities:

1. Pydantic schema for the LLM's structured per-source output (direction,
   confidence, key items) plus the cross-source synthesis fields.
2. Python-side parsing of the *already-fetched* news/StockTwits/Reddit
   prompt blocks into count/availability fields — the LLM never computes
   these numbers, it only reads the pre-fetched text and calls a direction.
3. Python-side derivation of the top-level ``signal``/``confidence`` from
   the per-source directions/confidences (weighted mean, fixed thresholds).
4. JSON envelope building per ``skills/SCHEMA.md``.

Key design principle (unchanged from news/market analysts): the LLM reads
and judges qualitatively, Python counts and derives the headline
signal/confidence.
"""

import json
import re

from pydantic import BaseModel, Field, field_validator

_DIRECTIONS = ("POSITIVE", "NEUTRAL", "NEGATIVE")
_OVERALL_DIRECTIONS = ("BULLISH", "BEARISH", "NEUTRAL", "MIXED")


class SourceAssessment(BaseModel):
    """LLM-provided directional read for one sentiment source.

    ``direction``/``confidence`` are null when the source has no usable
    data (Python has already marked it ``available: false`` or the source
    had zero messages/posts) — the LLM is instructed not to guess.
    """

    direction: str | None = Field(
        None,
        description="POSITIVE, NEUTRAL, NEGATIVE, or null if the source has no usable data",
    )
    confidence: float | None = Field(
        None, ge=0.0, le=1.0, description="Confidence 0.0-1.0, or null if direction is null"
    )
    key_items: list[str] = Field(
        default_factory=list,
        description="Up to 3 short (<=120 chars) supporting items, no URLs/scores",
        max_length=3,
    )

    @field_validator("direction")
    @classmethod
    def _validate_direction(cls, value):
        if value is not None and value not in _DIRECTIONS:
            raise ValueError(f"direction must be one of {_DIRECTIONS} or null")
        return value


class SentimentAnalystOutput(BaseModel):
    """Structured output from the sentiment analyst LLM.

    The LLM reads the three pre-fetched data blocks and provides a
    directional read + short evidence per source, plus a cross-source
    synthesis (overall direction, divergences, narratives, catalysts,
    risks). It does not compute any counts — those are Python-derived
    from the same blocks before the LLM is invoked.
    """

    news: SourceAssessment
    stocktwits: SourceAssessment
    reddit: SourceAssessment
    overall_direction: str = Field(pattern="^(BULLISH|BEARISH|NEUTRAL|MIXED)$")
    divergences: list[str] = Field(default_factory=list, max_length=3)
    narratives: list[str] = Field(default_factory=list, max_length=3)
    catalysts: list[str] = Field(default_factory=list, max_length=3)
    risks: list[str] = Field(default_factory=list, max_length=3)


# ---------------------------------------------------------------------------
# Python-side count/availability parsing of the pre-fetched prompt blocks
# ---------------------------------------------------------------------------
#
# The blocks are the same plaintext injected into the prompt (see
# sentiment_analyst.py). Parsing them here rather than re-fetching keeps
# this to a single network round-trip per source, at the cost of coupling
# to the fetchers' fixed text formats (tradingagents/dataflows/stocktwits.py,
# reddit.py, and the yfinance "### <headline>" news format). If those
# formats ever diverge, prefer extending the fetchers to also return
# structured data rather than growing these regexes.

_STOCKTWITS_UNAVAILABLE_RE = re.compile(r"^<stocktwits unavailable:\s*(.+?)>")
_STOCKTWITS_SUMMARY_RE = re.compile(
    r"Bullish:\s*(\d+)\s*\(\d+%\)\s*·\s*Bearish:\s*(\d+)\s*\(\d+%\)\s*·\s*"
    r"Unlabeled:\s*(\d+)\s*·\s*Total:\s*(\d+)"
)
_REDDIT_HEADER_RE = re.compile(r"r/\S+\s+—\s+(\d+)\s+recent posts")
_REDDIT_ENGAGEMENT_RE = re.compile(r"(\d+)↑\s*·\s*(\d+)c")
_NEWS_HEADLINE_RE = re.compile(r"^### ", re.MULTILINE)


def compute_news_counts(block) -> dict:
    """Parse the pre-fetched news block into ``{available, headline_count, reason}``.

    Headlines are yfinance-format ``### <title> (source: ...)`` lines
    (see ``tradingagents/dataflows/yfinance_news.py``); a fetch/routing
    error surfaces as an ``"Error fetching news..."`` string, and a genuinely
    empty result as ``"No news found..."``.

    ``get_news`` is vendor-routed (``data_vendors.news_data``; default
    ``yfinance``, alternative ``alpha_vantage``) and ``route_to_vendor`` does
    NOT normalize output across vendors — ``alpha_vantage_news.get_news``
    returns a dict/JSON payload, and a no-coverage route yields a
    ``"NO_DATA_AVAILABLE: ..."`` sentinel string, neither of which is the
    yfinance ``"### "`` prose this parser recognizes. Rather than silently
    reporting such a block as a confirmed-empty ``available`` source (issue
    #71, acceptance criterion 4), any block that matches no recognized
    yfinance shape is surfaced as ``available: false`` with a caveat reason,
    so a vendor-format mismatch shows up in ``data_quality.caveats`` instead
    of masquerading as "no news".
    """
    # Non-string vendor responses (e.g. alpha_vantage's dict/JSON payload) are
    # by definition not the yfinance text format — flag, don't crash or miscount.
    if not isinstance(block, str):
        return {"available": False, "headline_count": 0, "reason": "unrecognized news format"}
    stripped = block.strip()
    if not stripped:
        return {"available": False, "headline_count": 0, "reason": "empty response"}
    if stripped.startswith("Error fetching news"):
        return {"available": False, "headline_count": 0, "reason": stripped[:80]}
    if stripped.startswith("NO_DATA_AVAILABLE"):
        return {"available": False, "headline_count": 0, "reason": stripped[:80]}
    if stripped.startswith("No news found"):
        return {"available": True, "headline_count": 0, "reason": None}
    count = len(_NEWS_HEADLINE_RE.findall(stripped))
    if count == 0:
        # Non-empty, no recognized sentinel, and zero "### " headlines: the
        # block is in an unexpected format (a non-yfinance vendor payload, an
        # unknown sentinel, etc.). Never present it as confirmed-empty
        # available data — surface it as unavailable + caveat (criterion 4).
        return {"available": False, "headline_count": 0, "reason": "unrecognized news format"}
    return {"available": True, "headline_count": count, "reason": None}


def compute_stocktwits_counts(block: str) -> dict:
    """Parse the pre-fetched StockTwits block into
    ``{available, message_count, bullish, bearish, unlabeled, reason}``.

    Messages without a user-labeled Bullish/Bearish tag count as
    ``unlabeled`` — they are not neutral-direction evidence.
    """
    stripped = (block or "").strip()
    empty = {"available": False, "message_count": 0, "bullish": 0, "bearish": 0, "unlabeled": 0}
    if not stripped:
        return {**empty, "reason": "empty response"}

    unavailable = _STOCKTWITS_UNAVAILABLE_RE.match(stripped)
    if unavailable:
        return {**empty, "reason": unavailable.group(1)}
    if stripped.startswith("<no StockTwits messages found"):
        return {**empty, "available": True, "reason": None}

    match = _STOCKTWITS_SUMMARY_RE.search(stripped)
    if not match:
        return {**empty, "reason": "unparseable response"}
    bullish, bearish, unlabeled, total = (int(x) for x in match.groups())
    return {
        "available": True,
        "message_count": total,
        "bullish": bullish,
        "bearish": bearish,
        "unlabeled": unlabeled,
        "reason": None,
    }


def compute_reddit_counts(block: str) -> dict:
    """Parse the pre-fetched Reddit block into
    ``{available, post_count, top_engagement, reason}``.

    ``fetch_reddit_posts`` degrades silently on network failure (returns the
    same "no posts found" placeholder as a genuinely quiet ticker), so this
    cannot distinguish "Reddit is down" from "nobody is talking about this
    ticker" — both surface as ``post_count: 0`` with ``available: true``.
    """
    stripped = (block or "").strip()
    if not stripped:
        return {"available": False, "post_count": 0, "top_engagement": None, "reason": "empty response"}
    if stripped.startswith("<no Reddit posts found"):
        return {"available": True, "post_count": 0, "top_engagement": None, "reason": None}

    counts = [int(n) for n in _REDDIT_HEADER_RE.findall(stripped)]
    post_count = sum(counts)
    engagements = [(int(s), int(c)) for s, c in _REDDIT_ENGAGEMENT_RE.findall(stripped)]
    top_engagement = None
    if engagements:
        score, comments = max(engagements, key=lambda pair: pair[0])
        top_engagement = {"score": score, "comments": comments}
    return {"available": True, "post_count": post_count, "top_engagement": top_engagement, "reason": None}


def build_sources_skeleton(news_block: str, stocktwits_block: str, reddit_block: str) -> dict:
    """Build the Python-computed portion of ``details.sources`` (counts/
    availability, direction/confidence/key_items left null pending the LLM).
    """
    news_counts = compute_news_counts(news_block)
    stocktwits_counts = compute_stocktwits_counts(stocktwits_block)
    reddit_counts = compute_reddit_counts(reddit_block)

    return {
        "news": {
            "available": news_counts["available"],
            "headline_count": news_counts["headline_count"],
            "direction": None,
            "confidence": None,
            "key_items": [],
            "_reason": news_counts["reason"],
        },
        "stocktwits": {
            "available": stocktwits_counts["available"],
            "message_count": stocktwits_counts["message_count"],
            "bullish": stocktwits_counts["bullish"],
            "bearish": stocktwits_counts["bearish"],
            "unlabeled": stocktwits_counts["unlabeled"],
            "direction": None,
            "confidence": None,
            "key_items": [],
            "_reason": stocktwits_counts["reason"],
        },
        "reddit": {
            "available": reddit_counts["available"],
            "post_count": reddit_counts["post_count"],
            "top_engagement": reddit_counts["top_engagement"],
            "direction": None,
            "confidence": None,
            "key_items": [],
            "_reason": reddit_counts["reason"],
        },
    }


_SOURCE_LABELS = {"news": "News", "stocktwits": "StockTwits", "reddit": "Reddit"}


def build_details(
    start_date: str,
    end_date: str,
    sources_skeleton: dict,
    llm_output: SentimentAnalystOutput | None = None,
) -> dict:
    """Merge the Python-computed source skeleton with the LLM's directional
    read (if any) into the final ``details`` payload for the envelope.

    ``llm_output=None`` (parse/validation failure, or no LLM call made)
    produces the graceful-degradation payload: Python-computed counts,
    null directions/confidences, empty synthesis fields.
    """
    sources = {name: {k: v for k, v in fields.items() if k != "_reason"} for name, fields in sources_skeleton.items()}

    if llm_output is not None:
        for name in ("news", "stocktwits", "reddit"):
            source = sources[name]
            if not source["available"]:
                continue
            assessment = getattr(llm_output, name)
            source["direction"] = assessment.direction
            source["confidence"] = assessment.confidence
            source["key_items"] = assessment.key_items[:3]
        overall_direction = llm_output.overall_direction
        divergences = llm_output.divergences[:3]
        narratives = llm_output.narratives[:3]
        catalysts = llm_output.catalysts[:3]
        risks = llm_output.risks[:3]
    else:
        overall_direction = None
        divergences = []
        narratives = []
        catalysts = []
        risks = []

    caveats = []
    for name, fields in sources_skeleton.items():
        if fields["available"]:
            continue
        label = _SOURCE_LABELS[name]
        reason = fields.get("_reason")
        caveats.append(f"{label} unavailable ({reason})" if reason else f"{label} unavailable")

    sources_available = sum(1 for fields in sources_skeleton.values() if fields["available"])

    return {
        "window": {"start": start_date, "end": end_date},
        "sources": sources,
        "overall_direction": overall_direction,
        "divergences": divergences,
        "narratives": narratives,
        "catalysts": catalysts,
        "risks": risks,
        "data_quality": {
            "sources_available": sources_available,
            "caveats": caveats,
        },
    }


def derive_signal_and_confidence(details: dict) -> tuple[str | None, str | None]:
    """Derive top-level ``signal``/``confidence`` from per-source directions.

    Per issue #71's derivation rules:
    - Per available source with a non-null direction: score
      POSITIVE=+1, NEUTRAL=0, NEGATIVE=-1, weighted by that source's
      confidence. Weighted mean ``s``: ``s > 0.33`` -> BUY,
      ``s < -0.33`` -> SELL, else HOLD.
    - Confidence: mean of the contributing sources' confidences ->
      ``> 0.7`` HIGH, ``> 0.4`` MEDIUM, else LOW; capped at LOW when
      fewer than 2 sources contributed.
    - No source has a direction -> ``(None, None)``.
    """
    score_map = {"POSITIVE": 1.0, "NEUTRAL": 0.0, "NEGATIVE": -1.0}
    sources = details.get("sources") or {}

    confidences = []
    weighted_sum = 0.0
    for source in sources.values():
        direction = source.get("direction")
        confidence = source.get("confidence")
        if direction not in score_map or confidence is None:
            continue
        confidences.append(float(confidence))
        weighted_sum += score_map[direction] * float(confidence)

    if not confidences:
        return None, None

    total_confidence = sum(confidences)
    s = weighted_sum / total_confidence if total_confidence else 0.0

    if s > 0.33:
        signal = "BUY"
    elif s < -0.33:
        signal = "SELL"
    else:
        signal = "HOLD"

    mean_conf = total_confidence / len(confidences)
    if len(confidences) < 2:
        confidence = "LOW"
    elif mean_conf > 0.7:
        confidence = "HIGH"
    elif mean_conf > 0.4:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    return signal, confidence


def build_json_envelope(
    signal: str | None,
    confidence: str | None,
    summary: str,
    details: dict,
    ticker: str,
    date: str,
) -> str:
    """Build the JSON envelope per ``skills/SCHEMA.md`` and issue #71.

    Args:
        signal: BUY/HOLD/SELL or None
        confidence: HIGH/MEDIUM/LOW or None
        summary: One-line human-readable verdict
        details: The merged sentiment analysis payload
        ticker: Stock ticker symbol
        date: ISO 8601 date (YYYY-MM-DD)

    Returns:
        JSON string serialized for storage in ``sentiment_report``
    """
    envelope = {
        "skill": "sentiment-analyst",
        "ticker": ticker,
        "date": date,
        "signal": signal,
        "confidence": confidence,
        "summary": summary,
        "details": details,
    }
    return json.dumps(envelope, indent=2)
