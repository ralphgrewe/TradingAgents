"""News analyst computation module for aggregated per-article scoring + JSON envelope.

This module handles:
1. Pydantic schemas for structured LLM output (per-article scoring)
2. Python-side aggregation of top items and counts
3. Python-side derivation of signal/confidence from conservative/risky ratings
4. JSON envelope building per skills/SCHEMA.md

Key design principle: LLM scores articles, Python aggregates and derives top-level signal.
"""

import json

from pydantic import BaseModel, Field


class ArticleScore(BaseModel):
    """Score for a single article (internal, not in envelope)."""
    headline: str = Field(description="Article headline/title")
    fundamentals_impact: str = Field(
        description="POSITIVE, NEUTRAL, or NEGATIVE",
        pattern="^(POSITIVE|NEUTRAL|NEGATIVE)$"
    )
    fundamentals_confidence: float = Field(
        description="Confidence 0.0-1.0 for fundamentals impact",
        ge=0.0, le=1.0
    )
    sentiment_impact: str = Field(
        description="POSITIVE, NEUTRAL, or NEGATIVE",
        pattern="^(POSITIVE|NEUTRAL|NEGATIVE)$"
    )
    sentiment_confidence: float = Field(
        description="Confidence 0.0-1.0 for sentiment impact",
        ge=0.0, le=1.0
    )


class RatingWithConfidence(BaseModel):
    """A rating (BUY/HOLD/SELL) with numeric confidence."""
    rating: str = Field(description="BUY, HOLD, or SELL", pattern="^(BUY|HOLD|SELL)$")
    confidence: float = Field(description="Confidence 0.0-1.0", ge=0.0, le=1.0)


class NewsAnalystOutput(BaseModel):
    """Structured output from the news analyst LLM.

    The LLM scores articles internally and provides top items aggregated
    per quadrant (fundamentals/sentiment × positive/negative), plus
    conservative and risky ratings that drive the top-level signal/confidence.
    """
    articles_analyzed: int = Field(description="Number of articles analyzed", ge=0)
    window_days: int = Field(description="Days of news coverage analyzed", ge=1)

    # Top items per quadrant (≤3 items each, ≤120 chars, no URLs/scores)
    top_positive_fundamentals: list[str] = Field(
        description="Up to 3 strongest positive fundamental headlines (≤120 chars each, no URLs/scores)",
        min_length=0, max_length=3
    )
    top_negative_fundamentals: list[str] = Field(
        description="Up to 3 strongest negative fundamental headlines (≤120 chars each)",
        min_length=0, max_length=3
    )
    top_positive_sentiment: list[str] = Field(
        description="Up to 3 strongest positive sentiment headlines (≤120 chars each)",
        min_length=0, max_length=3
    )
    top_negative_sentiment: list[str] = Field(
        description="Up to 3 strongest negative sentiment headlines (≤120 chars each)",
        min_length=0, max_length=3
    )

    # Aggregate counts and confidences
    fundamentals_counts: dict = Field(
        description="Counts and avg confidences for fundamentals: {positive: {count, avg_confidence}, neutral: {...}, negative: {...}}"
    )
    sentiment_counts: dict = Field(
        description="Counts and avg confidences for sentiment: {positive: {count, avg_confidence}, ...}"
    )

    # Conservative and risky ratings (fixed before signal/confidence derivation)
    conservative: RatingWithConfidence = Field(
        description="Conservative rating: what the news argues for with low risk tolerance"
    )
    risky: RatingWithConfidence = Field(
        description="Risky rating: what the news argues for with high risk tolerance"
    )


def build_json_envelope(
    signal: str | None,
    confidence: str | None,
    summary: str,
    details: dict,
    ticker: str,
    date: str,
) -> str:
    """
    Build the JSON envelope per skills/SCHEMA.md and the news skill specification.

    Args:
        signal: BUY/HOLD/SELL or None
        confidence: HIGH/MEDIUM/LOW or None
        summary: One-line human-readable verdict
        details: The aggregated news analysis payload
        ticker: Stock ticker symbol
        date: ISO 8601 date (YYYY-MM-DD)

    Returns:
        JSON string serialized for storage in news_report
    """
    envelope = {
        "skill": "financial-news-analyst",
        "ticker": ticker,
        "date": date,
        "signal": signal,
        "confidence": confidence,
        "summary": summary,
        "details": details,
    }
    return json.dumps(envelope, indent=2)


def derive_signal_and_confidence(
    details: dict,
) -> tuple[str | None, str | None]:
    """
    Derive top-level signal and confidence from conservative/risky ratings.

    Per skills/news/SKILL.md Step 5:
    - signal = conservative.rating (fixed before Past Context is consulted)
    - confidence:
      - HIGH if mean(conservative.confidence, risky.confidence) > 0.7
      - MEDIUM if > 0.4
      - else LOW

    Args:
        details: The LLM's aggregated output, containing conservative/risky ratings

    Returns:
        (signal, confidence) tuple with both as strings or None if ratings missing
    """
    conservative = details.get("conservative")
    risky = details.get("risky")

    if not conservative or not risky:
        return None, None

    signal = conservative.get("rating") if isinstance(conservative, dict) else conservative.rating
    cons_conf = conservative.get("confidence") if isinstance(conservative, dict) else conservative.confidence
    risky_conf = risky.get("confidence") if isinstance(risky, dict) else risky.confidence

    if not signal or cons_conf is None or risky_conf is None:
        return None, None

    # Confidence mapping: mean > 0.7 → HIGH, > 0.4 → MEDIUM, else LOW
    mean_conf = (float(cons_conf) + float(risky_conf)) / 2.0
    if mean_conf > 0.7:
        confidence = "HIGH"
    elif mean_conf > 0.4:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    return str(signal), confidence
