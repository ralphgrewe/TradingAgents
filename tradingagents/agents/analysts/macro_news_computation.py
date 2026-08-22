"""Macro news analyst computation module for JSON envelope building.

This module handles:
1. Pydantic schema for structured LLM output (per-category sentiment scoring)
2. Python-side aggregation of category-level signals into one overall signal
3. JSON envelope building per skills/SCHEMA.md

Key design principle (issue #134, following #132's macro fundamentals pattern):
the LLM scores articles per category; Python aggregates category signals into
one overall signal. The category taxonomy is fixed (issue #128/#133):
monetary_policy, inflation_prices, labor_market, growth_output,
markets_volatility, geopolitical_trade. Per-category sentiment axes collapse
into one signal per category (Python-computed), then into one overall signal
(Python-computed), matching #128's specification and the LEARNINGS.md principle
of pre-computing in Python.

Envelope shape: the envelope carries only ``signal``/``confidence``/``summary``/
``details``, exactly like every other analyst envelope. ``details`` includes
per-category sentiment scores and the top articles cited. The full macro news
pack is deliberately *not* included in the envelope: it stays available on
demand via ``get_macro_news`` (the #133 tool), and downstream consumers inject
report strings verbatim into trader/portfolio-manager prompts
(``format_analyst_reports_section``), so duplicating the pack would bloat those
prompts.
"""

import json

from pydantic import BaseModel, Field


class CategorySentiment(BaseModel):
    """Sentiment scores for one macro news category."""
    category: str = Field(
        description="One of the six macro categories: monetary_policy, inflation_prices, "
        "labor_market, growth_output, markets_volatility, geopolitical_trade"
    )
    bullish_count: int = Field(
        description="Number of articles with bullish/positive sentiment",
        ge=0
    )
    bearish_count: int = Field(
        description="Number of articles with bearish/negative sentiment",
        ge=0
    )
    neutral_count: int = Field(
        description="Number of articles with neutral sentiment",
        ge=0
    )
    sentiment_score: float = Field(
        description="Aggregate sentiment for this category (range -1.0 to 1.0; "
        "-1 = all bearish, 0 = balanced, 1 = all bullish)",
        ge=-1.0, le=1.0
    )
    top_articles: list[str] = Field(
        description="Up to 2 key article headlines from this category (≤100 chars each)",
        min_length=0, max_length=2
    )


class RatingWithConfidence(BaseModel):
    """A rating (BUY/HOLD/SELL) with numeric confidence."""
    rating: str = Field(description="BUY, HOLD, or SELL", pattern="^(BUY|HOLD|SELL)$")
    confidence: float = Field(description="Confidence 0.0-1.0", ge=0.0, le=1.0)


class MacroNewsAnalystOutput(BaseModel):
    """Structured output from the macro news analyst LLM.

    The LLM reads the deterministic macro news pack (#133, per-category articles
    that are deduplicated, category-tagged, recency-ordered, and capped in Python)
    and scores sentiment per category, then provides conservative and risky
    ratings that drive the top-level signal/confidence (same derivation contract
    as the news analyst and macro fundamentals analyst).
    """
    articles_analyzed: int = Field(description="Total number of articles analyzed", ge=0)
    categories_with_articles: list[str] = Field(
        description="Categories that had articles available",
        min_length=0
    )

    # Per-category sentiment breakdown
    category_sentiments: list[CategorySentiment] = Field(
        description="Sentiment analysis for each category with articles (up to 6)",
        min_length=0, max_length=6
    )

    # Conservative and risky ratings (fixed before signal/confidence derivation)
    conservative: RatingWithConfidence = Field(
        description="Conservative rating: what the macro news argues for with low risk tolerance"
    )
    risky: RatingWithConfidence = Field(
        description="Risky rating: what the macro news argues for with high risk tolerance"
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
    Build the JSON envelope per skills/SCHEMA.md and the macro news analyst
    specification (issue #134).

    Args:
        signal: BUY/HOLD/SELL or None
        confidence: HIGH/MEDIUM/LOW or None
        summary: One-line human-readable verdict
        details: The per-category sentiment breakdown (never the full macro news pack)
        ticker: Stock/asset ticker symbol
        date: ISO 8601 date (YYYY-MM-DD)

    Returns:
        JSON string serialized for storage in macro_news_report
    """
    envelope = {
        "skill": "macro-news-analyst",
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

    Mirrors ``news_computation.derive_signal_and_confidence`` and
    ``macro_fundamentals_computation.derive_signal_and_confidence``:
    - signal = conservative.rating (fixed before Past Context is consulted)
    - confidence:
      - HIGH if mean(conservative.confidence, risky.confidence) > 0.7
      - MEDIUM if > 0.4
      - else LOW

    Args:
        details: The LLM's structured output, containing conservative/risky ratings

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

    # Confidence mapping: mean > 0.7 -> HIGH, > 0.4 -> MEDIUM, else LOW
    mean_conf = (float(cons_conf) + float(risky_conf)) / 2.0
    if mean_conf > 0.7:
        confidence = "HIGH"
    elif mean_conf > 0.4:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    return str(signal), confidence
