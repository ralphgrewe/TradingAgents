"""Macro fundamentals analyst computation module for JSON envelope building.

This module handles:
1. Pydantic schema for structured LLM output (macro regime read + drivers)
2. Python-side derivation of signal/confidence from conservative/risky ratings
3. JSON envelope building per skills/SCHEMA.md

Key design principle (issue #132, following #131's macro indicator pack): the
LLM never computes deltas, z-scores, or ratios — those are already attached
to every pack entry by ``tradingagents/dataflows/macro_pack.py``. The LLM's
only job is to *read* the prepared pack and characterize the macro regime,
mirroring the news analyst's "LLM scores, Python aggregates/derives"
division of labor (``news_computation.py``).

Envelope shape: the envelope carries only ``signal``/``confidence``/``summary``/
``details``, exactly like every other analyst envelope (``news_computation.py``).
``details`` stays compact (macro regime, rationale, and the handful of
indicators that drove the read) so it stays cheap to inject into downstream
prompts and to truncate for SQLite memory storage. The full deterministic pack
is deliberately *not* included anywhere in the envelope: the report file and
the envelope are the same object here (``reporting.py`` writes the envelope
string verbatim as ``macro.json``), and downstream consumers inject report
strings into trader/portfolio-manager prompts verbatim
(``format_analyst_reports_section``), so duplicating the pack would bloat those
prompts. The pack remains available on demand via ``get_macro_pack``.
"""

import json

from pydantic import BaseModel, Field


class RatingWithConfidence(BaseModel):
    """A rating (BUY/HOLD/SELL) with numeric confidence."""
    rating: str = Field(description="BUY, HOLD, or SELL", pattern="^(BUY|HOLD|SELL)$")
    confidence: float = Field(description="Confidence 0.0-1.0", ge=0.0, le=1.0)


class MacroDriver(BaseModel):
    """One indicator cited as a driver behind the macro regime read."""
    indicator: str = Field(
        description="Alias of the macro indicator pack entry this driver cites, "
        "e.g. 'fed_funds_rate', 'cpi', 'vix', 'gold'"
    )
    reading: str = Field(
        description="Short interpretation of this indicator's latest value/direction/"
        "z-score for the regime read, ≤160 chars",
    )


class MacroFundamentalsAnalystOutput(BaseModel):
    """Structured output from the macro fundamentals analyst LLM.

    The LLM reads the deterministic indicator pack (#131) and characterizes
    the overall macro regime, citing up to 5 driver indicators, plus
    conservative and risky ratings that drive the top-level signal/confidence
    (same derivation contract as the news analyst).
    """
    macro_regime: str = Field(
        description="Overall macro regime read",
        pattern="^(RISK_ON|RISK_OFF|NEUTRAL|MIXED)$",
    )
    regime_rationale: str = Field(
        description="One or two sentences on why, ≤240 chars",
    )
    drivers: list[MacroDriver] = Field(
        description="Up to 5 indicators that most drove the regime read",
        min_length=0, max_length=5,
    )

    # Conservative and risky ratings (fixed before signal/confidence derivation)
    conservative: RatingWithConfidence = Field(
        description="Conservative rating: what the macro backdrop argues for with low risk tolerance"
    )
    risky: RatingWithConfidence = Field(
        description="Risky rating: what the macro backdrop argues for with high risk tolerance"
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
    Build the JSON envelope per skills/SCHEMA.md and the news-analyst-derived
    macro fundamentals specification (issue #132).

    Args:
        signal: BUY/HOLD/SELL or None
        confidence: HIGH/MEDIUM/LOW or None
        summary: One-line human-readable verdict
        details: The compact macro regime read (regime, rationale, drivers,
            conservative/risky ratings) — never the full indicator pack.
        ticker: Stock/asset ticker symbol
        date: ISO 8601 date (YYYY-MM-DD)

    Returns:
        JSON string serialized for storage in macro_report
    """
    envelope = {
        "skill": "macro-fundamentals-analyst",
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

    Mirrors ``news_computation.derive_signal_and_confidence``:
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
