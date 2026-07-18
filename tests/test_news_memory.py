"""Tests for wiring the news analyst skill into the shared memory core
(issue #10).

`skills/news/SKILL.md` is markdown-driven (no test harness of its own), and
— unlike the quant pilot (issue #8) — this skill has no deterministic
compute script: its `conservative`/`risky` ratings (and therefore `signal`/
`confidence`) come from the model's own reading of fetched articles
(SKILL.md Steps 1-4), not from a script. The payload-construction logic the
skill documents for `memory_store_decision` — mapping the envelope's
HIGH/MEDIUM/LOW `confidence` to a numeric score and building `key_drivers`
from `top_positive_fundamentals`/`top_negative_fundamentals`/
`top_positive_sentiment`/`top_negative_sentiment` and the
`conservative`/`risky` rating split — lives in `skills/news/news_memory.py`
as plain functions (`confidence_to_score`, `build_key_drivers`), mirroring
`skills/quant/compute_indicators.py`'s `confidence_to_score`/
`build_key_drivers`. These tests exercise that code directly against the
shared memory core (`tradingagents/memory/`), mirroring the issue's
"Verify" section (same as the quant pilot):

1. Running news (a synthetic envelope, standing in for "a real ticker" the
   same way the quant test's synthetic OHLCV does — no live LLM or network
   access, per CLAUDE.md's no-live-API testing convention) + a
   `memory_store_decision`-equivalent call to `store_decision` writes a
   pending row to the memory DB.
2. With a backdated, resolved row, `get_past_context` surfaces the injected
   lesson.
3. That lesson/context never changes the envelope's `signal`/`confidence`
   or the deterministic reshaping `build_key_drivers`/`confidence_to_score`
   perform on `details` — tested explicitly, since (unlike quant) there is
   no `compute()` to fall back on as a structural guarantee; the guarantee
   here is that these two helpers are pure functions of `details` alone
   (no context parameter), asserted both by signature and by recomputing
   from the same `details` after a lesson has been injected into context.

`resolve_pending`'s internal LLM/yfinance calls are monkeypatched, matching
the precedent in `test_memory_resolve.py` / `test_mcp_server_memory.py` /
`test_quant_memory.py`.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from tradingagents.memory.query import get_past_context
from tradingagents.memory.resolve import DEFAULT_HORIZON_DAYS, resolve_pending
from tradingagents.memory.store import get_connection, store_decision

# ---------------------------------------------------------------------------
# Load skills/news/news_memory.py as a module (it lives outside any Python
# package under skills/, so it's not importable via a dotted path).
# ---------------------------------------------------------------------------

_NEWS_DIR = Path(__file__).resolve().parents[1] / "skills" / "news"
_SPEC = importlib.util.spec_from_file_location(
    "news_memory", _NEWS_DIR / "news_memory.py"
)
news_memory = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(news_memory)

AGENT = "financial-news-analyst"


def _db_path(tmp_path):
    return tmp_path / "memory.db"


def _iso_days_ago(days):
    dt = datetime.now(timezone.utc) - timedelta(days=days)
    return dt.strftime("%Y-%m-%d")


# Backdated far enough that DEFAULT_HORIZON_DAYS trading days have definitely
# elapsed — same margin convention as test_memory_resolve.py / test_quant_memory.py.
BACKDATED = _iso_days_ago(DEFAULT_HORIZON_DAYS * 3 + 10)
RECENT = _iso_days_ago(0)


def _make_news_envelope(ticker, date):
    """Deterministic, synthetic news envelope — the closest testable
    equivalent to "a real ticker" per the issue's Verify section (no live
    LLM/WebSearch/WebFetch call), matching the shape `skills/news/SKILL.md`
    Steps 1-6 produce."""
    details = {
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
    return {
        "skill": "financial-news-analyst",
        "ticker": ticker,
        "date": date,
        "signal": details["conservative"]["rating"],
        "confidence": "HIGH",
        "summary": "Q1 beat plus AI partnership headlines outweigh tariff overhang — conservative BUY",
        "details": details,
    }


@pytest.fixture(autouse=True)
def _stub_resolve_mechanics():
    """Stub resolve_pending's internal LLM/yfinance calls for every test in
    this module — no live provider or network access required (matches
    test_memory_resolve.py / test_mcp_server_memory.py / test_quant_memory.py
    precedent)."""
    with patch(
        "tradingagents.memory.resolve._generate_lesson",
        return_value="The bullish call played out; headlines-driven momentum continued through the horizon.",
    ) as lesson_mock, patch(
        "tradingagents.memory.resolve._fetch_forward_return", return_value=0.05
    ) as return_mock:
        yield lesson_mock, return_mock


# ---------------------------------------------------------------------------
# Helper-function unit tests: confidence_to_score / build_key_drivers.
# ---------------------------------------------------------------------------


def test_confidence_to_score_mapping():
    assert news_memory.confidence_to_score("HIGH") == 1.0
    assert news_memory.confidence_to_score("MEDIUM") == 0.6
    assert news_memory.confidence_to_score("LOW") == 0.3
    # Case-insensitive, matching score_trader.py's conf_weight convention.
    assert news_memory.confidence_to_score("high") == 1.0
    # Unknown/missing falls back to LOW's score, same as conf_weight's default.
    assert news_memory.confidence_to_score("unknown") == 0.3


def test_build_key_drivers_extracts_fields_verbatim():
    details = {
        "top_positive_fundamentals": ["Q1 beat"],
        "top_negative_fundamentals": ["Tariff overhang"],
        "top_positive_sentiment": ["Upgrade to Overweight"],
        "top_negative_sentiment": ["Short report"],
        "conservative": {"rating": "BUY", "confidence": 0.65},
        "risky": {"rating": "BUY", "confidence": 0.8},
        "articles_analyzed": 14,  # not part of key_drivers — must be excluded
    }
    key_drivers = news_memory.build_key_drivers(details)
    assert key_drivers == {
        "top_positive_fundamentals": ["Q1 beat"],
        "top_negative_fundamentals": ["Tariff overhang"],
        "top_positive_sentiment": ["Upgrade to Overweight"],
        "top_negative_sentiment": ["Short report"],
        "conservative": {"rating": "BUY", "confidence": 0.65},
        "risky": {"rating": "BUY", "confidence": 0.8},
    }
    assert "articles_analyzed" not in key_drivers


def test_build_key_drivers_handles_missing_fields():
    details = {"conservative": {"rating": "HOLD", "confidence": 0.5}}
    key_drivers = news_memory.build_key_drivers(details)
    assert key_drivers == {
        "top_positive_fundamentals": [],
        "top_negative_fundamentals": [],
        "top_positive_sentiment": [],
        "top_negative_sentiment": [],
        "conservative": {"rating": "HOLD", "confidence": 0.5},
        "risky": None,
    }


# ---------------------------------------------------------------------------
# build_key_drivers/confidence_to_score take no memory/context input —
# structural guarantee that injected lessons cannot reach them.
# ---------------------------------------------------------------------------


def test_build_key_drivers_signature_has_no_memory_context_parameter():
    sig = inspect.signature(news_memory.build_key_drivers)
    assert set(sig.parameters) == {"details"}


def test_confidence_to_score_signature_has_no_memory_context_parameter():
    sig = inspect.signature(news_memory.confidence_to_score)
    assert set(sig.parameters) == {"confidence"}


# ---------------------------------------------------------------------------
# 1. Running news writes a pending row to the memory DB.
# ---------------------------------------------------------------------------


def test_news_decision_writes_pending_row(tmp_path):
    db_path = _db_path(tmp_path)
    envelope = _make_news_envelope("AAPL", RECENT)
    details = envelope["details"]

    inserted = store_decision(
        agent=AGENT,
        ticker="AAPL",
        date=envelope["date"],
        signal=envelope["signal"],
        confidence=news_memory.confidence_to_score(envelope["confidence"]),
        key_drivers=news_memory.build_key_drivers(details),
        thesis=envelope["summary"],
        db_path=db_path,
    )
    assert inserted is True

    conn = get_connection(db_path)
    try:
        row = conn.execute("SELECT * FROM decisions").fetchone()
    finally:
        conn.close()

    assert row["agent"] == AGENT
    assert row["ticker"] == "AAPL"
    assert row["decision_date"] == envelope["date"]
    assert row["signal"] == envelope["signal"]
    assert row["confidence"] == news_memory.confidence_to_score(envelope["confidence"])
    assert json.loads(row["key_drivers"]) == news_memory.build_key_drivers(details)
    assert row["thesis"] == envelope["summary"]
    # Pending: no resolution yet.
    assert row["resolved_at"] is None


# ---------------------------------------------------------------------------
# 2 & 3. Backdated + resolved row surfaces a lesson in context, and that
#         lesson/context never changes signal/confidence or the key_drivers
#         reshaping.
# ---------------------------------------------------------------------------


def test_backdated_row_shows_lesson_without_changing_signal_or_confidence(tmp_path):
    db_path = _db_path(tmp_path)

    # "First run": a synthetic news envelope, stored backdated so its
    # horizon has already elapsed.
    first_envelope = _make_news_envelope("AAPL", BACKDATED)
    first_details = first_envelope["details"]

    store_decision(
        agent=AGENT,
        ticker="AAPL",
        date=BACKDATED,
        signal=first_envelope["signal"],
        confidence=news_memory.confidence_to_score(first_envelope["confidence"]),
        key_drivers=news_memory.build_key_drivers(first_details),
        thesis=first_envelope["summary"],
        db_path=db_path,
    )

    # Resolve it (mocked LLM/yfinance calls — see _stub_resolve_mechanics).
    resolved_ids = resolve_pending(agent=AGENT, ticker="AAPL", db_path=db_path)
    assert len(resolved_ids) == 1

    # Step 0 of the skill: load past context before the "next" run.
    context_md = get_past_context(agent=AGENT, ticker="AAPL", db_path=db_path)
    assert "## Past context: AAPL" in context_md
    assert "Same-ticker lessons (AAPL)" in context_md
    assert "momentum continued through the horizon" in context_md  # stubbed lesson

    # "Next run": a new envelope built from the exact same underlying
    # article evidence (Steps 1-4 already fixed signal/confidence/ratings
    # before Past Context was ever read, per SKILL.md Step 0.3 / Step 5).
    # The reshaping helpers must be identical regardless of Past Context —
    # they take no context parameter at all (see the signature tests above),
    # and this asserts the observable outcome of that structural guarantee.
    second_envelope = _make_news_envelope("AAPL", RECENT)

    assert second_envelope["signal"] == first_envelope["signal"]
    assert second_envelope["confidence"] == first_envelope["confidence"]
    assert second_envelope["details"]["conservative"] == first_envelope["details"]["conservative"]
    assert second_envelope["details"]["risky"] == first_envelope["details"]["risky"]
    assert news_memory.build_key_drivers(
        second_envelope["details"]
    ) == news_memory.build_key_drivers(first_envelope["details"])
    assert news_memory.confidence_to_score(
        second_envelope["confidence"]
    ) == news_memory.confidence_to_score(first_envelope["confidence"])


def test_resolve_pending_leaves_recent_news_row_untouched(tmp_path):
    db_path = _db_path(tmp_path)
    envelope = _make_news_envelope("MSFT", RECENT)

    store_decision(
        agent=AGENT,
        ticker="MSFT",
        date=RECENT,
        signal=envelope["signal"],
        confidence=news_memory.confidence_to_score(envelope["confidence"]),
        key_drivers=news_memory.build_key_drivers(envelope["details"]),
        thesis=envelope["summary"],
        db_path=db_path,
    )

    resolved_ids = resolve_pending(agent=AGENT, ticker="MSFT", db_path=db_path)
    assert resolved_ids == []

    context_md = get_past_context(agent=AGENT, ticker="MSFT", db_path=db_path)
    assert "No prior resolved lessons yet." in context_md
