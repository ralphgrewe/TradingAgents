"""Tests for wiring the trader skill into the shared memory core (issue #11).

`skills/trader/SKILL.md` is markdown-driven (no test harness of its own), but
the payload-construction logic it documents now lives in
`skills/trader/score_trader.py` as plain functions: `compute(fund, news,
quant)` (the composite-score math, extracted from what used to be inline in
`main()`) and `build_key_drivers(result, rationale, risk_note)` (new, issue
#11). These tests exercise that code directly against the shared memory core
(`tradingagents/memory/`), mirroring the issue's "Verify" section:

1. Running trader (`compute` + a `memory_store_decision`-equivalent call to
   `store_decision`) writes a pending row to the memory DB.
2. Analyst lessons (quant/fundamental/news) and hit-rate stats are visible
   in the context trader would receive at retrieval time — the Tier 2 /
   synthesis-level requirement this issue adds on top of the Tier 1 pattern
   (quant #8, fundamental #9, news #10): trader reads not just its own past
   lessons but the three analysts' most recent lessons for the ticker
   (`get_past_context(agent=<analyst>, ticker=<TICKER>, n_same=1,
   n_cross=0)`) plus a per-agent hit-rate table for the ticker
   (`get_statistics(ticker=<TICKER>)`, no `agent` filter, so
   `per_agent_ticker` covers all four agents).
3. `signal`/`confidence` produced by `compute()` are unchanged by the
   presence of memory data — tested explicitly via a regression test that
   runs `compute()` with the exact same three analyst envelopes both before
   and after populating/resolving memory-core rows for this ticker, and
   asserts the composite-score output is byte-for-byte identical. This is
   also guaranteed structurally: `compute()` takes only `fund`/`news`/
   `quant` as parameters, with no memory/context argument at all (asserted
   by signature, same precedent as `test_quant_memory.py`/
   `test_fundamental_memory.py`).

No live LLM or network access is used: analyst envelopes are
synthetic/deterministic, and `resolve_pending`'s internal LLM/yfinance calls
are monkeypatched, matching the precedent in `test_memory_resolve.py` /
`test_mcp_server_memory.py` / `test_quant_memory.py` / `test_news_memory.py`.
"""

from __future__ import annotations

import copy
import importlib.util
import inspect
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from tradingagents.memory.query import get_past_context
from tradingagents.memory.resolve import DEFAULT_HORIZON_DAYS, resolve_pending
from tradingagents.memory.stats import get_statistics
from tradingagents.memory.store import get_connection, store_decision

# ---------------------------------------------------------------------------
# Load skills/trader/score_trader.py as a module (it lives outside any
# Python package under skills/, so it's not importable via a dotted path).
# ---------------------------------------------------------------------------

_TRADER_DIR = Path(__file__).resolve().parents[1] / "skills" / "trader"
_SPEC = importlib.util.spec_from_file_location(
    "trader_score_trader", _TRADER_DIR / "score_trader.py"
)
score_trader = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(score_trader)

TRADER_AGENT = "trader"
FUNDAMENTAL_AGENT = "fundamental-analyst"
NEWS_AGENT = "financial-news-analyst"
QUANT_AGENT = "quant-indicator-analyst"


def _db_path(tmp_path):
    return tmp_path / "memory.db"


def _iso_days_ago(days):
    dt = datetime.now(timezone.utc) - timedelta(days=days)
    return dt.strftime("%Y-%m-%d")


# Backdated far enough that DEFAULT_HORIZON_DAYS trading days have definitely
# elapsed — same margin convention as test_memory_resolve.py / test_quant_memory.py.
BACKDATED = _iso_days_ago(DEFAULT_HORIZON_DAYS * 3 + 10)
RECENT = _iso_days_ago(0)


@pytest.fixture(autouse=True)
def _stub_resolve_mechanics():
    """Stub resolve_pending's internal LLM/yfinance calls for every test in
    this module — no live provider or network access required (matches
    test_memory_resolve.py / test_mcp_server_memory.py / test_quant_memory.py
    / test_news_memory.py precedent)."""
    with patch(
        "tradingagents.memory.resolve._generate_lesson",
        return_value="The bullish call played out; the composite score's confidence was validated.",
    ) as lesson_mock, patch(
        "tradingagents.memory.resolve._fetch_forward_return", return_value=0.04
    ) as return_mock:
        yield lesson_mock, return_mock


# ---------------------------------------------------------------------------
# Synthetic analyst envelopes — the closest testable equivalent to "a real
# ticker" per the issue's Verify section (no live LLM/network call), matching
# the shapes skills/fundamental, skills/news, skills/quant SKILL.md steps
# produce and score_trader.py's compute() expects.
# ---------------------------------------------------------------------------


def _make_fund_envelope(ticker="AAPL"):
    return {
        "skill": "fundamental-analyst",
        "ticker": ticker,
        "date": RECENT,
        "signal": "BUY",
        "confidence": "HIGH",
        "details": {
            "value": {"signal": "BUY", "confidence": "HIGH", "key_ratios": ["pe", "roic"]},
            "growth": {"signal": "BUY", "confidence": "MEDIUM", "key_ratios": ["fcf_margin"]},
            "insider_sentiment": "BULLISH",
        },
    }


def _make_news_envelope(ticker="AAPL"):
    return {
        "skill": "financial-news-analyst",
        "ticker": ticker,
        "date": RECENT,
        "signal": "BUY",
        "confidence": "HIGH",
        "details": {
            "conservative": {"rating": "BUY", "confidence": 0.7},
            "risky": {"rating": "BUY", "confidence": 0.8},
            "fundamentals_counts": {
                "positive": {"count": 6, "avg_confidence": 0.7},
                "neutral": {"count": 2, "avg_confidence": 0.5},
                "negative": {"count": 1, "avg_confidence": 0.4},
            },
            "sentiment_counts": {
                "positive": {"count": 5, "avg_confidence": 0.65},
                "neutral": {"count": 3, "avg_confidence": 0.5},
                "negative": {"count": 1, "avg_confidence": 0.3},
            },
        },
    }


def _make_quant_envelope(ticker="AAPL"):
    return {
        "skill": "quant-indicator-analyst",
        "ticker": ticker,
        "date": RECENT,
        "signal": "BUY",
        "confidence": "HIGH",
        "details": {
            "trade_setup": {"bias": "BUY", "entry_trigger": "breakout", "stop_loss": 100.0},
            "indicators": [
                {"name": "sma_50", "signal": "Bullish"},
                {"name": "macdh", "signal": "Bullish"},
                {"name": "rsi", "signal": "Bearish"},
            ],
        },
    }


# ---------------------------------------------------------------------------
# Helper-function unit tests: build_key_drivers.
# ---------------------------------------------------------------------------


def test_build_key_drivers_extracts_fields_verbatim():
    result = {
        "signals": [{"id": "F1"}],  # not part of key_drivers — must be excluded
        "composite_score": 0.42,
        "thresholds": {"buy": 0.15, "sell": -0.15},  # excluded
        "analyst_aggregates": {"fundamental": "BUY", "news": "BUY", "quant": "BUY"},
        "conflicts": [],
        "signal": "BUY",  # excluded — signal/confidence aren't part of key_drivers
        "confidence": "HIGH",
    }
    key_drivers = score_trader.build_key_drivers(
        result, rationale="Strong agreement across analysts.", risk_note="Valuation risk."
    )
    assert key_drivers == {
        "composite_score": 0.42,
        "analyst_aggregates": {"fundamental": "BUY", "news": "BUY", "quant": "BUY"},
        "conflicts": [],
        "rationale": "Strong agreement across analysts.",
        "risk_note": "Valuation risk.",
    }
    assert "signals" not in key_drivers
    assert "thresholds" not in key_drivers


def test_build_key_drivers_handles_missing_fields():
    key_drivers = score_trader.build_key_drivers({})
    assert key_drivers == {
        "composite_score": None,
        "analyst_aggregates": None,
        "conflicts": [],
        "rationale": None,
        "risk_note": None,
    }


# ---------------------------------------------------------------------------
# compute() takes no memory/context input — structural guarantee that
# injected lessons cannot reach the deterministic composite-score math.
# ---------------------------------------------------------------------------


def test_compute_signature_has_no_memory_context_parameter():
    sig = inspect.signature(score_trader.compute)
    assert set(sig.parameters) == {"fund", "news", "quant"}


# ---------------------------------------------------------------------------
# 1. Running trader writes a pending row to the memory DB.
# ---------------------------------------------------------------------------


def test_trader_decision_writes_pending_row(tmp_path):
    db_path = _db_path(tmp_path)
    fund, news, quant = _make_fund_envelope(), _make_news_envelope(), _make_quant_envelope()

    result = score_trader.compute(fund, news, quant)
    rationale = "Fundamentals, news, and quant all confirm a bullish setup."
    risk_note = "A reversal in momentum would invalidate the quant signal."

    inserted = store_decision(
        agent=TRADER_AGENT,
        ticker="AAPL",
        date=RECENT,
        signal=result["signal"],
        confidence=score_trader.conf_weight(result["confidence"]),
        key_drivers=score_trader.build_key_drivers(result, rationale, risk_note),
        thesis=rationale,
        db_path=db_path,
    )
    assert inserted is True

    conn = get_connection(db_path)
    try:
        row = conn.execute("SELECT * FROM decisions").fetchone()
    finally:
        conn.close()

    assert row["agent"] == TRADER_AGENT
    assert row["ticker"] == "AAPL"
    assert row["decision_date"] == RECENT
    assert row["signal"] == result["signal"]
    assert row["confidence"] == score_trader.conf_weight(result["confidence"])
    assert json.loads(row["key_drivers"]) == score_trader.build_key_drivers(
        result, rationale, risk_note
    )
    assert row["thesis"] == rationale
    # Pending: no resolution yet.
    assert row["resolved_at"] is None


# ---------------------------------------------------------------------------
# 2. Analyst lessons (quant/fundamental/news) and hit-rate stats are visible
#    in the context trader receives at retrieval time.
# ---------------------------------------------------------------------------


def test_analyst_lessons_and_hit_rate_stats_visible_to_trader(tmp_path):
    db_path = _db_path(tmp_path)
    ticker = "AAPL"

    # Seed one backdated, resolved decision per analyst plus trader itself —
    # standing in for prior runs of each Tier-1 skill (issues #8/#9/#10) and
    # a prior trader run, all for the same ticker.
    for agent, signal in (
        (FUNDAMENTAL_AGENT, "BUY"),
        (NEWS_AGENT, "BUY"),
        (QUANT_AGENT, "SELL"),
        (TRADER_AGENT, "BUY"),
    ):
        store_decision(
            agent=agent,
            ticker=ticker,
            date=BACKDATED,
            signal=signal,
            confidence=0.8,
            key_drivers={"note": f"{agent} backdated decision"},
            thesis=f"{agent} thesis",
            db_path=db_path,
        )

    # Resolve every pending row across all four agents (mocked LLM/yfinance
    # calls — see _stub_resolve_mechanics).
    resolved_ids = resolve_pending(ticker=ticker, db_path=db_path)
    assert len(resolved_ids) == 4

    # Step 1.2 of the skill: trader's own past context.
    trader_context = get_past_context(agent=TRADER_AGENT, ticker=ticker, db_path=db_path)
    assert "## Past context: AAPL" in trader_context
    assert "Same-ticker lessons (AAPL)" in trader_context
    assert "composite score's confidence was validated" in trader_context

    # Step 1.3 of the skill: each analyst's most recent lesson for this
    # ticker only (n_same=1, n_cross=0).
    for agent in (FUNDAMENTAL_AGENT, NEWS_AGENT, QUANT_AGENT):
        analyst_context = get_past_context(
            agent=agent, ticker=ticker, n_same=1, n_cross=0, db_path=db_path
        )
        assert f"## Past context: {ticker}" in analyst_context
        assert f"Same-ticker lessons ({ticker})" in analyst_context
        assert "composite score's confidence was validated" in analyst_context
        # n_cross=0 means no cross-ticker section, even if one existed.
        assert "Cross-ticker lessons" not in analyst_context

    # Step 1.4 of the skill: per-agent hit-rate table for this ticker, no
    # agent filter, so all four agents show up.
    stats = get_statistics(ticker=ticker, db_path=db_path)
    agents_in_table = {rec["agent"] for rec in stats["per_agent_ticker"]}
    assert agents_in_table == {FUNDAMENTAL_AGENT, NEWS_AGENT, QUANT_AGENT, TRADER_AGENT}
    for rec in stats["per_agent_ticker"]:
        assert rec["ticker"] == ticker
        assert rec["n"] == 1
        # forward_return stubbed to 0.04 (positive) for every row above.
        if rec["agent"] == QUANT_AGENT:
            # SELL with a positive forward return is "incorrect" per
            # tradingagents.memory.stats's correctness rule.
            assert rec["hit_rate"] == 0.0
        else:
            assert rec["hit_rate"] == 1.0


# ---------------------------------------------------------------------------
# 3. signal/confidence from compute() are unchanged by the presence of
#    memory data — both structurally (no context parameter) and empirically
#    (identical output before vs. after populating/resolving memory rows).
# ---------------------------------------------------------------------------


def test_compute_output_unchanged_with_or_without_memory_data(tmp_path):
    db_path = _db_path(tmp_path)
    fund, news, quant = _make_fund_envelope(), _make_news_envelope(), _make_quant_envelope()

    # "Before": no memory data exists yet for this ticker/agent at all.
    before = score_trader.compute(copy.deepcopy(fund), copy.deepcopy(news), copy.deepcopy(quant))

    # Populate and resolve memory-core rows for trader and all three
    # analysts on this exact ticker (as Step 1 of the skill would produce
    # "Past Context" from) — this is the memory data whose mere existence
    # must not affect compute()'s output.
    for agent in (FUNDAMENTAL_AGENT, NEWS_AGENT, QUANT_AGENT, TRADER_AGENT):
        store_decision(
            agent=agent,
            ticker="AAPL",
            date=BACKDATED,
            signal="SELL",
            confidence=0.9,
            key_drivers={"note": "a very different prior call"},
            thesis="prior thesis",
            db_path=db_path,
        )
    resolve_pending(ticker="AAPL", db_path=db_path)
    past_context = get_past_context(agent=TRADER_AGENT, ticker="AAPL", db_path=db_path)
    assert "No prior resolved lessons yet." not in past_context  # sanity: lessons do exist now
    get_statistics(ticker="AAPL", db_path=db_path)  # exercised, result intentionally unused

    # "After": recompute from the exact same three analyst envelopes.
    # compute() has no way to consume past_context/statistics at all (see
    # test_compute_signature_has_no_memory_context_parameter) — this
    # asserts the observable outcome of that structural guarantee.
    after = score_trader.compute(copy.deepcopy(fund), copy.deepcopy(news), copy.deepcopy(quant))

    assert after == before
    assert after["signal"] == before["signal"]
    assert after["confidence"] == before["confidence"]
    assert after["composite_score"] == before["composite_score"]


def test_resolve_pending_leaves_recent_trader_row_untouched(tmp_path):
    db_path = _db_path(tmp_path)
    fund, news, quant = _make_fund_envelope("MSFT"), _make_news_envelope("MSFT"), _make_quant_envelope("MSFT")
    result = score_trader.compute(fund, news, quant)

    store_decision(
        agent=TRADER_AGENT,
        ticker="MSFT",
        date=RECENT,
        signal=result["signal"],
        confidence=score_trader.conf_weight(result["confidence"]),
        key_drivers=score_trader.build_key_drivers(result, "rationale", "risk note"),
        thesis="rationale",
        db_path=db_path,
    )

    resolved_ids = resolve_pending(agent=TRADER_AGENT, ticker="MSFT", db_path=db_path)
    assert resolved_ids == []

    context_md = get_past_context(agent=TRADER_AGENT, ticker="MSFT", db_path=db_path)
    assert "No prior resolved lessons yet." in context_md
