"""Tests for wiring the quant skill into the shared memory core (issue #8).

`skills/quant/SKILL.md` is markdown-driven (no test harness of its own), but
the payload-construction logic it documents — mapping the envelope's
HIGH/MEDIUM/LOW `confidence` to a numeric score and building `key_drivers`
from `market_regime`/`convergence.confirms`/`.conflicts`/`trade_setup` — now
lives in `skills/quant/compute_indicators.py` as plain functions
(`confidence_to_score`, `build_key_drivers`), and the indicator computation
itself is exposed as a reusable `compute(records, ticker)` function. These
tests exercise that code directly against the shared memory core
(`tradingagents/memory/`), mirroring the issue's "Verify" section:

1. Running quant (`compute` + `memory_store_decision`-equivalent call to
   `store_decision`) writes a pending row to the memory DB.
2. With a backdated, resolved row, `get_past_context` surfaces the injected
   lesson.
3. That lesson/context never changes the deterministic `signal`/`confidence`
   `compute_indicators.py` produces — tested explicitly, not just implied by
   the fact that `compute()` happens to take no memory argument.

No live LLM or network access is used: OHLCV data is synthetic/deterministic
(no real ticker/vendor call), and `resolve_pending`'s internal LLM/yfinance
calls are monkeypatched, matching the precedent in
`test_memory_resolve.py` / `test_mcp_server_memory.py`.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from tradingagents.memory.resolve import DEFAULT_HORIZON_DAYS, resolve_pending
from tradingagents.memory.query import get_past_context
from tradingagents.memory.store import get_connection, store_decision

# ---------------------------------------------------------------------------
# Load skills/quant/compute_indicators.py as a module (it lives outside any
# Python package under skills/, so it's not importable via a dotted path).
# ---------------------------------------------------------------------------

_QUANT_DIR = Path(__file__).resolve().parents[1] / "skills" / "quant"
_SPEC = importlib.util.spec_from_file_location(
    "quant_compute_indicators", _QUANT_DIR / "compute_indicators.py"
)
compute_indicators = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(compute_indicators)

AGENT = "quant-indicator-analyst"


def _db_path(tmp_path):
    return tmp_path / "memory.db"


def _iso_days_ago(days):
    dt = datetime.now(timezone.utc) - timedelta(days=days)
    return dt.strftime("%Y-%m-%d")


# Backdated far enough that DEFAULT_HORIZON_DAYS trading days have definitely
# elapsed — same margin convention as test_memory_resolve.py.
BACKDATED = _iso_days_ago(DEFAULT_HORIZON_DAYS * 3 + 10)
RECENT = _iso_days_ago(0)


def _make_price_records(n=120, start=100.0, trend=0.5):
    """Deterministic, synthetic daily OHLCV series — a steady uptrend with a
    small oscillation, long enough (n >= 50) for every indicator
    (SMA-50 in particular) to be non-null. No network/vendor call — the
    closest testable equivalent to "a real ticker" per the issue's Verify
    section, per CLAUDE.md's no-live-API testing convention."""
    date0 = datetime(2026, 1, 1)
    price = start
    records = []
    for i in range(n):
        price += trend + 0.3 * math.sin(i / 5)
        date = date0 + timedelta(days=i)
        records.append(
            {
                "Date": date.strftime("%Y-%m-%d"),
                "Open": round(price - 0.2, 2),
                "High": round(price + 0.5, 2),
                "Low": round(price - 0.5, 2),
                "Close": round(price, 2),
                "Volume": 1_000_000 + (i % 10) * 1_000,
            }
        )
    return records


@pytest.fixture(autouse=True)
def _stub_resolve_mechanics():
    """Stub resolve_pending's internal LLM/yfinance calls for every test in
    this module — no live provider or network access required (matches
    test_memory_resolve.py / test_mcp_server_memory.py precedent)."""
    with patch(
        "tradingagents.memory.resolve._generate_lesson",
        return_value="The bullish call played out; the uptrend continued through the horizon.",
    ) as lesson_mock, patch(
        "tradingagents.memory.resolve._fetch_forward_return", return_value=0.06
    ) as return_mock:
        yield lesson_mock, return_mock


# ---------------------------------------------------------------------------
# Helper-function unit tests: confidence_to_score / build_key_drivers.
# ---------------------------------------------------------------------------


def test_confidence_to_score_mapping():
    assert compute_indicators.confidence_to_score("HIGH") == 1.0
    assert compute_indicators.confidence_to_score("MEDIUM") == 0.6
    assert compute_indicators.confidence_to_score("LOW") == 0.3
    # Case-insensitive, matching score_trader.py's conf_weight convention.
    assert compute_indicators.confidence_to_score("high") == 1.0
    # Unknown/missing falls back to LOW's score, same as conf_weight's default.
    assert compute_indicators.confidence_to_score("unknown") == 0.3


def test_build_key_drivers_extracts_fields_verbatim():
    details = {
        "market_regime": "trending_up",
        "convergence": {
            "confirms": ["sma_50", "macdh"],
            "conflicts": ["rsi bearish"],
            "missing": [],
        },
        "trade_setup": {"bias": "BUY", "entry_trigger": "x", "stop_loss": 1.0},
        "close": 123.4,  # not part of key_drivers — must be excluded
    }
    key_drivers = compute_indicators.build_key_drivers(details)
    assert key_drivers == {
        "market_regime": "trending_up",
        "confirms": ["sma_50", "macdh"],
        "conflicts": ["rsi bearish"],
        "trade_setup": {"bias": "BUY", "entry_trigger": "x", "stop_loss": 1.0},
    }
    assert "close" not in key_drivers


def test_build_key_drivers_handles_missing_convergence():
    details = {"market_regime": "ranging", "trade_setup": None}
    key_drivers = compute_indicators.build_key_drivers(details)
    assert key_drivers == {
        "market_regime": "ranging",
        "confirms": [],
        "conflicts": [],
        "trade_setup": None,
    }


# ---------------------------------------------------------------------------
# compute() takes no memory/context input — structural guarantee that
# injected lessons cannot reach the deterministic indicator math.
# ---------------------------------------------------------------------------


def test_compute_signature_has_no_memory_context_parameter():
    sig = inspect.signature(compute_indicators.compute)
    assert set(sig.parameters) == {"records", "ticker"}


# ---------------------------------------------------------------------------
# 1. Running quant writes a pending row to the memory DB.
# ---------------------------------------------------------------------------


def test_quant_decision_writes_pending_row(tmp_path):
    db_path = _db_path(tmp_path)
    records = _make_price_records()

    envelope = compute_indicators.compute(records, "AAPL")
    details = envelope["details"]

    inserted = store_decision(
        agent=AGENT,
        ticker="AAPL",
        date=details["as_of"],
        signal=envelope["signal"],
        confidence=compute_indicators.confidence_to_score(envelope["confidence"]),
        key_drivers=compute_indicators.build_key_drivers(details),
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
    assert row["decision_date"] == details["as_of"]
    assert row["signal"] == envelope["signal"]
    assert row["confidence"] == compute_indicators.confidence_to_score(envelope["confidence"])
    assert json.loads(row["key_drivers"]) == compute_indicators.build_key_drivers(details)
    assert row["thesis"] == envelope["summary"]
    # Pending: no resolution yet.
    assert row["resolved_at"] is None


# ---------------------------------------------------------------------------
# 2 & 3. Backdated + resolved row surfaces a lesson in context, and that
#         lesson/context never changes the deterministic signal/confidence.
# ---------------------------------------------------------------------------


def test_backdated_row_shows_lesson_without_changing_deterministic_signal(tmp_path):
    db_path = _db_path(tmp_path)
    records = _make_price_records()

    # "First run": compute the deterministic envelope and store a pending
    # decision, backdated so its horizon has already elapsed.
    first_envelope = compute_indicators.compute(records, "AAPL")
    first_details = first_envelope["details"]

    store_decision(
        agent=AGENT,
        ticker="AAPL",
        date=BACKDATED,
        signal=first_envelope["signal"],
        confidence=compute_indicators.confidence_to_score(first_envelope["confidence"]),
        key_drivers=compute_indicators.build_key_drivers(first_details),
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
    assert "uptrend continued through the horizon" in context_md  # the stubbed lesson

    # "Next run": recompute from the exact same price data. The deterministic
    # signal/confidence must be identical whether or not Past Context was
    # ever retrieved — compute() has no way to consume context_md at all
    # (see test_compute_signature_has_no_memory_context_parameter), and this
    # asserts the observable outcome of that structural guarantee too.
    second_envelope = compute_indicators.compute(records, "AAPL")

    assert second_envelope["signal"] == first_envelope["signal"]
    assert second_envelope["confidence"] == first_envelope["confidence"]
    assert second_envelope["details"] == first_envelope["details"]
    assert second_envelope == first_envelope


def test_resolve_pending_leaves_recent_quant_row_untouched(tmp_path):
    db_path = _db_path(tmp_path)
    records = _make_price_records()
    envelope = compute_indicators.compute(records, "MSFT")

    store_decision(
        agent=AGENT,
        ticker="MSFT",
        date=RECENT,
        signal=envelope["signal"],
        confidence=compute_indicators.confidence_to_score(envelope["confidence"]),
        key_drivers=compute_indicators.build_key_drivers(envelope["details"]),
        thesis=envelope["summary"],
        db_path=db_path,
    )

    resolved_ids = resolve_pending(agent=AGENT, ticker="MSFT", db_path=db_path)
    assert resolved_ids == []

    context_md = get_past_context(agent=AGENT, ticker="MSFT", db_path=db_path)
    assert "No prior resolved lessons yet." in context_md
