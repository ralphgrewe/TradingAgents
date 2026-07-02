"""Tests for wiring the fundamental skill into the shared memory core (issue #9).

`skills/fundamental/SKILL.md` is markdown-driven (no test harness of its own), but
the payload-construction logic it documents — mapping the envelope's
HIGH/MEDIUM/LOW `confidence` to a numeric score and building `key_drivers`
from `value`/`growth` sub-signals (`signal`/`confidence`/`key_ratios`) plus
`insider_sentiment` — now lives in `skills/fundamental/compute_ratios.py` as
plain functions (`confidence_to_score`, `build_key_drivers`), and the ratio
computation itself is exposed as a reusable
`compute(info_raw, income_raw, balance_raw, cashflow_raw, holders_raw)`
function. These tests exercise that code directly against the shared memory
core (`tradingagents/memory/`), mirroring the issue's "Verify" section (the
same section validated by the quant pilot, issue #8):

1. Running fundamental (`compute` + `memory_store_decision`-equivalent call
   to `store_decision`) writes a pending row to the memory DB.
2. With a backdated, resolved row, `get_past_context` surfaces the injected
   lesson.
3. That lesson/context never changes the deterministic ratio table
   `compute_ratios.py` produces — tested explicitly. Unlike quant,
   `value.signal`/`growth.signal` are the model's own judgment (not computed
   by this script), so the structural guarantee this module provides is one
   level down the pipeline: `compute()` (the ratio table those judgments are
   based on) takes no memory/context input at all, so it cannot vary run to
   run based on injected lessons, and `build_key_drivers`/`confidence_to_score`
   are themselves pure functions of an already-written envelope with no
   memory-context parameter either.

No live LLM or network access is used: ticker/financials data is
synthetic/deterministic (no real ticker/vendor call), and `resolve_pending`'s
internal LLM/yfinance calls are monkeypatched, matching the precedent in
`test_memory_resolve.py` / `test_mcp_server_memory.py` / `test_quant_memory.py`.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from tradingagents.memory.resolve import DEFAULT_HORIZON_DAYS, resolve_pending
from tradingagents.memory.query import get_past_context
from tradingagents.memory.store import get_connection, store_decision

# ---------------------------------------------------------------------------
# Load skills/fundamental/compute_ratios.py as a module (it lives outside any
# Python package under skills/, so it's not importable via a dotted path).
# ---------------------------------------------------------------------------

_FUNDAMENTAL_DIR = Path(__file__).resolve().parents[1] / "skills" / "fundamental"
_SPEC = importlib.util.spec_from_file_location(
    "fundamental_compute_ratios", _FUNDAMENTAL_DIR / "compute_ratios.py"
)
compute_ratios = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(compute_ratios)

AGENT = "fundamental-analyst"


def _db_path(tmp_path):
    return tmp_path / "memory.db"


def _iso_days_ago(days):
    dt = datetime.now(timezone.utc) - timedelta(days=days)
    return dt.strftime("%Y-%m-%d")


# Backdated far enough that DEFAULT_HORIZON_DAYS trading days have definitely
# elapsed — same margin convention as test_memory_resolve.py / test_quant_memory.py.
BACKDATED = _iso_days_ago(DEFAULT_HORIZON_DAYS * 3 + 10)
RECENT = _iso_days_ago(0)


def _make_fundamental_inputs(revenue=1_000_000_000.0, net_income=150_000_000.0):
    """Deterministic, synthetic yfinance-shaped raw inputs — the closest
    testable equivalent to "a real ticker" per the issue's Verify section,
    per CLAUDE.md's no-live-API testing convention."""
    info_raw = {
        "marketCap": 2_000_000_000,
        "enterpriseValue": 2_100_000_000,
        "sector": "Technology",
        "industry": "Software",
        "fiftyTwoWeekHigh": 120.0,
        "fiftyTwoWeekLow": 60.0,
        "recommendationKey": "buy",
        "trailingPE": 13.3,
        "priceToBook": 2.1,
        "priceToSalesTrailing12Months": 2.0,
        "pegRatio": 0.9,
        "enterpriseToEbitda": 9.5,
        "dividendYield": 0.01,
        "payoutRatio": 0.15,
    }
    income_raw = {
        "Total Revenue": {"2025-12-31": revenue, "2024-12-31": revenue * 0.9},
        "Gross Profit": {"2025-12-31": revenue * 0.6, "2024-12-31": revenue * 0.9 * 0.58},
        "Operating Income": {"2025-12-31": revenue * 0.25, "2024-12-31": revenue * 0.9 * 0.22},
        "Net Income": {"2025-12-31": net_income, "2024-12-31": net_income * 0.85},
        "Interest Expense": {"2025-12-31": 5_000_000, "2024-12-31": 5_000_000},
        "EBITDA": {"2025-12-31": revenue * 0.3, "2024-12-31": revenue * 0.9 * 0.27},
    }
    balance_raw = {
        "Total Assets": {"2025-12-31": 1_500_000_000, "2024-12-31": 1_400_000_000},
        "Total Stockholder Equity": {"2025-12-31": 950_000_000, "2024-12-31": 900_000_000},
        "Total Debt": {"2025-12-31": 300_000_000, "2024-12-31": 320_000_000},
        "Current Assets": {"2025-12-31": 600_000_000, "2024-12-31": 550_000_000},
        "Current Liabilities": {"2025-12-31": 300_000_000, "2024-12-31": 280_000_000},
        "Inventory": {"2025-12-31": 50_000_000, "2024-12-31": 45_000_000},
    }
    cashflow_raw = {
        "Operating Cash Flow": {"2025-12-31": 300_000_000, "2024-12-31": 260_000_000},
        "Capital Expenditure": {"2025-12-31": -60_000_000, "2024-12-31": -55_000_000},
    }
    holders_raw = {
        "insiderTransactions": [
            {"Transaction": "Buy", "Shares": 10_000},
            {"Transaction": "Sale", "Shares": 2_000},
        ]
    }
    return info_raw, income_raw, balance_raw, cashflow_raw, holders_raw


def _make_envelope_details(annual):
    """Build a fundamental-analyst `details` payload the way the model would
    after Steps 2-4 of SKILL.md: `compute()`'s deterministic output plus a
    model-written value/growth verdict (with `key_ratios` cited)."""
    return {
        "context": annual["context"],
        "annual": annual["annual"],
        "insider_sentiment": annual["insider_sentiment"],
        "forecast": {},
        "value": {
            "signal": "BUY",
            "confidence": "HIGH",
            "data_confidence": "HIGH",
            "key_ratios": ["pe", "roic", "debt_to_equity"],
        },
        "growth": {
            "signal": "HOLD",
            "confidence": "MEDIUM",
            "data_confidence": "MEDIUM",
            "key_ratios": ["gross_margin", "fcf_margin"],
        },
    }


@pytest.fixture(autouse=True)
def _stub_resolve_mechanics():
    """Stub resolve_pending's internal LLM/yfinance calls for every test in
    this module — no live provider or network access required (matches
    test_memory_resolve.py / test_mcp_server_memory.py / test_quant_memory.py
    precedent)."""
    with patch(
        "tradingagents.memory.resolve._generate_lesson",
        return_value="The value call played out; the re-rating toward fair value continued through the horizon.",
    ) as lesson_mock, patch(
        "tradingagents.memory.resolve._fetch_forward_return", return_value=0.04
    ) as return_mock:
        yield lesson_mock, return_mock


# ---------------------------------------------------------------------------
# Helper-function unit tests: confidence_to_score / build_key_drivers.
# ---------------------------------------------------------------------------


def test_confidence_to_score_mapping():
    assert compute_ratios.confidence_to_score("HIGH") == 1.0
    assert compute_ratios.confidence_to_score("MEDIUM") == 0.6
    assert compute_ratios.confidence_to_score("LOW") == 0.3
    # Case-insensitive, matching score_trader.py's conf_weight convention.
    assert compute_ratios.confidence_to_score("high") == 1.0
    # Unknown/missing falls back to LOW's score, same as conf_weight's default.
    assert compute_ratios.confidence_to_score("unknown") == 0.3


def test_build_key_drivers_extracts_fields_verbatim():
    details = {
        "value": {
            "signal": "BUY",
            "confidence": "HIGH",
            "data_confidence": "HIGH",
            "key_ratios": ["pe", "roic"],
        },
        "growth": {
            "signal": "HOLD",
            "confidence": "MEDIUM",
            "data_confidence": "MEDIUM",
            "key_ratios": ["fcf_margin"],
        },
        "insider_sentiment": "BULLISH",
        "annual": {"2025": {}},  # not part of key_drivers — must be excluded
    }
    key_drivers = compute_ratios.build_key_drivers(details)
    assert key_drivers == {
        "value": {"signal": "BUY", "confidence": "HIGH", "key_ratios": ["pe", "roic"]},
        "growth": {"signal": "HOLD", "confidence": "MEDIUM", "key_ratios": ["fcf_margin"]},
        "insider_sentiment": "BULLISH",
    }
    assert "annual" not in key_drivers
    assert "data_confidence" not in key_drivers["value"]


def test_build_key_drivers_handles_missing_value_growth():
    details = {"insider_sentiment": "NEUTRAL"}
    key_drivers = compute_ratios.build_key_drivers(details)
    assert key_drivers == {
        "value": {"signal": None, "confidence": None, "key_ratios": []},
        "growth": {"signal": None, "confidence": None, "key_ratios": []},
        "insider_sentiment": "NEUTRAL",
    }


# ---------------------------------------------------------------------------
# compute() takes no memory/context input — structural guarantee that
# injected lessons cannot reach the deterministic ratio table.
# ---------------------------------------------------------------------------


def test_compute_signature_has_no_memory_context_parameter():
    sig = inspect.signature(compute_ratios.compute)
    assert set(sig.parameters) == {
        "info_raw",
        "income_raw",
        "balance_raw",
        "cashflow_raw",
        "holders_raw",
    }


# ---------------------------------------------------------------------------
# 1. Running fundamental writes a pending row to the memory DB.
# ---------------------------------------------------------------------------


def test_fundamental_decision_writes_pending_row(tmp_path):
    db_path = _db_path(tmp_path)
    info_raw, income_raw, balance_raw, cashflow_raw, holders_raw = _make_fundamental_inputs()

    ratio_table = compute_ratios.compute(info_raw, income_raw, balance_raw, cashflow_raw, holders_raw)
    details = _make_envelope_details(ratio_table)
    # Explicit envelope-level signal/confidence/date/summary (model output,
    # per SKILL.md Step 4), independent of compute()'s ratio table.
    envelope_signal = "HOLD"
    envelope_confidence = "MEDIUM"
    envelope_date = "2026-06-01"
    envelope_summary = "Below intrinsic value but margins compressing — value BUY, growth HOLD"

    inserted = store_decision(
        agent=AGENT,
        ticker="AAPL",
        date=envelope_date,
        signal=envelope_signal,
        confidence=compute_ratios.confidence_to_score(envelope_confidence),
        key_drivers=compute_ratios.build_key_drivers(details),
        thesis=envelope_summary,
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
    assert row["decision_date"] == envelope_date
    assert row["signal"] == envelope_signal
    assert row["confidence"] == compute_ratios.confidence_to_score(envelope_confidence)
    assert json.loads(row["key_drivers"]) == compute_ratios.build_key_drivers(details)
    assert row["thesis"] == envelope_summary
    # Pending: no resolution yet.
    assert row["resolved_at"] is None


# ---------------------------------------------------------------------------
# 2 & 3. Backdated + resolved row surfaces a lesson in context, and that
#         lesson/context never changes the deterministic ratio table.
# ---------------------------------------------------------------------------


def test_backdated_row_shows_lesson_without_changing_deterministic_ratios(tmp_path):
    db_path = _db_path(tmp_path)
    info_raw, income_raw, balance_raw, cashflow_raw, holders_raw = _make_fundamental_inputs()

    # "First run": compute the deterministic ratio table and store a pending
    # decision, backdated so its horizon has already elapsed.
    first_ratio_table = compute_ratios.compute(info_raw, income_raw, balance_raw, cashflow_raw, holders_raw)
    first_details = _make_envelope_details(first_ratio_table)

    store_decision(
        agent=AGENT,
        ticker="AAPL",
        date=BACKDATED,
        signal="BUY",
        confidence=compute_ratios.confidence_to_score("HIGH"),
        key_drivers=compute_ratios.build_key_drivers(first_details),
        thesis="Below intrinsic value, strong balance sheet — value BUY, growth BUY",
        db_path=db_path,
    )

    # Resolve it (mocked LLM/yfinance calls — see _stub_resolve_mechanics).
    resolved_ids = resolve_pending(agent=AGENT, ticker="AAPL", db_path=db_path)
    assert len(resolved_ids) == 1

    # Step 0 of the skill: load past context before the "next" run.
    context_md = get_past_context(agent=AGENT, ticker="AAPL", db_path=db_path)
    assert "## Past context: AAPL" in context_md
    assert "Same-ticker lessons (AAPL)" in context_md
    assert "re-rating toward fair value continued through the horizon" in context_md  # stubbed lesson

    # "Next run": recompute the ratio table from the exact same raw inputs.
    # The deterministic ratio table must be identical whether or not Past
    # Context was ever retrieved — compute() has no way to consume
    # context_md at all (see
    # test_compute_signature_has_no_memory_context_parameter), and this
    # asserts the observable outcome of that structural guarantee too.
    second_ratio_table = compute_ratios.compute(info_raw, income_raw, balance_raw, cashflow_raw, holders_raw)

    assert second_ratio_table == first_ratio_table


def test_resolve_pending_leaves_recent_fundamental_row_untouched(tmp_path):
    db_path = _db_path(tmp_path)
    info_raw, income_raw, balance_raw, cashflow_raw, holders_raw = _make_fundamental_inputs()
    ratio_table = compute_ratios.compute(info_raw, income_raw, balance_raw, cashflow_raw, holders_raw)
    details = _make_envelope_details(ratio_table)

    store_decision(
        agent=AGENT,
        ticker="MSFT",
        date=RECENT,
        signal="HOLD",
        confidence=compute_ratios.confidence_to_score("MEDIUM"),
        key_drivers=compute_ratios.build_key_drivers(details),
        thesis="Mixed signals — value BUY, growth HOLD",
        db_path=db_path,
    )

    resolved_ids = resolve_pending(agent=AGENT, ticker="MSFT", db_path=db_path)
    assert resolved_ids == []

    context_md = get_past_context(agent=AGENT, ticker="MSFT", db_path=db_path)
    assert "No prior resolved lessons yet." in context_md


# ---------------------------------------------------------------------------
# compute() correctness sanity check — confirms the synthetic fixture
# exercises real ratio math (not just structural plumbing).
# ---------------------------------------------------------------------------


def test_compute_produces_expected_ratio_shape():
    info_raw, income_raw, balance_raw, cashflow_raw, holders_raw = _make_fundamental_inputs()
    result = compute_ratios.compute(info_raw, income_raw, balance_raw, cashflow_raw, holders_raw)

    assert set(result.keys()) == {"context", "annual", "insider_sentiment", "forecast"}
    assert result["context"]["market_cap"] == 2_000_000_000
    assert "2025" in result["annual"]
    assert result["annual"]["2025"]["valuation"]["pe"] == pytest.approx(13.3)
    # net buy (10,000 bought - 2,000 sold) -> BULLISH
    assert result["insider_sentiment"] == "BULLISH"
