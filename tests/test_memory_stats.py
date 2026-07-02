"""Tests for tradingagents.memory.stats — get_statistics / format_statistics_markdown (issue #23)."""

from datetime import datetime, timezone

import pytest

from tradingagents.memory.stats import (
    HOLD_CORRECT_THRESHOLD,
    format_statistics_markdown,
    get_statistics,
)
from tradingagents.memory.store import get_connection, store_decision


def _db_path(tmp_path):
    return tmp_path / "memory.db"


def _seed_resolved(
    db_path,
    agent,
    ticker,
    date,
    signal,
    confidence,
    forward_return,
    lesson="Some lesson.",
):
    """Write a fully resolved decision row directly (bypasses resolve_pending's
    LLM/yfinance mechanics — this module only needs resolved rows to exist,
    same precedent as test_memory_query.py's _resolve_row helper)."""
    store_decision(
        agent=agent,
        ticker=ticker,
        date=date,
        signal=signal,
        confidence=confidence,
        key_drivers=None,
        thesis=None,
        db_path=db_path,
    )
    resolved_at = datetime.now(timezone.utc).isoformat()
    conn = get_connection(db_path)
    try:
        conn.execute(
            """
            UPDATE decisions
            SET horizon_days = ?, forward_return = ?, lesson = ?, resolved_at = ?
            WHERE agent = ? AND ticker = ? AND decision_date = ?
            """,
            (10, forward_return, lesson, resolved_at, agent, ticker, date),
        )
        conn.commit()
    finally:
        conn.close()


def _seed_pending(db_path, agent, ticker, date, signal="Buy", confidence=0.5):
    store_decision(
        agent=agent,
        ticker=ticker,
        date=date,
        signal=signal,
        confidence=confidence,
        key_drivers=None,
        thesis=None,
        db_path=db_path,
    )


def _seed_known_dataset(db_path):
    """Seed a known, hand-computable set of resolved rows.

    quant/NVDA (6 rows):
      BUY  conf=0.90 (HIGH)   forward_return=+0.05  -> correct
      BUY  conf=0.30 (LOW)    forward_return=-0.02  -> incorrect
      SELL conf=0.85 (HIGH)   forward_return=-0.03  -> correct
      SELL conf=0.50 (MEDIUM) forward_return=+0.01  -> incorrect
      HOLD conf=0.60 (MEDIUM) forward_return=+0.01  -> correct (|.01| <= 0.02)
      HOLD conf=0.20 (LOW)    forward_return=+0.05  -> incorrect (|.05| > 0.02)

    quant/TSLA (1 row):
      BUY  conf=0.95 (HIGH)   forward_return=+0.10  -> correct

    fundamental/NVDA (2 rows):
      Overweight  conf=0.70 (MEDIUM) forward_return=+0.02 -> correct (normalizes to BUY)
      Underweight conf=0.40 (LOW)    forward_return=+0.03 -> incorrect (normalizes to SELL)
    """
    _seed_resolved(db_path, "quant", "NVDA", "2026-01-01", "Buy", 0.90, 0.05)
    _seed_resolved(db_path, "quant", "NVDA", "2026-01-02", "Buy", 0.30, -0.02)
    _seed_resolved(db_path, "quant", "NVDA", "2026-01-03", "Sell", 0.85, -0.03)
    _seed_resolved(db_path, "quant", "NVDA", "2026-01-04", "Sell", 0.50, 0.01)
    _seed_resolved(db_path, "quant", "NVDA", "2026-01-05", "Hold", 0.60, 0.01)
    _seed_resolved(db_path, "quant", "NVDA", "2026-01-06", "Hold", 0.20, 0.05)
    _seed_resolved(db_path, "quant", "TSLA", "2026-01-07", "Buy", 0.95, 0.10)
    _seed_resolved(db_path, "fundamental", "NVDA", "2026-01-08", "Overweight", 0.70, 0.02)
    _seed_resolved(db_path, "fundamental", "NVDA", "2026-01-09", "Underweight", 0.40, 0.03)


# ---------------------------------------------------------------------------
# Per (agent, ticker) hit-rate + average return
# ---------------------------------------------------------------------------

def test_per_agent_ticker_hit_rate_and_avg_return(tmp_path):
    db_path = _db_path(tmp_path)
    _seed_known_dataset(db_path)

    stats = get_statistics(db_path=db_path)

    by_key = {(r["agent"], r["ticker"]): r for r in stats["per_agent_ticker"]}
    quant_nvda = by_key[("quant", "NVDA")]

    assert quant_nvda["n"] == 6
    assert quant_nvda["hit_rate"] == pytest.approx(0.5)

    buy = quant_nvda["by_signal"]["BUY"]
    assert buy["n"] == 2
    assert buy["hit_rate"] == pytest.approx(0.5)
    assert buy["avg_forward_return"] == pytest.approx((0.05 + -0.02) / 2)

    sell = quant_nvda["by_signal"]["SELL"]
    assert sell["n"] == 2
    assert sell["hit_rate"] == pytest.approx(0.5)
    assert sell["avg_forward_return"] == pytest.approx((-0.03 + 0.01) / 2)

    hold = quant_nvda["by_signal"]["HOLD"]
    assert hold["n"] == 2
    assert hold["hit_rate"] == pytest.approx(0.5)
    assert hold["avg_forward_return"] == pytest.approx((0.01 + 0.05) / 2)

    quant_tsla = by_key[("quant", "TSLA")]
    assert quant_tsla["n"] == 1
    assert quant_tsla["hit_rate"] == pytest.approx(1.0)
    assert quant_tsla["by_signal"]["BUY"]["avg_forward_return"] == pytest.approx(0.10)


# ---------------------------------------------------------------------------
# Per agent overall (aggregated across tickers)
# ---------------------------------------------------------------------------

def test_per_agent_overall_aggregates_across_tickers(tmp_path):
    db_path = _db_path(tmp_path)
    _seed_known_dataset(db_path)

    stats = get_statistics(db_path=db_path)

    by_agent = {r["agent"]: r for r in stats["per_agent"]}
    quant = by_agent["quant"]

    assert quant["n"] == 7
    assert quant["hit_rate"] == pytest.approx(4 / 7)

    buy = quant["by_signal"]["BUY"]
    assert buy["n"] == 3
    assert buy["hit_rate"] == pytest.approx(2 / 3)
    assert buy["avg_forward_return"] == pytest.approx((0.05 - 0.02 + 0.10) / 3)

    fundamental = by_agent["fundamental"]
    assert fundamental["n"] == 2
    assert fundamental["hit_rate"] == pytest.approx(0.5)
    # 5-tier signals normalized onto BUY/SELL
    assert fundamental["by_signal"]["BUY"]["n"] == 1
    assert fundamental["by_signal"]["BUY"]["hit_rate"] == pytest.approx(1.0)
    assert fundamental["by_signal"]["SELL"]["n"] == 1
    assert fundamental["by_signal"]["SELL"]["hit_rate"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Calibration by confidence bucket: HIGH should clearly beat LOW here
# ---------------------------------------------------------------------------

def test_calibration_by_confidence_bucket(tmp_path):
    db_path = _db_path(tmp_path)
    _seed_known_dataset(db_path)

    stats = get_statistics(db_path=db_path)

    by_key = {(r["agent"], r["bucket"]): r for r in stats["calibration"]}

    quant_high = by_key[("quant", "HIGH")]
    assert quant_high["n"] == 3  # 0.90, 0.85, 0.95 -- all correct
    assert quant_high["hit_rate"] == pytest.approx(1.0)

    quant_medium = by_key[("quant", "MEDIUM")]
    assert quant_medium["n"] == 2  # 0.50 (incorrect), 0.60 (correct)
    assert quant_medium["hit_rate"] == pytest.approx(0.5)

    quant_low = by_key[("quant", "LOW")]
    assert quant_low["n"] == 2  # 0.30, 0.20 -- both incorrect
    assert quant_low["hit_rate"] == pytest.approx(0.0)

    # HIGH-confidence quant calls were right more often than LOW-confidence ones.
    assert quant_high["hit_rate"] > quant_low["hit_rate"]

    fundamental_medium = by_key[("fundamental", "MEDIUM")]
    assert fundamental_medium["n"] == 1
    assert fundamental_medium["hit_rate"] == pytest.approx(1.0)

    fundamental_low = by_key[("fundamental", "LOW")]
    assert fundamental_low["n"] == 1
    assert fundamental_low["hit_rate"] == pytest.approx(0.0)


def test_calibration_bucket_boundaries(tmp_path):
    db_path = _db_path(tmp_path)
    # Exactly on the documented boundaries.
    _seed_resolved(db_path, "quant", "AAPL", "2026-01-01", "Buy", 0.45, 0.01)  # -> MEDIUM
    _seed_resolved(db_path, "quant", "AAPL", "2026-01-02", "Buy", 0.80, 0.01)  # -> HIGH
    _seed_resolved(db_path, "quant", "AAPL", "2026-01-03", "Buy", 0.4499, 0.01)  # -> LOW

    stats = get_statistics(agent="quant", ticker="AAPL", db_path=db_path)
    buckets = {r["bucket"]: r["n"] for r in stats["calibration"]}

    assert buckets.get("LOW") == 1
    assert buckets.get("MEDIUM") == 1
    assert buckets.get("HIGH") == 1


# ---------------------------------------------------------------------------
# Filters: agent / ticker (exact match) / since
# ---------------------------------------------------------------------------

def test_agent_filter(tmp_path):
    db_path = _db_path(tmp_path)
    _seed_known_dataset(db_path)

    stats = get_statistics(agent="fundamental", db_path=db_path)

    assert {r["agent"] for r in stats["per_agent_ticker"]} == {"fundamental"}
    assert {r["agent"] for r in stats["per_agent"]} == {"fundamental"}
    assert {r["agent"] for r in stats["calibration"]} == {"fundamental"}


def test_ticker_filter_is_exact_match(tmp_path):
    db_path = _db_path(tmp_path)
    _seed_known_dataset(db_path)

    stats = get_statistics(ticker="TSLA", db_path=db_path)

    assert len(stats["per_agent_ticker"]) == 1
    assert stats["per_agent_ticker"][0]["ticker"] == "TSLA"
    assert stats["per_agent_ticker"][0]["n"] == 1


def test_since_filter_excludes_earlier_decisions(tmp_path):
    db_path = _db_path(tmp_path)
    _seed_resolved(db_path, "quant", "AAPL", "2026-01-01", "Buy", 0.9, 0.05)
    _seed_resolved(db_path, "quant", "AAPL", "2026-02-01", "Buy", 0.9, 0.05)

    stats = get_statistics(agent="quant", ticker="AAPL", since="2026-01-15", db_path=db_path)

    assert stats["per_agent_ticker"][0]["n"] == 1


# ---------------------------------------------------------------------------
# Exclusions: pending rows and unrecognized signal strings
# ---------------------------------------------------------------------------

def test_pending_rows_excluded(tmp_path):
    db_path = _db_path(tmp_path)
    _seed_resolved(db_path, "quant", "AAPL", "2026-01-01", "Buy", 0.9, 0.05)
    _seed_pending(db_path, "quant", "MSFT", "2026-01-02")

    stats = get_statistics(agent="quant", db_path=db_path)

    tickers = {r["ticker"] for r in stats["per_agent_ticker"]}
    assert tickers == {"AAPL"}


def test_unrecognized_signal_excluded(tmp_path):
    db_path = _db_path(tmp_path)
    _seed_resolved(db_path, "quant", "AAPL", "2026-01-01", "Buy", 0.9, 0.05)
    _seed_resolved(db_path, "quant", "AAPL", "2026-01-02", "Neutral", 0.9, 0.05)

    stats = get_statistics(agent="quant", ticker="AAPL", db_path=db_path)

    assert stats["per_agent_ticker"][0]["n"] == 1


# ---------------------------------------------------------------------------
# HOLD threshold behavior
# ---------------------------------------------------------------------------

def test_hold_threshold_boundary(tmp_path):
    db_path = _db_path(tmp_path)
    assert HOLD_CORRECT_THRESHOLD == 0.02
    _seed_resolved(db_path, "quant", "AAPL", "2026-01-01", "Hold", 0.9, 0.02)  # exactly at threshold -> correct
    _seed_resolved(db_path, "quant", "AAPL", "2026-01-02", "Hold", 0.9, 0.0201)  # just over -> incorrect
    _seed_resolved(db_path, "quant", "AAPL", "2026-01-03", "Hold", 0.9, -0.02)  # negative, at threshold -> correct

    stats = get_statistics(agent="quant", ticker="AAPL", db_path=db_path)
    hold = stats["per_agent_ticker"][0]["by_signal"]["HOLD"]
    assert hold["n"] == 3
    assert hold["hit_rate"] == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def test_markdown_rendering_contains_expected_headers_and_values(tmp_path):
    db_path = _db_path(tmp_path)
    _seed_known_dataset(db_path)

    stats = get_statistics(agent="quant", ticker="NVDA", db_path=db_path)
    md = format_statistics_markdown(stats)

    assert "## Memory statistics" in md
    assert "agent=quant" in md
    assert "ticker=NVDA" in md
    assert "### Hit-rate by agent x ticker" in md
    assert "### Hit-rate by agent (overall)" in md
    assert "### Calibration by confidence bucket" in md
    assert "| quant | NVDA | 6 | 50.0% |" in md
    assert "HIGH" in md
    assert "LOW" in md


def test_markdown_rendering_no_matching_rows(tmp_path):
    db_path = _db_path(tmp_path)

    stats = get_statistics(agent="nobody", db_path=db_path)
    md = format_statistics_markdown(stats)

    assert "## Memory statistics" in md
    assert "no resolved decisions" in md.lower()
