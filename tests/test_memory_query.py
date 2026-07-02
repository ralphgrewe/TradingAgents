"""Tests for tradingagents.memory.query — get_past_context (issue #7)."""

from datetime import datetime, timezone

import pytest

from tradingagents.memory.query import get_past_context
from tradingagents.memory.store import get_connection, store_decision


def _db_path(tmp_path):
    return tmp_path / "memory.db"


def _seed_pending(db_path, agent="trader", ticker="AAPL", date="2026-01-01", **kwargs):
    defaults = dict(
        signal="Buy",
        confidence=0.7,
        key_drivers=["strong earnings"],
        thesis="Momentum plus fundamentals align.",
    )
    defaults.update(kwargs)
    store_decision(agent=agent, ticker=ticker, date=date, db_path=db_path, **defaults)


def _resolve_row(
    db_path,
    agent,
    ticker,
    date,
    lesson,
    forward_return=0.05,
    horizon_days=10,
    resolved_at=None,
):
    """Directly write a resolved row (bypasses resolve_pending/LLM/yfinance —
    this module only needs resolved rows to exist, not the resolve mechanics,
    which are already covered by test_memory_resolve.py)."""
    _seed_pending(db_path, agent=agent, ticker=ticker, date=date)
    resolved_at = resolved_at or datetime.now(timezone.utc).isoformat()
    conn = get_connection(db_path)
    try:
        conn.execute(
            """
            UPDATE decisions
            SET horizon_days = ?, forward_return = ?, lesson = ?, resolved_at = ?
            WHERE agent = ? AND ticker = ? AND decision_date = ?
            """,
            (horizon_days, forward_return, lesson, resolved_at, agent, ticker, date),
        )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Core behavior: same-ticker + cross-ticker both present, correctly formatted
# ---------------------------------------------------------------------------

def test_same_and_cross_ticker_entries_present_and_formatted(tmp_path):
    db_path = _db_path(tmp_path)
    _resolve_row(db_path, "trader", "AAPL", "2026-01-01", "AAPL lesson one.")
    _resolve_row(db_path, "trader", "AAPL", "2026-01-05", "AAPL lesson two.")
    _resolve_row(db_path, "trader", "MSFT", "2026-01-03", "MSFT lesson.")

    md = get_past_context("trader", "AAPL", n_same=5, n_cross=5, db_path=db_path)

    assert "## Past context: AAPL" in md
    assert "### Same-ticker lessons (AAPL)" in md
    assert "### Cross-ticker lessons (trader)" in md
    assert "AAPL lesson one." in md
    assert "AAPL lesson two." in md
    assert "MSFT lesson." in md
    # Same-ticker most-recent-first (by decision_date DESC)
    same_idx_two = md.index("AAPL lesson two.")
    same_idx_one = md.index("AAPL lesson one.")
    assert same_idx_two < same_idx_one
    # Cross-ticker section includes the ticker (mixed section)
    assert "MSFT" in md.split("### Cross-ticker lessons (trader)")[1]
    # Same-ticker bullets should not repeat the ticker redundantly in the bullet text
    same_section = md.split("### Same-ticker lessons (AAPL)")[1].split("### Cross-ticker")[0]
    assert "AAPL |" not in same_section


def test_confidence_and_signal_rendered_in_bullet(tmp_path):
    db_path = _db_path(tmp_path)
    _resolve_row(db_path, "trader", "AAPL", "2026-01-01", "Some lesson text.")

    md = get_past_context("trader", "AAPL", db_path=db_path)

    assert "Buy" in md
    assert "0.7" in md
    assert "Some lesson text." in md


# ---------------------------------------------------------------------------
# Edge case: no resolved rows at all -> graceful, non-crashing output
# ---------------------------------------------------------------------------

def test_no_resolved_rows_returns_graceful_markdown(tmp_path):
    db_path = _db_path(tmp_path)
    md = get_past_context("trader", "AAPL", db_path=db_path)

    assert md
    assert "## Past context: AAPL" in md
    assert "no prior" in md.lower()


def test_only_pending_rows_returns_graceful_markdown(tmp_path):
    db_path = _db_path(tmp_path)
    _seed_pending(db_path, agent="trader", ticker="AAPL", date="2026-01-01")

    md = get_past_context("trader", "AAPL", db_path=db_path)

    assert "no prior" in md.lower()
    assert "2026-01-01" not in md


# ---------------------------------------------------------------------------
# Pending rows excluded even when resolved rows also exist
# ---------------------------------------------------------------------------

def test_pending_rows_excluded_from_both_sections(tmp_path):
    db_path = _db_path(tmp_path)
    _resolve_row(db_path, "trader", "AAPL", "2026-01-01", "Resolved AAPL lesson.")
    _seed_pending(db_path, agent="trader", ticker="AAPL", date="2026-02-01")
    _seed_pending(db_path, agent="trader", ticker="MSFT", date="2026-02-01")

    md = get_past_context("trader", "AAPL", n_same=5, n_cross=5, db_path=db_path)

    assert "Resolved AAPL lesson." in md
    assert "2026-02-01" not in md  # pending rows never rendered
    assert "### Cross-ticker lessons (trader)" not in md  # no resolved MSFT row


# ---------------------------------------------------------------------------
# n_same / n_cross limits respected
# ---------------------------------------------------------------------------

def test_limits_respected(tmp_path):
    db_path = _db_path(tmp_path)
    for i in range(5):
        _resolve_row(db_path, "trader", "AAPL", f"2026-01-{i+1:02d}", f"AAPL lesson {i}.")
    for i in range(5):
        _resolve_row(db_path, "trader", f"TICK{i}", f"2026-02-{i+1:02d}", f"Cross lesson {i}.")

    md = get_past_context("trader", "AAPL", n_same=2, n_cross=1, db_path=db_path)

    same_count = sum(1 for line in md.splitlines() if line.startswith("- ") and "AAPL lesson" in line)
    cross_count = sum(1 for line in md.splitlines() if line.startswith("- ") and "Cross lesson" in line)
    assert same_count == 2
    assert cross_count == 1
    # Most recent same-ticker entries kept (higher-numbered dates)
    assert "AAPL lesson 4." in md
    assert "AAPL lesson 3." in md
    assert "AAPL lesson 0." not in md


# ---------------------------------------------------------------------------
# Different agent's rows excluded
# ---------------------------------------------------------------------------

def test_different_agent_rows_excluded(tmp_path):
    db_path = _db_path(tmp_path)
    _resolve_row(db_path, "trader", "AAPL", "2026-01-01", "Trader AAPL lesson.")
    _resolve_row(db_path, "fundamental", "AAPL", "2026-01-02", "Fundamental AAPL lesson.")
    _resolve_row(db_path, "fundamental", "MSFT", "2026-01-03", "Fundamental MSFT lesson.")

    md = get_past_context("trader", "AAPL", n_same=5, n_cross=5, db_path=db_path)

    assert "Trader AAPL lesson." in md
    assert "Fundamental AAPL lesson." not in md
    assert "Fundamental MSFT lesson." not in md
    assert "### Cross-ticker lessons (trader)" not in md


# ---------------------------------------------------------------------------
# Cross-ticker section excludes the queried ticker
# ---------------------------------------------------------------------------

def test_cross_ticker_excludes_queried_ticker(tmp_path):
    db_path = _db_path(tmp_path)
    _resolve_row(db_path, "trader", "AAPL", "2026-01-01", "AAPL lesson.")
    _resolve_row(db_path, "trader", "MSFT", "2026-01-02", "MSFT lesson.")

    md = get_past_context("trader", "AAPL", n_same=5, n_cross=5, db_path=db_path)

    cross_section = md.split("### Cross-ticker lessons (trader)")[1]
    assert "AAPL lesson." not in cross_section
    assert "MSFT lesson." in cross_section
