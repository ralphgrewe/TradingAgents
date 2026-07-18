"""Tests for tradingagents.memory.query — get_past_context (issue #7) and gather_context_rows (issue #52)."""

import json
from datetime import datetime, timezone

from tradingagents.memory.query import gather_context_rows, get_past_context
from tradingagents.memory.store import get_connection, store_decision


def _db_path(tmp_path):
    return tmp_path / "memory.db"


def _seed_pending(db_path, agent="trader", ticker="AAPL", date="2026-01-01", **kwargs):
    defaults = {
        "signal": "Buy",
        "confidence": 0.7,
        "key_drivers": ["strong earnings"],
        "thesis": "Momentum plus fundamentals align.",
    }
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


# ---------------------------------------------------------------------------
# gather_context_rows: raw decision rows with key_drivers/thesis/lesson (issue #52)
# ---------------------------------------------------------------------------


def _seed_resolved_with_drivers(
    db_path,
    agent,
    ticker,
    date,
    signal,
    confidence,
    forward_return,
    key_drivers=None,
    thesis="Some thesis.",
    lesson="Some lesson.",
):
    """Seed a fully resolved decision row with key_drivers — same precedent as
    test_memory_review.py's _seed_resolved helper."""
    store_decision(
        agent=agent,
        ticker=ticker,
        date=date,
        signal=signal,
        confidence=confidence,
        key_drivers=key_drivers,
        thesis=thesis,
        db_path=db_path,
    )
    conn = get_connection(db_path)
    try:
        conn.execute(
            """
            UPDATE decisions
            SET horizon_days = ?, forward_return = ?, lesson = ?,
                resolved_at = ?
            WHERE agent = ? AND ticker = ? AND decision_date = ?
            """,
            (10, forward_return, lesson, "2026-02-01T00:00:00+00:00", agent, ticker, date),
        )
        conn.commit()
    finally:
        conn.close()


def test_gather_context_rows_basic_retrieval(tmp_path):
    """Basic retrieval: gather_context_rows returns all columns for resolved rows."""
    db_path = _db_path(tmp_path)
    key_drivers = {"signal": "strong_earnings", "confidence_factor": "high"}
    _seed_resolved_with_drivers(
        db_path,
        "fundamental",
        "AAPL",
        "2026-01-01",
        "Buy",
        0.8,
        0.05,
        key_drivers=key_drivers,
        thesis="Strong earnings growth.",
        lesson="Earnings beat drove a 5% gain.",
    )

    rows = gather_context_rows("fundamental", "AAPL", db_path=db_path)

    assert len(rows) == 1
    row = rows[0]
    assert row["decision_date"] == "2026-01-01"
    assert row["signal"] == "Buy"
    assert row["confidence"] == 0.8
    assert row["key_drivers"] == key_drivers
    assert row["thesis"] == "Strong earnings growth."
    assert row["lesson"] == "Earnings beat drove a 5% gain."
    assert row["forward_return"] == 0.05
    assert row["correct"] is True  # BUY with positive return


def test_gather_context_rows_correct_scoring(tmp_path):
    """Correctness scoring: BUY/HOLD/SELL against forward_return."""
    db_path = _db_path(tmp_path)
    # BUY call, positive return -> correct
    _seed_resolved_with_drivers(db_path, "trader", "AAPL", "2026-01-01", "Buy", 0.7, 0.03)
    # BUY call, negative return -> incorrect
    _seed_resolved_with_drivers(db_path, "trader", "AAPL", "2026-01-02", "Buy", 0.7, -0.02)
    # SELL call, negative return -> correct
    _seed_resolved_with_drivers(db_path, "trader", "AAPL", "2026-01-03", "Sell", 0.6, -0.04)
    # HOLD call, small move -> correct
    _seed_resolved_with_drivers(db_path, "trader", "AAPL", "2026-01-04", "Hold", 0.5, 0.01)
    # HOLD call, large move -> incorrect
    _seed_resolved_with_drivers(db_path, "trader", "AAPL", "2026-01-05", "Hold", 0.5, 0.05)

    rows = gather_context_rows("trader", "AAPL", db_path=db_path, limit=20)
    by_date = {r["decision_date"]: r for r in rows}

    assert by_date["2026-01-01"]["correct"] is True
    assert by_date["2026-01-02"]["correct"] is False
    assert by_date["2026-01-03"]["correct"] is True
    assert by_date["2026-01-04"]["correct"] is True
    assert by_date["2026-01-05"]["correct"] is False


def test_gather_context_rows_misses_only_filtering(tmp_path):
    """misses_only: filter to incorrect rows only."""
    db_path = _db_path(tmp_path)
    _seed_resolved_with_drivers(db_path, "quant", "TSLA", "2026-01-01", "Buy", 0.7, 0.02)  # hit
    _seed_resolved_with_drivers(db_path, "quant", "TSLA", "2026-01-02", "Buy", 0.7, -0.03)  # miss
    _seed_resolved_with_drivers(db_path, "quant", "TSLA", "2026-01-03", "Buy", 0.7, 0.01)  # hit

    all_rows = gather_context_rows("quant", "TSLA", db_path=db_path, limit=20, misses_only=False)
    miss_rows = gather_context_rows("quant", "TSLA", db_path=db_path, limit=20, misses_only=True)

    assert len(all_rows) == 3
    assert len(miss_rows) == 1
    assert miss_rows[0]["decision_date"] == "2026-01-02"
    assert miss_rows[0]["correct"] is False


def test_gather_context_rows_limit_respected(tmp_path):
    """limit: respect the maximum row count, most recent first."""
    db_path = _db_path(tmp_path)
    for i in range(5):
        _seed_resolved_with_drivers(
            db_path, "analyst", "SPY", f"2026-01-{i+1:02d}", "Buy", 0.6, 0.02
        )

    rows_limited = gather_context_rows("analyst", "SPY", db_path=db_path, limit=2)
    rows_all = gather_context_rows("analyst", "SPY", db_path=db_path, limit=20)

    assert len(rows_limited) == 2
    assert len(rows_all) == 5
    # Most recent (highest date) first
    assert rows_limited[0]["decision_date"] == "2026-01-05"
    assert rows_limited[1]["decision_date"] == "2026-01-04"


def test_gather_context_rows_resolved_only(tmp_path):
    """Resolved-only guarantee: pending rows never appear."""
    db_path = _db_path(tmp_path)
    # Resolved row
    _seed_resolved_with_drivers(
        db_path, "trader", "AAPL", "2026-01-01", "Buy", 0.7, 0.03, lesson="Resolved lesson."
    )
    # Pending row (no resolved_at, no forward_return, no lesson)
    _seed_pending(db_path, agent="trader", ticker="AAPL", date="2026-01-02")

    rows = gather_context_rows("trader", "AAPL", db_path=db_path, limit=20)

    assert len(rows) == 1
    assert rows[0]["decision_date"] == "2026-01-01"
    assert "Resolved lesson." in rows[0]["lesson"]


def test_gather_context_rows_exact_ticker_match(tmp_path):
    """Exact ticker match: no normalization or fuzzy matching."""
    db_path = _db_path(tmp_path)
    _seed_resolved_with_drivers(db_path, "trader", "AAPL", "2026-01-01", "Buy", 0.7, 0.02)
    _seed_resolved_with_drivers(db_path, "trader", "MSFT", "2026-01-02", "Buy", 0.7, 0.03)

    aapl_rows = gather_context_rows("trader", "AAPL", db_path=db_path)
    msft_rows = gather_context_rows("trader", "MSFT", db_path=db_path)
    other_rows = gather_context_rows("trader", "OTHER", db_path=db_path)

    assert len(aapl_rows) == 1
    assert aapl_rows[0]["decision_date"] == "2026-01-01"
    assert len(msft_rows) == 1
    assert msft_rows[0]["decision_date"] == "2026-01-02"
    assert len(other_rows) == 0


def test_gather_context_rows_empty_case(tmp_path):
    """Empty case: no matching rows returns empty list."""
    db_path = _db_path(tmp_path)
    rows = gather_context_rows("nonexistent_agent", "NONEXISTENT", db_path=db_path)

    assert rows == []


def test_gather_context_rows_key_drivers_parsing(tmp_path):
    """key_drivers: JSON parsing and None handling."""
    db_path = _db_path(tmp_path)
    # With key_drivers
    _seed_resolved_with_drivers(
        db_path,
        "analyst",
        "QQQ",
        "2026-01-01",
        "Buy",
        0.8,
        0.04,
        key_drivers={"reason": "momentum", "strength": "high"},
    )
    # Without key_drivers (JSON column is NULL)
    _seed_resolved_with_drivers(
        db_path, "analyst", "QQQ", "2026-01-02", "Sell", 0.6, -0.02, key_drivers=None
    )

    rows = gather_context_rows("analyst", "QQQ", db_path=db_path, limit=20)

    # Most recent first (by decision_date DESC, id DESC)
    assert rows[0]["decision_date"] == "2026-01-02"
    assert rows[0]["key_drivers"] is None
    assert rows[1]["decision_date"] == "2026-01-01"
    assert rows[1]["key_drivers"] == {"reason": "momentum", "strength": "high"}
