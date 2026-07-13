"""Tests for tradingagents.memory.inventory — get_inventory / get_ticker_entries (issue #65)."""

from datetime import datetime, timezone

from tradingagents.memory.inventory import get_inventory, get_ticker_entries
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
    thesis=None,
):
    """Write a fully resolved decision row directly (like test_memory_stats.py)."""
    store_decision(
        agent=agent,
        ticker=ticker,
        date=date,
        signal=signal,
        confidence=confidence,
        key_drivers=None,
        thesis=thesis,
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


def _seed_pending(db_path, agent, ticker, date, signal="Buy", confidence=0.5, thesis=None):
    """Write a pending decision row."""
    store_decision(
        agent=agent,
        ticker=ticker,
        date=date,
        signal=signal,
        confidence=confidence,
        key_drivers=None,
        thesis=thesis,
        db_path=db_path,
    )


def _seed_known_dataset(db_path):
    """Seed a known set of pending and resolved rows for testing.

    Mix of resolved and pending rows across multiple agents and tickers:
      - quant/NVDA: 3 resolved, 1 pending
      - quant/TSLA: 1 resolved
      - fundamental/NVDA: 2 resolved
      - fundamental/MSFT: 1 pending
    """
    _seed_resolved(db_path, "quant", "NVDA", "2026-01-01", "Buy", 0.90, 0.05)
    _seed_resolved(db_path, "quant", "NVDA", "2026-01-02", "Sell", 0.85, -0.03)
    _seed_resolved(db_path, "quant", "NVDA", "2026-01-03", "Hold", 0.60, 0.01)
    _seed_pending(db_path, "quant", "NVDA", "2026-01-04", "Buy", 0.70)
    _seed_resolved(db_path, "quant", "TSLA", "2026-01-05", "Buy", 0.95, 0.10)
    _seed_resolved(db_path, "fundamental", "NVDA", "2026-01-06", "Overweight", 0.70, 0.02)
    _seed_resolved(db_path, "fundamental", "NVDA", "2026-01-07", "Underweight", 0.40, 0.03)
    _seed_pending(db_path, "fundamental", "MSFT", "2026-01-08", "Sell", 0.50)


# ---------------------------------------------------------------------------
# Default inventory: total count and per-ticker breakdown
# ---------------------------------------------------------------------------


def test_empty_database(tmp_path):
    """Empty database should report 0 total and no tickers."""
    db_path = _db_path(tmp_path)
    inventory = get_inventory(db_path=db_path)

    assert inventory["total"] == 0
    assert inventory["by_ticker"] == []


def test_total_count_includes_pending_and_resolved(tmp_path):
    """Inventory count must include both pending and resolved rows."""
    db_path = _db_path(tmp_path)
    _seed_resolved(db_path, "quant", "NVDA", "2026-01-01", "Buy", 0.90, 0.05)
    _seed_pending(db_path, "quant", "NVDA", "2026-01-02", "Sell", 0.85)

    inventory = get_inventory(db_path=db_path)

    assert inventory["total"] == 2


def test_per_ticker_breakdown(tmp_path):
    """Per-ticker breakdown must list each ticker with its entry count (pending + resolved)."""
    db_path = _db_path(tmp_path)
    _seed_known_dataset(db_path)

    inventory = get_inventory(db_path=db_path)

    # Expected: NVDA (3 + 1 + 2 = 6), TSLA (1), MSFT (1)
    assert inventory["total"] == 8
    by_ticker_dict = {item["ticker"]: item["n"] for item in inventory["by_ticker"]}

    assert by_ticker_dict["NVDA"] == 6
    assert by_ticker_dict["TSLA"] == 1
    assert by_ticker_dict["MSFT"] == 1


def test_ticker_breakdown_is_sorted_alphabetically(tmp_path):
    """Per-ticker list must be sorted alphabetically for deterministic output."""
    db_path = _db_path(tmp_path)
    _seed_pending(db_path, "agent", "ZEBRA", "2026-01-01", "Buy", 0.5)
    _seed_pending(db_path, "agent", "APPLE", "2026-01-02", "Sell", 0.5)
    _seed_pending(db_path, "agent", "MANGO", "2026-01-03", "Buy", 0.5)

    inventory = get_inventory(db_path=db_path)

    tickers = [item["ticker"] for item in inventory["by_ticker"]]
    assert tickers == ["APPLE", "MANGO", "ZEBRA"]


# ---------------------------------------------------------------------------
# Detail view: get entries for a specific ticker
# ---------------------------------------------------------------------------


def test_ticker_entries_returns_all_agents_for_ticker(tmp_path):
    """Detail view for a ticker must show all entries from all agents for that ticker."""
    db_path = _db_path(tmp_path)
    _seed_resolved(db_path, "quant", "NVDA", "2026-01-01", "Buy", 0.90, 0.05)
    _seed_resolved(db_path, "fundamental", "NVDA", "2026-01-02", "Sell", 0.85, -0.03)
    _seed_pending(db_path, "technical", "NVDA", "2026-01-03", "Hold", 0.60)
    _seed_resolved(db_path, "quant", "TSLA", "2026-01-04", "Buy", 0.95, 0.10)

    entries = get_ticker_entries("NVDA", db_path=db_path)

    assert len(entries) == 3
    agents = {e["agent"] for e in entries}
    assert agents == {"quant", "fundamental", "technical"}


def test_ticker_entries_includes_pending_and_resolved(tmp_path):
    """Detail view must include both pending and resolved entries for a ticker."""
    db_path = _db_path(tmp_path)
    _seed_resolved(db_path, "quant", "NVDA", "2026-01-01", "Buy", 0.90, 0.05)
    _seed_pending(db_path, "quant", "NVDA", "2026-01-02", "Sell", 0.85)

    entries = get_ticker_entries("NVDA", db_path=db_path)

    assert len(entries) == 2
    resolved_count = sum(1 for e in entries if e["resolved_at"] is not None)
    pending_count = sum(1 for e in entries if e["resolved_at"] is None)

    assert resolved_count == 1
    assert pending_count == 1


def test_ticker_entries_ordered_decision_date_desc_then_id_desc(tmp_path):
    """Detail view must order entries by decision_date DESC, id DESC (same convention as query.py)."""
    db_path = _db_path(tmp_path)
    _seed_pending(db_path, "quant", "NVDA", "2026-01-01", "Buy", 0.90)
    _seed_pending(db_path, "quant", "NVDA", "2026-01-03", "Sell", 0.85)
    _seed_pending(db_path, "quant", "NVDA", "2026-01-02", "Hold", 0.60)

    entries = get_ticker_entries("NVDA", db_path=db_path)

    # Most recent date first: 2026-01-03, then 2026-01-02, then 2026-01-01
    assert [e["decision_date"] for e in entries] == ["2026-01-03", "2026-01-02", "2026-01-01"]


def test_ticker_entries_no_entries_returns_empty_list(tmp_path):
    """Detail view for a ticker with no entries should return an empty list, not error."""
    db_path = _db_path(tmp_path)
    _seed_pending(db_path, "quant", "NVDA", "2026-01-01", "Buy", 0.90)

    entries = get_ticker_entries("TSLA", db_path=db_path)

    assert entries == []


def test_ticker_entries_exact_match_not_normalized(tmp_path):
    """Ticker matching must be exact-string, not normalized (no symbol_utils call)."""
    db_path = _db_path(tmp_path)
    _seed_pending(db_path, "quant", "NVDA", "2026-01-01", "Buy", 0.90)

    # Query with different casing or variation should not match
    entries = get_ticker_entries("nvda", db_path=db_path)  # lowercase
    assert entries == []

    entries = get_ticker_entries("NVDA", db_path=db_path)  # exact match
    assert len(entries) == 1


def test_ticker_entries_contains_all_required_fields(tmp_path):
    """Detail view rows must contain all required fields."""
    db_path = _db_path(tmp_path)
    _seed_resolved(
        db_path,
        "quant",
        "NVDA",
        "2026-01-01",
        "Buy",
        0.90,
        0.05,
        lesson="NVDA rallied on earnings.",
        thesis="Strong guidance.",
    )

    entries = get_ticker_entries("NVDA", db_path=db_path)

    assert len(entries) == 1
    entry = entries[0]

    # Check required fields are present
    required_fields = [
        "id",
        "agent",
        "decision_date",
        "signal",
        "confidence",
        "thesis",
        "resolved_at",
        "forward_return",
        "lesson",
    ]
    for field in required_fields:
        assert field in entry

    # Check values
    assert entry["agent"] == "quant"
    assert entry["ticker"] == "NVDA"
    assert entry["decision_date"] == "2026-01-01"
    assert entry["signal"] == "Buy"
    assert entry["confidence"] == 0.90
    assert entry["thesis"] == "Strong guidance."
    assert entry["resolved_at"] is not None
    assert entry["forward_return"] == 0.05
    assert entry["lesson"] == "NVDA rallied on earnings."


def test_ticker_entries_null_fields_handled(tmp_path):
    """Detail view must handle NULL fields gracefully."""
    db_path = _db_path(tmp_path)
    # Pending entry: no thesis, no confidence, no forward_return, no lesson
    store_decision(
        agent="quant",
        ticker="NVDA",
        date="2026-01-01",
        signal="Buy",
        confidence=None,  # NULL confidence
        key_drivers=None,
        thesis=None,  # NULL thesis
        db_path=db_path,
    )

    entries = get_ticker_entries("NVDA", db_path=db_path)

    assert len(entries) == 1
    entry = entries[0]

    assert entry["confidence"] is None
    assert entry["thesis"] is None
    assert entry["resolved_at"] is None
    assert entry["forward_return"] is None
    assert entry["lesson"] is None


# ---------------------------------------------------------------------------
# Integration: seed known data and verify both views
# ---------------------------------------------------------------------------


def test_inventory_and_entries_consistency(tmp_path):
    """Total count from inventory should match sum of per-ticker counts."""
    db_path = _db_path(tmp_path)
    _seed_known_dataset(db_path)

    inventory = get_inventory(db_path=db_path)
    total_from_breakdown = sum(item["n"] for item in inventory["by_ticker"])

    assert inventory["total"] == total_from_breakdown
    assert inventory["total"] == 8
