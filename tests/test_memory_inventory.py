"""Tests for tradingagents.memory.inventory — get_inventory / get_ticker_entries (issue #65)."""

from datetime import datetime, timezone

from tradingagents.memory.inventory import (
    _format_ticker_entries_text,
    get_inventory,
    get_ticker_entries,
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


# ---------------------------------------------------------------------------
# Text formatter: _format_ticker_entries_text (issue #73)
# ---------------------------------------------------------------------------


def test_format_ticker_entries_text_no_entries():
    """Formatter should return 'No entries' message for empty list."""
    output = _format_ticker_entries_text("NVDA", [])
    assert output == "No entries for ticker NVDA."


def test_format_ticker_entries_text_header_and_structure():
    """Formatter should include header and have correct structure per entry."""
    entries = [
        {
            "agent": "quant",
            "decision_date": "2026-01-01",
            "signal": "Buy",
            "confidence": 0.90,
            "resolved_at": None,
            "forward_return": None,
            "thesis": "Strong momentum",
            "lesson": "Pattern held",
        }
    ]

    output = _format_ticker_entries_text("NVDA", entries)

    # Should have header
    assert "Entries for NVDA (1 total):" in output
    # Should have compact line with short fields
    assert "quant  2026-01-01  Buy  0.9  pending  n/a" in output
    # Should have thesis and lesson blocks
    assert "  Thesis: Strong momentum" in output
    assert "  Lesson: Pattern held" in output


def test_format_ticker_entries_text_null_thesis_and_lesson():
    """Formatter should render null thesis/lesson as '(none)'."""
    entries = [
        {
            "agent": "quant",
            "decision_date": "2026-01-01",
            "signal": "Buy",
            "confidence": None,
            "resolved_at": None,
            "forward_return": None,
            "thesis": None,
            "lesson": None,
        }
    ]

    output = _format_ticker_entries_text("NVDA", entries)

    assert "  Thesis: (none)" in output
    assert "  Lesson: (none)" in output


def test_format_ticker_entries_text_forward_return_formatting():
    """Formatter should render forward_return as '+.2%' or 'n/a'."""
    entries_resolved = [
        {
            "agent": "quant",
            "decision_date": "2026-01-01",
            "signal": "Buy",
            "confidence": 0.90,
            "resolved_at": "2026-01-11T00:00:00Z",
            "forward_return": 0.0523,  # 5.23%
            "thesis": None,
            "lesson": None,
        }
    ]

    output = _format_ticker_entries_text("NVDA", entries_resolved)
    assert "+5.23%" in output

    # Test negative return
    entries_negative = [
        {
            "agent": "quant",
            "decision_date": "2026-01-01",
            "signal": "Sell",
            "confidence": 0.85,
            "resolved_at": "2026-01-11T00:00:00Z",
            "forward_return": -0.0325,  # -3.25%
            "thesis": None,
            "lesson": None,
        }
    ]

    output = _format_ticker_entries_text("NVDA", entries_negative)
    assert "-3.25%" in output


def test_format_ticker_entries_text_status_resolved_vs_pending():
    """Formatter should show 'resolved' or 'pending' based on resolved_at."""
    entries = [
        {
            "agent": "quant",
            "decision_date": "2026-01-01",
            "signal": "Buy",
            "confidence": 0.90,
            "resolved_at": "2026-01-11T00:00:00Z",
            "forward_return": 0.05,
            "thesis": None,
            "lesson": None,
        },
        {
            "agent": "technical",
            "decision_date": "2026-01-02",
            "signal": "Hold",
            "confidence": 0.60,
            "resolved_at": None,
            "forward_return": None,
            "thesis": None,
            "lesson": None,
        },
    ]

    output = _format_ticker_entries_text("NVDA", entries)

    # Find the line with quant (resolved)
    lines = output.split("\n")
    quant_line = [line for line in lines if line.startswith("quant")][0]
    assert "resolved" in quant_line

    # Find the line with technical (pending)
    technical_line = [line for line in lines if line.startswith("technical")][0]
    assert "pending" in technical_line


def test_format_ticker_entries_text_long_thesis_and_lesson_not_truncated(tmp_path):
    """Formatter should not truncate thesis and lesson, even if very long."""
    db_path = _db_path(tmp_path)
    long_thesis = "This is a very long thesis " * 10  # Intentionally long
    long_lesson = "This is a very long lesson " * 10

    _seed_resolved(
        db_path,
        "quant",
        "NVDA",
        "2026-01-01",
        "Buy",
        0.90,
        0.05,
        thesis=long_thesis,
        lesson=long_lesson,
    )

    entries = get_ticker_entries("NVDA", db_path=db_path)
    output = _format_ticker_entries_text("NVDA", entries)

    # Both should appear in full in the output
    assert long_thesis in output
    assert long_lesson in output
    # Should NOT be truncated (no "..." at the end)
    assert long_thesis + "..." not in output
    assert long_lesson + "..." not in output


def test_format_ticker_entries_text_multiline_thesis_and_lesson():
    """Formatter should handle thesis/lesson with newlines."""
    entries = [
        {
            "agent": "quant",
            "decision_date": "2026-01-01",
            "signal": "Buy",
            "confidence": 0.90,
            "resolved_at": None,
            "forward_return": None,
            "thesis": "Strong momentum\n\nBased on:\n- Volume increase\n- Price breakout",
            "lesson": "Pattern worked\nStock rallied 5% in next week",
        }
    ]

    output = _format_ticker_entries_text("NVDA", entries)

    # Both multiline texts should be preserved
    assert "Strong momentum\n\nBased on:\n- Volume increase\n- Price breakout" in output
    assert "Pattern worked\nStock rallied 5% in next week" in output


def test_format_ticker_entries_text_multiple_entries_spacing():
    """Formatter should have blank lines between entries."""
    entries = [
        {
            "agent": "quant",
            "decision_date": "2026-01-01",
            "signal": "Buy",
            "confidence": 0.90,
            "resolved_at": None,
            "forward_return": None,
            "thesis": "Thesis 1",
            "lesson": "Lesson 1",
        },
        {
            "agent": "fundamental",
            "decision_date": "2026-01-02",
            "signal": "Sell",
            "confidence": 0.80,
            "resolved_at": None,
            "forward_return": None,
            "thesis": "Thesis 2",
            "lesson": "Lesson 2",
        },
    ]

    output = _format_ticker_entries_text("NVDA", entries)

    # Should have 2 entries with blank lines between them
    assert "Entries for NVDA (2 total):" in output
    # Check that entries are separated
    lines = output.split("\n")
    # Find line indices
    quant_idx = next(i for i, line in enumerate(lines) if line.startswith("quant"))
    fund_idx = next(i for i, line in enumerate(lines) if line.startswith("fundamental"))
    # There should be blank lines in between
    assert quant_idx < fund_idx
    assert "" in lines[quant_idx : fund_idx + 1]


def test_format_ticker_entries_text_confidence_formatting():
    """Formatter should use _format_confidence for null/numeric confidence."""
    entries = [
        {
            "agent": "quant",
            "decision_date": "2026-01-01",
            "signal": "Buy",
            "confidence": None,
            "resolved_at": None,
            "forward_return": None,
            "thesis": None,
            "lesson": None,
        },
        {
            "agent": "technical",
            "decision_date": "2026-01-02",
            "signal": "Hold",
            "confidence": 0.5,
            "resolved_at": None,
            "forward_return": None,
            "thesis": None,
            "lesson": None,
        },
    ]

    output = _format_ticker_entries_text("NVDA", entries)

    # Find lines with confidence
    lines = output.split("\n")
    quant_line = [line for line in lines if line.startswith("quant")][0]
    tech_line = [line for line in lines if line.startswith("technical")][0]

    # quant should have "n/a" for confidence
    assert "n/a" in quant_line
    # technical should have "0.5"
    assert "0.5" in tech_line
