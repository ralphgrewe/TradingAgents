"""Tests for tradingagents.memory.store — schema creation and store_decision."""

import json
import sqlite3

import pytest

from tradingagents.memory.store import (
    DEFAULT_DB_PATH,
    get_connection,
    resolve_db_path,
    store_decision,
)


def _db_path(tmp_path):
    return tmp_path / "memory.db"


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def test_resolve_db_path_default():
    assert resolve_db_path() == DEFAULT_DB_PATH


def test_resolve_db_path_explicit_arg_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADINGAGENTS_MEMORY_DB_PATH", str(tmp_path / "env.db"))
    explicit = tmp_path / "explicit.db"
    assert resolve_db_path(explicit) == explicit


def test_resolve_db_path_env_var(tmp_path, monkeypatch):
    env_path = tmp_path / "env.db"
    monkeypatch.setenv("TRADINGAGENTS_MEMORY_DB_PATH", str(env_path))
    assert resolve_db_path() == env_path


# ---------------------------------------------------------------------------
# Schema creation
# ---------------------------------------------------------------------------

def test_schema_created_with_expected_columns(tmp_path):
    db_path = _db_path(tmp_path)
    assert not db_path.exists()
    conn = get_connection(db_path)
    try:
        cols = {row["name"] for row in conn.execute("PRAGMA table_info(decisions)")}
    finally:
        conn.close()

    assert db_path.exists()
    assert cols == {
        "id", "agent", "ticker", "decision_date", "signal", "confidence",
        "key_drivers", "thesis", "created_at",
        "horizon_days", "forward_return", "benchmark_return", "lesson", "resolved_at",
    }


def test_schema_creates_parent_dirs(tmp_path):
    nested = tmp_path / "a" / "b" / "memory.db"
    assert not nested.parent.exists()
    conn = get_connection(nested)
    conn.close()
    assert nested.exists()


def test_unique_constraint_enforced_at_db_level(tmp_path):
    db_path = _db_path(tmp_path)
    conn = get_connection(db_path)
    try:
        conn.execute(
            """
            INSERT INTO decisions (agent, ticker, decision_date, signal, created_at)
            VALUES ('trader', 'AAPL', '2026-07-01', 'Buy', 'now')
            """
        )
        conn.commit()
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO decisions (agent, ticker, decision_date, signal, created_at)
                VALUES ('trader', 'AAPL', '2026-07-01', 'Sell', 'later')
                """
            )
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# store_decision
# ---------------------------------------------------------------------------

def test_store_decision_inserts_pending_row(tmp_path):
    db_path = _db_path(tmp_path)
    inserted = store_decision(
        agent="trader",
        ticker="AAPL",
        date="2026-07-01",
        signal="Buy",
        confidence=0.8,
        key_drivers=["strong earnings", "upgrade cycle"],
        thesis="Momentum plus fundamentals align.",
        db_path=db_path,
    )
    assert inserted is True

    conn = get_connection(db_path)
    try:
        rows = conn.execute("SELECT * FROM decisions").fetchall()
    finally:
        conn.close()

    assert len(rows) == 1
    row = rows[0]
    assert row["agent"] == "trader"
    assert row["ticker"] == "AAPL"
    assert row["decision_date"] == "2026-07-01"
    assert row["signal"] == "Buy"
    assert row["confidence"] == 0.8
    assert row["thesis"] == "Momentum plus fundamentals align."
    assert row["created_at"] is not None
    # Pending state: all resolution columns NULL.
    assert row["resolved_at"] is None
    assert row["horizon_days"] is None
    assert row["forward_return"] is None
    assert row["benchmark_return"] is None
    assert row["lesson"] is None


def test_store_decision_key_drivers_json_round_trips(tmp_path):
    db_path = _db_path(tmp_path)
    drivers = {"earnings_beat": True, "analyst_upgrades": 3, "notes": ["a", "b"]}
    store_decision(
        agent="fundamental",
        ticker="MSFT",
        date="2026-07-01",
        signal="Overweight",
        confidence=0.6,
        key_drivers=drivers,
        thesis="Solid quarter.",
        db_path=db_path,
    )

    conn = get_connection(db_path)
    try:
        row = conn.execute("SELECT key_drivers FROM decisions").fetchone()
    finally:
        conn.close()

    assert json.loads(row["key_drivers"]) == drivers


def test_store_decision_none_key_drivers_stored_as_null(tmp_path):
    db_path = _db_path(tmp_path)
    store_decision(
        agent="trader",
        ticker="AAPL",
        date="2026-07-01",
        signal="Hold",
        confidence=None,
        key_drivers=None,
        thesis=None,
        db_path=db_path,
    )
    conn = get_connection(db_path)
    try:
        row = conn.execute("SELECT key_drivers, confidence, thesis FROM decisions").fetchone()
    finally:
        conn.close()
    assert row["key_drivers"] is None
    assert row["confidence"] is None
    assert row["thesis"] is None


def test_store_decision_duplicate_key_is_noop(tmp_path):
    db_path = _db_path(tmp_path)
    first = store_decision(
        agent="trader",
        ticker="AAPL",
        date="2026-07-01",
        signal="Buy",
        confidence=0.8,
        key_drivers=["a"],
        thesis="First call.",
        db_path=db_path,
    )
    second = store_decision(
        agent="trader",
        ticker="AAPL",
        date="2026-07-01",
        signal="Sell",  # different payload — should NOT overwrite (no-op semantics)
        confidence=0.1,
        key_drivers=["b"],
        thesis="Second call, should be ignored.",
        db_path=db_path,
    )

    assert first is True
    assert second is False

    conn = get_connection(db_path)
    try:
        rows = conn.execute("SELECT * FROM decisions").fetchall()
    finally:
        conn.close()

    assert len(rows) == 1
    row = rows[0]
    # Original values from the first call are preserved.
    assert row["signal"] == "Buy"
    assert row["confidence"] == 0.8
    assert row["thesis"] == "First call."


def test_store_decision_different_key_creates_new_row(tmp_path):
    db_path = _db_path(tmp_path)
    store_decision(
        agent="trader", ticker="AAPL", date="2026-07-01",
        signal="Buy", confidence=0.8, key_drivers=None, thesis=None,
        db_path=db_path,
    )
    # Different agent, same ticker/date -> distinct key.
    store_decision(
        agent="fundamental", ticker="AAPL", date="2026-07-01",
        signal="Hold", confidence=0.5, key_drivers=None, thesis=None,
        db_path=db_path,
    )
    # Different ticker, same agent/date -> distinct key.
    store_decision(
        agent="trader", ticker="MSFT", date="2026-07-01",
        signal="Sell", confidence=0.3, key_drivers=None, thesis=None,
        db_path=db_path,
    )
    # Different date, same agent/ticker -> distinct key.
    store_decision(
        agent="trader", ticker="AAPL", date="2026-07-02",
        signal="Hold", confidence=0.4, key_drivers=None, thesis=None,
        db_path=db_path,
    )

    conn = get_connection(db_path)
    try:
        count = conn.execute("SELECT COUNT(*) AS c FROM decisions").fetchone()["c"]
    finally:
        conn.close()
    assert count == 4


def test_store_decision_uses_configurable_db_path(tmp_path):
    db_path = tmp_path / "custom" / "location.db"
    assert not db_path.exists()
    store_decision(
        agent="trader", ticker="AAPL", date="2026-07-01",
        signal="Buy", confidence=0.8, key_drivers=None, thesis=None,
        db_path=db_path,
    )
    assert db_path.exists()
