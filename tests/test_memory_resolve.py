"""Tests for tradingagents.memory.resolve — resolve_pending (issue #6)."""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from tradingagents.memory.resolve import DEFAULT_HORIZON_DAYS, resolve_pending
from tradingagents.memory.store import get_connection, store_decision


def _db_path(tmp_path):
    return tmp_path / "memory.db"


def _iso_days_ago(days):
    """A decision date far enough back (in calendar days) that at least
    DEFAULT_HORIZON_DAYS *trading* days have definitely elapsed."""
    dt = datetime.now(timezone.utc) - timedelta(days=days)
    return dt.strftime("%Y-%m-%d")


# A safely "old" decision date: DEFAULT_HORIZON_DAYS trading days is at most
# DEFAULT_HORIZON_DAYS * 7/5 calendar days; add a big margin for holidays/DST.
BACKDATED = _iso_days_ago(DEFAULT_HORIZON_DAYS * 3 + 10)
# A decision made "today" — window has certainly not elapsed.
RECENT = _iso_days_ago(0)


def _row(conn, row_id):
    return conn.execute("SELECT * FROM decisions WHERE id = ?", (row_id,)).fetchone()


def _seed(db_path, agent="trader", ticker="AAPL", date=BACKDATED, **kwargs):
    defaults = {
        "signal": "Buy",
        "confidence": 0.7,
        "key_drivers": ["strong earnings"],
        "thesis": "Momentum plus fundamentals align.",
    }
    defaults.update(kwargs)
    store_decision(agent=agent, ticker=ticker, date=date, db_path=db_path, **defaults)


@pytest.fixture(autouse=True)
def _stub_lesson(monkeypatch):
    """Stub the LLM/reflection call for every test in this module — no live
    LLM provider is required. Individual tests can still assert on calls via
    the returned mock if needed."""
    mock = patch(
        "tradingagents.memory.resolve._generate_lesson",
        return_value="The call was correct and momentum held. Earnings strength was the key driver "
        "that played out as expected. Lesson: keep weighting earnings surprises heavily.",
    )
    stub = mock.start()
    yield stub
    mock.stop()


@pytest.fixture(autouse=True)
def _stub_forward_return(monkeypatch):
    """Stub the yfinance fetch for every test — no live network required."""
    mock = patch("tradingagents.memory.resolve._fetch_forward_return", return_value=0.05)
    stub = mock.start()
    yield stub
    mock.stop()


# ---------------------------------------------------------------------------
# Window elapsed -> resolved
# ---------------------------------------------------------------------------

def test_resolves_backdated_pending_row(tmp_path, _stub_lesson, _stub_forward_return):
    db_path = _db_path(tmp_path)
    _seed(db_path, agent="trader", ticker="AAPL", date=BACKDATED)

    resolved_ids = resolve_pending(db_path=db_path)

    assert len(resolved_ids) == 1
    conn = get_connection(db_path)
    try:
        row = _row(conn, resolved_ids[0])
    finally:
        conn.close()

    assert row["forward_return"] == pytest.approx(0.05)
    assert row["lesson"]
    assert row["lesson"].strip() != ""
    assert row["resolved_at"] is not None
    assert row["horizon_days"] == DEFAULT_HORIZON_DAYS
    assert row["benchmark_return"] is None

    _stub_forward_return.assert_called_once()
    _stub_lesson.assert_called_once()


# ---------------------------------------------------------------------------
# Window not elapsed -> untouched, no external calls
# ---------------------------------------------------------------------------

def test_recent_pending_row_left_untouched(tmp_path, _stub_lesson, _stub_forward_return):
    db_path = _db_path(tmp_path)
    _seed(db_path, agent="trader", ticker="AAPL", date=RECENT)

    resolved_ids = resolve_pending(db_path=db_path)

    assert resolved_ids == []
    conn = get_connection(db_path)
    try:
        rows = conn.execute("SELECT * FROM decisions").fetchall()
    finally:
        conn.close()

    assert len(rows) == 1
    row = rows[0]
    assert row["resolved_at"] is None
    assert row["forward_return"] is None
    assert row["horizon_days"] is None
    assert row["lesson"] is None

    _stub_forward_return.assert_not_called()
    _stub_lesson.assert_not_called()


# ---------------------------------------------------------------------------
# agent= filter
# ---------------------------------------------------------------------------

def test_agent_filter_only_resolves_matching_agent(tmp_path, _stub_lesson, _stub_forward_return):
    db_path = _db_path(tmp_path)
    _seed(db_path, agent="trader", ticker="AAPL", date=BACKDATED)
    _seed(db_path, agent="fundamental", ticker="MSFT", date=BACKDATED)

    resolved_ids = resolve_pending(agent="trader", db_path=db_path)

    conn = get_connection(db_path)
    try:
        rows = {row["agent"]: row for row in conn.execute("SELECT * FROM decisions").fetchall()}
    finally:
        conn.close()

    assert len(resolved_ids) == 1
    assert rows["trader"]["resolved_at"] is not None
    assert rows["fundamental"]["resolved_at"] is None
    assert rows["fundamental"]["forward_return"] is None


# ---------------------------------------------------------------------------
# ticker= filter
# ---------------------------------------------------------------------------

def test_ticker_filter_only_resolves_matching_ticker(tmp_path, _stub_lesson, _stub_forward_return):
    db_path = _db_path(tmp_path)
    _seed(db_path, agent="trader", ticker="AAPL", date=BACKDATED)
    _seed(db_path, agent="trader", ticker="MSFT", date=BACKDATED)

    resolved_ids = resolve_pending(ticker="AAPL", db_path=db_path)

    conn = get_connection(db_path)
    try:
        rows = {row["ticker"]: row for row in conn.execute("SELECT * FROM decisions").fetchall()}
    finally:
        conn.close()

    assert len(resolved_ids) == 1
    assert rows["AAPL"]["resolved_at"] is not None
    assert rows["MSFT"]["resolved_at"] is None
    assert rows["MSFT"]["forward_return"] is None


# ---------------------------------------------------------------------------
# Mixed elapsed/not-elapsed: only eligible rows touched, single batch
# ---------------------------------------------------------------------------

def test_only_elapsed_rows_are_resolved_among_multiple(tmp_path, _stub_lesson, _stub_forward_return):
    db_path = _db_path(tmp_path)
    _seed(db_path, agent="trader", ticker="AAPL", date=BACKDATED)
    _seed(db_path, agent="trader", ticker="MSFT", date=RECENT)

    resolved_ids = resolve_pending(db_path=db_path)

    assert len(resolved_ids) == 1
    conn = get_connection(db_path)
    try:
        rows = {row["ticker"]: row for row in conn.execute("SELECT * FROM decisions").fetchall()}
    finally:
        conn.close()
    assert rows["AAPL"]["resolved_at"] is not None
    assert rows["MSFT"]["resolved_at"] is None
    # Only one external fetch/lesson call — for the eligible row only.
    assert _stub_forward_return.call_count == 1
    assert _stub_lesson.call_count == 1


# ---------------------------------------------------------------------------
# Unresolvable price data (e.g. delisted / too recent per yfinance) -> stays pending
# ---------------------------------------------------------------------------

def test_unavailable_price_data_leaves_row_pending(tmp_path, _stub_lesson):
    db_path = _db_path(tmp_path)
    _seed(db_path, agent="trader", ticker="AAPL", date=BACKDATED)

    with patch("tradingagents.memory.resolve._fetch_forward_return", return_value=None) as fr:
        resolved_ids = resolve_pending(db_path=db_path)

    assert resolved_ids == []
    fr.assert_called_once()
    _stub_lesson.assert_not_called()

    conn = get_connection(db_path)
    try:
        row = conn.execute("SELECT * FROM decisions").fetchone()
    finally:
        conn.close()
    assert row["resolved_at"] is None


# ---------------------------------------------------------------------------
# No pending rows -> empty list, no errors
# ---------------------------------------------------------------------------

def test_no_pending_rows_returns_empty_list(tmp_path, _stub_lesson, _stub_forward_return):
    db_path = _db_path(tmp_path)
    resolved_ids = resolve_pending(db_path=db_path)
    assert resolved_ids == []
    _stub_forward_return.assert_not_called()
    _stub_lesson.assert_not_called()
