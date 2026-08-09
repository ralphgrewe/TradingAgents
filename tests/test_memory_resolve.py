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


def _seed(db_path, agent="trader", ticker="AAPL", date=BACKDATED, horizon_days=None, **kwargs):
    defaults = {
        "signal": "Buy",
        "confidence": 0.7,
        "key_drivers": ["strong earnings"],
        "thesis": "Momentum plus fundamentals align.",
    }
    defaults.update(kwargs)
    store_decision(
        agent=agent, ticker=ticker, date=date, db_path=db_path,
        horizon_days=horizon_days, **defaults
    )


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


# ---------------------------------------------------------------------------
# Per-row lesson generation failure -> other rows still resolved
# ---------------------------------------------------------------------------

def test_lesson_generation_failure_does_not_abort_batch(tmp_path, _stub_forward_return):
    """When lesson generation fails for one row, other rows in the same
    resolve_pending() call should still be resolved and written."""
    db_path = _db_path(tmp_path)
    _seed(db_path, agent="trader", ticker="AAPL", date=BACKDATED)
    _seed(db_path, agent="trader", ticker="MSFT", date=BACKDATED)

    # Mock _generate_lesson to raise an exception only for the second call
    call_count = [0]

    def lesson_side_effect(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 2:  # fail on second call (MSFT)
            raise ConnectionError("Connection error.")
        return "Lesson for the successful row."

    with patch("tradingagents.memory.resolve._generate_lesson", side_effect=lesson_side_effect) as mock_lesson:
        resolved_ids = resolve_pending(db_path=db_path)

    # Only one row should be resolved (the first one succeeded, second failed)
    assert len(resolved_ids) == 1
    assert mock_lesson.call_count == 2

    conn = get_connection(db_path)
    try:
        rows = {row["ticker"]: row for row in conn.execute("SELECT * FROM decisions").fetchall()}
    finally:
        conn.close()

    # The first row (AAPL) should be resolved
    assert rows["AAPL"]["resolved_at"] is not None
    assert rows["AAPL"]["forward_return"] == pytest.approx(0.05)
    assert rows["AAPL"]["lesson"] == "Lesson for the successful row."
    assert rows["AAPL"]["horizon_days"] == DEFAULT_HORIZON_DAYS

    # The second row (MSFT) should remain pending (not resolved)
    assert rows["MSFT"]["resolved_at"] is None
    assert rows["MSFT"]["forward_return"] is None
    assert rows["MSFT"]["lesson"] is None
    assert rows["MSFT"]["horizon_days"] is None


# ---------------------------------------------------------------------------
# Per-decision horizon_days (issue #91)
# ---------------------------------------------------------------------------

def test_per_decision_horizons_mixed_batch(tmp_path, _stub_lesson, _stub_forward_return):
    """A batch with mixed horizons: one row with explicit horizon, one with NULL.
    Both should resolve correctly using their own horizons."""
    db_path = _db_path(tmp_path)

    # Store a row with explicit short horizon (5 days)
    _seed(db_path, agent="swing", ticker="AAPL", date=BACKDATED, horizon_days=5)

    # Store a row without horizon (will use DEFAULT_HORIZON_DAYS at resolution)
    _seed(db_path, agent="trader", ticker="MSFT", date=BACKDATED, horizon_days=None)

    resolved_ids = resolve_pending(db_path=db_path)

    # Both should have resolved (both windows elapsed, both had price data)
    assert len(resolved_ids) == 2

    conn = get_connection(db_path)
    try:
        rows = {row["ticker"]: row for row in conn.execute("SELECT * FROM decisions").fetchall()}
    finally:
        conn.close()

    # The explicit-horizon row keeps its own horizon
    assert rows["AAPL"]["horizon_days"] == 5
    assert rows["AAPL"]["resolved_at"] is not None
    assert rows["AAPL"]["forward_return"] == pytest.approx(0.05)

    # The NULL-horizon row gets the default at resolution
    assert rows["MSFT"]["horizon_days"] == DEFAULT_HORIZON_DAYS
    assert rows["MSFT"]["resolved_at"] is not None
    assert rows["MSFT"]["forward_return"] == pytest.approx(0.05)

    # Verify both external calls were made with the correct windows
    assert _stub_forward_return.call_count == 2
    calls = _stub_forward_return.call_args_list
    # First call should be with horizon 5 (AAPL)
    assert calls[0][0][2] == 5  # third argument is holding_days
    # Second call should be with default horizon (MSFT)
    assert calls[1][0][2] == DEFAULT_HORIZON_DAYS


def test_short_horizon_eligibility_sooner_than_default(tmp_path, _stub_lesson):
    """A row with a 2-day horizon becomes eligible sooner than a row with
    DEFAULT_HORIZON_DAYS. Test that the short-horizon row is resolvable while
    the longer-horizon row remains pending."""
    db_path = _db_path(tmp_path)

    # A decision from ~3 calendar days ago (ensures at least 2 trading days have elapsed)
    short_horizon_date = _iso_days_ago(3)

    # The same decision date, but this row needs the default horizon
    _seed(db_path, agent="swing", ticker="AAPL", date=short_horizon_date, horizon_days=2)
    _seed(db_path, agent="trader", ticker="MSFT", date=short_horizon_date, horizon_days=None)

    # Mock forward return to fail for the second row (to keep it pending)
    def return_side_effect(ticker, decision_date, holding_days):
        if ticker == "MSFT" and holding_days == DEFAULT_HORIZON_DAYS:
            return None  # data not available for the long horizon
        return 0.05  # data available for short horizon

    with patch("tradingagents.memory.resolve._fetch_forward_return", side_effect=return_side_effect):
        resolved_ids = resolve_pending(db_path=db_path)

    # Only the short-horizon row should be resolved
    assert len(resolved_ids) == 1

    conn = get_connection(db_path)
    try:
        rows = {row["ticker"]: row for row in conn.execute("SELECT * FROM decisions").fetchall()}
    finally:
        conn.close()

    # AAPL (2-day horizon) should be resolved
    assert rows["AAPL"]["resolved_at"] is not None
    assert rows["AAPL"]["horizon_days"] == 2

    # MSFT (default horizon) should still be pending
    assert rows["MSFT"]["resolved_at"] is None
    assert rows["MSFT"]["horizon_days"] is None  # still NULL


def test_existing_null_horizon_rows_resolve_exactly_as_before(tmp_path, _stub_lesson, _stub_forward_return):
    """Regression test: rows stored without horizon_days (all existing agents)
    should resolve exactly as before — using DEFAULT_HORIZON_DAYS throughout."""
    db_path = _db_path(tmp_path)

    # Store several rows without explicit horizons (like the current system)
    for ticker in ["AAPL", "MSFT", "GOOGL"]:
        _seed(db_path, agent="trader", ticker=ticker, date=BACKDATED, horizon_days=None)

    resolved_ids = resolve_pending(db_path=db_path)

    assert len(resolved_ids) == 3

    conn = get_connection(db_path)
    try:
        rows = {row["ticker"]: row for row in conn.execute("SELECT * FROM decisions").fetchall()}
    finally:
        conn.close()

    # All rows should have DEFAULT_HORIZON_DAYS recorded
    for ticker in ["AAPL", "MSFT", "GOOGL"]:
        assert rows[ticker]["horizon_days"] == DEFAULT_HORIZON_DAYS
        assert rows[ticker]["resolved_at"] is not None
        assert rows[ticker]["forward_return"] == pytest.approx(0.05)
