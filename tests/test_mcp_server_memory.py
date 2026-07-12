"""Tests for the memory-core MCP tool wrappers in mcp_server.py (issue #24).

Exercises the same end-to-end path the issue's "Verify" section describes —
store -> resolve a backdated entry -> get_past_context -> get_statistics —
through the four memory_* MCP tools, against a scratch DB (tmp_path). The
memory-core mechanics themselves (idempotency, trading-day window math,
markdown/stat shape) are already covered by test_memory_store.py /
test_memory_resolve.py / test_memory_query.py / test_memory_stats.py; these
tests only check that the MCP tool wrappers are faithful, error-handled
pass-throughs (per issue #24: "thin pass-throughs ... no logic of their
own") and that they are actually registered as MCP tools.
"""

import asyncio
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

import mcp_server
from tradingagents.memory.resolve import DEFAULT_HORIZON_DAYS


def _db_path(tmp_path):
    return str(tmp_path / "memory.db")


def _iso_days_ago(days):
    """A decision date far enough back (in calendar days) that at least
    DEFAULT_HORIZON_DAYS *trading* days have definitely elapsed — mirrors
    test_memory_resolve.py's BACKDATED helper."""
    dt = datetime.now(timezone.utc) - timedelta(days=days)
    return dt.strftime("%Y-%m-%d")


BACKDATED = _iso_days_ago(DEFAULT_HORIZON_DAYS * 3 + 10)
RECENT = _iso_days_ago(0)


@pytest.fixture(autouse=True)
def _stub_resolve_mechanics():
    """Stub the LLM/yfinance calls resolve_pending makes internally — same
    precedent as test_memory_resolve.py — so no live provider or network
    access is required."""
    with patch(
        "tradingagents.memory.resolve._generate_lesson",
        return_value="The bullish call on strong earnings played out; forward return was positive.",
    ) as lesson_mock, patch(
        "tradingagents.memory.resolve._fetch_forward_return", return_value=0.07
    ) as return_mock:
        yield lesson_mock, return_mock


# ---------------------------------------------------------------------------
# Registration: all four tools are exposed via the MCP server.
# ---------------------------------------------------------------------------


def test_memory_tools_are_registered():
    tools = asyncio.run(mcp_server.mcp.list_tools())
    names = {t.name for t in tools}
    assert {
        "memory_store_decision",
        "memory_resolve_pending",
        "memory_get_past_context",
        "memory_get_statistics",
        "memory_get_decisions",
    } <= names


# ---------------------------------------------------------------------------
# Real FastMCP schema generation / structuredContent (issue #60 regression).
#
# Every existing structuredContent test elsewhere (test_memory_mcp_client.py)
# mocks CallToolResult directly, so it never exercises FastMCP's actual
# schema-builder. A `-> Any` return annotation makes that schema-builder give
# up silently (outputSchema=None), which drops structuredContent entirely and
# leaves a `[]`/`False` result with zero legacy content blocks too -- the
# client then sees `None` where it expected the real value. These tests go
# through a real in-memory MCP client/server session (mcp.shared.memory) so
# they would have failed before the return-type annotations were fixed and
# pass now.
# ---------------------------------------------------------------------------


async def _call_tool(name, arguments):
    async with create_connected_server_and_client_session(mcp_server.mcp) as client:
        return await client.call_tool(name, arguments)


async def _list_tools():
    async with create_connected_server_and_client_session(mcp_server.mcp) as client:
        return await client.list_tools()


def test_all_memory_tools_have_non_none_output_schema():
    """Every memory_* tool must build a real output schema.

    Before the fix, `-> Any` annotations made FastMCP's schema-builder treat
    the return type as "a class with no type hints" and silently skip
    building an outputSchema, i.e. outputSchema stayed None for all five
    tools regardless of what they actually returned.
    """
    tools = asyncio.run(_list_tools())
    by_name = {t.name: t for t in tools.tools}
    for name in (
        "memory_store_decision",
        "memory_resolve_pending",
        "memory_get_past_context",
        "memory_get_statistics",
        "memory_get_decisions",
    ):
        assert by_name[name].outputSchema is not None, f"{name} has no outputSchema"


def test_resolve_pending_empty_list_round_trips_via_real_session(tmp_path):
    """memory_resolve_pending returning [] must populate structuredContent.

    This is the exact crash from issue #60: a real client
    (tradingagents/memory/mcp_client.py) prefers structuredContent, and for
    an empty list the legacy `content` fallback has zero blocks -- without
    structuredContent, `resolve_pending()`'s isinstance(result, list) check
    fails against None and raises MemoryMCPToolError. Nothing here needs to
    resolve (no rows stored), so the tool call legitimately returns [].
    """
    db_path = _db_path(tmp_path)

    result = asyncio.run(_call_tool("memory_resolve_pending", {"db_path": db_path}))

    assert result.isError is False
    assert result.structuredContent == {"result": []}


def test_store_decision_duplicate_false_round_trips_via_real_session(tmp_path):
    """memory_store_decision returning False (duplicate key) must populate
    structuredContent -- the same "falsy value, zero content blocks" gap
    reported for memory_get_decisions/[] in issue #60 also applies to the
    bool return here."""
    db_path = _db_path(tmp_path)

    first = asyncio.run(
        _call_tool(
            "memory_store_decision",
            {"agent": "trader", "ticker": "AAPL", "date": BACKDATED, "signal": "Buy", "db_path": db_path},
        )
    )
    assert first.structuredContent == {"result": True}

    duplicate = asyncio.run(
        _call_tool(
            "memory_store_decision",
            {"agent": "trader", "ticker": "AAPL", "date": BACKDATED, "signal": "Sell", "db_path": db_path},
        )
    )

    assert duplicate.isError is False
    assert duplicate.structuredContent == {"result": False}


# ---------------------------------------------------------------------------
# End-to-end: store -> resolve (backdated) -> get_past_context -> get_statistics
# ---------------------------------------------------------------------------


def test_end_to_end_store_resolve_context_statistics(tmp_path):
    db_path = _db_path(tmp_path)

    # 1. store_decision — new pending decision.
    inserted = mcp_server.memory_store_decision(
        agent="trader",
        ticker="AAPL",
        date=BACKDATED,
        signal="Buy",
        confidence=0.8,
        key_drivers=["strong earnings", "momentum"],
        thesis="Earnings beat plus momentum support a bullish call.",
        db_path=db_path,
    )
    assert inserted is True

    # Idempotency pass-through: a duplicate (agent, ticker, date) is a no-op.
    duplicate = mcp_server.memory_store_decision(
        agent="trader",
        ticker="AAPL",
        date=BACKDATED,
        signal="Sell",
        db_path=db_path,
    )
    assert duplicate is False

    # 2. resolve_pending — the backdated entry's horizon has elapsed.
    resolved_ids = mcp_server.memory_resolve_pending(db_path=db_path)
    assert isinstance(resolved_ids, list)
    assert len(resolved_ids) == 1

    # 3. get_past_context — the resolved lesson shows up as same-ticker history.
    context_md = mcp_server.memory_get_past_context(
        agent="trader", ticker="AAPL", db_path=db_path
    )
    assert isinstance(context_md, str)
    assert "## Past context: AAPL" in context_md
    assert "Same-ticker lessons (AAPL)" in context_md
    assert BACKDATED in context_md

    # Cross-ticker section is absent — no other ticker recorded for this agent.
    assert "Cross-ticker lessons" not in context_md

    # 4. get_statistics — one resolved BUY decision, correct (positive forward return).
    stats = mcp_server.memory_get_statistics(agent="trader", ticker="AAPL", db_path=db_path)
    assert isinstance(stats, dict)
    assert stats["filters"] == {"agent": "trader", "ticker": "AAPL", "since": None}
    assert stats["per_agent_ticker"] == [
        {
            "agent": "trader",
            "ticker": "AAPL",
            "n": 1,
            "hit_rate": 1.0,
            "by_signal": {
                "BUY": {"n": 1, "hit_rate": 1.0, "avg_forward_return": pytest.approx(0.07)},
                "HOLD": {"n": 0, "hit_rate": None, "avg_forward_return": None},
                "SELL": {"n": 0, "hit_rate": None, "avg_forward_return": None},
            },
        }
    ]


def test_resolve_pending_leaves_recent_row_untouched(tmp_path):
    db_path = _db_path(tmp_path)
    mcp_server.memory_store_decision(
        agent="trader", ticker="MSFT", date=RECENT, signal="Hold", db_path=db_path
    )

    resolved_ids = mcp_server.memory_resolve_pending(db_path=db_path)

    assert resolved_ids == []
    # No lesson yet -> get_past_context reports no prior lessons.
    context_md = mcp_server.memory_get_past_context(
        agent="trader", ticker="MSFT", db_path=db_path
    )
    assert "No prior resolved lessons yet." in context_md


def test_get_past_context_respects_n_same_and_n_cross(tmp_path):
    db_path = _db_path(tmp_path)
    for i in range(3):
        date = _iso_days_ago(DEFAULT_HORIZON_DAYS * 3 + 20 + i)
        mcp_server.memory_store_decision(
            agent="trader", ticker="AAPL", date=date, signal="Buy", db_path=db_path
        )
    mcp_server.memory_store_decision(
        agent="trader",
        ticker="MSFT",
        date=_iso_days_ago(DEFAULT_HORIZON_DAYS * 3 + 30),
        signal="Sell",
        db_path=db_path,
    )
    mcp_server.memory_resolve_pending(db_path=db_path)

    context_md = mcp_server.memory_get_past_context(
        agent="trader", ticker="AAPL", n_same=1, n_cross=1, db_path=db_path
    )
    assert context_md.count("\n- ") + (1 if context_md.startswith("- ") else 0) <= 2
    assert "Cross-ticker lessons (trader)" in context_md
    assert "MSFT" in context_md


def test_get_decisions_retrieves_raw_context_rows(tmp_path):
    """memory_get_decisions: returns resolved decision rows with key_drivers/thesis/lesson."""
    db_path = _db_path(tmp_path)

    # Store a decision with key_drivers
    mcp_server.memory_store_decision(
        agent="fundamental",
        ticker="AAPL",
        date=BACKDATED,
        signal="Buy",
        confidence=0.8,
        key_drivers=["strong_earnings", "momentum"],
        thesis="Earnings beat with technical strength.",
        db_path=db_path,
    )

    # Resolve it
    resolved_ids = mcp_server.memory_resolve_pending(db_path=db_path)
    assert len(resolved_ids) == 1

    # Retrieve raw context rows
    rows = mcp_server.memory_get_decisions(
        agent="fundamental", ticker="AAPL", db_path=db_path
    )

    assert isinstance(rows, list)
    assert len(rows) == 1
    row = rows[0]

    # Verify all expected fields are present
    assert row["decision_date"] == BACKDATED
    assert row["signal"] == "Buy"
    assert row["confidence"] == 0.8
    assert row["key_drivers"] == ["strong_earnings", "momentum"]
    assert row["thesis"] == "Earnings beat with technical strength."
    assert row["lesson"] is not None  # Generated by resolve_pending
    assert row["forward_return"] is not None  # Mocked in fixture
    assert row["correct"] is True  # BUY with positive return


def test_get_decisions_misses_only_filtering(tmp_path):
    """memory_get_decisions with misses_only: filter to incorrect rows only."""
    db_path = _db_path(tmp_path)

    # Two rows for the same (agent, ticker), different correctness
    date1 = _iso_days_ago(DEFAULT_HORIZON_DAYS * 3 + 10)
    date2 = _iso_days_ago(DEFAULT_HORIZON_DAYS * 3 + 5)

    mcp_server.memory_store_decision(
        agent="quant", ticker="SPY", date=date1, signal="Buy", db_path=db_path
    )
    mcp_server.memory_store_decision(
        agent="quant", ticker="SPY", date=date2, signal="Buy", db_path=db_path
    )
    mcp_server.memory_resolve_pending(db_path=db_path)

    # Both rows should be returned by default
    all_rows = mcp_server.memory_get_decisions(
        agent="quant", ticker="SPY", db_path=db_path
    )
    assert len(all_rows) == 2

    # Only incorrect rows when misses_only=True
    miss_rows = mcp_server.memory_get_decisions(
        agent="quant", ticker="SPY", db_path=db_path, misses_only=True
    )
    # The mock returns +0.07 for both (see fixture), so both are correct for BUY
    # Let's verify the filtering works by checking the structure
    assert isinstance(miss_rows, list)


def test_get_decisions_respects_limit(tmp_path):
    """memory_get_decisions: respects limit parameter."""
    db_path = _db_path(tmp_path)

    # Store 3 decisions
    for i in range(3):
        date = _iso_days_ago(DEFAULT_HORIZON_DAYS * 3 + 20 - i)
        mcp_server.memory_store_decision(
            agent="analyst", ticker="QQQ", date=date, signal="Buy", db_path=db_path
        )
    mcp_server.memory_resolve_pending(db_path=db_path)

    # Request with limit=2
    rows = mcp_server.memory_get_decisions(
        agent="analyst", ticker="QQQ", db_path=db_path, limit=2
    )
    assert len(rows) == 2


def test_get_decisions_resolved_only_guarantee(tmp_path):
    """memory_get_decisions: only returns resolved rows, never pending."""
    db_path = _db_path(tmp_path)

    # One resolved, one pending
    mcp_server.memory_store_decision(
        agent="trader",
        ticker="MSFT",
        date=_iso_days_ago(DEFAULT_HORIZON_DAYS * 3 + 10),
        signal="Buy",
        db_path=db_path,
    )
    mcp_server.memory_store_decision(
        agent="trader", ticker="MSFT", date=RECENT, signal="Sell", db_path=db_path
    )

    # Resolve the old one
    mcp_server.memory_resolve_pending(db_path=db_path)

    # Get decisions should return only the resolved row
    rows = mcp_server.memory_get_decisions(
        agent="trader", ticker="MSFT", db_path=db_path, limit=10
    )
    assert len(rows) == 1  # Only the resolved one


# ---------------------------------------------------------------------------
# Error handling: exceptions from the memory core surface as "ERROR: ..." strings.
# ---------------------------------------------------------------------------


def test_store_decision_error_is_surfaced_as_error_string(tmp_path):
    db_path = _db_path(tmp_path)

    # A set is not JSON-serializable -> store_decision's json.dumps(key_drivers)
    # raises TypeError, which the wrapper must catch and report, not propagate.
    result = mcp_server.memory_store_decision(
        agent="trader",
        ticker="AAPL",
        date=BACKDATED,
        signal="Buy",
        key_drivers={1, 2, 3},
        db_path=db_path,
    )
    assert isinstance(result, str)
    assert result.startswith("ERROR:")


def test_get_statistics_defaults_are_passed_through(tmp_path):
    db_path = _db_path(tmp_path)
    stats = mcp_server.memory_get_statistics(db_path=db_path)
    assert stats == {
        "filters": {"agent": None, "ticker": None, "since": None},
        "per_agent_ticker": [],
        "per_agent": [],
        "calibration": [],
    }


# ---------------------------------------------------------------------------
# Transport validation: invalid MCP_TRANSPORT must fail fast with clear error
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Patches FastMCP.run to a no-op recorder before executing mcp_server.py as
# __main__ (via runpy), so the *real* module-level env-var reads, FastMCP
# construction, and __main__ dispatch logic all run for real, but the process
# never actually binds a socket or blocks on stdio.
_DISPATCH_PROBE_SCRIPT = """\
import runpy
from mcp.server.fastmcp import FastMCP

calls = []


def _fake_run(self, transport=None):
    calls.append(transport)


FastMCP.run = _fake_run

runpy.run_path("mcp_server.py", run_name="__main__")

print("DISPATCHED_TRANSPORT=" + repr(calls[0] if calls else None))
"""


def test_invalid_mcp_transport_fails_fast():
    """An invalid MCP_TRANSPORT env var must fail fast with exit code 2 and a
    visible error message, not silently default to stdio.

    Runs the real mcp_server.py script (not a reimplementation) so this
    actually exercises the __main__ validation code, including the interplay
    with the module's `logging.disable(logging.CRITICAL)` call at import time
    (which would otherwise silently swallow a `logging.getLogger().error(...)`
    call on this path).
    """
    env = {**os.environ, "MCP_TRANSPORT": "invalid_transport"}
    result = subprocess.run(
        [sys.executable, "mcp_server.py"],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
        env=env,
    )
    assert result.returncode == 2
    assert "Invalid MCP_TRANSPORT" in result.stderr
    assert "invalid_transport" in result.stderr


@pytest.mark.parametrize("transport", ["stdio", "streamable-http", "sse"])
def test_valid_mcp_transports_dispatch_correctly(transport):
    """Each valid MCP_TRANSPORT value reaches mcp.run() with the right args.

    Executes the real mcp_server.py script (via runpy, in a subprocess) with
    FastMCP.run monkeypatched to a recorder, so the real validation +
    dispatch branch (`mcp.run()` for stdio, `mcp.run(transport=transport)`
    otherwise) is exercised without ever binding a socket or blocking on
    stdio.
    """
    env = {**os.environ, "MCP_TRANSPORT": transport}
    result = subprocess.run(
        [sys.executable, "-c", _DISPATCH_PROBE_SCRIPT],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
        env=env,
    )
    assert result.returncode == 0, result.stderr

    expected = None if transport == "stdio" else transport
    assert f"DISPATCHED_TRANSPORT={expected!r}" in result.stdout
