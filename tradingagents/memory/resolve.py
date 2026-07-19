"""SQLite-backed shared memory core — resolve path (issue #6).

Fills in the resolution columns (``horizon_days``, ``forward_return``,
``lesson``, ``resolved_at``) on ``decisions`` rows written by
``store_decision`` (issue #5), once enough trading days have elapsed since
``decision_date`` to compute a realized forward return. Retrieval/formatting
for prompt injection is separate, later work (issue #7).

Design choices (see issue #6 discussion / the #5 handoff notes for the full
rationale):

- **Horizon**: per-row, with a module-level default fallback. Each row may
  have been stored with a declared ``horizon_days`` (e.g., a swing trader's
  1–15 trading day range, issue #91); when ``resolve_pending`` processes a
  row, it uses that stored horizon if non-``NULL``, falling back to
  ``DEFAULT_HORIZON_DAYS`` (default 10 trading days) if ``NULL``. Both the
  eligibility window check and the forward-return window use the row's own
  horizon. A row with ``NULL`` horizon at store time gets
  ``DEFAULT_HORIZON_DAYS`` written to ``horizon_days`` at resolution time,
  so all resolved rows have a non-``NULL`` horizon recorded.
- **Elapsed check is trading-day-aware and happens *before* any external
  call**: ``_trading_days_elapsed`` uses ``numpy.busday_count`` (Mon-Fri,
  no market-holiday calendar — the same level of precision
  ``TradingAgentsGraph._fetch_returns`` already works at, which just buffers
  by 7 *calendar* days rather than consulting an exchange calendar) so that
  rows whose window has not elapsed yet are skipped without ever touching
  yfinance or the LLM.
- **Ticker normalization**: applied here, at the yfinance-lookup read
  boundary (via ``tradingagents.dataflows.symbol_utils.normalize_symbol``,
  issue #4) — never at ``store_decision`` write time (see
  ``tradingagents/memory/store.py`` module docstring for why). The row's
  stored ``ticker`` column is never rewritten; only the outbound yfinance
  query uses the normalized symbol.
- **Raw return only**: mirrors ``TradingAgentsGraph._fetch_returns``'s
  yfinance start/end-window math, but computes only the raw price return —
  ``benchmark_return`` is intentionally left ``NULL`` on every row written
  here. It is reserved for the deferred Tier-3 depot entries (see issue #6
  body), which are out of scope for this module.
- **Lesson generation**: adapted from ``tradingagents.graph.reflection.Reflector``
  (see ``Reflector.reflect_on_decision``, added alongside the existing
  ``reflect_on_final_decision`` for this issue) — same "2-4 sentence plain
  prose" prompting style and the same ``quick_think_llm``/``create_llm_client``
  wiring used by the legacy graph's reflection path, just fed the stored
  decision row's fields instead of a free-text final decision plus alpha.
- **Not-yet-resolvable rows are skipped, not failed**: if the elapsed check
  passes but yfinance still has no usable price data (too recent, delisted,
  network error), the row is left pending to retry on a future call — same
  precedent as ``TradingAgentsGraph._resolve_pending_entries``.
- **Single transaction**: pending+eligible rows are read in one query, all
  external calls (yfinance, LLM) happen against that in-memory snapshot,
  and then every resolved row is written via one ``sqlite3`` connection
  with one batch of ``UPDATE`` statements and a single ``commit()`` — no
  connection is opened per row and no partial state is ever written for a
  row that didn't fully resolve.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yfinance as yf

from tradingagents.dataflows.symbol_utils import normalize_symbol
from tradingagents.memory.store import get_connection

logger = logging.getLogger(__name__)

# Trading-day horizon used to decide when a pending decision is eligible to
# be resolved, and the value written to ``decisions.horizon_days`` for every
# row resolved by this module. Single, module-level knob by design (issue #6).
DEFAULT_HORIZON_DAYS = 10


def _trading_days_elapsed(decision_date: str, as_of: str) -> int:
    """Count trading (weekday) days elapsed from ``decision_date`` to ``as_of``.

    Uses ``numpy.busday_count`` (Mon-Fri only; no exchange holiday calendar —
    consistent with the level of precision ``_fetch_returns`` already uses
    elsewhere, i.e. a calendar-day buffer rather than a market calendar).
    Half-open interval matching ``numpy.busday_count`` semantics:
    ``decision_date`` itself is not counted, ``as_of`` is excluded. Returns
    ``0`` (rather than a negative count) if ``as_of`` is not after
    ``decision_date``.
    """
    if as_of <= decision_date:
        return 0
    return int(np.busday_count(decision_date, as_of))


def _fetch_forward_return(
    ticker: str, decision_date: str, holding_days: int
) -> float | None:
    """Fetch the raw forward return for ``ticker`` over ``holding_days`` trading days.

    Mirrors the yfinance start/end-window approach used by
    ``TradingAgentsGraph._fetch_returns`` (``tradingagents/graph/trading_graph.py``)
    but computes only the raw price return — no benchmark/alpha (see module
    docstring). ``ticker`` is expected to already be normalized by the
    caller (``normalize_symbol``).

    Returns ``None`` if price data is unavailable (too recent, delisted, or
    a network error) — same "not yet resolvable, retry later" semantics as
    ``_fetch_returns``.
    """
    try:
        start = datetime.strptime(decision_date, "%Y-%m-%d")
        end = start + timedelta(days=holding_days + 7)  # buffer for weekends/holidays
        end_str = end.strftime("%Y-%m-%d")

        stock = yf.Ticker(ticker).history(start=decision_date, end=end_str)
        if len(stock) < 2:
            return None

        actual_days = min(holding_days, len(stock) - 1)
        return float(
            (stock["Close"].iloc[actual_days] - stock["Close"].iloc[0])
            / stock["Close"].iloc[0]
        )
    except Exception as e:
        logger.warning(
            "Could not resolve forward return for %s on %s (will retry next run): %s",
            ticker, decision_date, e,
        )
        return None


# Why resolving a pending row needs an LLM call at all: this isn't a plain
# lookup. `_generate_lesson` below turns a resolved decision (signal,
# confidence, key drivers, realized forward return) into a 2-4 sentence
# prose "lesson" for later prompt injection — that's a synthesis step, and
# synthesis needs an LLM. See the "Lesson generation" bullet in this
# module's top docstring for the fuller rationale (issue #60).
def _build_reflector():
    """Build a ``Reflector`` wired to the configured quick-thinking LLM.

    Lazy-imported so that ``import tradingagents.memory`` alone does not
    pull in the full ``tradingagents.graph`` / ``tradingagents.agents``
    import chain — that cost is only paid when a lesson actually needs to
    be generated. Tests bypass this entirely by monkeypatching
    ``_generate_lesson``.
    """
    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.graph.reflection import Reflector
    from tradingagents.llm_clients import create_llm_client

    client = create_llm_client(
        provider=DEFAULT_CONFIG["llm_provider"],
        model=DEFAULT_CONFIG["quick_think_llm"],
        base_url=DEFAULT_CONFIG.get("backend_url"),
    )
    return Reflector(client.get_llm())


def _generate_lesson(
    signal: str,
    confidence: float | None,
    key_drivers: Any | None,
    thesis: str | None,
    forward_return: float,
) -> str:
    """Produce a 2-4 sentence lesson for one resolved decision row.

    Thin wrapper around ``Reflector.reflect_on_decision``
    (``tradingagents/graph/reflection.py``, adapted from the existing
    ``reflect_on_final_decision`` for issue #6). Kept as its own
    module-level function so callers/tests can monkeypatch
    ``tradingagents.memory.resolve._generate_lesson`` directly, without
    needing a live LLM provider configured.
    """
    reflector = _build_reflector()
    return reflector.reflect_on_decision(
        signal=signal,
        confidence=confidence,
        key_drivers=key_drivers,
        thesis=thesis,
        forward_return=forward_return,
    )


def resolve_pending(
    agent: str | None = None,
    ticker: str | None = None,
    db_path: str | Path | None = None,
) -> list[int]:
    """Resolve pending decisions whose N-trading-day window has elapsed.

    Selects rows from ``decisions`` where ``resolved_at IS NULL``
    (optionally filtered to a single ``agent`` and/or ``ticker``, matched
    exactly against the stored — un-normalized — values), then for each row
    checks whether enough trading days have elapsed since ``decision_date``
    (``_trading_days_elapsed``). Each row uses its own ``horizon_days`` if
    non-``NULL``, falling back to ``DEFAULT_HORIZON_DAYS`` if ``NULL``.

    Rows whose window has **not** elapsed are left completely untouched —
    no write, and no external call (yfinance or LLM) is made for them.

    For rows whose window **has** elapsed:
      1. The ticker is normalized via
         ``tradingagents.dataflows.symbol_utils.normalize_symbol`` (issue #4)
         before the yfinance lookup, mirroring ``_fetch_returns``.
      2. The raw forward return is fetched (``_fetch_forward_return``,
         same yfinance window math as ``_fetch_returns``) using the row's
         own horizon window. Only the raw return is computed —
         ``benchmark_return`` is always left ``NULL`` by this function.
      3. If price data isn't available yet (``_fetch_forward_return``
         returns ``None`` — too recent, delisted, network error), the row
         is left pending to retry on a future call.
      4. Otherwise a 2-4 sentence lesson is generated
         (``_generate_lesson`` / ``Reflector.reflect_on_decision``) from the
         row's original ``signal``/``confidence``/``key_drivers``/``thesis``
         and the realized forward return.

    All resolved rows are written in a single transaction: the
    pending+eligible rows are read once up front, every external call
    happens against that snapshot, and then one ``sqlite3`` connection
    issues every ``UPDATE`` and commits once. Each resolved row keeps its
    own horizon (or has the default written if it was ``NULL``).

    Args:
        agent: If given, only consider pending rows for this agent.
        ticker: If given, only consider pending rows whose stored (raw,
            un-normalized) ``ticker`` column equals this value.
        db_path: Optional override for the DB path (see
            ``tradingagents.memory.store.resolve_db_path``).

    Returns:
        A list of ``decisions.id`` values that were resolved (written) by
        this call, in ascending id order. Empty if no pending row was both
        selected by the filters and eligible (window elapsed with usable
        price data).
    """
    conn = get_connection(db_path)
    try:
        query = "SELECT * FROM decisions WHERE resolved_at IS NULL"
        params: list[Any] = []
        if agent is not None:
            query += " AND agent = ?"
            params.append(agent)
        if ticker is not None:
            query += " AND ticker = ?"
            params.append(ticker)
        query += " ORDER BY id"
        pending_rows = conn.execute(query, params).fetchall()
    finally:
        conn.close()

    today = datetime.now(timezone.utc).date().isoformat()
    resolved_at = datetime.now(timezone.utc).isoformat()

    updates: list[dict[str, Any]] = []
    for row in pending_rows:
        # Use the row's own horizon if set, otherwise fall back to the default.
        row_horizon = row["horizon_days"] if row["horizon_days"] is not None else DEFAULT_HORIZON_DAYS

        if _trading_days_elapsed(row["decision_date"], today) < row_horizon:
            continue  # window not elapsed — leave pending, no external calls

        normalized_ticker = normalize_symbol(row["ticker"])
        forward_return = _fetch_forward_return(
            normalized_ticker, row["decision_date"], row_horizon
        )
        if forward_return is None:
            continue  # price data not available yet — retry on a future call

        key_drivers = json.loads(row["key_drivers"]) if row["key_drivers"] else None
        try:
            lesson = _generate_lesson(
                signal=row["signal"],
                confidence=row["confidence"],
                key_drivers=key_drivers,
                thesis=row["thesis"],
                forward_return=forward_return,
            )
        except Exception as e:
            logger.warning(
                "Could not generate lesson for %s on %s (will retry next run): %s",
                row["ticker"], row["decision_date"], e,
            )
            continue  # leave row pending, do not add to updates

        updates.append(
            {
                "id": row["id"],
                "horizon_days": row_horizon,
                "forward_return": forward_return,
                "lesson": lesson,
                "resolved_at": resolved_at,
            }
        )

    if not updates:
        return []

    conn = get_connection(db_path)
    try:
        for u in updates:
            conn.execute(
                """
                UPDATE decisions
                SET horizon_days = ?, forward_return = ?, lesson = ?, resolved_at = ?
                WHERE id = ?
                """,
                (u["horizon_days"], u["forward_return"], u["lesson"], u["resolved_at"], u["id"]),
            )
        conn.commit()
    finally:
        conn.close()

    return [u["id"] for u in updates]
