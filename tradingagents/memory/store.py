"""SQLite-backed shared memory core — schema + write path.

This module is the source of truth for trading-decision history shared
across the legacy LangGraph pipeline and the standalone skill scripts
(issue #2 / #5). It intentionally covers *only* the schema and the
``store_decision`` write path — resolution (filling in ``forward_return`` /
``lesson`` once an outcome is known) and retrieval/formatting for prompt
injection are separate, later pieces of work (issues #6 and #7).

Design choices (see issue #5 discussion for the full rationale):

- **Storage**: a single SQLite database, one ``decisions`` table. SQLite was
  chosen over markdown/DuckDB in #2 specifically because per-agent /
  per-ticker hit-rate statistics are a hard requirement later (#23); a plain
  SQL table makes that a simple aggregate query instead of a text-parsing
  exercise.
- **Path configuration**: the DB path defaults to ``runs/memory/memory.db``
  (relative to the current working directory — matches the existing
  ``runs/<YYYYMMDD>/<TICKER>`` convention used by the skill scripts, see
  ``skills/trader/SKILL.md``). It can be overridden per-call via the
  ``db_path`` parameter, or globally via the ``TRADINGAGENTS_MEMORY_DB_PATH``
  environment variable (mirrors the ``TRADINGAGENTS_MEMORY_LOG_PATH`` pattern
  already used for the legacy markdown log in ``default_config.py``).
  Precedence: explicit ``db_path`` argument > env var > built-in default.
- **Idempotency on duplicate ``(agent, ticker, decision_date)``**: a
  **no-op**. This matches the existing precedent in the legacy
  ``TradingMemoryLog.store_decision`` (``tradingagents/agents/utils/memory.py``),
  which already treats a duplicate pending tag as "nothing to do" rather
  than overwriting it. It also gives the simplest possible idempotent-write
  semantics for a decision log: the first decision recorded for a given
  agent/ticker/day is what actually happened and should stick, even if the
  caller (e.g. a retried skill run) submits a slightly different thesis on
  a re-run. Enforced at the DB level via a ``UNIQUE`` constraint plus
  ``INSERT OR IGNORE``, so it is race-safe without needing a SELECT-then-INSERT
  round trip.
- **Ticker normalization**: ``store_decision`` stores ``ticker`` exactly as
  given, with **no** call to ``dataflows.symbol_utils.normalize_symbol``.
  Normalization only matters where a broker-style symbol (e.g. ``XAUUSD``)
  needs to be turned into a vendor-specific lookup key (e.g. ``GC=F`` for a
  yfinance call) — that only happens on the *resolve* path (a future
  ``resolve_pending`` in issue #6, which fetches forward returns from
  yfinance) and the *retrieval* path (issue #7, which needs to match a
  requested ticker against stored rows). ``store_decision`` never calls
  yfinance and has no ticker-matching concern, so normalizing here would
  only add a dependency without changing behavior — deferred to #6/#7.
- **Connection model**: one ``sqlite3`` connection per call (opened, used,
  closed), no long-lived connection or exclusive locking held across calls.
  This keeps ``store_decision`` safe to call from concurrent skill
  invocations and does not preclude the batch-read-then-atomic-write shape
  used elsewhere (``TradingMemoryLog.batch_update_with_outcomes``) from being
  layered on top for the future resolve path.
- **confidence**: stored as ``REAL`` (a numeric score). The legacy log has
  no distinct confidence field — its 5-tier ``rating`` (Buy/Overweight/Hold/
  Underweight/Sell, see ``agents/utils/rating.py``) is closer to this
  schema's ``signal`` column. ``confidence`` is new and numeric so that
  later aggregate stats (issue #23, hit-rate weighted by confidence) can do
  arithmetic directly instead of parsing a text enum.
- **horizon_days**: stored as ``INTEGER`` (trading days). Each decision can
  declare its own intended holding period (e.g., a 3-day swing trade vs. a
  10-day momentum play). When a decision row is stored, an optional
  ``horizon_days`` parameter may be passed; if not given, the column is left
  ``NULL`` and ``resolve_pending`` later fills it in with
  ``DEFAULT_HORIZON_DAYS``. Per-decision horizons flow through the MCP
  boundary (``MemoryMCPClient.store_decision``) so that the swing trader's
  1–15 trading day range is properly tracked (issue #91).
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_DB_PATH = Path("runs") / "memory" / "memory.db"
_ENV_VAR = "TRADINGAGENTS_MEMORY_DB_PATH"
_MEMORY_ID_ENV_VAR = "TRADINGAGENTS_MEMORY_ID"
_MEMORY_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent TEXT NOT NULL,
    ticker TEXT NOT NULL,
    decision_date TEXT NOT NULL,
    signal TEXT NOT NULL,
    confidence REAL,
    key_drivers TEXT,
    thesis TEXT,
    created_at TEXT NOT NULL,
    horizon_days INTEGER,
    forward_return REAL,
    benchmark_return REAL,
    lesson TEXT,
    resolved_at TEXT,
    UNIQUE (agent, ticker, decision_date)
);
"""


def resolve_memory_id_to_db_path(memory_id: str | None = None) -> Path:
    """Resolve a memory ID to its corresponding DB path.

    A memory ID is a named identifier that maps to a separate SQLite database
    file for isolation across multiple runs (e.g., different models or configurations).
    This allows different model/config combinations to maintain independent decision
    histories for the same stock list.

    Precedence: ``memory_id`` argument > ``TRADINGAGENTS_MEMORY_ID`` env var >
    ``TRADINGAGENTS_MEMORY_DB_PATH`` env var > built-in default
    (``runs/memory/memory.db``).

    If a ``memory_id`` is provided (not None/empty), it is validated against
    the pattern ``^[A-Za-z0-9][A-Za-z0-9._-]*$`` and the path becomes
    ``runs/memory/<id>/memory.db``.

    Args:
        memory_id: Named memory ID to use. If None/empty, falls back to the
            env var or explicit TRADINGAGENTS_MEMORY_DB_PATH, or the default.

    Returns:
        Path to the resolved database file.

    Raises:
        ValueError: If the memory_id contains invalid characters or is invalid.
    """
    # Check if memory_id is provided and valid
    if memory_id is None or memory_id == "":
        # Fall back to env var or explicit path
        env_id = os.environ.get(_MEMORY_ID_ENV_VAR)
        if env_id and env_id != "":
            memory_id = env_id
        else:
            # Use explicit db_path if set, or default
            return resolve_db_path()

    # Validate the memory_id. ".." is checked explicitly (in addition to the
    # pattern) because the pattern alone would accept it anywhere except as
    # the leading character (e.g. "foo..bar") — the issue requires rejecting
    # any occurrence of ".." outright, not just a leading one.
    if not _MEMORY_ID_PATTERN.match(memory_id) or ".." in memory_id:
        raise ValueError(
            f"Invalid memory_id {memory_id!r}. Must match pattern ^[A-Za-z0-9][A-Za-z0-9._-]*$ "
            f"(starts with alphanumeric, contains alphanumeric/underscore/dash/dot). "
            f"No empty strings, slashes, backslashes, or '..' allowed."
        )

    # Derive path from memory_id: runs/memory/<id>/memory.db
    parent = DEFAULT_DB_PATH.parent  # runs/memory
    return parent / memory_id / "memory.db"


def resolve_db_path(db_path: str | Path | None = None) -> Path:
    """Resolve the effective DB path.

    Precedence: explicit ``db_path`` argument > ``TRADINGAGENTS_MEMORY_DB_PATH``
    env var > built-in default (``runs/memory/memory.db``, relative to cwd).
    """
    if db_path is not None:
        return Path(db_path).expanduser()
    env_value = os.environ.get(_ENV_VAR)
    if env_value:
        return Path(env_value).expanduser()
    return DEFAULT_DB_PATH


def get_connection(db_path: str | Path | None = None) -> sqlite3.Connection:
    """Open a new connection to the memory DB, creating the schema if needed.

    Creates parent directories (e.g. ``runs/memory/``) as needed. One
    connection per call — callers are expected to use it as a context
    manager (or close it explicitly) rather than hold it open long-term.
    """
    path = resolve_db_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


def store_decision(
    agent: str,
    ticker: str,
    date: str,
    signal: str,
    confidence: float | None,
    key_drivers: Sequence[Any] | Mapping[str, Any] | None,
    thesis: str | None,
    db_path: str | Path | None = None,
    horizon_days: int | None = None,
) -> bool:
    """Insert a pending decision row.

    Inserts a new row with all resolution columns (``forward_return``,
    ``benchmark_return``, ``lesson``, ``resolved_at``) left ``NULL`` —
    i.e. "pending" (``resolved_at IS NULL``). The ``horizon_days`` column
    may be set at store time (if the caller has a declared holding period),
    otherwise it is left ``NULL`` and ``resolve_pending`` fills it in with
    ``DEFAULT_HORIZON_DAYS`` at resolution time. Resolving the other
    resolution columns is out of scope here (issue #6).

    Idempotency: if a row already exists for the same
    ``(agent, ticker, date)`` key, this call is a **no-op** — the existing
    row (including its original ``signal``/``confidence``/``thesis``) is left
    untouched and no duplicate row is created. See the module docstring for
    the rationale. Use the return value to distinguish the two cases.

    Args:
        agent: Free-form identifier — a skill name (e.g. ``"trader"``) or a
            legacy agent id. Not constrained to an enum, to stay
            design-compatible with the legacy graph's agent naming.
        ticker: Stored exactly as given — no normalization is performed
            here (see module docstring).
        date: Decision date, e.g. ``"2026-07-02"``. Stored as given; not
            parsed or validated.
        signal: The decision's directional call (e.g. one of the 5-tier
            ratings used elsewhere in this repo — Buy/Overweight/Hold/
            Underweight/Sell — or any other free-form signal string).
        confidence: A numeric confidence score. May be ``None`` if the
            caller has no confidence estimate.
        key_drivers: A JSON-serializable list/dict of the key drivers behind
            the decision. Stored as a JSON-encoded TEXT column. May be
            ``None``.
        thesis: Short free-text rationale. May be ``None``.
        db_path: Optional override for the DB path (see ``resolve_db_path``).
        horizon_days: Optional intended holding period in trading days. If
            given, stored in the ``horizon_days`` column at insert time; if
            ``None``, the column is left ``NULL`` and filled in later by
            ``resolve_pending`` with ``DEFAULT_HORIZON_DAYS``.

    Returns:
        ``True`` if a new pending row was inserted, ``False`` if a row for
        this ``(agent, ticker, date)`` key already existed (no-op).
    """
    created_at = datetime.now(timezone.utc).isoformat()
    key_drivers_json = json.dumps(key_drivers) if key_drivers is not None else None

    conn = get_connection(db_path)
    try:
        cursor = conn.execute(
            """
            INSERT OR IGNORE INTO decisions (
                agent, ticker, decision_date, signal, confidence,
                key_drivers, thesis, created_at, horizon_days
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (agent, ticker, date, signal, confidence, key_drivers_json, thesis, created_at, horizon_days),
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()
