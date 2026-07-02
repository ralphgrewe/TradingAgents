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
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence, Union

DEFAULT_DB_PATH = Path("runs") / "memory" / "memory.db"
_ENV_VAR = "TRADINGAGENTS_MEMORY_DB_PATH"

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


def resolve_db_path(db_path: Union[str, Path, None] = None) -> Path:
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


def get_connection(db_path: Union[str, Path, None] = None) -> sqlite3.Connection:
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
    key_drivers: Union[Sequence[Any], Mapping[str, Any], None],
    thesis: str | None,
    db_path: Union[str, Path, None] = None,
) -> bool:
    """Insert a pending decision row.

    Inserts a new row with all resolution columns (``horizon_days``,
    ``forward_return``, ``benchmark_return``, ``lesson``, ``resolved_at``)
    left ``NULL`` — i.e. "pending" (``resolved_at IS NULL``). Resolving
    those columns later is out of scope here (issue #6).

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
                key_drivers, thesis, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (agent, ticker, date, signal, confidence, key_drivers_json, thesis, created_at),
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()
