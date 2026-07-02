"""Shared memory core (SQLite) — schema + write + resolve paths.

Public API for now: ``store_decision`` (write path, issue #5) and
``resolve_pending`` (resolve path, issue #6), plus the connection/path
helpers and the ``DEFAULT_HORIZON_DAYS`` constant they are built on.
Retrieval (issue #7) will extend this package with its own functions rather
than its own module where practical, keeping ``tradingagents.memory`` as
the single import surface for both the legacy graph and skill scripts.
"""

from tradingagents.memory.resolve import DEFAULT_HORIZON_DAYS, resolve_pending
from tradingagents.memory.store import (
    DEFAULT_DB_PATH,
    get_connection,
    resolve_db_path,
    store_decision,
)

__all__ = [
    "DEFAULT_DB_PATH",
    "DEFAULT_HORIZON_DAYS",
    "get_connection",
    "resolve_db_path",
    "resolve_pending",
    "store_decision",
]
