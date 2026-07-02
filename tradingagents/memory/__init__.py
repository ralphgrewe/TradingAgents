"""Shared memory core (SQLite) — schema + write + resolve + retrieval paths.

Public API: ``store_decision`` (write path, issue #5), ``resolve_pending``
(resolve path, issue #6), and ``get_past_context`` (retrieval path, issue
#7), plus the connection/path helpers and the ``DEFAULT_HORIZON_DAYS``
constant they are built on. This completes the core read/write API planned
in #2; hard statistics (issue #23) and the MCP wrapper (issue #24) are
separate, later pieces of work that build on this module rather than
extending it further.
"""

from tradingagents.memory.query import get_past_context
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
    "get_past_context",
    "resolve_db_path",
    "resolve_pending",
    "store_decision",
]
