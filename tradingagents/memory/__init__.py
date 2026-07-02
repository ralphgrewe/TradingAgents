"""Shared memory core (SQLite) — schema + write + resolve + retrieval + stats paths.

Public API: ``store_decision`` (write path, issue #5), ``resolve_pending``
(resolve path, issue #6), ``get_past_context`` (retrieval path, issue #7),
and ``get_statistics`` / ``format_statistics_markdown`` (statistics path,
issue #23), plus the connection/path helpers and the ``DEFAULT_HORIZON_DAYS``
constant they are built on. This completes the core read/write API planned
in #2; the MCP wrapper (issue #24) is separate, later work that builds on
this module rather than extending it further.
"""

from tradingagents.memory.query import get_past_context
from tradingagents.memory.resolve import DEFAULT_HORIZON_DAYS, resolve_pending
from tradingagents.memory.stats import format_statistics_markdown, get_statistics
from tradingagents.memory.store import (
    DEFAULT_DB_PATH,
    get_connection,
    resolve_db_path,
    store_decision,
)

__all__ = [
    "DEFAULT_DB_PATH",
    "DEFAULT_HORIZON_DAYS",
    "format_statistics_markdown",
    "get_connection",
    "get_past_context",
    "get_statistics",
    "resolve_db_path",
    "resolve_pending",
    "store_decision",
]
