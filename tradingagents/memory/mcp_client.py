"""Synchronous MCP client for the shared memory core's networked server (issue #51).

Connects to ``mcp_server.py`` running under a networked transport
(``streamable-http`` or ``sse`` — see ``start_server.sh`` /
``MCP_TRANSPORT``/``FASTMCP_HOST``/``FASTMCP_PORT`` at the top of
``mcp_server.py``) and wraps its five memory tools
(``memory_store_decision``, ``memory_resolve_pending``,
``memory_get_past_context``, ``memory_get_statistics``,
``memory_get_decisions``) with typed Python methods mirroring the signatures
of the functions they proxy to
(``tradingagents.memory.{store,resolve,query,stats}``).

This module is the shared foundation #50's ``trading_graph.py`` (issue #53)
and ``find_patterns.py`` (issue #54) conversions build on. It is scoped to
*only* the client: no caller is converted to use it here — that happens in
each consuming module.

Usage::

    from tradingagents.memory.mcp_client import MemoryMCPClient

    with MemoryMCPClient() as client:
        client.store_decision("trader", "AAPL", "2026-07-01", "BUY", confidence=0.7)
        markdown = client.get_past_context("trader", "AAPL")

Design choices (mirrors ``tradingagents/simulation/__init__.py``'s
``SimulationClient`` — connect/close/context-manager shape and event-loop
lifecycle — adapted from ``mcp.client.stdio`` to ``mcp.client.streamable_http``
/ ``mcp.client.sse``; see commits c42d73c and 4bf7125 for the bugs that shape
was hardened against):

- **Transport selection**: ``streamable-http`` (default) or ``sse``, chosen
  via the ``transport`` constructor argument, the
  ``TRADINGAGENTS_MEMORY_MCP_TRANSPORT`` env var, or the
  ``memory_mcp_transport`` config key (in that precedence order — mirrors
  ``SimulationClient``'s ``server_command``/``server_args`` precedence).
- **URL**: an explicit ``url`` argument (or ``TRADINGAGENTS_MEMORY_MCP_URL`` /
  ``memory_mcp_url`` config key) always wins. Otherwise a default is derived
  from the resolved transport and FastMCP's own default mount paths for that
  transport — verified against the installed ``mcp`` SDK
  (``mcp/server/fastmcp/server.py``: ``streamable_http_path="/mcp"``,
  ``sse_path="/sse"``) rather than assumed — combined with the same
  ``127.0.0.1:8000`` host/port ``start_server.sh`` binds
  (``FASTMCP_HOST``/``FASTMCP_PORT`` there bind the *server*; the client
  talks to it over loopback by default, same as any other local client).
- **Event-loop lifecycle**: identical to ``SimulationClient`` — a single
  event loop and ``AsyncExitStack`` are created in ``connect()`` and kept
  alive for the client's whole lifetime (the transport's streams and the
  ``ClientSession`` are bound to whichever loop entered their async context
  managers), reused for every subsequent tool call, and unwound together in
  ``close()``. ``close()`` never raises (same contract and rationale as
  ``SimulationClient.close()``, see commit c42d73c) — any error unwinding the
  exit stack or closing the loop is logged as a warning rather than
  propagated, and internal state is always cleared.
- **Error handling**: ``MemoryMCPConnectionError`` for "can't reach / can't
  initialize a session with the server" (mirrors
  ``SimulationConnectionError``); ``MemoryMCPToolError`` for "the tool call
  itself failed" (mirrors ``SimulationToolError``) — covering both an
  MCP-protocol-level error (``CallToolResult.isError``) *and* the memory
  tools' own error-signaling convention (a successful JSON-RPC result whose
  payload is a string starting with ``"ERROR:"`` — see ``mcp_server.py``'s
  module docstring: each ``memory_*`` tool catches its own exceptions and
  returns that string rather than raising through FastMCP). Callers need to
  be able to distinguish the two failure modes, so both are raised as
  distinct, catchable exception types rather than one generic error.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from typing import Any

from mcp.client.session import ClientSession

logger = logging.getLogger(__name__)

_VALID_TRANSPORTS = ("streamable-http", "sse")

# Default host/port a locally-run ``mcp_server.py`` binds to under a
# networked transport (see ``start_server.sh``). The client connects to it
# over loopback by default; override via an explicit ``url``.
_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 8000

# FastMCP's own default mount paths per transport (verified against the
# installed SDK — ``mcp/server/fastmcp/server.py``'s ``Settings`` defaults —
# rather than assumed to both be "/mcp").
_DEFAULT_TRANSPORT_PATHS = {
    "streamable-http": "/mcp",
    "sse": "/sse",
}


class MemoryMCPClientError(Exception):
    """Base exception for MemoryMCPClient errors."""

    pass


class MemoryMCPConnectionError(MemoryMCPClientError):
    """Raised when unable to connect to the memory MCP server."""

    pass


class MemoryMCPToolError(MemoryMCPClientError):
    """Raised when a tool call fails or returns an error."""

    pass


def _default_url(transport: str) -> str:
    """Build the default server URL for ``transport`` (loopback, FastMCP's default path)."""
    return f"http://{_DEFAULT_HOST}:{_DEFAULT_PORT}{_DEFAULT_TRANSPORT_PATHS[transport]}"


def _resolve_connection(url: str | None, transport: str | None) -> tuple[str, str]:
    """Resolve the effective (url, transport) pair.

    Precedence for each, independently: explicit constructor argument >
    ``TRADINGAGENTS_MEMORY_MCP_*`` env var (already folded into config by
    ``default_config._apply_env_overrides`` at import time, so reading
    ``get_config()`` here picks it up) > built-in default.

    Raises:
        MemoryMCPConnectionError: If the resolved transport is not one of
            ``streamable-http``/``sse``.
    """
    from tradingagents.dataflows.config import get_config

    config = get_config()

    resolved_transport = transport or config.get("memory_mcp_transport") or "streamable-http"
    if resolved_transport not in _VALID_TRANSPORTS:
        raise MemoryMCPConnectionError(
            f"Invalid memory MCP transport {resolved_transport!r}; "
            f"must be one of {_VALID_TRANSPORTS}"
        )

    resolved_url = url or config.get("memory_mcp_url") or _default_url(resolved_transport)
    return resolved_url, resolved_transport


class MemoryMCPClient:
    """Synchronous client for the memory MCP server's ``memory_*`` tools.

    Connects over a networked transport (``streamable-http`` or ``sse``) and
    provides typed Python methods for each tool. The session is managed via
    context manager or explicit connect/close.

    Example:
        with MemoryMCPClient() as client:
            client.store_decision("trader", "AAPL", "2026-07-01", "BUY")
            stats = client.get_statistics(agent="trader")
    """

    def __init__(self, url: str | None = None, transport: str | None = None):
        """Initialize the client.

        Args:
            url: Full server URL (e.g. ``"http://127.0.0.1:8000/mcp"``).
                Defaults to config/env value, then a URL derived from the
                resolved transport's default path on ``127.0.0.1:8000``.
            transport: ``"streamable-http"`` or ``"sse"``. Defaults to
                config/env value, then ``"streamable-http"``.
        """
        self.url = url
        self.transport = transport
        self._session: ClientSession | None = None
        # See module docstring "Event-loop lifecycle" — mirrors
        # SimulationClient exactly: one loop + one AsyncExitStack kept alive
        # for connect() -> N tool calls -> close(), never recreated per call.
        self._loop: asyncio.AbstractEventLoop | None = None
        self._exit_stack: contextlib.AsyncExitStack | None = None

    def __enter__(self) -> MemoryMCPClient:
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()

    def connect(self) -> None:
        """Connect to the memory MCP server.

        Raises:
            MemoryMCPConnectionError: If unable to reach the server or
                initialize a session (including an invalid transport).
        """
        if self._session is not None:
            return  # Already connected

        if self.url is None or self.transport is None:
            self.url, self.transport = _resolve_connection(self.url, self.transport)
        elif self.transport not in _VALID_TRANSPORTS:
            raise MemoryMCPConnectionError(
                f"Invalid memory MCP transport {self.transport!r}; "
                f"must be one of {_VALID_TRANSPORTS}"
            )

        try:
            # A dedicated event loop is kept alive for the lifetime of the
            # connection — the streams/session yielded below are bound to
            # whichever loop entered their context managers, so every
            # subsequent call (tool calls, close) must reuse this same loop.
            loop = asyncio.new_event_loop()
            try:
                session = loop.run_until_complete(self._async_connect())
            except Exception:
                loop.close()
                raise

            self._loop = loop
            self._session = session

        except Exception as exc:
            raise MemoryMCPConnectionError(
                f"Failed to connect to memory MCP server at {self.url!r} "
                f"via {self.transport!r} transport: {exc}"
            ) from exc

    async def _async_connect(self) -> ClientSession:
        """Async connection logic.

        Enters the transport (``streamable_http_client`` or ``sse_client``)
        and the session as async context managers via an ``AsyncExitStack``
        stored on ``self``, so ``close()`` can unwind them later on the same
        event loop. If anything after the transport is entered fails (e.g.
        ``session.initialize()``), whatever was already entered is unwound
        here before the exception propagates, so ``connect()`` never leaks a
        half-open connection.
        """
        exit_stack = contextlib.AsyncExitStack()
        try:
            if self.transport == "sse":
                from mcp.client.sse import sse_client

                read_stream, write_stream = await exit_stack.enter_async_context(
                    sse_client(self.url)
                )
            else:  # "streamable-http"
                from mcp.client.streamable_http import streamable_http_client

                read_stream, write_stream, _get_session_id = await exit_stack.enter_async_context(
                    streamable_http_client(self.url)
                )

            session = await exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await session.initialize()
        except Exception:
            await exit_stack.aclose()
            raise

        self._exit_stack = exit_stack
        return session

    def close(self) -> None:
        """Close the connection to the memory MCP server.

        Never raises: unwinding the session/HTTP transport can fail for
        reasons outside our control (e.g. the server already went away, or
        an ``__aexit__`` throws), but ``close()`` must always leave the
        client in a clean, disconnected state — the loop closed and every
        reference cleared — regardless (same contract as
        ``SimulationClient.close()``, see commit c42d73c). Any error from the
        unwind is logged as a warning rather than propagated.
        """
        if self._session is None:
            return

        loop = self._loop
        exit_stack = self._exit_stack
        self._session = None
        self._exit_stack = None
        self._loop = None

        try:
            if exit_stack is not None and loop is not None:
                loop.run_until_complete(exit_stack.aclose())
        except Exception as exc:  # noqa: BLE001 - close() must never raise
            logger.warning(
                "Error while closing MemoryMCPClient session: %s", exc, exc_info=True
            )
        finally:
            if loop is not None:
                try:
                    loop.close()
                except Exception as exc:  # noqa: BLE001 - close() must never raise
                    logger.warning(
                        "Error while closing MemoryMCPClient event loop: %s",
                        exc,
                        exc_info=True,
                    )

    def _call_tool_sync(self, tool_name: str, arguments: dict[str, Any] | None = None) -> Any:
        """Call a tool synchronously by running async code.

        Args:
            tool_name: Name of the MCP tool to call.
            arguments: Tool arguments dict.

        Returns:
            The tool result, JSON-decoded when possible.

        Raises:
            MemoryMCPToolError: If the tool call fails, the MCP protocol
                reports an error, or the tool's own payload is an
                ``"ERROR:"``-prefixed string (see module docstring).
            MemoryMCPConnectionError: If not connected.
        """
        if self._session is None:
            raise MemoryMCPConnectionError("Not connected to memory MCP server")
        # connect()/close() always set and clear _session and _loop together,
        # so a live session without a loop is an internal invariant
        # violation. Fail loudly instead of silently falling back to
        # asyncio.run(), which would reintroduce the per-call-new-loop bug
        # (the session's streams are bound to the loop that created them and
        # cannot be driven from a fresh one).
        if self._loop is None:
            raise MemoryMCPConnectionError(
                "Inconsistent client state: session is set but event loop is missing"
            )

        try:
            # Reuse the event loop the session was created on (see
            # connect()) — the session's streams are bound to it and cannot
            # be driven from a different loop.
            result = self._loop.run_until_complete(
                self._session.call_tool(tool_name, arguments or {})
            )

            if result.isError:
                raise MemoryMCPToolError(
                    f"Tool '{tool_name}' returned error: {result.content}"
                )

            parsed: Any = None
            # Prefer structuredContent when available — it's always the
            # correctly-shaped {"result": ...} dict regardless of whether the
            # underlying value is a list, dict, bool, or string. The content
            # list is only for backward compatibility and is unreliable for
            # list-typed returns (empty list → zero blocks, N-item list → N
            # blocks each containing one item — see issue #57 and the FastMCP
            # conversion logic in mcp/server/fastmcp/utilities/func_metadata.py).
            if result.structuredContent is not None:
                parsed = result.structuredContent.get("result")
            elif result.content and hasattr(result.content[0], "text"):
                try:
                    parsed = json.loads(result.content[0].text)
                except (json.JSONDecodeError, AttributeError):
                    parsed = result.content[0].text

            # The memory_* tools never raise an MCP-protocol-level error for
            # their own exceptions — they catch internally and return a
            # string starting with "ERROR:" (see mcp_server.py's module
            # docstring). Surface that the same way as a protocol error.
            if isinstance(parsed, str) and parsed.startswith("ERROR:"):
                raise MemoryMCPToolError(f"Tool '{tool_name}' returned error: {parsed}")

            return parsed

        except MemoryMCPToolError:
            raise
        except Exception as exc:
            raise MemoryMCPToolError(f"Tool call '{tool_name}' failed: {exc}") from exc

    # ─── Public tool methods ────────────────────────────────────────────────

    def store_decision(
        self,
        agent: str,
        ticker: str,
        date: str,
        signal: str,
        confidence: float | None = None,
        key_drivers: Any = None,
        thesis: str | None = None,
        db_path: str | None = None,
    ) -> bool:
        """Record a pending trading decision via ``memory_store_decision``.

        Mirrors ``tradingagents.memory.store_decision``. Idempotent on
        ``(agent, ticker, date)``: a duplicate call is a no-op that leaves
        the original row untouched.

        Args:
            agent: Free-form agent identifier (e.g. ``"trader"``).
            ticker: Ticker symbol, stored exactly as given (no normalization).
            date: Decision date, ``"YYYY-MM-DD"``.
            signal: Directional call (e.g. ``"BUY"``/``"HOLD"``/``"SELL"``).
            confidence: Optional numeric confidence score.
            key_drivers: Optional JSON-serializable list/dict of key drivers.
            thesis: Optional short free-text rationale.
            db_path: Optional override for the memory DB path on the server.

        Returns:
            ``True`` if a new pending row was inserted, ``False`` if a row
            already existed for this key.

        Raises:
            MemoryMCPToolError: If the tool call fails.
            MemoryMCPConnectionError: If not connected.
        """
        result = self._call_tool_sync(
            "memory_store_decision",
            {
                "agent": agent,
                "ticker": ticker,
                "date": date,
                "signal": signal,
                "confidence": confidence,
                "key_drivers": key_drivers,
                "thesis": thesis,
                "db_path": db_path,
            },
        )
        if not isinstance(result, bool):
            raise MemoryMCPToolError(f"memory_store_decision returned non-bool: {result!r}")
        return result

    def resolve_pending(
        self,
        agent: str | None = None,
        ticker: str | None = None,
        db_path: str | None = None,
    ) -> list:
        """Resolve pending decisions via ``memory_resolve_pending``.

        Mirrors ``tradingagents.memory.resolve_pending``.

        Args:
            agent: Optional exact agent id to restrict to.
            ticker: Optional exact (un-normalized) ticker to restrict to.
            db_path: Optional override for the memory DB path on the server.

        Returns:
            A list of ``decisions.id`` values resolved by this call
            (possibly empty).

        Raises:
            MemoryMCPToolError: If the tool call fails.
            MemoryMCPConnectionError: If not connected.
        """
        result = self._call_tool_sync(
            "memory_resolve_pending",
            {"agent": agent, "ticker": ticker, "db_path": db_path},
        )
        if not isinstance(result, list):
            raise MemoryMCPToolError(f"memory_resolve_pending returned non-list: {result!r}")
        return result

    def get_past_context(
        self,
        agent: str,
        ticker: str,
        n_same: int = 5,
        n_cross: int = 3,
        db_path: str | None = None,
    ) -> str:
        """Render past resolved lessons via ``memory_get_past_context``.

        Mirrors ``tradingagents.memory.get_past_context``.

        Args:
            agent: Exact agent identifier to match.
            ticker: Exact ticker to match for the same-ticker section (and to
                exclude from the cross-ticker section).
            n_same: Max number of same-ticker entries to include.
            n_cross: Max number of cross-ticker entries to include.
            db_path: Optional override for the memory DB path on the server.

        Returns:
            A non-empty markdown string, safe to inject into a prompt
            directly.

        Raises:
            MemoryMCPToolError: If the tool call fails.
            MemoryMCPConnectionError: If not connected.
        """
        result = self._call_tool_sync(
            "memory_get_past_context",
            {
                "agent": agent,
                "ticker": ticker,
                "n_same": n_same,
                "n_cross": n_cross,
                "db_path": db_path,
            },
        )
        if not isinstance(result, str):
            raise MemoryMCPToolError(f"memory_get_past_context returned non-str: {result!r}")
        return result

    def get_statistics(
        self,
        agent: str | None = None,
        ticker: str | None = None,
        since: str | None = None,
        db_path: str | None = None,
    ) -> dict:
        """Compute hit-rate/return/calibration statistics via ``memory_get_statistics``.

        Mirrors ``tradingagents.memory.get_statistics``.

        Args:
            agent: Optional exact agent id to restrict to.
            ticker: Optional exact (un-normalized) ticker to restrict to.
            since: Optional inclusive lower bound on decision_date
                (``"YYYY-MM-DD"``).
            db_path: Optional override for the memory DB path on the server.

        Returns:
            A JSON-serializable dict with ``"filters"``, ``"per_agent_ticker"``,
            ``"per_agent"``, and ``"calibration"`` keys — see
            ``tradingagents.memory.stats.get_statistics`` for the exact shape.

        Raises:
            MemoryMCPToolError: If the tool call fails.
            MemoryMCPConnectionError: If not connected.
        """
        result = self._call_tool_sync(
            "memory_get_statistics",
            {"agent": agent, "ticker": ticker, "since": since, "db_path": db_path},
        )
        if not isinstance(result, dict):
            raise MemoryMCPToolError(f"memory_get_statistics returned non-dict: {result!r}")
        return result

    def get_decisions(
        self,
        agent: str,
        ticker: str,
        db_path: str | None = None,
        limit: int | None = None,
        misses_only: bool = False,
    ) -> list:
        """Fetch raw per-decision context rows via ``memory_get_decisions``.

        Mirrors ``tradingagents.memory.query.gather_context_rows``. Only resolved
        decisions (``resolved_at IS NOT NULL``) are ever returned; pending rows are
        always excluded.

        Args:
            agent: Exact agent identifier to match (no normalization).
            ticker: Exact (un-normalized) ticker to match against the stored
                column (no normalization, consistent with ``resolve_pending``
                and ``get_past_context``).
            db_path: Optional override for the memory DB path on the server.
            limit: Maximum number of rows to return, most recent first
                (``decision_date`` DESC, ``id`` DESC). If ``None``, the server
                defaults to ``DEFAULT_CONTEXT_LIMIT`` (10).
            misses_only: If True, only rows scored "incorrect" are returned —
                useful for focusing on failures when investigating a
                divergent-performance pattern.

        Returns:
            A JSON-serializable list of dicts, one per resolved decision:
            ``{"decision_date", "signal", "confidence", "key_drivers" (parsed
            JSON object/list or None), "thesis", "lesson", "forward_return",
            "correct"}``. ``correct`` is ``None`` if the signal doesn't
            normalize to BUY/HOLD/SELL, and ``True``/``False`` otherwise.

        Raises:
            MemoryMCPToolError: If the tool call fails.
            MemoryMCPConnectionError: If not connected.
        """
        result = self._call_tool_sync(
            "memory_get_decisions",
            {
                "agent": agent,
                "ticker": ticker,
                "db_path": db_path,
                "limit": limit,
                "misses_only": misses_only,
            },
        )
        if not isinstance(result, list):
            raise MemoryMCPToolError(f"memory_get_decisions returned non-list: {result!r}")
        return result
