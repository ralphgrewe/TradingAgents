"""MCP client for McpTradingSimulation.

Provides synchronous typed Python methods wrapping the trading simulator's MCP
tools (list_depots, create_depot, get_portfolio, get_quote, place_order,
get_trades).

Usage:
    from tradingagents.simulation import SimulationClient

    with SimulationClient() as client:
        portfolio = client.get_portfolio("my-depot")
        quote = client.get_quote("AAPL")
        order = client.place_order("AAPL", "buy", 10, "my-depot")

Design choices (mirrors ``tradingagents/memory/mcp_client.py``'s
``MemoryMCPClient`` — connect/close/context-manager shape and event-loop
lifecycle — adapted from ``mcp.client.streamable_http``/``mcp.client.sse`` to
``mcp.client.stdio``, which manages a server subprocess rather than an HTTP
connection; see that module's docstring section "Event-loop lifecycle" for
the original rationale and commit 8d60230/issue #58 for the bug it fixes):

- **Event-loop lifecycle**: a single event loop is created in ``connect()``
  and kept alive for the client's whole lifetime, reused for every
  subsequent tool call, and closed in ``close()`` — but entering and exiting
  the transport's ``AsyncExitStack`` (stdio streams + ``ClientSession``) is
  **not** split across two independent ``loop.run_until_complete()`` calls.
  Each ``run_until_complete()`` wraps its coroutine in a brand-new
  ``asyncio.Task`` even on the same loop, and the stdio transport
  (``mcp.client.stdio.stdio_client``) opens an anyio ``TaskGroup`` internally
  whose cancel scope must be entered *and* exited by the same ``Task`` —
  entering it in one ``run_until_complete()`` call (``connect()``) and
  exiting it in another (``close()``) raises ``RuntimeError: Attempted to
  exit cancel scope in a different task than it was entered in`` (issue #63,
  the same bug as #58).

  Instead, ``connect()`` starts one persistent coroutine as a single
  ``asyncio.Task`` (``self._lifecycle_task``, via ``loop.create_task()``)
  that owns the entire session lifetime: it enters the transport/session
  into an ``AsyncExitStack`` (via ``_enter_session()``), signals readiness
  through a dedicated event, then blocks on ``self._shutdown_event.wait()``
  until told to unwind, and only then calls ``exit_stack.aclose()`` — all
  inside that one Task. ``connect()`` drives the loop just far enough to
  observe readiness (or failure) and capture the session; ``close()`` sets
  ``self._shutdown_event`` and drives the loop with
  ``run_until_complete(self._lifecycle_task)`` until that *same* task
  finishes unwinding — which also terminates the server subprocess, since
  ``stdio_client``'s ``__aexit__`` tears it down — rather than closing the
  stack from a task of its own. Tool calls (``_call_tool_sync``) still run
  each ``session.call_tool()`` in its own ad hoc ``run_until_complete()``
  task — that's fine, since they only drive the already-open streams/session
  and never touch the transport's task group directly; only the stack's
  entry/exit needed to move into the persistent task. ``close()`` never
  raises (same contract and rationale as before, see commit c42d73c) — any
  error unwinding the exit stack or closing the loop is logged as a warning
  rather than propagated, and internal state is always cleared.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from pathlib import Path
from typing import Any

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

logger = logging.getLogger(__name__)


class SimulationClientError(Exception):
    """Base exception for SimulationClient errors."""

    pass


class SimulationConnectionError(SimulationClientError):
    """Raised when unable to connect to the MCP server."""

    pass


class SimulationToolError(SimulationClientError):
    """Raised when a tool call fails or returns an error."""

    pass


class SimulationClient:
    """Synchronous client for McpTradingSimulation MCP server.

    Connects to the simulator via stdio and provides typed Python methods for
    each tool. The session is managed via context manager or explicit connect/close.

    Example:
        with SimulationClient() as client:
            depots = client.list_depots()
            portfolio = client.get_portfolio("default")
    """

    def __init__(self, server_command: str | None = None, server_args: list[str] | None = None):
        """Initialize the client.

        Args:
            server_command: Path to the MCP server executable (python interpreter).
                Defaults to config value or ../McpTradingSimulation/venv/bin/python.
            server_args: Arguments to pass to the server (script path, etc).
                Defaults to config value or [path-to-mcp_server.py].
        """
        self.server_command = server_command
        self.server_args = server_args
        self._session: ClientSession | None = None
        # See module docstring "Event-loop lifecycle" — one loop plus one
        # persistent lifecycle Task (which owns the AsyncExitStack
        # internally) are kept alive for connect() -> N tool calls ->
        # close(), never recreated per call. The exit stack itself lives in
        # the lifecycle task's coroutine frame, not as an attribute here —
        # only the same Task that entered it may exit it (issue #63).
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lifecycle_task: asyncio.Task | None = None
        self._shutdown_event: asyncio.Event | None = None

    def __enter__(self) -> SimulationClient:
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()

    def connect(self) -> None:
        """Connect to the MCP server.

        Starts a single persistent lifecycle Task (see module docstring
        "Event-loop lifecycle") that owns entering and — much later, in
        ``close()`` — exiting the stdio transport/session ``AsyncExitStack``,
        so the anyio cancel scope opened by the stdio transport is entered
        and exited by the same ``asyncio.Task`` throughout (issue #63).

        Raises:
            SimulationConnectionError: If unable to start the server or
                initialize a session.
        """
        if self._session is not None:
            return  # Already connected

        try:
            # Get server command and args from config if not provided
            if self.server_command is None or self.server_args is None:
                self.server_command, self.server_args = _get_server_config()

            # Create stdio transport
            server_params = StdioServerParameters(
                command=self.server_command,
                args=self.server_args,
                env=None,  # Inherit from parent
            )

            # A dedicated event loop is kept alive for the lifetime of the
            # connection — the streams/session entered below are bound to
            # whichever loop/Task entered their context managers, so every
            # subsequent call (tool calls, close) must reuse this same loop.
            loop = asyncio.new_event_loop()
            ready_event = asyncio.Event()
            shutdown_event = asyncio.Event()
            state: dict[str, Any] = {}

            async def _lifecycle() -> None:
                """Own the transport/session AsyncExitStack for its entire
                life: enter it, publish the session (or the failure) and
                signal readiness, then wait to be told to shut down before
                exiting the stack — all within this one Task, so entry and
                exit are never split across different Tasks.
                """
                exit_stack = contextlib.AsyncExitStack()
                try:
                    session = await self._enter_session(server_params, exit_stack)
                except Exception as exc:
                    state["error"] = exc
                    ready_event.set()
                    await exit_stack.aclose()
                    return

                state["session"] = session
                ready_event.set()
                try:
                    await shutdown_event.wait()
                finally:
                    await exit_stack.aclose()

            task = loop.create_task(_lifecycle())
            try:
                loop.run_until_complete(ready_event.wait())
            except Exception:
                loop.close()
                raise

            if "error" in state:
                # Drive the lifecycle task to completion (it still needs to
                # unwind whatever it already entered, including terminating
                # the subprocess) before surfacing the failure, so connect()
                # never leaks a half-open subprocess.
                with contextlib.suppress(Exception):
                    loop.run_until_complete(task)
                loop.close()
                raise state["error"]

            self._loop = loop
            self._session = state["session"]
            self._lifecycle_task = task
            self._shutdown_event = shutdown_event

        except Exception as exc:
            raise SimulationConnectionError(
                f"Failed to connect to McpTradingSimulation server: {exc}"
            ) from exc

    async def _enter_session(
        self, server_params: StdioServerParameters, exit_stack: contextlib.AsyncExitStack
    ) -> ClientSession:
        """Enter the stdio transport and ``ClientSession`` into ``exit_stack``
        and initialize the session.

        ``exit_stack`` is owned by the caller (the persistent lifecycle Task
        started in ``connect()``), not by this method — this method does not
        unwind it on failure; the caller is responsible for calling
        ``exit_stack.aclose()`` (from the same Task) if this raises, so that
        whatever was already entered (e.g. the subprocess/transport, if
        ``session.initialize()`` is what failed) is unwound rather than
        leaked.
        """
        read_stream, write_stream = await exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        session = await exit_stack.enter_async_context(ClientSession(read_stream, write_stream))
        await session.initialize()
        return session

    def close(self) -> None:
        """Close the connection to the MCP server.

        Signals the persistent lifecycle Task (started in ``connect()``) to
        shut down via ``self._shutdown_event``, then drives the event loop
        with ``run_until_complete(task)`` until that *same* Task finishes
        unwinding its ``AsyncExitStack`` — which also terminates the server
        subprocess — never by calling ``exit_stack.aclose()`` from a task of
        its own, which is what caused the cross-task cancel-scope error this
        lifecycle fixes (issue #63).

        Never raises: unwinding the session/subprocess can fail for reasons
        outside our control (e.g. the subprocess already died, or an
        ``__aexit__`` throws), but ``close()`` must always leave the client in
        a clean, disconnected state — the loop closed and every reference
        cleared — so the acceptance criterion "``close()`` cleanly shuts down
        the session and subprocess without raising" holds regardless. Any
        error from the unwind is logged as a warning rather than propagated.
        """
        if self._session is None:
            return

        loop = self._loop
        task = self._lifecycle_task
        shutdown_event = self._shutdown_event
        self._session = None
        self._lifecycle_task = None
        self._shutdown_event = None
        self._loop = None

        try:
            if shutdown_event is not None:
                shutdown_event.set()
            if task is not None and loop is not None:
                loop.run_until_complete(task)
        except Exception as exc:  # noqa: BLE001 - close() must never raise
            logger.warning(
                "Error while closing SimulationClient session: %s", exc, exc_info=True
            )
        finally:
            if loop is not None:
                try:
                    loop.close()
                except Exception as exc:  # noqa: BLE001 - close() must never raise
                    logger.warning(
                        "Error while closing SimulationClient event loop: %s",
                        exc,
                        exc_info=True,
                    )

    def _call_tool_sync(self, tool_name: str, arguments: dict[str, Any] | None = None) -> Any:
        """Call a tool synchronously by running async code.

        Args:
            tool_name: Name of the MCP tool to call.
            arguments: Tool arguments dict.

        Returns:
            Tool result as dict/list/etc.

        Raises:
            SimulationToolError: If the tool call fails or returns an error.
            SimulationConnectionError: If not connected.
        """
        if self._session is None:
            raise SimulationConnectionError("Not connected to simulation server")
        # connect()/close() always set and clear _session and _loop together, so
        # a live session without a loop is an internal invariant violation. Fail
        # loudly instead of silently falling back to asyncio.run(), which would
        # reintroduce the per-call-new-loop bug (the session's streams are bound
        # to the loop that created them and cannot be driven from a fresh one).
        if self._loop is None:
            raise SimulationConnectionError(
                "Inconsistent client state: session is set but event loop is missing"
            )

        try:
            # Reuse the event loop the session was created on (see connect()) —
            # the session's streams are bound to it and cannot be driven from a
            # different loop.
            result = self._loop.run_until_complete(
                self._session.call_tool(tool_name, arguments or {})
            )

            # Check if there's an error
            if result.isError:
                raise SimulationToolError(
                    f"Tool '{tool_name}' returned error: {result.content}"
                )

            # Parse result following the reference implementation in mcp_client.py.
            # Prefer structuredContent when available — it preserves the original
            # type including empty and multi-element lists. The content list is
            # only for backward compatibility and is unreliable for list-typed
            # returns (empty list → zero blocks, N-item list → N blocks each
            # containing one item — see issue #62).
            import json

            parsed: Any = None
            if result.structuredContent is not None:
                parsed = result.structuredContent.get("result")
            elif result.content:
                # Parse all content blocks as JSON; if there is more than one
                # block, return them as a list.
                parsed_items = []
                for block in result.content:
                    if hasattr(block, "text"):
                        try:
                            item = json.loads(block.text)
                            parsed_items.append(item)
                        except json.JSONDecodeError:
                            # If not JSON, return the text as-is
                            parsed_items.append(block.text)

                # If single block, return it unwrapped; if multiple blocks,
                # return as list; if zero blocks, return None.
                if len(parsed_items) == 1:
                    parsed = parsed_items[0]
                elif len(parsed_items) > 1:
                    parsed = parsed_items
                # else: zero blocks → parsed stays None

            return parsed

        except SimulationToolError:
            raise
        except Exception as exc:
            raise SimulationToolError(f"Tool call '{tool_name}' failed: {exc}") from exc

    # ─── Public tool methods ────────────────────────────────────────────────

    def list_depots(self) -> list:
        """Return all depots that exist on disk.

        Returns:
            List of depot dicts with 'id', 'db_path', and 'size_bytes'.

        Raises:
            SimulationToolError: If the tool call fails.
        """
        result = self._call_tool_sync("list_depots")
        if not isinstance(result, list):
            raise SimulationToolError(f"list_depots returned non-list: {result}")
        return result

    def create_depot(self, depot_id: str, initial_cash: float = 10_000.0) -> dict:
        """Create a new isolated trading depot.

        Args:
            depot_id: Unique identifier for the depot (alphanumeric + hyphens/underscores).
            initial_cash: Starting cash balance in USD (default 10,000).

        Returns:
            Dict with 'status', 'depot_id', 'initial_cash', and 'db_path' keys.
            If creation fails, returns dict with 'error' key.

        Raises:
            SimulationToolError: If the tool call fails.
        """
        result = self._call_tool_sync(
            "create_depot",
            {"depot_id": depot_id, "initial_cash": initial_cash},
        )
        if not isinstance(result, dict):
            raise SimulationToolError(f"create_depot returned non-dict: {result}")
        return result

    def get_portfolio(self, depot_id: str = "default") -> dict:
        """Return the current portfolio snapshot.

        Includes cash balance, all open positions with live prices, unrealized
        P&L per position, and total equity.

        Args:
            depot_id: Which depot to query (default "default").

        Returns:
            Dict with 'depot_id', 'cash', 'currency', 'positions',
            'positions_value', and 'total_equity' keys. On error, dict with 'error' key.

        Raises:
            SimulationToolError: If the tool call fails.
        """
        result = self._call_tool_sync("get_portfolio", {"depot_id": depot_id})
        if not isinstance(result, dict):
            raise SimulationToolError(f"get_portfolio returned non-dict: {result}")
        return result

    def get_quote(self, symbol: str) -> dict:
        """Fetch the latest price quote for a symbol.

        Returns price, timestamp, and freshness (whether ≤ 15 min old).
        Quotes are cached to avoid rate limits.

        Args:
            symbol: Ticker symbol (e.g. "AAPL").

        Returns:
            Dict with 'symbol', 'price', 'ts' (ISO-8601), and 'is_fresh' keys.
            On error, dict with 'error' key.

        Raises:
            SimulationToolError: If the tool call fails.
        """
        result = self._call_tool_sync("get_quote", {"symbol": symbol})
        if not isinstance(result, dict):
            raise SimulationToolError(f"get_quote returned non-dict: {result}")
        return result

    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        depot_id: str = "default",
    ) -> dict:
        """Place a market order.

        Args:
            symbol: Ticker symbol (e.g. "AAPL").
            side: "buy" or "sell".
            quantity: Number of shares (positive integer).
            depot_id: Which depot to trade in (default "default").

        Returns:
            Dict with 'status' ("executed", "pending", or "rejected"), 'message',
            'depot_id', 'trade_id', 'pending_order_id', 'symbol', 'side',
            'quantity', 'price', and 'fee'.

        Raises:
            SimulationToolError: If the tool call fails.
        """
        result = self._call_tool_sync(
            "place_order",
            {
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "depot_id": depot_id,
            },
        )
        if not isinstance(result, dict):
            raise SimulationToolError(f"place_order returned non-dict: {result}")
        return result

    def get_trades(
        self,
        limit: int = 50,
        symbol: str = "",
        depot_id: str = "default",
    ) -> list:
        """Return recent trades, newest first.

        Args:
            limit: Maximum number of trades to return (default 50).
            symbol: If provided, filter to this ticker only.
            depot_id: Which depot to query (default "default").

        Returns:
            List of trade dicts with 'id', 'ts', 'symbol', 'side', 'quantity',
            'price', 'fee', 'gross', 'net', 'cash_after'.

        Raises:
            SimulationToolError: If the tool call fails.
        """
        result = self._call_tool_sync(
            "get_trades",
            {
                "limit": limit,
                "symbol": symbol,
                "depot_id": depot_id,
            },
        )
        if not isinstance(result, list):
            raise SimulationToolError(f"get_trades returned non-list: {result}")
        return result


def _get_server_config() -> tuple[str, list[str]]:
    """Get the MCP server command and args from config.

    Returns:
        Tuple of (command, args) where command is the Python interpreter path
        and args is [path_to_mcp_server.py].

    Raises:
        SimulationConnectionError: If config is missing or invalid.
    """
    from tradingagents.dataflows.config import get_config

    config = get_config()

    # Try to get from config first
    server_command = config.get("simulation_server_command")
    server_args = config.get("simulation_server_args")

    if server_command and server_args:
        return server_command, server_args

    # Fallback: detect sibling checkout and default to its venv/bin/python
    this_repo = Path(__file__).parent.parent.parent  # /repo/tradingagents/simulation/
    sibling = this_repo.parent / "McpTradingSimulation"

    if not sibling.exists():
        raise SimulationConnectionError(
            f"McpTradingSimulation checkout not found at {sibling}. "
            "Set TRADINGAGENTS_SIMULATION_SERVER_COMMAND and "
            "TRADINGAGENTS_SIMULATION_SERVER_ARGS in .env or via config."
        )

    # Check for venv in sibling
    venv_python = sibling / "venv" / "bin" / "python"
    mcp_server = sibling / "mcp_server.py"

    if not venv_python.exists():
        raise SimulationConnectionError(
            f"McpTradingSimulation venv not found at {venv_python}"
        )
    if not mcp_server.exists():
        raise SimulationConnectionError(
            f"McpTradingSimulation mcp_server.py not found at {mcp_server}"
        )

    return str(venv_python), [str(mcp_server)]
