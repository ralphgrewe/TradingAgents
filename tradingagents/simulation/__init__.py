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
"""

from __future__ import annotations

import asyncio
import contextlib
import subprocess
import sys
from pathlib import Path
from typing import Any

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


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
        # The stdio transport and ClientSession are async context managers whose
        # yielded streams are bound to the event loop that entered them. A single
        # loop is kept alive for the client's whole lifetime (connect -> N tool
        # calls -> close) instead of spinning up a fresh one per call, and the
        # AsyncExitStack that entered both context managers is kept around so
        # close() can unwind them on that same loop.
        self._loop: asyncio.AbstractEventLoop | None = None
        self._exit_stack: contextlib.AsyncExitStack | None = None

    def __enter__(self) -> SimulationClient:
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()

    def connect(self) -> None:
        """Connect to the MCP server.

        Raises:
            SimulationConnectionError: If unable to start the server or initialize session.
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

            # Create a dedicated event loop and keep it alive for the lifetime of
            # the connection — the streams/session yielded below are bound to
            # whichever loop entered their context managers, so every subsequent
            # call (tool calls, close) must reuse this same loop.
            loop = asyncio.new_event_loop()
            try:
                session = loop.run_until_complete(self._async_connect(server_params))
            except Exception:
                loop.close()
                raise

            self._loop = loop
            self._session = session

        except Exception as exc:
            raise SimulationConnectionError(
                f"Failed to connect to McpTradingSimulation server: {exc}"
            ) from exc

    async def _async_connect(self, server_params: StdioServerParameters) -> ClientSession:
        """Async connection logic.

        Enters the stdio transport and the session as async context managers via
        an `AsyncExitStack` stored on `self`, so `close()` can unwind them later
        on the same event loop. If anything after the transport is entered fails
        (e.g. `session.initialize()`), whatever was already entered is unwound
        here before the exception propagates, so `connect()` never leaks a
        half-open subprocess.
        """
        exit_stack = contextlib.AsyncExitStack()
        try:
            read_stream, write_stream = await exit_stack.enter_async_context(
                stdio_client(server_params)
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
        """Close the connection to the MCP server."""
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
        finally:
            if loop is not None:
                loop.close()

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

        try:
            # Reuse the event loop the session was created on (see connect()) —
            # the session's streams are bound to it and cannot be driven from a
            # different loop.
            run = self._loop.run_until_complete if self._loop is not None else asyncio.run
            result = run(self._session.call_tool(tool_name, arguments or {}))

            # Check if result has content (success)
            if result.content:
                # result.content is a list of TextContent/ImageContent/etc
                if result.content and hasattr(result.content[0], "text"):
                    # Try to parse JSON response
                    import json

                    try:
                        return json.loads(result.content[0].text)
                    except (json.JSONDecodeError, AttributeError):
                        # Return as-is if not JSON
                        return result.content[0].text if result.content else None

            # Check if there's an error
            if result.isError:
                raise SimulationToolError(
                    f"Tool '{tool_name}' returned error: {result.content}"
                )

            return None

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
