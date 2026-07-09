"""Unit tests for SimulationClient (MCP client for McpTradingSimulation).

Tests run with mocked MCP sessions — no connection to real simulator.
"""

import asyncio
import contextlib
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp.client.stdio import StdioServerParameters

from tradingagents.simulation import (
    SimulationClient,
    SimulationConnectionError,
    SimulationToolError,
)


@pytest.fixture
def mock_client_session():
    """Create a mock ClientSession for testing."""
    session = AsyncMock()
    return session


@pytest.fixture
def mock_call_tool_result():
    """Create a mock tool result object."""
    result = MagicMock()
    result.isError = False
    result.content = []
    return result


class TestSimulationClientConnection:
    """Tests for connection/initialization."""

    def test_init_no_args(self):
        """Client can be initialized without arguments."""
        client = SimulationClient()
        assert client.server_command is None
        assert client.server_args is None
        assert client._session is None

    def test_context_manager_connect_disconnect(self, mock_client_session):
        """Context manager connects, keeps a persistent loop alive, and tears it
        down cleanly on exit (the loop/session lifecycle this issue fixes)."""
        with patch(
            "tradingagents.simulation._get_server_config"
        ) as mock_config, patch.object(
            SimulationClient, "_async_connect", new_callable=AsyncMock
        ) as mock_async_connect:
            mock_config.return_value = ("/path/to/python", ["/path/to/mcp_server.py"])
            mock_async_connect.return_value = mock_client_session

            with SimulationClient() as client:
                assert client._session is mock_client_session
                assert client._loop is not None
                assert not client._loop.is_closed()
                loop = client._loop

            # After exiting, session/exit-stack/loop are all cleared and the loop
            # itself is closed (not leaked, not reused across connections).
            assert client._session is None
            assert client._loop is None
            assert client._exit_stack is None
            assert loop.is_closed()

    def test_async_connect_uses_real_stdio_and_session_wiring(self):
        """_async_connect must call ClientSession(read_stream, write_stream) — the
        exact bug from issue #46 (`ClientSession(transport)` with a single
        positional arg) — and enter both context managers via the exit stack."""
        read_stream, write_stream = object(), object()

        @contextlib.asynccontextmanager
        async def fake_stdio_client(server_params):
            yield (read_stream, write_stream)

        fake_session = AsyncMock()
        fake_session.__aenter__.return_value = fake_session
        fake_session.__aexit__.return_value = None

        with patch(
            "tradingagents.simulation.stdio_client", side_effect=fake_stdio_client
        ), patch(
            "tradingagents.simulation.ClientSession", return_value=fake_session
        ) as mock_session_cls:
            client = SimulationClient()
            loop = asyncio.new_event_loop()
            try:
                session = loop.run_until_complete(
                    client._async_connect(
                        StdioServerParameters(command="python", args=["server.py"])
                    )
                )
            finally:
                loop.close()

            mock_session_cls.assert_called_once_with(read_stream, write_stream)
            assert session is fake_session
            fake_session.initialize.assert_awaited_once()
            assert client._exit_stack is not None

    def test_connect_unwinds_partial_state_on_initialize_failure(self):
        """If session.initialize() fails, whatever was already entered (the stdio
        transport) must be unwound — no leaked subprocess/session."""
        exited = []

        @contextlib.asynccontextmanager
        async def fake_stdio_client(server_params):
            try:
                yield (object(), object())
            finally:
                exited.append("stdio")

        fake_session = AsyncMock()
        fake_session.__aenter__.return_value = fake_session
        fake_session.__aexit__.return_value = None
        fake_session.initialize.side_effect = RuntimeError("boom")

        with patch(
            "tradingagents.simulation._get_server_config",
            return_value=("/path/to/python", ["/path/to/mcp_server.py"]),
        ), patch(
            "tradingagents.simulation.stdio_client", side_effect=fake_stdio_client
        ), patch("tradingagents.simulation.ClientSession", return_value=fake_session):
            client = SimulationClient()
            with pytest.raises(SimulationConnectionError):
                client.connect()

            assert exited == ["stdio"]
            fake_session.__aexit__.assert_awaited_once()
            assert client._session is None
            assert client._loop is None

    def test_close_unwinds_real_exit_stack(self):
        """close() must actually drive the AsyncExitStack unwind end-to-end —
        i.e. the entered stdio/session context managers' __aexit__ run — not just
        close a loop. (test_context_manager_connect_disconnect mocks
        _async_connect wholesale, so _exit_stack stays None there and this path
        is never exercised.)"""
        exited = []

        @contextlib.asynccontextmanager
        async def fake_stdio_client(server_params):
            try:
                yield (object(), object())
            finally:
                exited.append("stdio")

        fake_session = AsyncMock()
        fake_session.__aenter__.return_value = fake_session
        fake_session.__aexit__.return_value = None

        with patch(
            "tradingagents.simulation._get_server_config",
            return_value=("/path/to/python", ["/path/to/mcp_server.py"]),
        ), patch(
            "tradingagents.simulation.stdio_client", side_effect=fake_stdio_client
        ), patch("tradingagents.simulation.ClientSession", return_value=fake_session):
            client = SimulationClient()
            client.connect()
            assert client._exit_stack is not None
            loop = client._loop

            client.close()  # must not raise

            # The exit stack actually unwound: session and stdio __aexit__ ran.
            fake_session.__aexit__.assert_awaited_once()
            assert exited == ["stdio"]
            # Internal state cleared and loop closed.
            assert client._session is None
            assert client._exit_stack is None
            assert client._loop is None
            assert loop.is_closed()

    def test_close_does_not_raise_when_unwind_fails(self):
        """Regression test for issue #46: if the exit stack raises while
        unwinding (e.g. a context manager's __aexit__ throws, or the subprocess
        already died), close() must still not raise and must still leave the
        client cleanly disconnected with the loop closed."""
        exited = []

        @contextlib.asynccontextmanager
        async def fake_stdio_client(server_params):
            try:
                yield (object(), object())
            finally:
                exited.append("stdio")

        fake_session = AsyncMock()
        fake_session.__aenter__.return_value = fake_session
        fake_session.__aexit__.side_effect = RuntimeError("aexit boom")

        with patch(
            "tradingagents.simulation._get_server_config",
            return_value=("/path/to/python", ["/path/to/mcp_server.py"]),
        ), patch(
            "tradingagents.simulation.stdio_client", side_effect=fake_stdio_client
        ), patch("tradingagents.simulation.ClientSession", return_value=fake_session):
            client = SimulationClient()
            client.connect()
            loop = client._loop

            # Must swallow the __aexit__ error rather than propagate it.
            client.close()

            fake_session.__aexit__.assert_awaited_once()
            assert client._session is None
            assert client._exit_stack is None
            assert client._loop is None
            assert loop.is_closed()

    def test_connection_error_propagates(self):
        """Connection errors are converted to SimulationConnectionError."""
        with patch("tradingagents.simulation._get_server_config") as mock_config:
            mock_config.side_effect = Exception("Config error")
            client = SimulationClient()
            with pytest.raises(SimulationConnectionError) as exc_info:
                client.connect()
            assert "Failed to connect" in str(exc_info.value)

    def test_no_connect_if_already_connected(self, mock_client_session):
        """Calling connect() twice is a no-op."""
        client = SimulationClient()
        client._session = mock_client_session
        with patch("tradingagents.simulation._get_server_config") as mock_config:
            client.connect()  # Should not call _get_server_config
            mock_config.assert_not_called()

    def test_close_when_not_connected(self):
        """Calling close() when not connected is safe."""
        client = SimulationClient()
        client.close()  # Should not raise
        assert client._session is None


class TestSimulationClientToolCalls:
    """Tests for tool call methods."""

    @pytest.fixture
    def connected_client(self, mock_client_session):
        """Create a connected client for testing."""
        client = SimulationClient()
        client._session = mock_client_session
        client._loop = asyncio.new_event_loop()
        yield client
        client._loop.close()

    def test_call_tool_sync_not_connected(self):
        """Calling a tool when not connected raises error."""
        client = SimulationClient()
        with pytest.raises(SimulationConnectionError):
            client._call_tool_sync("list_depots")

    def test_call_tool_sync_raises_when_loop_missing(self, mock_client_session):
        """A live session without an event loop is an invariant violation
        (connect/close always set/clear both together). _call_tool_sync must fail
        loudly rather than silently fall back to asyncio.run() and reintroduce
        the per-call-new-loop bug."""
        client = SimulationClient()
        client._session = mock_client_session
        client._loop = None
        with pytest.raises(SimulationConnectionError):
            client._call_tool_sync("list_depots")

    def test_call_tool_sync_success_json(self, connected_client, mock_call_tool_result):
        """Successful tool call returns parsed JSON."""
        test_data = {"id": "test", "value": 123}
        content = MagicMock()
        content.text = json.dumps(test_data)
        mock_call_tool_result.content = [content]

        connected_client._session.call_tool = AsyncMock(
            return_value=mock_call_tool_result
        )

        result = connected_client._call_tool_sync("test_tool", {"arg": "value"})
        assert result == test_data

    def test_call_tool_sync_error(self, connected_client):
        """Tool error is converted to SimulationToolError."""
        mock_result = MagicMock()
        mock_result.isError = True
        mock_result.content = "Tool failed"

        connected_client._session.call_tool = AsyncMock(return_value=mock_result)

        with pytest.raises(SimulationToolError) as exc_info:
            connected_client._call_tool_sync("bad_tool")
        assert "returned error" in str(exc_info.value)

    def test_call_tool_sync_exception(self, connected_client):
        """Tool call exception is converted to SimulationToolError."""
        connected_client._session.call_tool = AsyncMock(
            side_effect=Exception("Network error")
        )

        with pytest.raises(SimulationToolError) as exc_info:
            connected_client._call_tool_sync("test_tool")
        assert "failed" in str(exc_info.value).lower()


class TestListDepots:
    """Tests for list_depots method."""

    @pytest.fixture
    def connected_client(self, mock_client_session):
        client = SimulationClient()
        client._session = mock_client_session
        client._loop = asyncio.new_event_loop()
        yield client
        client._loop.close()

    def test_list_depots_success(self, connected_client, mock_call_tool_result):
        """list_depots returns list of depot dicts."""
        depots = [
            {"id": "default", "db_path": "/data/simulation.db", "size_bytes": 1024},
            {"id": "test-1", "db_path": "/data/depots/test-1.db", "size_bytes": 512},
        ]
        content = MagicMock()
        content.text = json.dumps(depots)
        mock_call_tool_result.content = [content]

        connected_client._session.call_tool = AsyncMock(
            return_value=mock_call_tool_result
        )

        result = connected_client.list_depots()
        assert result == depots
        assert len(result) == 2

    def test_list_depots_validates_list_type(self, connected_client):
        """list_depots raises error if result is not a list."""
        content = MagicMock()
        content.text = json.dumps({"error": "not a list"})
        mock_result = MagicMock()
        mock_result.content = [content]
        mock_result.isError = False

        connected_client._session.call_tool = AsyncMock(return_value=mock_result)

        with pytest.raises(SimulationToolError) as exc_info:
            connected_client.list_depots()
        assert "non-list" in str(exc_info.value)


class TestCreateDepot:
    """Tests for create_depot method."""

    @pytest.fixture
    def connected_client(self, mock_client_session):
        client = SimulationClient()
        client._session = mock_client_session
        client._loop = asyncio.new_event_loop()
        yield client
        client._loop.close()

    def test_create_depot_success(self, connected_client, mock_call_tool_result):
        """create_depot returns creation result dict."""
        response = {
            "status": "created",
            "depot_id": "agent-alpha",
            "initial_cash": 50000.0,
            "db_path": "/data/depots/agent-alpha.db",
        }
        content = MagicMock()
        content.text = json.dumps(response)
        mock_call_tool_result.content = [content]

        connected_client._session.call_tool = AsyncMock(
            return_value=mock_call_tool_result
        )

        result = connected_client.create_depot("agent-alpha", 50000.0)
        assert result["status"] == "created"
        assert result["depot_id"] == "agent-alpha"

    def test_create_depot_error_response(self, connected_client, mock_call_tool_result):
        """create_depot returns error dict if depot already exists."""
        response = {"error": "Depot 'agent-alpha' already exists"}
        content = MagicMock()
        content.text = json.dumps(response)
        mock_call_tool_result.content = [content]

        connected_client._session.call_tool = AsyncMock(
            return_value=mock_call_tool_result
        )

        result = connected_client.create_depot("agent-alpha", 50000.0)
        assert "error" in result

    def test_create_depot_args(self, connected_client, mock_call_tool_result):
        """create_depot passes correct arguments to tool."""
        response = {"status": "created"}
        content = MagicMock()
        content.text = json.dumps(response)
        mock_call_tool_result.content = [content]

        connected_client._session.call_tool = AsyncMock(
            return_value=mock_call_tool_result
        )

        connected_client.create_depot("my-depot", 25000.0)

        # Verify call_tool was called with correct args
        call_args = connected_client._session.call_tool.call_args
        assert call_args[0][0] == "create_depot"  # tool name
        assert call_args[0][1]["depot_id"] == "my-depot"
        assert call_args[0][1]["initial_cash"] == 25000.0


class TestGetPortfolio:
    """Tests for get_portfolio method."""

    @pytest.fixture
    def connected_client(self, mock_client_session):
        client = SimulationClient()
        client._session = mock_client_session
        client._loop = asyncio.new_event_loop()
        yield client
        client._loop.close()

    def test_get_portfolio_success(self, connected_client, mock_call_tool_result):
        """get_portfolio returns portfolio dict."""
        portfolio = {
            "depot_id": "default",
            "cash": 95000.0,
            "currency": "USD",
            "positions": {"AAPL": {"shares": 10, "price": 150.0, "market_value": 1500.0}},
            "positions_value": 1500.0,
            "total_equity": 96500.0,
        }
        content = MagicMock()
        content.text = json.dumps(portfolio)
        mock_call_tool_result.content = [content]

        connected_client._session.call_tool = AsyncMock(
            return_value=mock_call_tool_result
        )

        result = connected_client.get_portfolio("default")
        assert result["total_equity"] == 96500.0
        assert "AAPL" in result["positions"]

    def test_get_portfolio_default_depot(self, connected_client, mock_call_tool_result):
        """get_portfolio defaults to 'default' depot."""
        portfolio = {"depot_id": "default", "cash": 100000.0}
        content = MagicMock()
        content.text = json.dumps(portfolio)
        mock_call_tool_result.content = [content]

        connected_client._session.call_tool = AsyncMock(
            return_value=mock_call_tool_result
        )

        connected_client.get_portfolio()  # No depot_id argument

        call_args = connected_client._session.call_tool.call_args
        assert call_args[0][1]["depot_id"] == "default"


class TestGetQuote:
    """Tests for get_quote method."""

    @pytest.fixture
    def connected_client(self, mock_client_session):
        client = SimulationClient()
        client._session = mock_client_session
        client._loop = asyncio.new_event_loop()
        yield client
        client._loop.close()

    def test_get_quote_success(self, connected_client, mock_call_tool_result):
        """get_quote returns quote dict."""
        quote = {
            "symbol": "AAPL",
            "price": 150.50,
            "ts": "2025-07-05T15:30:00Z",
            "is_fresh": True,
        }
        content = MagicMock()
        content.text = json.dumps(quote)
        mock_call_tool_result.content = [content]

        connected_client._session.call_tool = AsyncMock(
            return_value=mock_call_tool_result
        )

        result = connected_client.get_quote("AAPL")
        assert result["symbol"] == "AAPL"
        assert result["price"] == 150.50
        assert result["is_fresh"] is True

    def test_get_quote_error(self, connected_client, mock_call_tool_result):
        """get_quote returns error dict for unknown symbol."""
        quote = {"error": "Symbol 'INVALID' not found"}
        content = MagicMock()
        content.text = json.dumps(quote)
        mock_call_tool_result.content = [content]

        connected_client._session.call_tool = AsyncMock(
            return_value=mock_call_tool_result
        )

        result = connected_client.get_quote("INVALID")
        assert "error" in result


class TestPlaceOrder:
    """Tests for place_order method."""

    @pytest.fixture
    def connected_client(self, mock_client_session):
        client = SimulationClient()
        client._session = mock_client_session
        client._loop = asyncio.new_event_loop()
        yield client
        client._loop.close()

    def test_place_order_success(self, connected_client, mock_call_tool_result):
        """place_order returns order result dict."""
        order_result = {
            "status": "executed",
            "message": "Order executed at market price",
            "depot_id": "default",
            "trade_id": "trade-001",
            "pending_order_id": None,
            "symbol": "AAPL",
            "side": "buy",
            "quantity": 10,
            "price": 150.50,
            "fee": 2.50,
        }
        content = MagicMock()
        content.text = json.dumps(order_result)
        mock_call_tool_result.content = [content]

        connected_client._session.call_tool = AsyncMock(
            return_value=mock_call_tool_result
        )

        result = connected_client.place_order("AAPL", "buy", 10, "default")
        assert result["status"] == "executed"
        assert result["symbol"] == "AAPL"
        assert result["side"] == "buy"

    def test_place_order_rejected(self, connected_client, mock_call_tool_result):
        """place_order handles rejected orders."""
        order_result = {
            "status": "rejected",
            "message": "Insufficient cash",
            "depot_id": "default",
            "symbol": "AAPL",
            "side": "buy",
            "quantity": 10,
        }
        content = MagicMock()
        content.text = json.dumps(order_result)
        mock_call_tool_result.content = [content]

        connected_client._session.call_tool = AsyncMock(
            return_value=mock_call_tool_result
        )

        result = connected_client.place_order("AAPL", "buy", 10)
        assert result["status"] == "rejected"

    def test_place_order_args(self, connected_client, mock_call_tool_result):
        """place_order passes correct arguments."""
        order_result = {"status": "pending"}
        content = MagicMock()
        content.text = json.dumps(order_result)
        mock_call_tool_result.content = [content]

        connected_client._session.call_tool = AsyncMock(
            return_value=mock_call_tool_result
        )

        connected_client.place_order("MSFT", "sell", 5, "test-depot")

        call_args = connected_client._session.call_tool.call_args
        assert call_args[0][1]["symbol"] == "MSFT"
        assert call_args[0][1]["side"] == "sell"
        assert call_args[0][1]["quantity"] == 5
        assert call_args[0][1]["depot_id"] == "test-depot"


class TestGetTrades:
    """Tests for get_trades method."""

    @pytest.fixture
    def connected_client(self, mock_client_session):
        client = SimulationClient()
        client._session = mock_client_session
        client._loop = asyncio.new_event_loop()
        yield client
        client._loop.close()

    def test_get_trades_success(self, connected_client, mock_call_tool_result):
        """get_trades returns list of trade dicts."""
        trades = [
            {
                "id": "trade-002",
                "ts": "2025-07-05T15:30:00Z",
                "symbol": "AAPL",
                "side": "sell",
                "quantity": 5,
                "price": 150.50,
                "fee": 2.00,
                "gross": 752.50,
                "net": 750.50,
                "cash_after": 102750.50,
            },
            {
                "id": "trade-001",
                "ts": "2025-07-05T14:00:00Z",
                "symbol": "MSFT",
                "side": "buy",
                "quantity": 10,
                "price": 400.00,
                "fee": 2.50,
                "gross": 4000.00,
                "net": 4002.50,
                "cash_after": 95997.50,
            },
        ]
        content = MagicMock()
        content.text = json.dumps(trades)
        mock_call_tool_result.content = [content]

        connected_client._session.call_tool = AsyncMock(
            return_value=mock_call_tool_result
        )

        result = connected_client.get_trades(limit=50, depot_id="default")
        assert len(result) == 2
        assert result[0]["id"] == "trade-002"  # Newest first

    def test_get_trades_default_args(self, connected_client, mock_call_tool_result):
        """get_trades uses defaults for limit, symbol, depot_id."""
        trades = []
        content = MagicMock()
        content.text = json.dumps(trades)
        mock_call_tool_result.content = [content]

        connected_client._session.call_tool = AsyncMock(
            return_value=mock_call_tool_result
        )

        connected_client.get_trades()

        call_args = connected_client._session.call_tool.call_args
        assert call_args[0][1]["limit"] == 50
        assert call_args[0][1]["symbol"] == ""
        assert call_args[0][1]["depot_id"] == "default"

    def test_get_trades_validates_list_type(self, connected_client):
        """get_trades raises error if result is not a list."""
        content = MagicMock()
        content.text = json.dumps({"error": "not a list"})
        mock_result = MagicMock()
        mock_result.content = [content]
        mock_result.isError = False

        connected_client._session.call_tool = AsyncMock(return_value=mock_result)

        with pytest.raises(SimulationToolError) as exc_info:
            connected_client.get_trades()
        assert "non-list" in str(exc_info.value)


class TestGetServerConfig:
    """Tests for _get_server_config function."""

    def test_get_server_config_from_settings(self):
        """_get_server_config returns values from config when set."""
        from tradingagents.simulation import _get_server_config
        from tradingagents.dataflows.config import set_config

        set_config({
            "simulation_server_command": "/custom/python",
            "simulation_server_args": ["/custom/mcp_server.py"],
        })

        cmd, args = _get_server_config()
        assert cmd == "/custom/python"
        assert args == ["/custom/mcp_server.py"]

    def test_get_server_config_fallback_to_sibling(self):
        """_get_server_config falls back to sibling checkout when config not set."""
        from tradingagents.simulation import _get_server_config
        from tradingagents.dataflows.config import set_config

        set_config({
            "simulation_server_command": None,
            "simulation_server_args": None,
        })

        # This test assumes the sibling checkout exists (it does in CI)
        sibling = Path(__file__).parent.parent.parent.parent / "McpTradingSimulation"
        if sibling.exists():
            cmd, args = _get_server_config()
            assert cmd.endswith("python")
            assert len(args) == 1
            assert "mcp_server.py" in args[0]

    def test_get_server_config_missing_sibling(self):
        """_get_server_config raises error when sibling not found and no config."""
        from tradingagents.simulation import _get_server_config
        from tradingagents.dataflows.config import set_config

        set_config({
            "simulation_server_command": None,
            "simulation_server_args": None,
        })

        with patch("tradingagents.simulation.Path.exists", return_value=False):
            with pytest.raises(SimulationConnectionError) as exc_info:
                _get_server_config()
            assert "McpTradingSimulation" in str(exc_info.value) or "not found" in str(exc_info.value)
