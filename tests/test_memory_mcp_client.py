"""Unit tests for MemoryMCPClient (networked MCP client for the memory core).

Tests run with mocked MCP sessions/transports — no real network calls, no
real LLM calls (issue #51).
"""

import asyncio
import contextlib
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tradingagents.memory.mcp_client import (
    MemoryMCPClient,
    MemoryMCPConnectionError,
    MemoryMCPToolError,
    _default_url,
    _resolve_connection,
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


class TestUrlResolution:
    """Tests for the URL/transport default-derivation helpers."""

    def test_default_url_streamable_http(self):
        assert _default_url("streamable-http") == "http://127.0.0.1:8000/mcp"

    def test_default_url_sse(self):
        assert _default_url("sse") == "http://127.0.0.1:8000/sse"

    def test_resolve_connection_builtin_defaults(self):
        url, transport = _resolve_connection(None, None)
        assert transport == "streamable-http"
        assert url == "http://127.0.0.1:8000/mcp"

    def test_resolve_connection_explicit_args_win(self):
        url, transport = _resolve_connection("http://example.com/mcp", "sse")
        assert url == "http://example.com/mcp"
        assert transport == "sse"

    def test_resolve_connection_reads_config(self):
        from tradingagents.dataflows.config import set_config

        set_config({"memory_mcp_transport": "sse", "memory_mcp_url": None})
        url, transport = _resolve_connection(None, None)
        assert transport == "sse"
        assert url == "http://127.0.0.1:8000/sse"

    def test_resolve_connection_config_url_overrides_derived_default(self):
        from tradingagents.dataflows.config import set_config

        set_config({"memory_mcp_url": "http://memory.internal:9000/mcp"})
        url, transport = _resolve_connection(None, None)
        assert url == "http://memory.internal:9000/mcp"
        assert transport == "streamable-http"

    def test_resolve_connection_invalid_transport(self):
        with pytest.raises(MemoryMCPConnectionError):
            _resolve_connection(None, "carrier-pigeon")


class TestMemoryMCPClientConnection:
    """Tests for connection/initialization."""

    def test_init_no_args(self):
        """Client can be initialized without arguments."""
        client = MemoryMCPClient()
        assert client.url is None
        assert client.transport is None
        assert client._session is None

    def test_context_manager_connect_disconnect(self, mock_client_session):
        """Context manager connects, keeps a persistent loop alive, resolves
        defaults, and tears everything down cleanly on exit."""
        with patch.object(
            MemoryMCPClient, "_async_connect", new_callable=AsyncMock
        ) as mock_async_connect:
            mock_async_connect.return_value = mock_client_session

            with MemoryMCPClient() as client:
                assert client._session is mock_client_session
                assert client._loop is not None
                assert not client._loop.is_closed()
                assert client.url == "http://127.0.0.1:8000/mcp"
                assert client.transport == "streamable-http"
                loop = client._loop

            # After exiting, session/exit-stack/loop are all cleared and the
            # loop itself is closed (not leaked, not reused across connections).
            assert client._session is None
            assert client._loop is None
            assert client._exit_stack is None
            assert loop.is_closed()

    def test_async_connect_streamable_http_wiring(self):
        """_async_connect enters streamable_http_client then constructs
        ClientSession(read_stream, write_stream) — the 3-tuple's session-id
        callback must be discarded, not passed to ClientSession."""
        read_stream, write_stream = object(), object()

        @contextlib.asynccontextmanager
        async def fake_streamable_http_client(url):
            yield (read_stream, write_stream, lambda: "session-id")

        fake_session = AsyncMock()
        fake_session.__aenter__.return_value = fake_session
        fake_session.__aexit__.return_value = None

        with patch(
            "mcp.client.streamable_http.streamable_http_client",
            side_effect=fake_streamable_http_client,
        ), patch(
            "tradingagents.memory.mcp_client.ClientSession", return_value=fake_session
        ) as mock_session_cls:
            client = MemoryMCPClient(url="http://example.com/mcp", transport="streamable-http")
            loop = asyncio.new_event_loop()
            try:
                session = loop.run_until_complete(client._async_connect())
            finally:
                loop.close()

            mock_session_cls.assert_called_once_with(read_stream, write_stream)
            assert session is fake_session
            fake_session.initialize.assert_awaited_once()
            assert client._exit_stack is not None

    def test_async_connect_sse_wiring(self):
        """_async_connect enters sse_client (2-tuple) for the sse transport."""
        read_stream, write_stream = object(), object()

        @contextlib.asynccontextmanager
        async def fake_sse_client(url):
            yield (read_stream, write_stream)

        fake_session = AsyncMock()
        fake_session.__aenter__.return_value = fake_session
        fake_session.__aexit__.return_value = None

        with patch(
            "mcp.client.sse.sse_client", side_effect=fake_sse_client
        ), patch(
            "tradingagents.memory.mcp_client.ClientSession", return_value=fake_session
        ) as mock_session_cls:
            client = MemoryMCPClient(url="http://example.com/sse", transport="sse")
            loop = asyncio.new_event_loop()
            try:
                session = loop.run_until_complete(client._async_connect())
            finally:
                loop.close()

            mock_session_cls.assert_called_once_with(read_stream, write_stream)
            assert session is fake_session
            fake_session.initialize.assert_awaited_once()

    def test_connect_unwinds_partial_state_on_initialize_failure(self):
        """If session.initialize() fails, whatever was already entered (the
        HTTP transport) must be unwound — no leaked connection."""
        exited = []

        @contextlib.asynccontextmanager
        async def fake_streamable_http_client(url):
            try:
                yield (object(), object(), lambda: None)
            finally:
                exited.append("transport")

        fake_session = AsyncMock()
        fake_session.__aenter__.return_value = fake_session
        fake_session.__aexit__.return_value = None
        fake_session.initialize.side_effect = RuntimeError("boom")

        with patch(
            "mcp.client.streamable_http.streamable_http_client",
            side_effect=fake_streamable_http_client,
        ), patch("tradingagents.memory.mcp_client.ClientSession", return_value=fake_session):
            client = MemoryMCPClient(url="http://example.com/mcp", transport="streamable-http")
            with pytest.raises(MemoryMCPConnectionError):
                client.connect()

            assert exited == ["transport"]
            fake_session.__aexit__.assert_awaited_once()
            assert client._session is None
            assert client._loop is None

    def test_close_unwinds_real_exit_stack(self):
        """close() must actually drive the AsyncExitStack unwind end-to-end —
        i.e. the entered transport/session context managers' __aexit__ run."""
        exited = []

        @contextlib.asynccontextmanager
        async def fake_streamable_http_client(url):
            try:
                yield (object(), object(), lambda: None)
            finally:
                exited.append("transport")

        fake_session = AsyncMock()
        fake_session.__aenter__.return_value = fake_session
        fake_session.__aexit__.return_value = None

        with patch(
            "mcp.client.streamable_http.streamable_http_client",
            side_effect=fake_streamable_http_client,
        ), patch("tradingagents.memory.mcp_client.ClientSession", return_value=fake_session):
            client = MemoryMCPClient(url="http://example.com/mcp", transport="streamable-http")
            client.connect()
            assert client._exit_stack is not None
            loop = client._loop

            client.close()  # must not raise

            fake_session.__aexit__.assert_awaited_once()
            assert exited == ["transport"]
            assert client._session is None
            assert client._exit_stack is None
            assert client._loop is None
            assert loop.is_closed()

    def test_close_does_not_raise_when_unwind_fails(self):
        """Regression-shaped test (mirrors issue #46's fix for SimulationClient):
        if the exit stack raises while unwinding, close() must still not raise
        and must still leave the client cleanly disconnected with the loop
        closed."""

        @contextlib.asynccontextmanager
        async def fake_streamable_http_client(url):
            yield (object(), object(), lambda: None)

        fake_session = AsyncMock()
        fake_session.__aenter__.return_value = fake_session
        fake_session.__aexit__.side_effect = RuntimeError("aexit boom")

        with patch(
            "mcp.client.streamable_http.streamable_http_client",
            side_effect=fake_streamable_http_client,
        ), patch("tradingagents.memory.mcp_client.ClientSession", return_value=fake_session):
            client = MemoryMCPClient(url="http://example.com/mcp", transport="streamable-http")
            client.connect()
            loop = client._loop

            client.close()  # must swallow the __aexit__ error

            fake_session.__aexit__.assert_awaited_once()
            assert client._session is None
            assert client._exit_stack is None
            assert client._loop is None
            assert loop.is_closed()

    def test_connect_raises_on_invalid_transport(self):
        """Connection failure path: an invalid transport is a connection error."""
        client = MemoryMCPClient(transport="carrier-pigeon")
        with pytest.raises(MemoryMCPConnectionError):
            client.connect()
        assert client._session is None

    def test_connect_wraps_transport_failure(self):
        """Connection failure path: the underlying transport raising is wrapped
        as MemoryMCPConnectionError, distinct from a tool-call error."""

        @contextlib.asynccontextmanager
        async def failing_streamable_http_client(url):
            raise ConnectionRefusedError("server unreachable")
            yield  # pragma: no cover - never reached

        with patch(
            "mcp.client.streamable_http.streamable_http_client",
            side_effect=failing_streamable_http_client,
        ):
            client = MemoryMCPClient(url="http://example.com/mcp", transport="streamable-http")
            with pytest.raises(MemoryMCPConnectionError) as exc_info:
                client.connect()
            assert "server unreachable" in str(exc_info.value)
            assert client._session is None

    def test_no_connect_if_already_connected(self, mock_client_session):
        """Calling connect() twice is a no-op."""
        client = MemoryMCPClient()
        client._session = mock_client_session
        with patch("tradingagents.memory.mcp_client._resolve_connection") as mock_resolve:
            client.connect()
            mock_resolve.assert_not_called()

    def test_close_when_not_connected(self):
        """Calling close() when not connected is safe."""
        client = MemoryMCPClient()
        client.close()  # Should not raise
        assert client._session is None


class TestMemoryMCPClientToolCalls:
    """Tests for the low-level _call_tool_sync used by every typed method."""

    @pytest.fixture
    def connected_client(self, mock_client_session):
        client = MemoryMCPClient(url="http://example.com/mcp", transport="streamable-http")
        client._session = mock_client_session
        client._loop = asyncio.new_event_loop()
        yield client
        client._loop.close()

    def test_call_tool_sync_not_connected(self):
        """Calling a tool when not connected raises a connection error."""
        client = MemoryMCPClient()
        with pytest.raises(MemoryMCPConnectionError):
            client._call_tool_sync("memory_get_statistics")

    def test_call_tool_sync_raises_when_loop_missing(self, mock_client_session):
        """A live session without an event loop is an invariant violation."""
        client = MemoryMCPClient()
        client._session = mock_client_session
        client._loop = None
        with pytest.raises(MemoryMCPConnectionError):
            client._call_tool_sync("memory_get_statistics")

    def test_call_tool_sync_success_json(self, connected_client, mock_call_tool_result):
        """Successful tool call returns parsed JSON."""
        test_data = {"id": "test", "value": 123}
        content = MagicMock()
        content.text = json.dumps(test_data)
        mock_call_tool_result.content = [content]

        connected_client._session.call_tool = AsyncMock(return_value=mock_call_tool_result)

        result = connected_client._call_tool_sync("test_tool", {"arg": "value"})
        assert result == test_data

    def test_call_tool_sync_protocol_error(self, connected_client):
        """An MCP-protocol-level error (isError=True) is a tool error, distinct
        from a connection error."""
        mock_result = MagicMock()
        mock_result.isError = True
        mock_result.content = "Tool failed"

        connected_client._session.call_tool = AsyncMock(return_value=mock_result)

        with pytest.raises(MemoryMCPToolError) as exc_info:
            connected_client._call_tool_sync("bad_tool")
        assert "returned error" in str(exc_info.value)

    def test_call_tool_sync_error_string_convention(
        self, connected_client, mock_call_tool_result
    ):
        """The memory_* tools signal their own failures as a successful
        JSON-RPC call whose text payload starts with "ERROR:" (see
        mcp_server.py) rather than an MCP protocol error. This must also
        raise MemoryMCPToolError."""
        content = MagicMock()
        content.text = "ERROR: db unavailable"
        mock_call_tool_result.content = [content]

        connected_client._session.call_tool = AsyncMock(return_value=mock_call_tool_result)

        with pytest.raises(MemoryMCPToolError) as exc_info:
            connected_client._call_tool_sync("memory_get_statistics")
        assert "ERROR: db unavailable" in str(exc_info.value)

    def test_call_tool_sync_exception(self, connected_client):
        """A raised exception during the call is converted to a tool error."""
        connected_client._session.call_tool = AsyncMock(side_effect=Exception("Network error"))

        with pytest.raises(MemoryMCPToolError) as exc_info:
            connected_client._call_tool_sync("test_tool")
        assert "failed" in str(exc_info.value).lower()


class TestStoreDecision:
    """Tests for store_decision (mirrors tradingagents.memory.store_decision)."""

    @pytest.fixture
    def connected_client(self, mock_client_session):
        client = MemoryMCPClient(url="http://example.com/mcp", transport="streamable-http")
        client._session = mock_client_session
        client._loop = asyncio.new_event_loop()
        yield client
        client._loop.close()

    def test_store_decision_success_round_trip(self, connected_client, mock_call_tool_result):
        content = MagicMock()
        content.text = json.dumps(True)
        mock_call_tool_result.content = [content]
        connected_client._session.call_tool = AsyncMock(return_value=mock_call_tool_result)

        result = connected_client.store_decision(
            "trader",
            "AAPL",
            "2026-07-01",
            "BUY",
            confidence=0.7,
            key_drivers=["earnings"],
            thesis="strong quarter",
        )
        assert result is True

        call_args = connected_client._session.call_tool.call_args
        assert call_args[0][0] == "memory_store_decision"
        assert call_args[0][1] == {
            "agent": "trader",
            "ticker": "AAPL",
            "date": "2026-07-01",
            "signal": "BUY",
            "confidence": 0.7,
            "key_drivers": ["earnings"],
            "thesis": "strong quarter",
            "db_path": None,
        }

    def test_store_decision_duplicate_is_noop(self, connected_client, mock_call_tool_result):
        content = MagicMock()
        content.text = json.dumps(False)
        mock_call_tool_result.content = [content]
        connected_client._session.call_tool = AsyncMock(return_value=mock_call_tool_result)

        result = connected_client.store_decision("trader", "AAPL", "2026-07-01", "BUY")
        assert result is False

    def test_store_decision_validates_bool_type(self, connected_client, mock_call_tool_result):
        content = MagicMock()
        content.text = json.dumps({"unexpected": "dict"})
        mock_call_tool_result.content = [content]
        connected_client._session.call_tool = AsyncMock(return_value=mock_call_tool_result)

        with pytest.raises(MemoryMCPToolError) as exc_info:
            connected_client.store_decision("trader", "AAPL", "2026-07-01", "BUY")
        assert "non-bool" in str(exc_info.value)

    def test_store_decision_tool_error_response(self, connected_client, mock_call_tool_result):
        content = MagicMock()
        content.text = "ERROR: disk full"
        mock_call_tool_result.content = [content]
        connected_client._session.call_tool = AsyncMock(return_value=mock_call_tool_result)

        with pytest.raises(MemoryMCPToolError):
            connected_client.store_decision("trader", "AAPL", "2026-07-01", "BUY")


class TestResolvePending:
    """Tests for resolve_pending (mirrors tradingagents.memory.resolve_pending)."""

    @pytest.fixture
    def connected_client(self, mock_client_session):
        client = MemoryMCPClient(url="http://example.com/mcp", transport="streamable-http")
        client._session = mock_client_session
        client._loop = asyncio.new_event_loop()
        yield client
        client._loop.close()

    def test_resolve_pending_success_round_trip(self, connected_client, mock_call_tool_result):
        ids = [1, 2, 3]
        content = MagicMock()
        content.text = json.dumps(ids)
        mock_call_tool_result.content = [content]
        connected_client._session.call_tool = AsyncMock(return_value=mock_call_tool_result)

        result = connected_client.resolve_pending(agent="trader", ticker="AAPL")
        assert result == ids

        call_args = connected_client._session.call_tool.call_args
        assert call_args[0][0] == "memory_resolve_pending"
        assert call_args[0][1] == {"agent": "trader", "ticker": "AAPL", "db_path": None}

    def test_resolve_pending_defaults(self, connected_client, mock_call_tool_result):
        content = MagicMock()
        content.text = json.dumps([])
        mock_call_tool_result.content = [content]
        connected_client._session.call_tool = AsyncMock(return_value=mock_call_tool_result)

        result = connected_client.resolve_pending()
        assert result == []

        call_args = connected_client._session.call_tool.call_args
        assert call_args[0][1] == {"agent": None, "ticker": None, "db_path": None}

    def test_resolve_pending_validates_list_type(self, connected_client, mock_call_tool_result):
        content = MagicMock()
        content.text = json.dumps({"not": "a list"})
        mock_call_tool_result.content = [content]
        connected_client._session.call_tool = AsyncMock(return_value=mock_call_tool_result)

        with pytest.raises(MemoryMCPToolError) as exc_info:
            connected_client.resolve_pending()
        assert "non-list" in str(exc_info.value)

    def test_resolve_pending_tool_error_response(self, connected_client, mock_call_tool_result):
        content = MagicMock()
        content.text = "ERROR: yfinance unavailable"
        mock_call_tool_result.content = [content]
        connected_client._session.call_tool = AsyncMock(return_value=mock_call_tool_result)

        with pytest.raises(MemoryMCPToolError):
            connected_client.resolve_pending()


class TestGetPastContext:
    """Tests for get_past_context (mirrors tradingagents.memory.get_past_context)."""

    @pytest.fixture
    def connected_client(self, mock_client_session):
        client = MemoryMCPClient(url="http://example.com/mcp", transport="streamable-http")
        client._session = mock_client_session
        client._loop = asyncio.new_event_loop()
        yield client
        client._loop.close()

    def test_get_past_context_success_round_trip(self, connected_client, mock_call_tool_result):
        markdown = "## Past context: AAPL\n\n*No prior resolved lessons yet.*"
        content = MagicMock()
        content.text = markdown  # returned raw, not JSON-encoded (see FastMCP _convert_to_content)
        mock_call_tool_result.content = [content]
        connected_client._session.call_tool = AsyncMock(return_value=mock_call_tool_result)

        result = connected_client.get_past_context("trader", "AAPL")
        assert result == markdown

        call_args = connected_client._session.call_tool.call_args
        assert call_args[0][0] == "memory_get_past_context"
        assert call_args[0][1] == {
            "agent": "trader",
            "ticker": "AAPL",
            "n_same": 5,
            "n_cross": 3,
            "db_path": None,
        }

    def test_get_past_context_custom_limits(self, connected_client, mock_call_tool_result):
        content = MagicMock()
        content.text = "## Past context: MSFT"
        mock_call_tool_result.content = [content]
        connected_client._session.call_tool = AsyncMock(return_value=mock_call_tool_result)

        connected_client.get_past_context("trader", "MSFT", n_same=10, n_cross=1)

        call_args = connected_client._session.call_tool.call_args
        assert call_args[0][1]["n_same"] == 10
        assert call_args[0][1]["n_cross"] == 1

    def test_get_past_context_validates_str_type(self, connected_client, mock_call_tool_result):
        content = MagicMock()
        content.text = json.dumps({"not": "a string"})
        mock_call_tool_result.content = [content]
        connected_client._session.call_tool = AsyncMock(return_value=mock_call_tool_result)

        with pytest.raises(MemoryMCPToolError) as exc_info:
            connected_client.get_past_context("trader", "AAPL")
        assert "non-str" in str(exc_info.value)


class TestGetStatistics:
    """Tests for get_statistics (mirrors tradingagents.memory.get_statistics)."""

    @pytest.fixture
    def connected_client(self, mock_client_session):
        client = MemoryMCPClient(url="http://example.com/mcp", transport="streamable-http")
        client._session = mock_client_session
        client._loop = asyncio.new_event_loop()
        yield client
        client._loop.close()

    def test_get_statistics_success_round_trip(self, connected_client, mock_call_tool_result):
        stats = {
            "filters": {"agent": "trader", "ticker": None, "since": None},
            "per_agent_ticker": [],
            "per_agent": [],
            "calibration": [],
        }
        content = MagicMock()
        content.text = json.dumps(stats)
        mock_call_tool_result.content = [content]
        connected_client._session.call_tool = AsyncMock(return_value=mock_call_tool_result)

        result = connected_client.get_statistics(agent="trader")
        assert result == stats

        call_args = connected_client._session.call_tool.call_args
        assert call_args[0][0] == "memory_get_statistics"
        assert call_args[0][1] == {
            "agent": "trader",
            "ticker": None,
            "since": None,
            "db_path": None,
        }

    def test_get_statistics_validates_dict_type(self, connected_client, mock_call_tool_result):
        content = MagicMock()
        content.text = json.dumps([1, 2, 3])
        mock_call_tool_result.content = [content]
        connected_client._session.call_tool = AsyncMock(return_value=mock_call_tool_result)

        with pytest.raises(MemoryMCPToolError) as exc_info:
            connected_client.get_statistics()
        assert "non-dict" in str(exc_info.value)

    def test_get_statistics_tool_error_response(self, connected_client, mock_call_tool_result):
        content = MagicMock()
        content.text = "ERROR: bad since format"
        mock_call_tool_result.content = [content]
        connected_client._session.call_tool = AsyncMock(return_value=mock_call_tool_result)

        with pytest.raises(MemoryMCPToolError):
            connected_client.get_statistics(since="not-a-date")


class TestGetDecisions:
    """Tests for get_decisions (mirrors tradingagents.memory.query.gather_context_rows)."""

    @pytest.fixture
    def connected_client(self, mock_client_session):
        client = MemoryMCPClient(url="http://example.com/mcp", transport="streamable-http")
        client._session = mock_client_session
        client._loop = asyncio.new_event_loop()
        yield client
        client._loop.close()

    def test_get_decisions_success_round_trip(self, connected_client, mock_call_tool_result):
        rows = [
            {
                "decision_date": "2026-01-01",
                "signal": "BUY",
                "confidence": 0.75,
                "key_drivers": {"ratio_signal": "undervalued"},
                "thesis": "Cheap on P/E.",
                "lesson": "Missed an earnings-driven guidance cut.",
                "forward_return": -0.04,
                "correct": False,
            }
        ]
        content = MagicMock()
        content.text = json.dumps(rows)
        mock_call_tool_result.content = [content]
        connected_client._session.call_tool = AsyncMock(return_value=mock_call_tool_result)

        result = connected_client.get_decisions("fundamental", "TSLA", limit=5, misses_only=True)
        assert result == rows

        call_args = connected_client._session.call_tool.call_args
        assert call_args[0][0] == "memory_get_decisions"
        assert call_args[0][1] == {
            "agent": "fundamental",
            "ticker": "TSLA",
            "db_path": None,
            "limit": 5,
            "misses_only": True,
        }

    def test_get_decisions_defaults(self, connected_client, mock_call_tool_result):
        content = MagicMock()
        content.text = json.dumps([])
        mock_call_tool_result.content = [content]
        connected_client._session.call_tool = AsyncMock(return_value=mock_call_tool_result)

        result = connected_client.get_decisions("trader", "AAPL")
        assert result == []

        call_args = connected_client._session.call_tool.call_args
        assert call_args[0][1] == {
            "agent": "trader",
            "ticker": "AAPL",
            "db_path": None,
            "limit": None,
            "misses_only": False,
        }

    def test_get_decisions_validates_list_type(self, connected_client, mock_call_tool_result):
        content = MagicMock()
        content.text = json.dumps({"not": "a list"})
        mock_call_tool_result.content = [content]
        connected_client._session.call_tool = AsyncMock(return_value=mock_call_tool_result)

        with pytest.raises(MemoryMCPToolError) as exc_info:
            connected_client.get_decisions("trader", "AAPL")
        assert "non-list" in str(exc_info.value)

    def test_get_decisions_tool_error_response(self, connected_client, mock_call_tool_result):
        content = MagicMock()
        content.text = "ERROR: something went wrong"
        mock_call_tool_result.content = [content]
        connected_client._session.call_tool = AsyncMock(return_value=mock_call_tool_result)

        with pytest.raises(MemoryMCPToolError):
            connected_client.get_decisions("trader", "AAPL")
