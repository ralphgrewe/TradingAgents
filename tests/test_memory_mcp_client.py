"""Unit tests for MemoryMCPClient (networked MCP client for the memory core).

Tests run with mocked MCP sessions/transports — no real network calls, no
real LLM calls (issue #51).
"""

import asyncio
import contextlib
import json
import socket
import threading
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tradingagents.memory.mcp_client import (
    MemoryMCPClient,
    MemoryMCPConnectionError,
    MemoryMCPToolError,
    _default_url,
    _is_loopback_url,
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
    result.structuredContent = None  # Explicitly None to test legacy content path
    return result


class TestUrlResolution:
    """Tests for the URL/transport default-derivation helpers."""

    def test_default_url_streamable_http(self):
        assert _default_url("streamable-http") == "http://127.0.0.1:8001/mcp"

    def test_default_url_sse(self):
        assert _default_url("sse") == "http://127.0.0.1:8001/sse"

    def test_resolve_connection_builtin_defaults(self):
        url, transport = _resolve_connection(None, None)
        assert transport == "streamable-http"
        assert url == "http://127.0.0.1:8001/mcp"

    def test_resolve_connection_explicit_args_win(self):
        url, transport = _resolve_connection("http://example.com/mcp", "sse")
        assert url == "http://example.com/mcp"
        assert transport == "sse"

    def test_resolve_connection_reads_config(self):
        from tradingagents.dataflows.config import set_config

        set_config({"memory_mcp_transport": "sse", "memory_mcp_url": None})
        url, transport = _resolve_connection(None, None)
        assert transport == "sse"
        assert url == "http://127.0.0.1:8001/sse"

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
            MemoryMCPClient, "_enter_session", new_callable=AsyncMock
        ) as mock_enter_session:
            mock_enter_session.return_value = mock_client_session

            with MemoryMCPClient() as client:
                assert client._session is mock_client_session
                assert client._loop is not None
                assert not client._loop.is_closed()
                assert client.url == "http://127.0.0.1:8001/mcp"
                assert client.transport == "streamable-http"
                loop = client._loop

            # After exiting, session/task/loop are all cleared and the
            # loop itself is closed (not leaked, not reused across connections).
            assert client._session is None
            assert client._loop is None
            assert client._lifecycle_task is None
            assert client._shutdown_event is None
            assert loop.is_closed()

    def test_async_connect_streamable_http_wiring(self):
        """_enter_session enters streamable_http_client then constructs
        ClientSession(read_stream, write_stream) — the 3-tuple's session-id
        callback must be discarded, not passed to ClientSession."""
        read_stream, write_stream = object(), object()

        @contextlib.asynccontextmanager
        async def fake_streamable_http_client(url, **kwargs):
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
                exit_stack = contextlib.AsyncExitStack()
                session = loop.run_until_complete(client._enter_session(exit_stack))
                loop.run_until_complete(exit_stack.aclose())
            finally:
                loop.close()

            mock_session_cls.assert_called_once_with(read_stream, write_stream)
            assert session is fake_session
            fake_session.initialize.assert_awaited_once()

    def test_async_connect_sse_wiring(self):
        """_enter_session enters sse_client (2-tuple) for the sse transport."""
        read_stream, write_stream = object(), object()

        @contextlib.asynccontextmanager
        async def fake_sse_client(url, **kwargs):
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
                exit_stack = contextlib.AsyncExitStack()
                session = loop.run_until_complete(client._enter_session(exit_stack))
                loop.run_until_complete(exit_stack.aclose())
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
        async def fake_streamable_http_client(url, **kwargs):
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
        async def fake_streamable_http_client(url, **kwargs):
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
            assert client._lifecycle_task is not None
            assert not client._lifecycle_task.done()
            loop = client._loop

            client.close()  # must not raise

            fake_session.__aexit__.assert_awaited_once()
            assert exited == ["transport"]
            assert client._session is None
            assert client._lifecycle_task is None
            assert client._shutdown_event is None
            assert client._loop is None
            assert loop.is_closed()

    def test_repeated_connect_close_no_cross_task_cancel_scope_error(self, caplog):
        """Regression test for issue #58: opening and closing several
        MemoryMCPClient instances in sequence (mirroring
        run_trading_agents.py processing multiple tickers, each opening its
        own client per ticker) must not raise or log a warning from close()
        — in particular, no "Attempted to exit cancel scope in a different
        task than it was entered in" RuntimeError, which is what happens
        when the transport's AsyncExitStack is entered in one Task
        (connect()'s) and exited in another (close()'s)."""
        exited = []

        @contextlib.asynccontextmanager
        async def fake_streamable_http_client(url, **kwargs):
            try:
                yield (object(), object(), lambda: None)
            finally:
                exited.append("transport")

        def make_fake_session():
            fake_session = AsyncMock()
            fake_session.__aenter__.return_value = fake_session
            fake_session.__aexit__.return_value = None
            return fake_session

        with patch(
            "mcp.client.streamable_http.streamable_http_client",
            side_effect=fake_streamable_http_client,
        ), patch(
            "tradingagents.memory.mcp_client.ClientSession",
            side_effect=lambda *a, **kw: make_fake_session(),
        ):
            with caplog.at_level("WARNING", logger="tradingagents.memory.mcp_client"):
                for _ in range(3):
                    client = MemoryMCPClient(
                        url="http://example.com/mcp", transport="streamable-http"
                    )
                    client.connect()
                    loop = client._loop
                    client.close()  # must not raise
                    assert client._session is None
                    assert client._lifecycle_task is None
                    assert client._shutdown_event is None
                    assert client._loop is None
                    assert loop.is_closed()

            assert exited == ["transport"] * 3
            assert caplog.records == []

    def test_close_does_not_raise_when_unwind_fails(self):
        """Regression-shaped test (mirrors issue #46's fix for SimulationClient):
        if the exit stack raises while unwinding, close() must still not raise
        and must still leave the client cleanly disconnected with the loop
        closed."""

        @contextlib.asynccontextmanager
        async def fake_streamable_http_client(url, **kwargs):
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
            assert client._lifecycle_task is None
            assert client._shutdown_event is None
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
        async def failing_streamable_http_client(url, **kwargs):
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


class TestConnectTimeout:
    """Regression tests for issue #108 — connect() must never hang forever.

    The MCP SDK raises transport-level failures (e.g. an HTTP error status
    returned by a proxy sitting in front of the server) inside
    ``streamable_http_client``'s internal anyio task group, where they only
    surface when that task group exits. The coroutine awaiting the JSON-RPC
    response is never woken, so ``session.initialize()`` blocks forever —
    the reported symptom was ``run_trading_agents.py`` sitting at
    "Processing AAPL..." indefinitely with no CPU load.
    """

    def test_connect_times_out_when_initialize_never_returns(self):
        """A session that never initializes fails fast instead of hanging."""

        @contextlib.asynccontextmanager
        async def fake_streamable_http_client(url, **kwargs):
            yield (object(), object(), lambda: None)

        fake_session = AsyncMock()
        fake_session.__aenter__.return_value = fake_session
        fake_session.__aexit__.return_value = None

        async def never_returns():
            await asyncio.sleep(3600)

        fake_session.initialize.side_effect = never_returns

        with patch(
            "mcp.client.streamable_http.streamable_http_client",
            side_effect=fake_streamable_http_client,
        ), patch("tradingagents.memory.mcp_client.ClientSession", return_value=fake_session):
            client = MemoryMCPClient(
                url="http://127.0.0.1:8001/mcp", transport="streamable-http", timeout=0.5
            )
            started = time.monotonic()
            with pytest.raises(MemoryMCPConnectionError) as exc_info:
                client.connect()
            elapsed = time.monotonic() - started

        assert elapsed < 10, "connect() hung well past its configured timeout"
        assert "Timed out" in str(exc_info.value)
        # The message must be actionable — it names the endpoint and points at
        # the two things that actually cause this.
        assert "http://127.0.0.1:8001/mcp" in str(exc_info.value)
        assert "proxy" in str(exc_info.value)
        # And the client is left cleanly disconnected, not half-open.
        assert client._session is None
        assert client._loop is None
        assert client._lifecycle_task is None

    def test_connect_timeout_is_not_double_wrapped(self):
        """The timeout error is raised as-is, not re-wrapped in a second
        'Failed to connect...' MemoryMCPConnectionError."""

        @contextlib.asynccontextmanager
        async def fake_streamable_http_client(url, **kwargs):
            yield (object(), object(), lambda: None)

        fake_session = AsyncMock()
        fake_session.__aenter__.return_value = fake_session
        fake_session.__aexit__.return_value = None

        async def never_returns():
            await asyncio.sleep(3600)

        fake_session.initialize.side_effect = never_returns

        with patch(
            "mcp.client.streamable_http.streamable_http_client",
            side_effect=fake_streamable_http_client,
        ), patch("tradingagents.memory.mcp_client.ClientSession", return_value=fake_session):
            client = MemoryMCPClient(
                url="http://127.0.0.1:8001/mcp", transport="streamable-http", timeout=0.5
            )
            with pytest.raises(MemoryMCPConnectionError) as exc_info:
                client.connect()

        assert "Failed to connect" not in str(exc_info.value)

    def test_timeout_defaults_to_config(self):
        """With no explicit timeout, connect() resolves memory_mcp_timeout."""
        from tradingagents.dataflows.config import set_config

        set_config({"memory_mcp_timeout": 0.25})

        @contextlib.asynccontextmanager
        async def fake_streamable_http_client(url, **kwargs):
            yield (object(), object(), lambda: None)

        fake_session = AsyncMock()
        fake_session.__aenter__.return_value = fake_session
        fake_session.__aexit__.return_value = None

        async def never_returns():
            await asyncio.sleep(3600)

        fake_session.initialize.side_effect = never_returns

        with patch(
            "mcp.client.streamable_http.streamable_http_client",
            side_effect=fake_streamable_http_client,
        ), patch("tradingagents.memory.mcp_client.ClientSession", return_value=fake_session):
            client = MemoryMCPClient(url="http://127.0.0.1:8001/mcp")
            with pytest.raises(MemoryMCPConnectionError):
                client.connect()
            assert client.timeout == 0.25

    def test_unresponsive_server_fails_fast(self):
        """End-to-end shape of the reported hang: a real TCP listener that
        accepts the connection and then never answers (standing in for the
        proxy that swallowed the request) must surface as a connection error
        in about the configured timeout, not block forever."""
        listener = socket.socket()
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(5)
        port = listener.getsockname()[1]
        accepted = []

        def accept_and_stall():
            while True:
                try:
                    accepted.append(listener.accept()[0])
                except OSError:
                    return

        thread = threading.Thread(target=accept_and_stall, daemon=True)
        thread.start()
        try:
            client = MemoryMCPClient(url=f"http://127.0.0.1:{port}/mcp", timeout=2)
            started = time.monotonic()
            with pytest.raises(MemoryMCPConnectionError):
                client.connect()
            elapsed = time.monotonic() - started
        finally:
            listener.close()
            for conn in accepted:
                conn.close()

        assert elapsed < 15, f"connect() took {elapsed:.1f}s against a stalled server"
        assert client._session is None


class TestProxyBypassForLoopback:
    """Regression tests for the root cause of issue #108.

    httpx reads ``http_proxy``/``https_proxy`` from the environment, and its
    ``no_proxy`` support only matches literal hosts — the common
    ``no_proxy=127.0.0.0/8`` CIDR entry becomes the mount pattern
    ``all://127.0.0.0/8``, which never matches ``127.0.0.1``. So requests to
    the loopback memory server were being handed to the corporate proxy,
    which answered ``503`` and wedged the client.
    """

    @pytest.mark.parametrize(
        "url",
        [
            "http://127.0.0.1:8001/mcp",
            "http://127.0.0.5:8001/mcp",
            "http://localhost:8001/mcp",
            "http://LocalHost:8001/sse",
            "http://[::1]:8001/mcp",
        ],
    )
    def test_loopback_urls_detected(self, url):
        assert _is_loopback_url(url) is True

    @pytest.mark.parametrize(
        "url",
        [
            "http://memory.internal:9000/mcp",
            "http://192.168.1.20:8001/mcp",
            "https://example.com/mcp",
            "not-a-url",
        ],
    )
    def test_non_loopback_urls_detected(self, url):
        assert _is_loopback_url(url) is False

    def test_loopback_client_ignores_environment_proxies(self, monkeypatch):
        """The httpx client used for a loopback server must not pick up
        http_proxy from the environment."""
        monkeypatch.setenv("http_proxy", "http://proxy.invalid:3128")
        monkeypatch.setenv("no_proxy", "127.0.0.0/8")  # the CIDR httpx can't match

        client = MemoryMCPClient(url="http://127.0.0.1:8001/mcp", transport="streamable-http")
        http_client = client._http_client_factory()()
        try:
            assert http_client.trust_env is False
            assert http_client._mounts == {}
        finally:
            asyncio.run(http_client.aclose())

    def test_remote_client_keeps_default_environment_behavior(self):
        """A non-loopback server keeps the SDK's env-aware client factory —
        the bypass is deliberately scoped to loopback."""
        from mcp.shared._httpx_utils import create_mcp_http_client

        client = MemoryMCPClient(url="http://memory.internal:9000/mcp")
        assert client._http_client_factory() is create_mcp_http_client


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

    def test_call_tool_sync_times_out(self, connected_client):
        """A tool call whose response never arrives raises instead of hanging
        (issue #108) — the transport can wedge mid-session just as it can
        during connect()."""

        async def never_returns(*args, **kwargs):
            await asyncio.sleep(3600)

        connected_client.timeout = 0.5
        connected_client._session.call_tool = never_returns

        started = time.monotonic()
        with pytest.raises(MemoryMCPToolError) as exc_info:
            connected_client._call_tool_sync("memory_get_statistics")
        elapsed = time.monotonic() - started

        assert elapsed < 10, "tool call hung well past its configured timeout"
        assert "timed out" in str(exc_info.value)
        assert "memory_get_statistics" in str(exc_info.value)

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
            "horizon_days": None,
        }

    def test_store_decision_horizon_days_round_trip(self, connected_client, mock_call_tool_result):
        content = MagicMock()
        content.text = json.dumps(True)
        mock_call_tool_result.content = [content]
        connected_client._session.call_tool = AsyncMock(return_value=mock_call_tool_result)

        result = connected_client.store_decision(
            "swing_trader",
            "AAPL",
            "2026-07-01",
            "BUY",
            confidence=0.7,
            key_drivers=["earnings"],
            thesis="strong quarter",
            horizon_days=5,
        )
        assert result is True

        call_args = connected_client._session.call_tool.call_args
        assert call_args[0][0] == "memory_store_decision"
        assert call_args[0][1] == {
            "agent": "swing_trader",
            "ticker": "AAPL",
            "date": "2026-07-01",
            "signal": "BUY",
            "confidence": 0.7,
            "key_drivers": ["earnings"],
            "thesis": "strong quarter",
            "db_path": None,
            "horizon_days": 5,
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


class TestStructuredContentParsing:
    """Tests for structuredContent parsing (issue #57).

    Verifies that the fix uses result.structuredContent["result"] when
    available, instead of manually parsing result.content[0].text. This
    is critical for list-typed returns, which are mishandled by the
    legacy content parsing: empty lists have zero content blocks, and
    N-item lists have N separate blocks (each containing one item, not
    the whole list).
    """

    @pytest.fixture
    def connected_client(self, mock_client_session):
        client = MemoryMCPClient(url="http://example.com/mcp", transport="streamable-http")
        client._session = mock_client_session
        client._loop = asyncio.new_event_loop()
        yield client
        client._loop.close()

    def test_empty_list_via_structured_content(self, connected_client):
        """Empty list result via structuredContent (the main bug fix for #57).

        Before the fix, result.content would be empty/falsy, parsed would stay
        None, and resolve_pending() would raise MemoryMCPToolError("returned
        non-list: None"). With the fix, structuredContent["result"] contains
        [], and it round-trips correctly.
        """
        mock_result = MagicMock()
        mock_result.isError = False
        # structuredContent is the fixed path: always has {"result": ...}
        mock_result.structuredContent = {"result": []}
        # content is empty for an empty list (shows the legacy path's problem)
        mock_result.content = []

        connected_client._session.call_tool = AsyncMock(return_value=mock_result)

        result = connected_client.resolve_pending()
        assert result == []

    def test_multi_item_list_via_structured_content(self, connected_client):
        """Multi-item list result via structuredContent (the other list bug).

        Before the fix, result.content would have N separate blocks (each
        containing one item's JSON), and only content[0].text would be read,
        silently truncating the list. With the fix, structuredContent["result"]
        contains the full list, and all items round-trip.
        """
        list_data = [1, 2, 3]
        mock_result = MagicMock()
        mock_result.isError = False
        # structuredContent is correct
        mock_result.structuredContent = {"result": list_data}
        # content is split per item (the legacy path's problem)
        content_items = [MagicMock() for _ in range(3)]
        for i, content_item in enumerate(content_items):
            content_item.text = json.dumps(list_data[i])
        mock_result.content = content_items

        connected_client._session.call_tool = AsyncMock(return_value=mock_result)

        result = connected_client.resolve_pending()
        assert result == list_data

    def test_dict_result_via_structured_content(self, connected_client):
        """Dict result (non-list) via structuredContent.

        Verifies that the fix handles non-list types correctly too.
        """
        dict_data = {"filters": {}, "per_agent": []}
        mock_result = MagicMock()
        mock_result.isError = False
        mock_result.structuredContent = {"result": dict_data}
        mock_result.content = [MagicMock(text=json.dumps(dict_data))]

        connected_client._session.call_tool = AsyncMock(return_value=mock_result)

        result = connected_client.get_statistics()
        assert result == dict_data

    def test_bool_result_via_structured_content(self, connected_client):
        """Bool result via structuredContent.

        Verifies that the fix handles non-list types correctly too.
        """
        bool_data = True
        mock_result = MagicMock()
        mock_result.isError = False
        mock_result.structuredContent = {"result": bool_data}
        mock_result.content = [MagicMock(text=json.dumps(bool_data))]

        connected_client._session.call_tool = AsyncMock(return_value=mock_result)

        result = connected_client.store_decision("trader", "AAPL", "2026-07-01", "BUY")
        assert result is True

    def test_string_result_via_structured_content(self, connected_client):
        """String result via structuredContent."""
        string_data = "## Past context"
        mock_result = MagicMock()
        mock_result.isError = False
        mock_result.structuredContent = {"result": string_data}
        # content would have the raw string (not JSON-encoded)
        mock_result.content = [MagicMock(text=string_data)]

        connected_client._session.call_tool = AsyncMock(return_value=mock_result)

        result = connected_client.get_past_context("trader", "AAPL")
        assert result == string_data

    def test_error_string_detected_after_structured_content(self, connected_client):
        """ERROR:-prefixed string in structuredContent is still detected as error.

        The error convention check (isinstance(parsed, str) and
        parsed.startswith("ERROR:")) must work the same whether parsing from
        structuredContent or legacy content.
        """
        error_string = "ERROR: db unavailable"
        mock_result = MagicMock()
        mock_result.isError = False
        mock_result.structuredContent = {"result": error_string}
        mock_result.content = [MagicMock(text=error_string)]

        connected_client._session.call_tool = AsyncMock(return_value=mock_result)

        with pytest.raises(MemoryMCPToolError) as exc_info:
            connected_client.get_statistics()
        assert "ERROR: db unavailable" in str(exc_info.value)

    def test_legacy_content_fallback_when_structured_content_missing(self, connected_client):
        """Legacy content parsing path still works when structuredContent is None.

        The fix includes a fallback for backward compatibility or when the MCP
        server doesn't provide structuredContent.
        """
        dict_data = {"test": "value"}
        mock_result = MagicMock()
        mock_result.isError = False
        mock_result.structuredContent = None  # Not present
        mock_result.content = [MagicMock(text=json.dumps(dict_data))]

        connected_client._session.call_tool = AsyncMock(return_value=mock_result)

        result = connected_client.get_statistics()
        assert result == dict_data

    def test_list_result_empty_via_structured_content_makes_resolve_pending_succeed(
        self, connected_client
    ):
        """Integration test: resolve_pending() succeeds with empty list via structuredContent.

        This is the concrete manifestation of the bug from issue #56: before
        the fix, an empty list result would cause resolve_pending() to raise
        MemoryMCPToolError("returned non-list: None").
        """
        mock_result = MagicMock()
        mock_result.isError = False
        mock_result.structuredContent = {"result": []}
        mock_result.content = []

        connected_client._session.call_tool = AsyncMock(return_value=mock_result)

        # This should not raise — empty list is a valid "nothing resolved yet" result
        result = connected_client.resolve_pending(agent="trader", ticker="AAPL")
        assert result == []
        assert isinstance(result, list)

    def test_list_result_with_multiple_items_via_structured_content_makes_get_decisions_complete(
        self, connected_client
    ):
        """Integration test: get_decisions() returns full list via structuredContent.

        Before the fix, only the first item would be returned (due to reading
        only content[0].text), and subsequent items would be silently lost.
        """
        list_data = [
            {"decision_date": "2026-01-01", "signal": "BUY", "correct": True},
            {"decision_date": "2026-01-02", "signal": "HOLD", "correct": False},
            {"decision_date": "2026-01-03", "signal": "SELL", "correct": True},
        ]
        mock_result = MagicMock()
        mock_result.isError = False
        mock_result.structuredContent = {"result": list_data}
        # Legacy content path would have 3 separate blocks
        content_items = [MagicMock() for _ in range(3)]
        for i, content_item in enumerate(content_items):
            content_item.text = json.dumps(list_data[i])
        mock_result.content = content_items

        connected_client._session.call_tool = AsyncMock(return_value=mock_result)

        result = connected_client.get_decisions("trader", "AAPL")
        assert result == list_data
        assert len(result) == 3
        assert result[0]["decision_date"] == "2026-01-01"
        assert result[1]["decision_date"] == "2026-01-02"
        assert result[2]["decision_date"] == "2026-01-03"
