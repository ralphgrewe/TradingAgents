"""Tests for the LLM client network timeout (#108).

``run_trading_agents.py`` was hanging indefinitely (no output, no CPU/GPU
load) with the default ollama provider. Root cause: ``langchain_openai.
ChatOpenAI`` forwards its ``request_timeout`` field (default ``None``)
straight through as the openai SDK's ``timeout`` kwarg. Passing ``timeout=
None`` explicitly to the openai SDK/httpx is *not* the same as omitting the
kwarg — it disables all timeouts rather than falling back to the SDK's own
600s default (see ``openai/_base_client.py``'s ``SyncAPIClient.__init__``:
the ``DEFAULT_TIMEOUT`` fallback only kicks in when the caller passes the
``NotGiven`` sentinel, not an explicit ``None``). Since nothing in this repo
set a ``timeout`` config value, every ``ChatOpenAI`` (including the ollama
provider's) was built with no bound at all, so a wedged local ollama daemon
(reachable, but never responding — e.g. mid model-pull) blocked the run
forever on a socket read with no CPU/GPU activity and no error.

The fix adds a ``llm_timeout`` config default (120s, env override
``TRADINGAGENTS_LLM_TIMEOUT``) that ``TradingAgentsGraph._get_provider_kwargs``
forwards as ``timeout`` to ``create_llm_client`` for every provider.

``TestGetProviderKwargsTimeout`` and ``TestTimeoutReachesClient`` verify the
wiring without any real network I/O. ``TestWedgedEndpointFailsFast`` proves
the fix actually bounds a stuck connection end-to-end: a local TCP server
that accepts connections but never responds stands in for a wedged ollama
daemon, and a short configured timeout demonstrates the call raises quickly
instead of hanging (kept fast — well under a second — by using a short
timeout rather than sleeping out a long one).
"""

from __future__ import annotations

import contextlib
import socket
import threading
import time
from unittest.mock import MagicMock

import openai
import pytest

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.llm_clients.factory import create_llm_client
from tradingagents.llm_clients.openai_client import NormalizedChatOpenAI


@pytest.mark.unit
class TestGetProviderKwargsTimeout:
    """Unit tests for the llm_timeout branch of ``_get_provider_kwargs``.

    Same MagicMock(spec=TradingAgentsGraph) pattern as
    tests/test_temperature_config.py's TestGetProviderKwargsTemperature.
    """

    def test_configured_timeout_passes_through(self):
        mock_graph = MagicMock(spec=TradingAgentsGraph)
        mock_graph.config = {"llm_provider": "ollama", "llm_timeout": 120}
        kwargs = TradingAgentsGraph._get_provider_kwargs(mock_graph)
        assert kwargs["timeout"] == 120

    def test_none_timeout_is_omitted(self):
        mock_graph = MagicMock(spec=TradingAgentsGraph)
        mock_graph.config = {"llm_provider": "ollama", "llm_timeout": None}
        kwargs = TradingAgentsGraph._get_provider_kwargs(mock_graph)
        assert "timeout" not in kwargs

    def test_default_config_has_bounded_timeout(self):
        # DEFAULT_CONFIG must ship a non-None llm_timeout so every run gets a
        # bound by default, not just runs that opt in explicitly (#108).
        from tradingagents.default_config import DEFAULT_CONFIG

        assert DEFAULT_CONFIG["llm_timeout"] is not None
        assert DEFAULT_CONFIG["llm_timeout"] > 0


@pytest.mark.unit
class TestTimeoutReachesClient:
    """Verify the configured timeout reaches the underlying ChatOpenAI
    instance for the ollama provider specifically (the provider named in
    the bug report), without any real network I/O.
    """

    @pytest.mark.parametrize(
        "provider,model",
        [
            ("ollama", "ministral-3:8b"),
            ("openai", "gpt-4.1"),
        ],
    )
    def test_timeout_reaches_chat_openai(self, provider, model):
        llm = create_llm_client(
            provider=provider, model=model, timeout=45, api_key="placeholder"
        ).get_llm()
        assert isinstance(llm, NormalizedChatOpenAI)
        assert llm.request_timeout == 45

    def test_timeout_omitted_leaves_request_timeout_none(self):
        # Documents the pre-fix default the hang relied on: without an
        # explicit timeout, ChatOpenAI.request_timeout is None (which, per
        # the module docstring, disables timeouts entirely once forwarded to
        # httpx) -- this is exactly why _get_provider_kwargs now always
        # forwards a configured llm_timeout.
        llm = create_llm_client(
            provider="ollama", model="ministral-3:8b", api_key="placeholder"
        ).get_llm()
        assert llm.request_timeout is None


@pytest.mark.unit
class TestWedgedEndpointFailsFast:
    """End-to-end proof that a stuck connection now fails fast.

    A raw TCP server that accepts the connection but never writes a
    response simulates a wedged local ollama daemon: reachable, but never
    answering. Before the fix (no timeout forwarded), ChatOpenAI would
    block on this forever. With a short configured timeout, it must raise
    well within a couple of seconds instead.
    """

    def test_unresponsive_server_raises_instead_of_hanging(self, monkeypatch):
        # httpx (via openai's SDK) honors HTTP(S)_PROXY env vars by default;
        # on a machine/CI runner with an ambient proxy configured, a request
        # to a local port can be transparently routed through it and answered
        # (or rejected) by the proxy itself well before our configured
        # timeout ever comes into play, which would make this test pass for
        # the wrong reason. Strip proxy env vars so the client talks to the
        # loopback server directly, the same way it would talk directly to a
        # local ollama daemon.
        for var in (
            "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
            "http_proxy", "https_proxy", "all_proxy",
        ):
            monkeypatch.delenv(var, raising=False)

        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]

        stop = threading.Event()

        def accept_loop():
            server.settimeout(0.1)
            while not stop.is_set():
                try:
                    conn, _addr = server.accept()
                except TimeoutError:
                    continue
                # Accept the connection but never send a response or close
                # it -- this is what a wedged-but-reachable daemon looks
                # like from the client's side.
                with contextlib.suppress(OSError):
                    conn.settimeout(5)

        thread = threading.Thread(target=accept_loop, daemon=True)
        thread.start()
        try:
            llm = create_llm_client(
                provider="ollama",
                model="ministral-3:8b",
                base_url=f"http://127.0.0.1:{port}/v1",
                timeout=0.5,
                max_retries=0,
                api_key="placeholder",
            ).get_llm()

            start = time.monotonic()
            with pytest.raises(openai.APITimeoutError):
                llm.invoke("hello")
            elapsed = time.monotonic() - start

            # Bounded by the configured 0.5s timeout, not hanging
            # indefinitely -- generous slack for CI jitter while still
            # proving this isn't a multi-minute (let alone infinite) wait.
            assert elapsed < 5
        finally:
            stop.set()
            thread.join(timeout=2)
            server.close()
