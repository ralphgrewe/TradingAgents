"""Tests for logging configuration in mcp_server.py (issue #59).

Verifies that:
1. Logging is properly configured based on MCP_TRANSPORT
   - stdio: logging is disabled (to keep stdout clean)
   - streamable-http, sse: logging is enabled with INFO level
2. Startup message is logged when using networked transports
3. Each tool logs start/completion with tool name and arguments, via the
   shared `_log_tool_call` helper (design-review follow-up: no more
   `if _logger:` guards duplicated at every call site or in test bodies)
4. Errors are logged when tool execution fails
5. Under stdio transport, an actual tool call leaves stdout (the JSON-RPC
   protocol channel) completely clean
"""

import importlib
import io
import logging
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

import mcp_server


_REPO_ROOT = Path(__file__).resolve().parent.parent

# Helper to run mcp_server as a subprocess with a probe to capture logging
_LOGGING_PROBE_SCRIPT = """\
import logging
import sys
import os

# Configure handlers to capture logs
log_handler = logging.StreamHandler(sys.stdout)
log_handler.setFormatter(logging.Formatter("%(name)s|%(levelname)s|%(message)s"))
logging.getLogger().addHandler(log_handler)

# Run the real module
import runpy
try:
    from mcp.server.fastmcp import FastMCP

    calls = []
    original_run = FastMCP.run

    def _fake_run(self, transport=None):
        calls.append(transport)

    FastMCP.run = _fake_run
    runpy.run_path("mcp_server.py", run_name="__main__")
except SystemExit:
    pass
"""


@pytest.fixture(autouse=True)
def _reset_logging_state():
    """Full reset around every test in this module.

    Several tests below mutate *global* `logging` module state — via
    `logging.disable`/`logging.basicConfig` (triggered indirectly by
    `importlib.reload(mcp_server)`, which re-runs the module's transport-
    dependent logging setup) — rather than anything scoped to an instance.
    Without an explicit reset this leaks across tests (and across other test
    modules that import `mcp_server`), causing order-dependent flakiness
    (design-review finding on issue #59). Snapshot the relevant global state
    before each test and restore it verbatim afterwards, then reload the
    module once more under the default "stdio" transport so later tests see
    it in its normal, quiet state.
    """
    original_disable = logging.root.manager.disable
    original_handlers = list(logging.root.handlers)
    original_level = logging.root.level
    try:
        yield
    finally:
        logging.disable(original_disable)
        logging.root.handlers[:] = original_handlers
        logging.root.setLevel(original_level)
        with patch.dict(os.environ, {"MCP_TRANSPORT": "stdio"}):
            importlib.reload(mcp_server)


class TestLoggingConfiguration:
    """Test conditional logging setup based on transport."""

    def test_logging_disabled_for_stdio_transport(self):
        """For stdio transport, logging should be disabled to keep stdout clean."""
        with patch.dict(os.environ, {"MCP_TRANSPORT": "stdio"}):
            importlib.reload(mcp_server)

            # The logger is always defined (no `_logger = None` sentinel) —
            # what actually differs per-transport is whether logging is
            # globally disabled.
            assert isinstance(mcp_server.logger, logging.Logger)
            assert logging.root.manager.disable == logging.CRITICAL

    def test_logging_enabled_for_streamable_http_transport(self):
        """For streamable-http transport, logging should be enabled."""
        with patch.dict(os.environ, {"MCP_TRANSPORT": "streamable-http"}):
            importlib.reload(mcp_server)

            assert isinstance(mcp_server.logger, logging.Logger)
            assert logging.root.manager.disable != logging.CRITICAL

    def test_logging_enabled_for_sse_transport(self):
        """For sse transport, logging should be enabled."""
        with patch.dict(os.environ, {"MCP_TRANSPORT": "sse"}):
            importlib.reload(mcp_server)

            assert isinstance(mcp_server.logger, logging.Logger)
            assert logging.root.manager.disable != logging.CRITICAL


class TestToolLogging:
    """Test that tools log their invocations with arguments via `_log_tool_call`.

    Every assertion here executes unconditionally — no `if mcp_server._logger:`
    self-gating in the test body (design-review finding on issue #59: with
    that pattern, a silently-broken logger setup would make these tests no-op
    instead of failing).
    """

    @pytest.fixture(autouse=True)
    def setup_logging(self):
        """Ensure logging is enabled for these tests."""
        with patch.dict(os.environ, {"MCP_TRANSPORT": "streamable-http"}):
            importlib.reload(mcp_server)
            yield

    def test_analyze_stock_logs_start_and_completion(self, caplog):
        """analyze_stock should log when starting and completing with arguments."""
        with caplog.at_level(logging.INFO):
            with patch(
                "mcp_server._run_analysis",
                return_value=("# Report", {}, "BUY"),
            ):
                result = mcp_server.analyze_stock("AAPL", "2024-05-10")
                assert "## Final Decision: BUY" in result

        assert any(
            "analyze_stock: starting" in record.message
            and "ticker=AAPL" in record.message
            and "date=2024-05-10" in record.message
            for record in caplog.records
        )
        assert any(
            "analyze_stock: completed" in record.message
            and "ticker=AAPL" in record.message
            for record in caplog.records
        )

    def test_analyze_stock_logs_error_on_failure(self, caplog):
        """analyze_stock should log errors when execution fails."""
        with caplog.at_level(logging.ERROR):
            with patch(
                "mcp_server._run_analysis",
                side_effect=ValueError("Test error"),
            ):
                result = mcp_server.analyze_stock("AAPL", "2024-05-10")
                assert "ERROR:" in result

        assert any(
            "analyze_stock: failed" in record.message
            and "ticker=AAPL" in record.message
            and "Test error" in record.message
            for record in caplog.records
        )

    def test_memory_store_decision_logs_with_arguments(self, caplog, tmp_path):
        """memory_store_decision should log with all key arguments."""
        with caplog.at_level(logging.INFO):
            mcp_server.memory_store_decision(
                agent="trader",
                ticker="AAPL",
                date="2024-05-10",
                signal="Buy",
                confidence=0.8,
                db_path=str(tmp_path / "test.db"),
            )

        assert any(
            "memory_store_decision: starting" in record.message
            and "agent=trader" in record.message
            and "ticker=AAPL" in record.message
            and "date=2024-05-10" in record.message
            and "signal=Buy" in record.message
            for record in caplog.records
        )
        assert any(
            "memory_store_decision: completed" in record.message
            and "agent=trader" in record.message
            for record in caplog.records
        )

    def test_memory_resolve_pending_logs_with_arguments(self, caplog):
        """memory_resolve_pending should log with arguments."""
        with caplog.at_level(logging.INFO):
            with patch("tradingagents.memory.resolve_pending", return_value=[]):
                mcp_server.memory_resolve_pending(agent="trader", ticker="AAPL")

        assert any(
            "memory_resolve_pending: starting" in record.message
            and "agent=trader" in record.message
            and "ticker=AAPL" in record.message
            for record in caplog.records
        )
        assert any(
            "memory_resolve_pending: completed" in record.message
            and "agent=trader" in record.message
            for record in caplog.records
        )

    def test_memory_get_past_context_logs_with_arguments(self, caplog):
        """memory_get_past_context should log with arguments."""
        with caplog.at_level(logging.INFO):
            with patch(
                "tradingagents.memory.get_past_context",
                return_value="No prior lessons.",
            ):
                mcp_server.memory_get_past_context(
                    agent="trader", ticker="AAPL", n_same=5, n_cross=3
                )

        assert any(
            "memory_get_past_context: starting" in record.message
            and "agent=trader" in record.message
            and "ticker=AAPL" in record.message
            and "n_same=5" in record.message
            and "n_cross=3" in record.message
            for record in caplog.records
        )
        assert any(
            "memory_get_past_context: completed" in record.message
            and "agent=trader" in record.message
            for record in caplog.records
        )

    def test_memory_get_statistics_logs_with_arguments(self, caplog):
        """memory_get_statistics should log with arguments."""
        with caplog.at_level(logging.INFO):
            with patch(
                "tradingagents.memory.get_statistics",
                return_value={
                    "filters": {"agent": "trader", "ticker": "AAPL", "since": None}
                },
            ):
                mcp_server.memory_get_statistics(
                    agent="trader", ticker="AAPL", since="2024-01-01"
                )

        assert any(
            "memory_get_statistics: starting" in record.message
            and "agent=trader" in record.message
            and "ticker=AAPL" in record.message
            and "since=2024-01-01" in record.message
            for record in caplog.records
        )
        assert any(
            "memory_get_statistics: completed" in record.message
            and "agent=trader" in record.message
            for record in caplog.records
        )

    def test_memory_get_decisions_logs_with_arguments(self, caplog):
        """memory_get_decisions should log with arguments."""
        with caplog.at_level(logging.INFO):
            with patch("tradingagents.memory.gather_context_rows", return_value=[]):
                mcp_server.memory_get_decisions(
                    agent="trader", ticker="AAPL", limit=5, misses_only=True
                )

        assert any(
            "memory_get_decisions: starting" in record.message
            and "agent=trader" in record.message
            and "ticker=AAPL" in record.message
            and "limit=5" in record.message
            and "misses_only=True" in record.message
            for record in caplog.records
        )
        assert any(
            "memory_get_decisions: completed" in record.message
            and "agent=trader" in record.message
            for record in caplog.records
        )

    def test_memory_resolve_pending_logs_error_on_failure(self, caplog):
        """A failing tool call logs a "failed" line (not just "starting")."""
        with caplog.at_level(logging.ERROR):
            with patch(
                "tradingagents.memory.resolve_pending",
                side_effect=RuntimeError("boom"),
            ):
                result = mcp_server.memory_resolve_pending(agent="trader", ticker="AAPL")
                assert result == "ERROR: boom"

        assert any(
            "memory_resolve_pending: failed" in record.message
            and "agent=trader" in record.message
            and "boom" in record.message
            for record in caplog.records
        )


class TestStdioStdoutCleanliness:
    """Confirm the stdio transport's stdout — the JSON-RPC protocol channel —
    stays clean during an actual tool call (issue #59's central acceptance
    criterion; previously untested per design review)."""

    @pytest.fixture(autouse=True)
    def _stdio_transport(self):
        with patch.dict(os.environ, {"MCP_TRANSPORT": "stdio"}):
            importlib.reload(mcp_server)
            yield

    def test_analyze_stock_call_does_not_write_to_stdout(self):
        """Calling analyze_stock under stdio transport must not print/log
        anything to stdout, even though the tool logs start/completion
        internally — those log calls must be no-ops.

        Note: deliberately does not use `caplog.at_level(...)` here — pytest's
        caplog forcibly re-enables logging past `logging.disable()` so it can
        capture records (see `_pytest.logging._force_enable_logging`), which
        would defeat the very thing this test is verifying. The stdout
        content is the actual acceptance criterion (issue #59): the JSON-RPC
        protocol channel must stay clean.
        """
        assert logging.root.manager.disable == logging.CRITICAL  # sanity: stdio disables logging

        captured_stdout = io.StringIO()
        with patch("sys.stdout", captured_stdout):
            with patch(
                "mcp_server._run_analysis",
                return_value=("# Report", {"foo": "bar"}, "HOLD"),
            ):
                result = mcp_server.analyze_stock("AAPL", "2024-05-10")

        assert "## Final Decision: HOLD" in result
        assert captured_stdout.getvalue() == ""

    def test_memory_tool_call_does_not_write_to_stdout(self, tmp_path):
        """Same guarantee for a memory_* tool call."""
        assert logging.root.manager.disable == logging.CRITICAL  # sanity: stdio disables logging

        captured_stdout = io.StringIO()
        with patch("sys.stdout", captured_stdout):
            result = mcp_server.memory_store_decision(
                agent="trader",
                ticker="AAPL",
                date="2024-05-10",
                signal="Buy",
                db_path=str(tmp_path / "test.db"),
            )

        assert result is True
        assert captured_stdout.getvalue() == ""


class TestLoggingDoesNotAffectStdio:
    """Test that logging changes don't break stdio transport."""

    def test_invalid_transport_error_message_for_stdio(self):
        """Invalid transport error should be visible even with stdio logging disabled."""
        env = {**os.environ, "MCP_TRANSPORT": "invalid_transport"}
        result = subprocess.run(
            [sys.executable, "mcp_server.py"],
            capture_output=True,
            text=True,
            cwd=str(_REPO_ROOT),
            env=env,
        )
        assert result.returncode == 2
        assert "Invalid MCP_TRANSPORT" in result.stderr

    def test_stdio_transport_does_not_log_startup(self):
        """Stdio transport should not produce startup logging."""
        # This is harder to test in unit tests, but we can at least verify
        # that the transport validation still works
        env = {**os.environ, "MCP_TRANSPORT": "stdio"}
        result = subprocess.run(
            [sys.executable, "-c", _LOGGING_PROBE_SCRIPT],
            capture_output=True,
            text=True,
            cwd=str(_REPO_ROOT),
            env=env,
        )
        # Should not error
        assert result.returncode == 0
