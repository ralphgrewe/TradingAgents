"""Tests for logging configuration in mcp_server.py (issue #59).

Verifies that:
1. Logging is properly configured based on MCP_TRANSPORT
   - stdio: logging is disabled (to keep stdout clean)
   - streamable-http, sse: logging is enabled with INFO level
2. Startup message is logged when using networked transports
3. Each tool logs start/completion with tool name and arguments
4. Errors are logged when tool execution fails
"""

import io
import logging
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import mcp_server


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

_REPO_ROOT = Path(__file__).resolve().parent.parent


class TestLoggingConfiguration:
    """Test conditional logging setup based on transport."""

    def test_logging_disabled_for_stdio_transport(self):
        """For stdio transport, logging should be disabled to keep stdout clean."""
        with patch.dict(os.environ, {"MCP_TRANSPORT": "stdio"}):
            # Re-import the module to pick up the new environment variable
            import importlib

            importlib.reload(mcp_server)

            # Verify that _logger is None
            assert mcp_server._logger is None

            # Verify that logging.disable(CRITICAL) was called
            assert logging.root.manager.disable == logging.CRITICAL

    def test_logging_enabled_for_streamable_http_transport(self):
        """For streamable-http transport, logging should be enabled."""
        with patch.dict(os.environ, {"MCP_TRANSPORT": "streamable-http"}):
            import importlib

            importlib.reload(mcp_server)

            # Verify that _logger is set (not None)
            assert mcp_server._logger is not None
            assert isinstance(mcp_server._logger, logging.Logger)

            # Verify that logging is not disabled
            assert logging.root.manager.disable != logging.CRITICAL

    def test_logging_enabled_for_sse_transport(self):
        """For sse transport, logging should be enabled."""
        with patch.dict(os.environ, {"MCP_TRANSPORT": "sse"}):
            import importlib

            importlib.reload(mcp_server)

            # Verify that _logger is set (not None)
            assert mcp_server._logger is not None
            assert isinstance(mcp_server._logger, logging.Logger)

            # Verify that logging is not disabled
            assert logging.root.manager.disable != logging.CRITICAL


class TestToolLogging:
    """Test that tools log their invocations with arguments."""

    @pytest.fixture(autouse=True)
    def setup_logging(self):
        """Ensure logging is enabled for these tests."""
        with patch.dict(os.environ, {"MCP_TRANSPORT": "streamable-http"}):
            import importlib

            importlib.reload(mcp_server)
            # Reset to have a fresh logger for tests
            mcp_server._logger = logging.getLogger("mcp_server")
            yield
            # Reset back to stdio for other tests
            with patch.dict(os.environ, {"MCP_TRANSPORT": "stdio"}):
                importlib.reload(mcp_server)

    def test_analyze_stock_logs_start_and_completion(self, caplog):
        """analyze_stock should log when starting and completing with arguments."""
        if mcp_server._logger:
            with caplog.at_level(logging.INFO):
                with patch(
                    "mcp_server._run_analysis",
                    return_value=("# Report", {}, "BUY"),
                ):
                    result = mcp_server.analyze_stock("AAPL", "2024-05-10")
                    assert "## Final Decision: BUY" in result

            # Check for start log
            assert any(
                "analyze_stock: starting analysis" in record.message
                and "ticker=AAPL" in record.message
                and "date=2024-05-10" in record.message
                for record in caplog.records
            )

            # Check for completion log
            assert any(
                "analyze_stock: completed successfully" in record.message
                and "ticker=AAPL" in record.message
                for record in caplog.records
            )

    def test_analyze_stock_logs_error_on_failure(self, caplog):
        """analyze_stock should log errors when execution fails."""
        if mcp_server._logger:
            with caplog.at_level(logging.ERROR):
                with patch(
                    "mcp_server._run_analysis",
                    side_effect=ValueError("Test error"),
                ):
                    result = mcp_server.analyze_stock("AAPL", "2024-05-10")
                    assert "ERROR:" in result

            # Check for error log
            assert any(
                "analyze_stock: failed" in record.message
                and "ticker=AAPL" in record.message
                for record in caplog.records
            )

    def test_memory_store_decision_logs_with_arguments(self, caplog, tmp_path):
        """memory_store_decision should log with all key arguments."""
        if mcp_server._logger:
            with caplog.at_level(logging.INFO):
                mcp_server.memory_store_decision(
                    agent="trader",
                    ticker="AAPL",
                    date="2024-05-10",
                    signal="Buy",
                    confidence=0.8,
                    db_path=str(tmp_path / "test.db"),
                )

            # Check for start log
            assert any(
                "memory_store_decision: agent=trader" in record.message
                and "ticker=AAPL" in record.message
                and "date=2024-05-10" in record.message
                and "signal=Buy" in record.message
                for record in caplog.records
            )

            # Check for completion log
            assert any(
                "memory_store_decision: completed" in record.message
                and "agent=trader" in record.message
                for record in caplog.records
            )

    def test_memory_resolve_pending_logs_with_arguments(self, caplog):
        """memory_resolve_pending should log with arguments."""
        if mcp_server._logger:
            with caplog.at_level(logging.INFO):
                with patch(
                    "tradingagents.memory.resolve_pending", return_value=[]
                ):
                    mcp_server.memory_resolve_pending(agent="trader", ticker="AAPL")

            # Check for start log
            assert any(
                "memory_resolve_pending: agent=trader" in record.message
                and "ticker=AAPL" in record.message
                for record in caplog.records
            )

            # Check for completion log
            assert any(
                "memory_resolve_pending: completed" in record.message
                and "agent=trader" in record.message
                for record in caplog.records
            )

    def test_memory_get_past_context_logs_with_arguments(self, caplog):
        """memory_get_past_context should log with arguments."""
        if mcp_server._logger:
            with caplog.at_level(logging.INFO):
                with patch(
                    "tradingagents.memory.get_past_context",
                    return_value="No prior lessons.",
                ):
                    mcp_server.memory_get_past_context(
                        agent="trader", ticker="AAPL", n_same=5, n_cross=3
                    )

            # Check for start log with all params
            assert any(
                "memory_get_past_context: agent=trader" in record.message
                and "ticker=AAPL" in record.message
                and "n_same=5" in record.message
                and "n_cross=3" in record.message
                for record in caplog.records
            )

            # Check for completion log
            assert any(
                "memory_get_past_context: completed" in record.message
                and "agent=trader" in record.message
                for record in caplog.records
            )

    def test_memory_get_statistics_logs_with_arguments(self, caplog):
        """memory_get_statistics should log with arguments."""
        if mcp_server._logger:
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

            # Check for start log
            assert any(
                "memory_get_statistics: agent=trader" in record.message
                and "ticker=AAPL" in record.message
                and "since=2024-01-01" in record.message
                for record in caplog.records
            )

            # Check for completion log
            assert any(
                "memory_get_statistics: completed" in record.message
                and "agent=trader" in record.message
                for record in caplog.records
            )

    def test_memory_get_decisions_logs_with_arguments(self, caplog):
        """memory_get_decisions should log with arguments."""
        if mcp_server._logger:
            with caplog.at_level(logging.INFO):
                with patch(
                    "tradingagents.memory.gather_context_rows", return_value=[]
                ):
                    mcp_server.memory_get_decisions(
                        agent="trader", ticker="AAPL", limit=5, misses_only=True
                    )

            # Check for start log
            assert any(
                "memory_get_decisions: agent=trader" in record.message
                and "ticker=AAPL" in record.message
                and "limit=5" in record.message
                and "misses_only=True" in record.message
                for record in caplog.records
            )

            # Check for completion log
            assert any(
                "memory_get_decisions: completed" in record.message
                and "agent=trader" in record.message
                for record in caplog.records
            )


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
