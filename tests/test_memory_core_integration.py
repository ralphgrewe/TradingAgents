"""Tests for SQLite memory core integration with the LangGraph pipeline (issue #28).

This module tests that the original trading agents pipeline writes decisions
to the SQLite memory core in addition to the legacy markdown log, and that
resolve_pending is called at the right time. The write path is write-only (no
prompt injection from the store yet).
"""

from unittest.mock import MagicMock, patch

import pytest

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.memory import store as memory_store
from tradingagents.memory import resolve as memory_resolve


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

def _make_final_state(
    investment_plan="Rating: Buy\nResearch plan text.",
    trader_plan="Rating: Overweight\nTrader plan text.",
    final_decision="Rating: Hold\nPortfolio manager decision.",
):
    """Build a minimal final_state dict for testing."""
    return {
        "company_of_interest": "NVDA",
        "trade_date": "2026-01-10",
        "asset_type": "stock",
        "investment_plan": investment_plan,
        "trader_investment_plan": trader_plan,
        "final_trade_decision": final_decision,
        "market_report": "Market report",
        "sentiment_report": "Sentiment report",
        "news_report": "News report",
        "perplexity_news_report": "",
        "fundamentals_report": "Fundamentals report",
        "investment_debate_state": {
            "bull_history": "", "bear_history": "", "history": "",
            "current_response": "", "judge_decision": "", "count": 1,
        },
        "risk_debate_state": {
            "aggressive_history": "", "conservative_history": "",
            "neutral_history": "", "history": "", "judge_decision": "",
            "current_aggressive_response": "", "current_conservative_response": "",
            "current_neutral_response": "", "count": 1, "latest_speaker": "",
        },
        "messages": [],
    }


# ---------------------------------------------------------------------------
# Core: store_decision called for three stages in _run_graph
# ---------------------------------------------------------------------------

class TestMemoryCoreStorageIntegration:
    """Test that _run_graph stores decisions for all three decision-bearing stages."""

    def test_three_decisions_stored_in_sqlite(self, tmp_path):
        """After _run_graph completes, three rows exist in the SQLite DB."""
        # Mock the graph and its dependencies to isolate the store path.
        mock_graph = MagicMock()
        mock_graph.config = {"results_dir": str(tmp_path), "checkpoint_enabled": False}
        mock_graph.memory_log = MagicMock()
        mock_graph.log_states_dict = {}
        mock_graph.debug = False
        mock_graph.signal_processor = MagicMock()
        mock_graph.signal_processor.process_signal.return_value = "Buy"
        mock_graph.propagator = MagicMock()
        mock_graph.propagator.create_initial_state.return_value = _make_final_state()
        mock_graph.propagator.get_graph_args.return_value = {}
        mock_graph.graph = MagicMock()
        mock_graph.graph.invoke.return_value = _make_final_state()

        final_state = _make_final_state()

        # Call the real _run_graph with a temporary DB path.
        db_path = str(tmp_path / "test_memory.db")
        with patch.dict("os.environ", {"TRADINGAGENTS_MEMORY_DB_PATH": db_path}):
            result = TradingAgentsGraph._run_graph(
                mock_graph, "NVDA", "2026-01-10", asset_type="stock"
            )

        # Verify the graph was called correctly and returned final_state.
        mock_graph.memory_log.store_decision.assert_called_once()
        assert result[0] == final_state

        # Verify three rows exist in the SQLite DB.
        conn = memory_store.get_connection(db_path)
        try:
            rows = conn.execute("SELECT agent, signal FROM decisions ORDER BY id").fetchall()
            assert len(rows) == 3
            agents = [row["agent"] for row in rows]
            assert "research_manager" in agents
            assert "trader" in agents
            assert "portfolio_manager" in agents
        finally:
            conn.close()

    def test_signal_parsed_from_each_stage_text(self, tmp_path):
        """Each row's signal is derived via parse_rating from its stage's own text."""
        final_state = _make_final_state(
            investment_plan="Rating: Buy\nStrong fundamentals.",
            trader_plan="Rating: Underweight\nNear-term caution.",
            final_decision="Rating: Sell\nExit position.",
        )

        mock_graph = MagicMock()
        mock_graph.config = {"results_dir": str(tmp_path), "checkpoint_enabled": False}
        mock_graph.memory_log = MagicMock()
        mock_graph.log_states_dict = {}
        mock_graph.debug = False
        mock_graph.signal_processor = MagicMock()
        mock_graph.signal_processor.process_signal.return_value = "Sell"
        mock_graph.propagator = MagicMock()
        mock_graph.propagator.create_initial_state.return_value = final_state
        mock_graph.propagator.get_graph_args.return_value = {}
        mock_graph.graph = MagicMock()
        mock_graph.graph.invoke.return_value = final_state

        db_path = str(tmp_path / "test_memory.db")
        with patch.dict("os.environ", {"TRADINGAGENTS_MEMORY_DB_PATH": db_path}):
            TradingAgentsGraph._run_graph(
                mock_graph, "NVDA", "2026-01-10", asset_type="stock"
            )

        # Verify each stage's signal matches its own text, not final_decision.
        conn = memory_store.get_connection(db_path)
        try:
            rows = {
                row["agent"]: row["signal"]
                for row in conn.execute("SELECT agent, signal FROM decisions").fetchall()
            }
            assert rows["research_manager"] == "Buy"
            assert rows["trader"] == "Underweight"
            assert rows["portfolio_manager"] == "Sell"
        finally:
            conn.close()

    def test_signal_fallback_to_hold_when_no_rating_found(self, tmp_path):
        """When a stage's text has no rating word, signal defaults to 'Hold'."""
        # investment_plan has no rating word, so should default to "Hold".
        final_state = _make_final_state(
            investment_plan="Complex situation, no clear decision.",
            trader_plan="Rating: Buy\nPositive outlook.",
            final_decision="Rating: Hold\nWait for catalyst.",
        )

        mock_graph = MagicMock()
        mock_graph.config = {"results_dir": str(tmp_path), "checkpoint_enabled": False}
        mock_graph.memory_log = MagicMock()
        mock_graph.log_states_dict = {}
        mock_graph.debug = False
        mock_graph.signal_processor = MagicMock()
        mock_graph.signal_processor.process_signal.return_value = "Hold"
        mock_graph.propagator = MagicMock()
        mock_graph.propagator.create_initial_state.return_value = final_state
        mock_graph.propagator.get_graph_args.return_value = {}
        mock_graph.graph = MagicMock()
        mock_graph.graph.invoke.return_value = final_state

        db_path = str(tmp_path / "test_memory.db")
        with patch.dict("os.environ", {"TRADINGAGENTS_MEMORY_DB_PATH": db_path}):
            TradingAgentsGraph._run_graph(
                mock_graph, "NVDA", "2026-01-10", asset_type="stock"
            )

        conn = memory_store.get_connection(db_path)
        try:
            rows = {
                row["agent"]: row["signal"]
                for row in conn.execute("SELECT agent, signal FROM decisions").fetchall()
            }
            assert rows["research_manager"] == "Hold"  # default
            assert rows["trader"] == "Buy"
            assert rows["portfolio_manager"] == "Hold"
        finally:
            conn.close()

    def test_thesis_truncated_to_500_chars(self, tmp_path):
        """Each row's thesis is truncated to 500 characters."""
        long_text = "Rating: Buy\n" + "x" * 1000  # 1012 chars total
        final_state = _make_final_state(
            investment_plan=long_text,
            trader_plan=long_text,
            final_decision=long_text,
        )

        mock_graph = MagicMock()
        mock_graph.config = {"results_dir": str(tmp_path), "checkpoint_enabled": False}
        mock_graph.memory_log = MagicMock()
        mock_graph.log_states_dict = {}
        mock_graph.debug = False
        mock_graph.signal_processor = MagicMock()
        mock_graph.signal_processor.process_signal.return_value = "Buy"
        mock_graph.propagator = MagicMock()
        mock_graph.propagator.create_initial_state.return_value = final_state
        mock_graph.propagator.get_graph_args.return_value = {}
        mock_graph.graph = MagicMock()
        mock_graph.graph.invoke.return_value = final_state

        db_path = str(tmp_path / "test_memory.db")
        with patch.dict("os.environ", {"TRADINGAGENTS_MEMORY_DB_PATH": db_path}):
            TradingAgentsGraph._run_graph(
                mock_graph, "NVDA", "2026-01-10", asset_type="stock"
            )

        conn = memory_store.get_connection(db_path)
        try:
            rows = conn.execute("SELECT thesis FROM decisions").fetchall()
            for row in rows:
                assert len(row["thesis"]) <= 500
        finally:
            conn.close()

    def test_confidence_and_key_drivers_none(self, tmp_path):
        """Confidence and key_drivers are stored as None (not extracted)."""
        final_state = _make_final_state()

        mock_graph = MagicMock()
        mock_graph.config = {"results_dir": str(tmp_path), "checkpoint_enabled": False}
        mock_graph.memory_log = MagicMock()
        mock_graph.log_states_dict = {}
        mock_graph.debug = False
        mock_graph.signal_processor = MagicMock()
        mock_graph.signal_processor.process_signal.return_value = "Buy"
        mock_graph.propagator = MagicMock()
        mock_graph.propagator.create_initial_state.return_value = final_state
        mock_graph.propagator.get_graph_args.return_value = {}
        mock_graph.graph = MagicMock()
        mock_graph.graph.invoke.return_value = final_state

        db_path = str(tmp_path / "test_memory.db")
        with patch.dict("os.environ", {"TRADINGAGENTS_MEMORY_DB_PATH": db_path}):
            TradingAgentsGraph._run_graph(
                mock_graph, "NVDA", "2026-01-10", asset_type="stock"
            )

        conn = memory_store.get_connection(db_path)
        try:
            rows = conn.execute("SELECT confidence, key_drivers FROM decisions").fetchall()
            for row in rows:
                assert row["confidence"] is None
                assert row["key_drivers"] is None
        finally:
            conn.close()

    def test_idempotency_guard_on_store(self, tmp_path):
        """Calling _run_graph twice with same ticker+date stores only one row per agent."""
        final_state = _make_final_state()

        mock_graph = MagicMock()
        mock_graph.config = {"results_dir": str(tmp_path), "checkpoint_enabled": False}
        mock_graph.memory_log = MagicMock()
        mock_graph.log_states_dict = {}
        mock_graph.debug = False
        mock_graph.signal_processor = MagicMock()
        mock_graph.signal_processor.process_signal.return_value = "Buy"
        mock_graph.propagator = MagicMock()
        mock_graph.propagator.create_initial_state.return_value = final_state
        mock_graph.propagator.get_graph_args.return_value = {}
        mock_graph.graph = MagicMock()
        mock_graph.graph.invoke.return_value = final_state

        db_path = str(tmp_path / "test_memory.db")

        with patch.dict("os.environ", {"TRADINGAGENTS_MEMORY_DB_PATH": db_path}):
            # First call
            TradingAgentsGraph._run_graph(
                mock_graph, "NVDA", "2026-01-10", asset_type="stock"
            )
            # Second call with same ticker and date
            TradingAgentsGraph._run_graph(
                mock_graph, "NVDA", "2026-01-10", asset_type="stock"
            )

        # Only 3 rows total (one per agent), not 6.
        conn = memory_store.get_connection(db_path)
        try:
            rows = conn.execute("SELECT COUNT(*) as cnt FROM decisions").fetchone()
            assert rows["cnt"] == 3
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Resolve path: resolve_pending called before graph runs
# ---------------------------------------------------------------------------

class TestMemoryCoreResolveIntegration:
    """Test that propagate() calls resolve_pending at the right time."""

    def test_resolve_pending_called_before_graph(self):
        """resolve_pending is called in propagate() before the graph runs."""
        mock_graph = MagicMock()
        mock_graph.config = {"checkpoint_enabled": False}

        call_order = []

        # Mock resolve_pending to track when it's called.
        original_resolve = memory_resolve.resolve_pending

        def tracked_resolve_pending(*args, **kwargs):
            call_order.append("resolve_pending")
            return original_resolve(*args, **kwargs)

        # Mock _run_graph to track when it's called.
        def tracked_run_graph(*args, **kwargs):
            call_order.append("_run_graph")
            return ({"final_trade_decision": "Rating: Buy"}, "Buy")

        with patch.object(memory_resolve, "resolve_pending", side_effect=tracked_resolve_pending):
            mock_graph._run_graph = tracked_run_graph
            # Also mock _resolve_pending_entries to avoid legacy log operations.
            with patch.object(TradingAgentsGraph, "_resolve_pending_entries"):
                TradingAgentsGraph.propagate(mock_graph, "NVDA", "2026-01-10")

        # resolve_pending must be called before _run_graph.
        assert call_order == ["resolve_pending", "_run_graph"]

    def test_resolve_pending_called_with_ticker_only(self):
        """resolve_pending is called with ticker=company_name, no agent filter."""
        mock_graph = MagicMock()
        mock_graph.config = {"checkpoint_enabled": False}
        mock_graph._run_graph = MagicMock(return_value=({"final_trade_decision": "Buy"}, "Buy"))

        with patch.object(memory_resolve, "resolve_pending") as mock_resolve:
            with patch.object(TradingAgentsGraph, "_resolve_pending_entries"):
                TradingAgentsGraph.propagate(mock_graph, "NVDA", "2026-01-10")

        # resolve_pending should be called with ticker="NVDA" (no agent filter).
        mock_resolve.assert_called_once_with(ticker="NVDA")

    def test_resolve_pending_failure_does_not_abort_pipeline(self):
        """If resolve_pending raises, the pipeline continues with a warning."""
        mock_graph = MagicMock()
        mock_graph.config = {"checkpoint_enabled": False}
        mock_graph._run_graph = MagicMock(return_value=({"final_trade_decision": "Buy"}, "Buy"))

        with patch.object(memory_resolve, "resolve_pending", side_effect=Exception("DB error")):
            with patch.object(TradingAgentsGraph, "_resolve_pending_entries"):
                # Should not raise — instead logs a warning and continues.
                result = TradingAgentsGraph.propagate(mock_graph, "NVDA", "2026-01-10")
                assert result == ({"final_trade_decision": "Buy"}, "Buy")


# ---------------------------------------------------------------------------
# Error handling: failures in store_decision do not break the pipeline
# ---------------------------------------------------------------------------

class TestMemoryCoreErrorHandling:
    """Test that errors in memory core writes follow the warn-and-continue pattern."""

    def test_research_manager_store_failure_continues(self, tmp_path):
        """If research_manager store_decision fails, the pipeline continues."""
        final_state = _make_final_state()

        mock_graph = MagicMock()
        mock_graph.config = {"results_dir": str(tmp_path), "checkpoint_enabled": False}
        mock_graph.memory_log = MagicMock()
        mock_graph.log_states_dict = {}
        mock_graph.debug = False
        mock_graph.signal_processor = MagicMock()
        mock_graph.signal_processor.process_signal.return_value = "Buy"
        mock_graph.propagator = MagicMock()
        mock_graph.propagator.create_initial_state.return_value = final_state
        mock_graph.propagator.get_graph_args.return_value = {}
        mock_graph.graph = MagicMock()
        mock_graph.graph.invoke.return_value = final_state

        db_path = str(tmp_path / "test_memory.db")

        # Mock store_decision to fail on the first call (research_manager).
        call_count = [0]
        original_store = memory_store.store_decision

        def failing_store(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:  # First call (research_manager)
                raise Exception("DB I/O error")
            # Otherwise call the real store_decision.
            return original_store(*args, **kwargs)

        with patch.dict("os.environ", {"TRADINGAGENTS_MEMORY_DB_PATH": db_path}):
            with patch.object(memory_store, "store_decision", side_effect=failing_store):
                # Should not raise; instead logs and continues.
                result = TradingAgentsGraph._run_graph(
                    mock_graph, "NVDA", "2026-01-10", asset_type="stock"
                )
                assert result[0] == final_state

            # Trader and Portfolio Manager rows were still stored.
            conn = memory_store.get_connection(db_path)
            try:
                rows = conn.execute(
                    "SELECT agent FROM decisions ORDER BY id"
                ).fetchall()
                agents = [row["agent"] for row in rows]
                # Only trader and portfolio_manager (research_manager failed).
                assert agents == ["trader", "portfolio_manager"]
            finally:
                conn.close()

    def test_trader_store_failure_continues(self, tmp_path):
        """If trader store_decision fails, the pipeline continues."""
        final_state = _make_final_state()

        mock_graph = MagicMock()
        mock_graph.config = {"results_dir": str(tmp_path), "checkpoint_enabled": False}
        mock_graph.memory_log = MagicMock()
        mock_graph.log_states_dict = {}
        mock_graph.debug = False
        mock_graph.signal_processor = MagicMock()
        mock_graph.signal_processor.process_signal.return_value = "Buy"
        mock_graph.propagator = MagicMock()
        mock_graph.propagator.create_initial_state.return_value = final_state
        mock_graph.propagator.get_graph_args.return_value = {}
        mock_graph.graph = MagicMock()
        mock_graph.graph.invoke.return_value = final_state

        db_path = str(tmp_path / "test_memory.db")

        call_count = [0]
        original_store = memory_store.store_decision

        def failing_store(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:  # Second call (trader)
                raise Exception("DB I/O error")
            return original_store(*args, **kwargs)

        with patch.dict("os.environ", {"TRADINGAGENTS_MEMORY_DB_PATH": db_path}):
            with patch.object(memory_store, "store_decision", side_effect=failing_store):
                result = TradingAgentsGraph._run_graph(
                    mock_graph, "NVDA", "2026-01-10", asset_type="stock"
                )
                assert result[0] == final_state

            conn = memory_store.get_connection(db_path)
            try:
                rows = conn.execute(
                    "SELECT agent FROM decisions ORDER BY id"
                ).fetchall()
                agents = [row["agent"] for row in rows]
                # research_manager and portfolio_manager (trader failed).
                assert agents == ["research_manager", "portfolio_manager"]
            finally:
                conn.close()

    def test_pm_store_failure_continues(self, tmp_path):
        """If portfolio_manager store_decision fails, the pipeline continues."""
        final_state = _make_final_state()

        mock_graph = MagicMock()
        mock_graph.config = {"results_dir": str(tmp_path), "checkpoint_enabled": False}
        mock_graph.memory_log = MagicMock()
        mock_graph.log_states_dict = {}
        mock_graph.debug = False
        mock_graph.signal_processor = MagicMock()
        mock_graph.signal_processor.process_signal.return_value = "Buy"
        mock_graph.propagator = MagicMock()
        mock_graph.propagator.create_initial_state.return_value = final_state
        mock_graph.propagator.get_graph_args.return_value = {}
        mock_graph.graph = MagicMock()
        mock_graph.graph.invoke.return_value = final_state

        db_path = str(tmp_path / "test_memory.db")

        call_count = [0]
        original_store = memory_store.store_decision

        def failing_store(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 3:  # Third call (portfolio_manager)
                raise Exception("DB I/O error")
            return original_store(*args, **kwargs)

        with patch.dict("os.environ", {"TRADINGAGENTS_MEMORY_DB_PATH": db_path}):
            with patch.object(memory_store, "store_decision", side_effect=failing_store):
                result = TradingAgentsGraph._run_graph(
                    mock_graph, "NVDA", "2026-01-10", asset_type="stock"
                )
                assert result[0] == final_state

            conn = memory_store.get_connection(db_path)
            try:
                rows = conn.execute(
                    "SELECT agent FROM decisions ORDER BY id"
                ).fetchall()
                agents = [row["agent"] for row in rows]
                # research_manager and trader (portfolio_manager failed).
                assert agents == ["research_manager", "trader"]
            finally:
                conn.close()


# ---------------------------------------------------------------------------
# Legacy log compatibility: TradingMemoryLog behavior unchanged
# ---------------------------------------------------------------------------

class TestLegacyLogCompatibility:
    """Test that legacy TradingMemoryLog write path is completely unchanged."""

    def test_legacy_store_decision_still_called(self, tmp_path):
        """_run_graph still calls memory_log.store_decision after SQLite writes."""
        mock_graph = MagicMock()
        mock_graph.config = {"results_dir": str(tmp_path), "checkpoint_enabled": False}
        mock_graph.memory_log = MagicMock()
        mock_graph.log_states_dict = {}
        mock_graph.debug = False
        mock_graph.signal_processor = MagicMock()
        mock_graph.signal_processor.process_signal.return_value = "Buy"
        mock_graph.propagator = MagicMock()
        mock_graph.graph = MagicMock()
        final_state = _make_final_state()
        mock_graph.propagator.create_initial_state.return_value = final_state
        mock_graph.propagator.get_graph_args.return_value = {}
        mock_graph.graph.invoke.return_value = final_state

        final_state = _make_final_state()
        db_path = str(tmp_path / "test_memory.db")

        with patch.dict("os.environ", {"TRADINGAGENTS_MEMORY_DB_PATH": db_path}):
            TradingAgentsGraph._run_graph(
                mock_graph, "NVDA", "2026-01-10", asset_type="stock"
            )

        # Legacy log's store_decision must be called exactly once.
        mock_graph.memory_log.store_decision.assert_called_once_with(
            ticker="NVDA",
            trade_date="2026-01-10",
            final_trade_decision=final_state["final_trade_decision"],
        )

    def test_legacy_resolve_pending_entries_still_called(self):
        """propagate() still calls _resolve_pending_entries before the graph."""
        mock_graph = MagicMock()
        mock_graph.config = {"checkpoint_enabled": False}
        mock_graph._run_graph = MagicMock(return_value=({"final_trade_decision": "Buy"}, "Buy"))
        # Spy on the _resolve_pending_entries method of the mock instance.
        mock_graph._resolve_pending_entries = MagicMock()

        with patch.object(memory_resolve, "resolve_pending"):
            TradingAgentsGraph.propagate(mock_graph, "NVDA", "2026-01-10")

        # Legacy _resolve_pending_entries must be called on the graph instance.
        mock_graph._resolve_pending_entries.assert_called_once_with("NVDA")
