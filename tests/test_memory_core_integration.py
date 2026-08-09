"""Tests for SQLite memory core integration with the LangGraph pipeline (issue #28,
converted to the networked memory MCP client in issue #53).

This module tests that the original trading agents pipeline writes decisions
through the memory MCP client (in addition to the legacy markdown log), that
``resolve_pending`` is called at the right time, and that connection/tool
failures now propagate (hard dependency) instead of being logged and
swallowed. The memory MCP client itself is faked here (no real MCP
subprocess/network); its own protocol-level behavior is covered by
``test_memory_mcp_client.py`` and the underlying SQLite functions it proxies
to are covered by ``test_memory_store.py``/``test_memory_resolve.py``.
"""

from unittest.mock import MagicMock, patch

import pytest

import tradingagents.graph.trading_graph as trading_graph_module
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.memory.mcp_client import MemoryMCPConnectionError, MemoryMCPToolError


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


class FakeMemoryMCPClient:
    """In-memory stand-in for tradingagents.memory.mcp_client.MemoryMCPClient.

    Tracks resolve_pending/store_decision calls (args and order) so tests can
    assert on trading_graph.py's integration with the client interface,
    without a real MCP connection. Supports injecting a connect-time failure
    (``connect_error``) and per-call failures (``resolve_side_effect``,
    ``store_side_effects`` — a list consumed in call order) to exercise the
    new hard-fail behavior.
    """

    def __init__(
        self,
        *,
        connect_error=None,
        resolve_side_effect=None,
        store_side_effects=None,
        get_past_context_return="",
        get_past_context_side_effect=None,
    ):
        self.connect_error = connect_error
        self.resolve_side_effect = resolve_side_effect
        self._store_side_effects = list(store_side_effects or [])
        self.get_past_context_return = get_past_context_return
        self.get_past_context_side_effect = get_past_context_side_effect
        self.resolve_calls = []
        self.store_calls = []
        self.get_past_context_calls = []
        self.connected = False
        self.closed = False

    def __enter__(self):
        if self.connect_error is not None:
            raise self.connect_error
        self.connected = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.closed = True
        return False

    def resolve_pending(self, agent=None, ticker=None, db_path=None):
        self.resolve_calls.append({"agent": agent, "ticker": ticker, "db_path": db_path})
        if self.resolve_side_effect is not None:
            raise self.resolve_side_effect
        return []

    def store_decision(
        self, agent, ticker, date, signal,
        confidence=None, key_drivers=None, thesis=None, db_path=None, horizon_days=None,
    ):
        call = {
            "agent": agent, "ticker": ticker, "date": date, "signal": signal,
            "confidence": confidence, "key_drivers": key_drivers, "thesis": thesis,
            "horizon_days": horizon_days,
        }
        self.store_calls.append(call)
        if self._store_side_effects:
            effect = self._store_side_effects.pop(0)
            if effect is not None:
                raise effect
        return True

    def get_past_context(self, agent, ticker, n_same=5, n_cross=3, db_path=None):
        self.get_past_context_calls.append({"agent": agent, "ticker": ticker, "db_path": db_path})
        if self.get_past_context_side_effect is not None:
            raise self.get_past_context_side_effect
        return self.get_past_context_return


def _store_call(fake, agent):
    """Look up the recorded store_decision call for a given agent."""
    return next(c for c in fake.store_calls if c["agent"] == agent)


# ---------------------------------------------------------------------------
# Core: store_decision called for three stages in _run_graph
# ---------------------------------------------------------------------------

class TestMemoryCoreStorageIntegration:
    """Test that _run_graph stores decisions for all three decision-bearing stages."""

    def _mock_graph(self, tmp_path, final_state, fake_client, research_stage="debate"):
        mock_graph = MagicMock()
        mock_graph.config = {
            "results_dir": str(tmp_path),
            "checkpoint_enabled": False,
            "research_stage": research_stage,
        }
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
        mock_graph._memory_client = fake_client
        return mock_graph

    def test_three_decisions_stored_via_client(self, tmp_path):
        """After _run_graph completes, store_decision was called once per stage."""
        final_state = _make_final_state()
        fake_client = FakeMemoryMCPClient()
        mock_graph = self._mock_graph(tmp_path, final_state, fake_client)

        result = TradingAgentsGraph._run_graph(mock_graph, "NVDA", "2026-01-10", asset_type="stock")

        mock_graph.memory_log.store_decision.assert_called_once()
        assert result[0] == final_state

        assert len(fake_client.store_calls) == 3
        agents = [c["agent"] for c in fake_client.store_calls]
        assert agents == ["research_manager", "trader", "portfolio_manager"]
        for call in fake_client.store_calls:
            assert call["ticker"] == "NVDA"
            assert call["date"] == "2026-01-10"

    def test_signal_parsed_from_each_stage_text(self, tmp_path):
        """Each call's signal is derived via parse_rating from its stage's own text."""
        final_state = _make_final_state(
            investment_plan="Rating: Buy\nStrong fundamentals.",
            trader_plan="Rating: Underweight\nNear-term caution.",
            final_decision="Rating: Sell\nExit position.",
        )
        fake_client = FakeMemoryMCPClient()
        mock_graph = self._mock_graph(tmp_path, final_state, fake_client)

        TradingAgentsGraph._run_graph(mock_graph, "NVDA", "2026-01-10", asset_type="stock")

        assert _store_call(fake_client, "research_manager")["signal"] == "Buy"
        assert _store_call(fake_client, "trader")["signal"] == "Underweight"
        assert _store_call(fake_client, "portfolio_manager")["signal"] == "Sell"

    def test_signal_fallback_to_hold_when_no_rating_found(self, tmp_path):
        """When a stage's text has no rating word, signal defaults to 'Hold'."""
        final_state = _make_final_state(
            investment_plan="Complex situation, no clear decision.",
            trader_plan="Rating: Buy\nPositive outlook.",
            final_decision="Rating: Hold\nWait for catalyst.",
        )
        fake_client = FakeMemoryMCPClient()
        mock_graph = self._mock_graph(tmp_path, final_state, fake_client)

        TradingAgentsGraph._run_graph(mock_graph, "NVDA", "2026-01-10", asset_type="stock")

        assert _store_call(fake_client, "research_manager")["signal"] == "Hold"  # default
        assert _store_call(fake_client, "trader")["signal"] == "Buy"
        assert _store_call(fake_client, "portfolio_manager")["signal"] == "Hold"

    def test_thesis_truncated_to_500_chars(self, tmp_path):
        """Each call's thesis is truncated to 500 characters."""
        long_text = "Rating: Buy\n" + "x" * 1000  # 1012 chars total
        final_state = _make_final_state(
            investment_plan=long_text, trader_plan=long_text, final_decision=long_text,
        )
        fake_client = FakeMemoryMCPClient()
        mock_graph = self._mock_graph(tmp_path, final_state, fake_client)

        TradingAgentsGraph._run_graph(mock_graph, "NVDA", "2026-01-10", asset_type="stock")

        assert len(fake_client.store_calls) == 3
        for call in fake_client.store_calls:
            assert len(call["thesis"]) <= 500

    def test_confidence_and_key_drivers_none(self, tmp_path):
        """Confidence and key_drivers are passed as None (not extracted)."""
        final_state = _make_final_state()
        fake_client = FakeMemoryMCPClient()
        mock_graph = self._mock_graph(tmp_path, final_state, fake_client)

        TradingAgentsGraph._run_graph(mock_graph, "NVDA", "2026-01-10", asset_type="stock")

        assert len(fake_client.store_calls) == 3
        for call in fake_client.store_calls:
            assert call["confidence"] is None
            assert call["key_drivers"] is None


# ---------------------------------------------------------------------------
# research_stage="none" (#79): no fabricated research_manager decision
# ---------------------------------------------------------------------------

class TestResearchStageMemoryGuard:
    """Test that the research_manager store_decision call is skipped when
    research_stage == "none" (no Bull/Bear/Research Manager node ran, so
    investment_plan is empty by design — not a genuine thin plan), and still
    happens normally in "debate" mode."""

    def _mock_graph(self, tmp_path, final_state, fake_client, research_stage):
        mock_graph = MagicMock()
        mock_graph.config = {
            "results_dir": str(tmp_path),
            "checkpoint_enabled": False,
            "research_stage": research_stage,
        }
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
        mock_graph._memory_client = fake_client
        return mock_graph

    def test_none_mode_skips_research_manager_store(self, tmp_path):
        """research_stage='none': investment_plan is empty by design and no
        research_manager decision is stored — only trader and portfolio_manager."""
        final_state = _make_final_state(investment_plan="")
        fake_client = FakeMemoryMCPClient()
        mock_graph = self._mock_graph(tmp_path, final_state, fake_client, "none")

        TradingAgentsGraph._run_graph(mock_graph, "NVDA", "2026-01-10", asset_type="stock")

        agents = [c["agent"] for c in fake_client.store_calls]
        assert agents == ["trader", "portfolio_manager"]
        assert not any(c["agent"] == "research_manager" for c in fake_client.store_calls)

    def test_none_mode_defaults_when_research_stage_key_absent(self, tmp_path):
        """If research_stage is missing from config entirely, it defaults to
        "none" (matching the production default in default_config.py and the
        GraphSetup wiring), so the research_manager store is still skipped."""
        final_state = _make_final_state(investment_plan="")
        fake_client = FakeMemoryMCPClient()
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
        mock_graph._memory_client = fake_client

        TradingAgentsGraph._run_graph(mock_graph, "NVDA", "2026-01-10", asset_type="stock")

        agents = [c["agent"] for c in fake_client.store_calls]
        assert "research_manager" not in agents

    def test_debate_mode_still_stores_research_manager(self, tmp_path):
        """research_stage='debate': the research_manager decision is still
        stored, with a signal parsed from a genuine (non-empty) investment_plan."""
        final_state = _make_final_state(investment_plan="Rating: Buy\nStrong fundamentals.")
        fake_client = FakeMemoryMCPClient()
        mock_graph = self._mock_graph(tmp_path, final_state, fake_client, "debate")

        TradingAgentsGraph._run_graph(mock_graph, "NVDA", "2026-01-10", asset_type="stock")

        agents = [c["agent"] for c in fake_client.store_calls]
        assert agents == ["research_manager", "trader", "portfolio_manager"]
        assert _store_call(fake_client, "research_manager")["signal"] == "Buy"
        assert _store_call(fake_client, "research_manager")["thesis"] == final_state["investment_plan"]

    def test_researcher_mode_stores_under_researcher_key(self, tmp_path):
        """research_stage='researcher' (#85): the decision is stored under the
        'researcher' agent key — not 'research_manager' — with no continuity
        implied between the two modes' rows."""
        final_state = _make_final_state(investment_plan="Rating: Buy\nResearcher brief.")
        fake_client = FakeMemoryMCPClient()
        mock_graph = self._mock_graph(tmp_path, final_state, fake_client, "researcher")

        TradingAgentsGraph._run_graph(mock_graph, "NVDA", "2026-01-10", asset_type="stock")

        agents = [c["agent"] for c in fake_client.store_calls]
        assert agents == ["researcher", "trader", "portfolio_manager"]
        assert "research_manager" not in agents
        assert _store_call(fake_client, "researcher")["signal"] == "Buy"
        assert _store_call(fake_client, "researcher")["thesis"] == final_state["investment_plan"]


# ---------------------------------------------------------------------------
# Resolve path: resolve_pending called before graph runs, via the MCP client
# ---------------------------------------------------------------------------

class TestMemoryCoreResolveIntegration:
    """Test that propagate() calls the memory MCP client's resolve_pending
    at the right time, and connects to it exactly once per run."""

    def test_resolve_pending_called_before_graph(self, monkeypatch):
        """resolve_pending is called in propagate() before _run_graph."""
        mock_graph = MagicMock()
        mock_graph.config = {"checkpoint_enabled": False}

        call_order = []
        fake_client = FakeMemoryMCPClient()
        original_resolve = fake_client.resolve_pending

        def tracked_resolve_pending(*args, **kwargs):
            call_order.append("resolve_pending")
            return original_resolve(*args, **kwargs)

        fake_client.resolve_pending = tracked_resolve_pending

        def tracked_run_graph(*args, **kwargs):
            call_order.append("_run_graph")
            return ({"final_trade_decision": "Rating: Buy"}, "Buy")

        monkeypatch.setattr(trading_graph_module, "MemoryMCPClient", lambda *a, **kw: fake_client)
        mock_graph._run_graph = tracked_run_graph
        with patch.object(TradingAgentsGraph, "_resolve_pending_entries"):
            TradingAgentsGraph.propagate(mock_graph, "NVDA", "2026-01-10")

        assert call_order == ["resolve_pending", "_run_graph"]

    def test_resolve_pending_called_with_ticker_only(self, monkeypatch):
        """resolve_pending is called with ticker=company_name, no agent filter."""
        mock_graph = MagicMock()
        mock_graph.config = {"checkpoint_enabled": False}
        mock_graph._run_graph = MagicMock(return_value=({"final_trade_decision": "Buy"}, "Buy"))

        fake_client = FakeMemoryMCPClient()
        monkeypatch.setattr(trading_graph_module, "MemoryMCPClient", lambda *a, **kw: fake_client)
        with patch.object(TradingAgentsGraph, "_resolve_pending_entries"):
            TradingAgentsGraph.propagate(mock_graph, "NVDA", "2026-01-10")

        assert fake_client.resolve_calls == [{"agent": None, "ticker": "NVDA", "db_path": None}]

    def test_memory_client_connected_once_per_run(self, monkeypatch):
        """The client is used as a context manager exactly once per propagate() call."""
        mock_graph = MagicMock()
        mock_graph.config = {"checkpoint_enabled": False}
        mock_graph._run_graph = MagicMock(return_value=({"final_trade_decision": "Buy"}, "Buy"))

        fake_client = FakeMemoryMCPClient()
        monkeypatch.setattr(trading_graph_module, "MemoryMCPClient", lambda *a, **kw: fake_client)
        with patch.object(TradingAgentsGraph, "_resolve_pending_entries"):
            TradingAgentsGraph.propagate(mock_graph, "NVDA", "2026-01-10")

        assert fake_client.connected is True
        assert fake_client.closed is True
        # self._memory_client is cleared after the run.
        assert mock_graph._memory_client is None


# ---------------------------------------------------------------------------
# Hard-fail behavior: unreachable server / failing tool calls now propagate
# ---------------------------------------------------------------------------

class TestMemoryCoreHardFailure:
    """Test the issue #53 behavior change: memory MCP failures fail the run
    instead of being logged and swallowed."""

    def test_unreachable_memory_server_fails_run(self, monkeypatch):
        """If the memory MCP server is unreachable, propagate() raises."""
        mock_graph = MagicMock()
        mock_graph.config = {"checkpoint_enabled": False}
        mock_graph._run_graph = MagicMock(return_value=({"final_trade_decision": "Buy"}, "Buy"))

        fake_client = FakeMemoryMCPClient(
            connect_error=MemoryMCPConnectionError("server unreachable")
        )
        monkeypatch.setattr(trading_graph_module, "MemoryMCPClient", lambda *a, **kw: fake_client)
        with patch.object(TradingAgentsGraph, "_resolve_pending_entries"):
            with pytest.raises(MemoryMCPConnectionError):
                TradingAgentsGraph.propagate(mock_graph, "NVDA", "2026-01-10")

        # The graph never ran — connection failure aborts before that.
        mock_graph._run_graph.assert_not_called()

    def test_resolve_pending_tool_error_propagates(self, monkeypatch):
        """A tool-level error from resolve_pending is not swallowed."""
        mock_graph = MagicMock()
        mock_graph.config = {"checkpoint_enabled": False}
        mock_graph._run_graph = MagicMock(return_value=({"final_trade_decision": "Buy"}, "Buy"))

        fake_client = FakeMemoryMCPClient(
            resolve_side_effect=MemoryMCPToolError("db unavailable")
        )
        monkeypatch.setattr(trading_graph_module, "MemoryMCPClient", lambda *a, **kw: fake_client)
        with patch.object(TradingAgentsGraph, "_resolve_pending_entries"):
            with pytest.raises(MemoryMCPToolError):
                TradingAgentsGraph.propagate(mock_graph, "NVDA", "2026-01-10")

        mock_graph._run_graph.assert_not_called()

    def test_research_manager_store_failure_aborts_run(self, tmp_path):
        """If the research_manager store_decision call fails, the exception
        propagates out of _run_graph rather than being logged and continued."""
        final_state = _make_final_state()
        fake_client = FakeMemoryMCPClient(
            store_side_effects=[MemoryMCPToolError("DB I/O error")]
        )
        mock_graph = MagicMock()
        mock_graph.config = {
            "results_dir": str(tmp_path),
            "checkpoint_enabled": False,
            "research_stage": "debate",
        }
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
        mock_graph._memory_client = fake_client

        with pytest.raises(MemoryMCPToolError):
            TradingAgentsGraph._run_graph(mock_graph, "NVDA", "2026-01-10", asset_type="stock")

        # Only the failing (research_manager) call was attempted — trader and
        # portfolio_manager never ran because the exception propagated.
        assert [c["agent"] for c in fake_client.store_calls] == ["research_manager"]

    def test_trader_store_failure_aborts_run(self, tmp_path):
        """If the trader store_decision call fails, research_manager's write
        already happened but trader/portfolio_manager do not complete."""
        final_state = _make_final_state()
        fake_client = FakeMemoryMCPClient(
            store_side_effects=[None, MemoryMCPToolError("DB I/O error")]
        )
        mock_graph = MagicMock()
        mock_graph.config = {
            "results_dir": str(tmp_path),
            "checkpoint_enabled": False,
            "research_stage": "debate",
        }
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
        mock_graph._memory_client = fake_client

        with pytest.raises(MemoryMCPToolError):
            TradingAgentsGraph._run_graph(mock_graph, "NVDA", "2026-01-10", asset_type="stock")

        assert [c["agent"] for c in fake_client.store_calls] == ["research_manager", "trader"]

    def test_pm_store_failure_aborts_run(self, tmp_path):
        """If the portfolio_manager store_decision call fails, the exception
        still propagates even though it's the last of the three calls."""
        final_state = _make_final_state()
        fake_client = FakeMemoryMCPClient(
            store_side_effects=[None, None, MemoryMCPToolError("DB I/O error")]
        )
        mock_graph = MagicMock()
        mock_graph.config = {
            "results_dir": str(tmp_path),
            "checkpoint_enabled": False,
            "research_stage": "debate",
        }
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
        mock_graph._memory_client = fake_client

        with pytest.raises(MemoryMCPToolError):
            TradingAgentsGraph._run_graph(mock_graph, "NVDA", "2026-01-10", asset_type="stock")

        assert [c["agent"] for c in fake_client.store_calls] == [
            "research_manager", "trader", "portfolio_manager",
        ]


# ---------------------------------------------------------------------------
# Legacy log compatibility: TradingMemoryLog behavior unchanged
# ---------------------------------------------------------------------------

class TestLegacyLogCompatibility:
    """Test that legacy TradingMemoryLog write path is completely unchanged."""

    def test_legacy_store_decision_still_called(self, tmp_path):
        """_run_graph still calls memory_log.store_decision after the SQLite
        memory-core writes (now via the memory MCP client)."""
        mock_graph = MagicMock()
        mock_graph.config = {
            "results_dir": str(tmp_path),
            "checkpoint_enabled": False,
            "research_stage": "debate",
        }
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
        mock_graph._memory_client = FakeMemoryMCPClient()

        TradingAgentsGraph._run_graph(mock_graph, "NVDA", "2026-01-10", asset_type="stock")

        mock_graph.memory_log.store_decision.assert_called_once_with(
            ticker="NVDA",
            trade_date="2026-01-10",
            final_trade_decision=final_state["final_trade_decision"],
        )

    def test_legacy_resolve_pending_entries_still_called(self, monkeypatch):
        """propagate() still calls _resolve_pending_entries before the graph."""
        mock_graph = MagicMock()
        mock_graph.config = {"checkpoint_enabled": False}
        mock_graph._run_graph = MagicMock(return_value=({"final_trade_decision": "Buy"}, "Buy"))
        # Spy on the _resolve_pending_entries method of the mock instance.
        mock_graph._resolve_pending_entries = MagicMock()

        fake_client = FakeMemoryMCPClient()
        monkeypatch.setattr(trading_graph_module, "MemoryMCPClient", lambda *a, **kw: fake_client)
        TradingAgentsGraph.propagate(mock_graph, "NVDA", "2026-01-10")

        # Legacy _resolve_pending_entries must be called on the graph instance.
        mock_graph._resolve_pending_entries.assert_called_once_with("NVDA")
