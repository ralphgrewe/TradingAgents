"""Tests for risk_stage config key and graph branching logic (issue #119).

Mirrors tests/test_research_stage.py, which covers the structurally identical
research_stage bypass added in #79.
"""

from __future__ import annotations

import importlib
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

import tradingagents.default_config as default_config_module
import tradingagents.graph.setup as graph_setup_module
from tradingagents.graph.conditional_logic import ConditionalLogic
from tradingagents.graph.setup import GraphSetup
from tradingagents.graph.trading_graph import TradingAgentsGraph

pytestmark = pytest.mark.unit


def _reload_with_env(monkeypatch, **overrides):
    """Set/clear env vars then reload default_config to re-evaluate DEFAULT_CONFIG."""
    for key in list(default_config_module._ENV_OVERRIDES):
        monkeypatch.delenv(key, raising=False)
    for key, val in overrides.items():
        monkeypatch.setenv(key, val)
    return importlib.reload(default_config_module)


class TestRiskStageConfig:
    """Tests for risk_stage config key and env override."""

    def test_risk_stage_defaults_to_debate(self, monkeypatch):
        """The risk_stage config key should default to 'debate' (today's behavior)."""
        dc = _reload_with_env(monkeypatch)
        assert dc.DEFAULT_CONFIG["risk_stage"] == "debate"

    def test_risk_stage_env_override(self, monkeypatch):
        """TRADINGAGENTS_RISK_STAGE env var should override the default."""
        dc = _reload_with_env(monkeypatch, TRADINGAGENTS_RISK_STAGE="none")
        assert dc.DEFAULT_CONFIG["risk_stage"] == "none"

    def test_risk_stage_empty_env_passthrough(self, monkeypatch):
        """Empty TRADINGAGENTS_RISK_STAGE should not clobber the default."""
        dc = _reload_with_env(monkeypatch, TRADINGAGENTS_RISK_STAGE="")
        assert dc.DEFAULT_CONFIG["risk_stage"] == "debate"

    def test_risk_stage_accepts_debate_string(self, monkeypatch):
        """TRADINGAGENTS_RISK_STAGE='debate' should be accepted (explicit default)."""
        dc = _reload_with_env(monkeypatch, TRADINGAGENTS_RISK_STAGE="debate")
        assert dc.DEFAULT_CONFIG["risk_stage"] == "debate"


class TestGraphSetupWithRiskStage:
    """Tests for GraphSetup behavior with different risk_stage values."""

    def _mock_llm(self):
        mock = MagicMock()
        mock.invoke = MagicMock(return_value=MagicMock(content="Test response"))
        return mock

    def _mock_tool_nodes(self):
        return {"social": MagicMock()}

    def test_graph_setup_with_risk_stage_none_omits_debate_nodes(self):
        """With risk_stage='none', Aggressive/Conservative/Neutral nodes should not be added."""
        quick_llm = self._mock_llm()
        deep_llm = self._mock_llm()
        tool_nodes = self._mock_tool_nodes()
        conditional_logic = ConditionalLogic(max_debate_rounds=1, max_risk_discuss_rounds=1)

        setup = GraphSetup(
            quick_llm,
            deep_llm,
            tool_nodes,
            conditional_logic,
            risk_stage="none",
        )

        workflow = setup.setup_graph(selected_analysts=["market"])

        node_names = list(workflow.nodes)
        assert "Aggressive Analyst" not in node_names
        assert "Conservative Analyst" not in node_names
        assert "Neutral Analyst" not in node_names
        # But Trader and Portfolio Manager should still be there
        assert "Trader" in node_names
        assert "Portfolio Manager" in node_names

    def test_graph_setup_with_risk_stage_debate_includes_debate_nodes(self):
        """With risk_stage='debate', Aggressive/Conservative/Neutral nodes should be added."""
        quick_llm = self._mock_llm()
        deep_llm = self._mock_llm()
        tool_nodes = self._mock_tool_nodes()
        conditional_logic = ConditionalLogic(max_debate_rounds=1, max_risk_discuss_rounds=1)

        setup = GraphSetup(
            quick_llm,
            deep_llm,
            tool_nodes,
            conditional_logic,
            risk_stage="debate",
        )

        workflow = setup.setup_graph(selected_analysts=["market"])

        node_names = list(workflow.nodes)
        assert "Aggressive Analyst" in node_names
        assert "Conservative Analyst" in node_names
        assert "Neutral Analyst" in node_names
        assert "Portfolio Manager" in node_names

    def test_graph_setup_defaults_to_debate_risk_stage(self):
        """When risk_stage is not provided, it should default to 'debate'."""
        quick_llm = self._mock_llm()
        deep_llm = self._mock_llm()
        tool_nodes = self._mock_tool_nodes()
        conditional_logic = ConditionalLogic(max_debate_rounds=1, max_risk_discuss_rounds=1)

        # Create without explicit risk_stage parameter (tests default)
        setup = GraphSetup(
            quick_llm,
            deep_llm,
            tool_nodes,
            conditional_logic,
        )

        workflow = setup.setup_graph(selected_analysts=["market"])

        node_names = list(workflow.nodes)
        assert "Aggressive Analyst" in node_names
        assert "Conservative Analyst" in node_names
        assert "Neutral Analyst" in node_names


class TestEdgeWiringForRiskStage:
    """Tests for edge wiring logic that differs between risk_stage modes."""

    def _mock_llm(self):
        mock = MagicMock()
        mock.invoke = MagicMock(return_value=MagicMock(content="Test response"))
        return mock

    def _mock_tool_nodes(self):
        return {"social": MagicMock()}

    def test_trader_edges_to_portfolio_manager_in_none_mode(self):
        """In 'none' mode, the Trader should edge directly to the Portfolio Manager."""
        quick_llm = self._mock_llm()
        deep_llm = self._mock_llm()
        tool_nodes = self._mock_tool_nodes()
        conditional_logic = ConditionalLogic(max_debate_rounds=1, max_risk_discuss_rounds=1)

        setup = GraphSetup(
            quick_llm,
            deep_llm,
            tool_nodes,
            conditional_logic,
            risk_stage="none",
        )

        workflow = setup.setup_graph(selected_analysts=["market"])
        compiled = workflow.compile()

        assert compiled is not None
        # The Trader -> Portfolio Manager edge should exist directly (no risk debaters
        # in between). workflow.edges is a set of (source, target) tuples.
        assert ("Trader", "Portfolio Manager") in workflow.edges
        assert ("Trader", "Aggressive Analyst") not in workflow.edges

    def test_graph_compiles_in_debate_mode(self):
        """In 'debate' mode, the graph should still compile with the risk-debate wiring."""
        quick_llm = self._mock_llm()
        deep_llm = self._mock_llm()
        tool_nodes = self._mock_tool_nodes()
        conditional_logic = ConditionalLogic(max_debate_rounds=1, max_risk_discuss_rounds=1)

        setup = GraphSetup(
            quick_llm,
            deep_llm,
            tool_nodes,
            conditional_logic,
            risk_stage="debate",
        )

        workflow = setup.setup_graph(selected_analysts=["market"])
        compiled = workflow.compile()

        assert compiled is not None
        assert ("Trader", "Aggressive Analyst") in workflow.edges
        assert ("Trader", "Portfolio Manager") not in workflow.edges


class TestRiskStageValidation:
    """Tests for risk_stage validation in TradingAgentsGraph."""

    def test_invalid_risk_stage_raises(self, monkeypatch):
        """An unrecognized risk_stage value raises ValueError with a clear message."""
        dc = _reload_with_env(monkeypatch)
        config = dc.DEFAULT_CONFIG.copy()
        config["risk_stage"] = "bogus"

        with pytest.raises(ValueError, match="Invalid risk_stage"):
            TradingAgentsGraph(selected_analysts=["market"], config=config)

        _reload_with_env(monkeypatch)

    def test_valid_risk_stage_values_do_not_raise_from_validation(self, monkeypatch):
        """'debate' and 'none' both pass the risk_stage validation step.

        Graph/LLM-client construction may still fail further down __init__ for
        unrelated reasons (no real provider credentials in the test env); we
        only assert that failure isn't the risk_stage validation itself.
        """
        dc = _reload_with_env(monkeypatch)

        for value in ("debate", "none"):
            config = dc.DEFAULT_CONFIG.copy()
            config["risk_stage"] = value
            try:
                TradingAgentsGraph(selected_analysts=["market"], config=config)
            except Exception as e:
                assert "invalid risk_stage" not in str(e).lower()

        _reload_with_env(monkeypatch)


def _empty_risk_state():
    return {
        "history": "",
        "aggressive_history": "",
        "conservative_history": "",
        "neutral_history": "",
        "latest_speaker": "",
        "current_aggressive_response": "",
        "current_conservative_response": "",
        "current_neutral_response": "",
        "judge_decision": "",
        "count": 0,
    }


class TestFullRunWithRiskStageNone:
    """End-to-end graph run with risk_stage='none' (acceptance criterion: 'a full run completes')."""

    def _install_stub_node_factories(self, monkeypatch):
        """Stub every node factory so the graph runs deterministically with no real LLM calls."""

        def sentiment_node(state):
            return {
                "sentiment_report": "sentiment prose",
                "messages": [AIMessage(content="sentiment done", tool_calls=[])],
            }

        monkeypatch.setattr(graph_setup_module, "create_sentiment_analyst", lambda llm: sentiment_node)

        def trader_node(state):
            return {"trader_investment_plan": "Buy 100 shares"}

        def portfolio_manager_node(state):
            return {
                "final_trade_decision": "BUY",
                "risk_debate_state": dict(state["risk_debate_state"], judge_decision="BUY"),
            }

        monkeypatch.setattr(graph_setup_module, "create_trader", lambda llm: trader_node)
        monkeypatch.setattr(
            graph_setup_module, "create_portfolio_manager", lambda llm: portfolio_manager_node
        )

        # These should never be called in risk_stage="none" mode; fail loudly if they are.
        def _unexpected(name):
            def _factory(llm):
                raise AssertionError(f"{name} node factory should not be invoked in risk_stage='none' mode")

            return _factory

        monkeypatch.setattr(graph_setup_module, "create_aggressive_debator", _unexpected("aggressive"))
        monkeypatch.setattr(graph_setup_module, "create_conservative_debator", _unexpected("conservative"))
        monkeypatch.setattr(graph_setup_module, "create_neutral_debator", _unexpected("neutral"))

    def test_full_run_completes_with_risk_stage_none(self, monkeypatch):
        self._install_stub_node_factories(monkeypatch)

        conditional_logic = ConditionalLogic(max_debate_rounds=1, max_risk_discuss_rounds=1)
        setup = GraphSetup(
            quick_thinking_llm=None,
            deep_thinking_llm=None,
            tool_nodes={"social": MagicMock()},
            conditional_logic=conditional_logic,
            research_stage="none",
            risk_stage="none",
        )

        workflow = setup.setup_graph(["social"])
        compiled = workflow.compile()

        initial_state = {
            "company_of_interest": "NVDA",
            "trade_date": "2026-07-07",
            "asset_type": "stock",
            "messages": [HumanMessage(content="NVDA")],
            "investment_debate_state": {
                "history": "", "bull_history": "", "bear_history": "",
                "current_response": "", "judge_decision": "", "count": 0,
            },
            "risk_debate_state": _empty_risk_state(),
            "investment_plan": "",
            "trader_investment_plan": "",
        }

        final_state = compiled.invoke(initial_state)

        assert final_state["final_trade_decision"] == "BUY"
        assert final_state["trader_investment_plan"] == "Buy 100 shares"
        # No risk-debate history accumulated: the debaters never ran.
        assert final_state["risk_debate_state"]["aggressive_history"] == ""
        assert final_state["risk_debate_state"]["conservative_history"] == ""
        assert final_state["risk_debate_state"]["neutral_history"] == ""


class _FakeGraphForLogState:
    """Minimal stand-in exposing only the attributes `_log_state` reads/writes.

    Avoids constructing a real TradingAgentsGraph (LLM clients, memory log, ...)
    just to exercise the pure state -> log_dict -> JSON-file logic.
    """

    def __init__(self, config, ticker):
        self.config = config
        self.ticker = ticker
        self.log_states_dict = {}


class TestLogStateRiskDebateGating:
    """Full-state JSON log should handle the empty risk state cleanly (#119, mirrors #79)."""

    def _final_state(self, risk_debate_state):
        return {
            "company_of_interest": "NVDA",
            "trade_date": "2026-07-07",
            "trader_investment_plan": "Buy 100 shares",
            "investment_plan": "",
            "final_trade_decision": "BUY",
            "risk_debate_state": risk_debate_state,
        }

    def test_risk_debate_state_included_when_debate_mode(self, tmp_path):
        fake = _FakeGraphForLogState(
            config={"results_dir": str(tmp_path), "risk_stage": "debate"},
            ticker="NVDA",
        )
        risk_state = dict(
            _empty_risk_state(),
            history="Aggressive: go big. Conservative: too risky.",
            aggressive_history="go big",
            judge_decision="BUY",
        )

        TradingAgentsGraph._log_state(fake, "2026-07-07", self._final_state(risk_state))

        log_dict = fake.log_states_dict["2026-07-07"]
        assert "risk_debate_state" in log_dict
        assert log_dict["risk_debate_state"]["aggressive_history"] == "go big"
        assert log_dict["final_trade_decision"] == "BUY"

    def test_risk_debate_state_omitted_when_none_mode(self, tmp_path):
        fake = _FakeGraphForLogState(
            config={"results_dir": str(tmp_path), "risk_stage": "none"},
            ticker="NVDA",
        )
        risk_state = dict(_empty_risk_state(), judge_decision="BUY")

        TradingAgentsGraph._log_state(fake, "2026-07-07", self._final_state(risk_state))

        log_dict = fake.log_states_dict["2026-07-07"]
        # The history breakdown is omitted entirely, but the PM's decision is
        # never lost -- it's already captured unconditionally as final_trade_decision.
        assert "risk_debate_state" not in log_dict
        assert log_dict["final_trade_decision"] == "BUY"

    def test_risk_debate_state_defaults_to_debate_when_key_absent(self, tmp_path):
        """If risk_stage is missing from config entirely, it defaults to 'debate'
        (today's behavior) for logging purposes too."""
        fake = _FakeGraphForLogState(
            config={"results_dir": str(tmp_path)},
            ticker="NVDA",
        )
        risk_state = dict(_empty_risk_state(), aggressive_history="go big", judge_decision="BUY")

        TradingAgentsGraph._log_state(fake, "2026-07-07", self._final_state(risk_state))

        log_dict = fake.log_states_dict["2026-07-07"]
        assert "risk_debate_state" in log_dict
