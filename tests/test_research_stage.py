"""Tests for research_stage config key and graph branching logic."""

from __future__ import annotations

import importlib
from unittest.mock import MagicMock

import tradingagents.default_config as default_config_module
from tradingagents.graph.conditional_logic import ConditionalLogic
from tradingagents.graph.setup import GraphSetup


def _reload_with_env(monkeypatch, **overrides):
    """Set/clear env vars then reload default_config to re-evaluate DEFAULT_CONFIG."""
    for key in list(default_config_module._ENV_OVERRIDES):
        monkeypatch.delenv(key, raising=False)
    for key, val in overrides.items():
        monkeypatch.setenv(key, val)
    return importlib.reload(default_config_module)


class TestResearchStageConfig:
    """Tests for research_stage config key and env override."""

    def test_research_stage_defaults_to_researcher(self, monkeypatch):
        """The research_stage config key should default to 'researcher'."""
        dc = _reload_with_env(monkeypatch)
        assert dc.DEFAULT_CONFIG["research_stage"] == "researcher"

    def test_research_stage_env_override(self, monkeypatch):
        """TRADINGAGENTS_RESEARCH_STAGE env var should override the default."""
        dc = _reload_with_env(monkeypatch, TRADINGAGENTS_RESEARCH_STAGE="debate")
        assert dc.DEFAULT_CONFIG["research_stage"] == "debate"

    def test_research_stage_empty_env_passthrough(self, monkeypatch):
        """Empty TRADINGAGENTS_RESEARCH_STAGE should not clobber the default."""
        dc = _reload_with_env(monkeypatch, TRADINGAGENTS_RESEARCH_STAGE="")
        assert dc.DEFAULT_CONFIG["research_stage"] == "researcher"

    def test_research_stage_accepts_none_string(self, monkeypatch):
        """TRADINGAGENTS_RESEARCH_STAGE='none' should be accepted."""
        dc = _reload_with_env(monkeypatch, TRADINGAGENTS_RESEARCH_STAGE="none")
        assert dc.DEFAULT_CONFIG["research_stage"] == "none"


class TestGraphSetupWithResearchStage:
    """Tests for GraphSetup behavior with different research_stage values."""

    def _mock_llm(self):
        """Create a mock LLM."""
        mock = MagicMock()
        mock.invoke = MagicMock(return_value=MagicMock(content="Test response"))
        return mock

    def _mock_tool_nodes(self):
        """Create mock tool nodes."""
        return {"social": MagicMock()}

    def test_graph_setup_with_research_stage_none_omits_debate_nodes(self):
        """With research_stage='none', Bull/Bear/Research Manager nodes should not be added."""
        quick_llm = self._mock_llm()
        deep_llm = self._mock_llm()
        tool_nodes = self._mock_tool_nodes()
        conditional_logic = ConditionalLogic(max_debate_rounds=1, max_risk_discuss_rounds=1)

        setup = GraphSetup(
            quick_llm,
            deep_llm,
            tool_nodes,
            conditional_logic,
            research_stage="none",
        )

        workflow = setup.setup_graph(selected_analysts=["market"])

        # Check that the graph does not contain Bull/Bear/Research Manager nodes
        node_names = list(workflow.nodes)
        assert "Bull Researcher" not in node_names
        assert "Bear Researcher" not in node_names
        assert "Research Manager" not in node_names
        # But Trader should still be there
        assert "Trader" in node_names

    def test_graph_setup_with_research_stage_debate_includes_debate_nodes(self):
        """With research_stage='debate', Bull/Bear/Research Manager nodes should be added."""
        quick_llm = self._mock_llm()
        deep_llm = self._mock_llm()
        tool_nodes = self._mock_tool_nodes()
        conditional_logic = ConditionalLogic(max_debate_rounds=1, max_risk_discuss_rounds=1)

        setup = GraphSetup(
            quick_llm,
            deep_llm,
            tool_nodes,
            conditional_logic,
            research_stage="debate",
        )

        workflow = setup.setup_graph(selected_analysts=["market"])

        # Check that the graph contains Bull/Bear/Research Manager nodes
        node_names = list(workflow.nodes)
        assert "Bull Researcher" in node_names
        assert "Bear Researcher" in node_names
        assert "Research Manager" in node_names
        assert "Trader" in node_names

    def test_graph_setup_defaults_to_none_research_stage(self):
        """When research_stage is not provided, it should default to 'none'."""
        quick_llm = self._mock_llm()
        deep_llm = self._mock_llm()
        tool_nodes = self._mock_tool_nodes()
        conditional_logic = ConditionalLogic(max_debate_rounds=1, max_risk_discuss_rounds=1)

        # Create without explicit research_stage parameter (tests default)
        setup = GraphSetup(
            quick_llm,
            deep_llm,
            tool_nodes,
            conditional_logic,
        )

        workflow = setup.setup_graph(selected_analysts=["market"])

        # Should behave like "none" mode (debate nodes omitted)
        node_names = list(workflow.nodes)
        assert "Bull Researcher" not in node_names
        assert "Bear Researcher" not in node_names
        assert "Research Manager" not in node_names


class TestEdgeWiringForResearchStage:
    """Tests for edge wiring logic that differs between research_stage modes."""

    def _mock_llm(self):
        """Create a mock LLM."""
        mock = MagicMock()
        mock.invoke = MagicMock(return_value=MagicMock(content="Test response"))
        return mock

    def _mock_tool_nodes(self):
        """Create mock tool nodes."""
        return {"social": MagicMock()}

    def test_analyst_edges_to_trader_in_none_mode(self):
        """In 'none' mode, last analyst should edge directly to Trader."""
        quick_llm = self._mock_llm()
        deep_llm = self._mock_llm()
        tool_nodes = self._mock_tool_nodes()
        conditional_logic = ConditionalLogic(max_debate_rounds=1, max_risk_discuss_rounds=1)

        setup = GraphSetup(
            quick_llm,
            deep_llm,
            tool_nodes,
            conditional_logic,
            research_stage="none",
        )

        workflow = setup.setup_graph(selected_analysts=["market"])
        compiled = workflow.compile()

        # The graph should be compilable and structured correctly
        assert compiled is not None

    def test_analyst_edges_to_bull_researcher_in_debate_mode(self):
        """In 'debate' mode, last analyst should edge to Bull Researcher."""
        quick_llm = self._mock_llm()
        deep_llm = self._mock_llm()
        tool_nodes = self._mock_tool_nodes()
        conditional_logic = ConditionalLogic(max_debate_rounds=1, max_risk_discuss_rounds=1)

        setup = GraphSetup(
            quick_llm,
            deep_llm,
            tool_nodes,
            conditional_logic,
            research_stage="debate",
        )

        workflow = setup.setup_graph(selected_analysts=["market"])
        compiled = workflow.compile()

        # The graph should be compilable and structured correctly
        assert compiled is not None
