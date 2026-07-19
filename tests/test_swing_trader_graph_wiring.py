"""Test Swing Trader graph wiring (issue #92).

Tests cover:
- Swing Trader node is present when enabled, absent when disabled
- Graph edge routing: Portfolio Manager → Swing Trader (when enabled) → END
- Graph edge routing: Portfolio Manager → END (when disabled)
"""

from __future__ import annotations

import pytest
from unittest.mock import Mock, MagicMock

from tradingagents.graph.setup import GraphSetup
from tradingagents.graph.conditional_logic import ConditionalLogic

pytestmark = pytest.mark.unit


class TestSwingTraderGraphWiring:
    """Tests for swing trader graph wiring."""

    def _create_mock_llms(self):
        """Create mock LLM clients."""
        quick_llm = Mock()
        deep_llm = Mock()
        return quick_llm, deep_llm

    def _create_mock_tool_nodes(self):
        """Create mock tool nodes."""
        return {
            "market": Mock(),
            "social": Mock(),
            "news": Mock(),
            "fundamentals": Mock(),
        }

    def _create_mock_conditional_logic(self):
        """Create mock conditional logic."""
        return ConditionalLogic(max_debate_rounds=1, max_risk_discuss_rounds=1)

    def test_swing_trader_node_present_when_enabled(self):
        """Test that Swing Trader node is added when swing_trader_enabled=True."""
        quick_llm, deep_llm = self._create_mock_llms()
        tool_nodes = self._create_mock_tool_nodes()
        conditional_logic = self._create_mock_conditional_logic()

        graph_setup = GraphSetup(
            quick_llm,
            deep_llm,
            tool_nodes,
            conditional_logic,
            swing_trader_enabled=True,
        )
        workflow = graph_setup.setup_graph()
        compiled = workflow.compile()

        # Check that Swing Trader node exists in the graph
        assert "Swing Trader" in compiled.get_graph().nodes

    def test_swing_trader_node_absent_when_disabled(self):
        """Test that Swing Trader node is not added when swing_trader_enabled=False."""
        quick_llm, deep_llm = self._create_mock_llms()
        tool_nodes = self._create_mock_tool_nodes()
        conditional_logic = self._create_mock_conditional_logic()

        graph_setup = GraphSetup(
            quick_llm,
            deep_llm,
            tool_nodes,
            conditional_logic,
            swing_trader_enabled=False,
        )
        workflow = graph_setup.setup_graph()
        compiled = workflow.compile()

        # Check that Swing Trader node does NOT exist in the graph
        assert "Swing Trader" not in compiled.get_graph().nodes

    def test_portfolio_manager_routes_to_swing_trader_when_enabled(self):
        """Test that Portfolio Manager → Swing Trader when enabled."""
        quick_llm, deep_llm = self._create_mock_llms()
        tool_nodes = self._create_mock_tool_nodes()
        conditional_logic = self._create_mock_conditional_logic()

        graph_setup = GraphSetup(
            quick_llm,
            deep_llm,
            tool_nodes,
            conditional_logic,
            swing_trader_enabled=True,
        )
        workflow = graph_setup.setup_graph()
        compiled = workflow.compile()

        graph = compiled.get_graph()

        # Find edges from Portfolio Manager
        edges = list(graph.edges)
        pm_edges = [e for e in edges if e[0] == "Portfolio Manager"]

        # Should have one edge from Portfolio Manager to Swing Trader
        assert len(pm_edges) >= 1
        assert any(e[1] == "Swing Trader" for e in pm_edges), \
            f"Portfolio Manager should route to Swing Trader, but edges are {pm_edges}"

    def test_portfolio_manager_routes_to_end_when_disabled(self):
        """Test that Portfolio Manager → END when swing trader disabled."""
        quick_llm, deep_llm = self._create_mock_llms()
        tool_nodes = self._create_mock_tool_nodes()
        conditional_logic = self._create_mock_conditional_logic()

        graph_setup = GraphSetup(
            quick_llm,
            deep_llm,
            tool_nodes,
            conditional_logic,
            swing_trader_enabled=False,
        )
        workflow = graph_setup.setup_graph()
        compiled = workflow.compile()

        graph = compiled.get_graph()

        # Find edges from Portfolio Manager
        edges = list(graph.edges)
        pm_edges = [e for e in edges if e[0] == "Portfolio Manager"]

        # Should have one edge from Portfolio Manager to END
        assert len(pm_edges) >= 1
        assert any(e[1] == "__end__" for e in pm_edges), \
            f"Portfolio Manager should route to END, but edges are {pm_edges}"

    def test_swing_trader_routes_to_end_when_enabled(self):
        """Test that Swing Trader → END when enabled."""
        quick_llm, deep_llm = self._create_mock_llms()
        tool_nodes = self._create_mock_tool_nodes()
        conditional_logic = self._create_mock_conditional_logic()

        graph_setup = GraphSetup(
            quick_llm,
            deep_llm,
            tool_nodes,
            conditional_logic,
            swing_trader_enabled=True,
        )
        workflow = graph_setup.setup_graph()
        compiled = workflow.compile()

        graph = compiled.get_graph()

        # Find edges from Swing Trader
        edges = list(graph.edges)
        st_edges = [e for e in edges if e[0] == "Swing Trader"]

        # Should have one edge from Swing Trader to END
        assert len(st_edges) >= 1
        assert any(e[1] == "__end__" for e in st_edges), \
            f"Swing Trader should route to END, but edges are {st_edges}"
