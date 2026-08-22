"""Test macro fundamentals analyst graph wiring (issue #132).

Tests cover:
- Macro Fundamentals Analyst node is present only when selected
- A default-selected-analysts run produces the exact same graph as before
  this issue (no new node) — the explicit regression-safety acceptance
  criterion from issue #132.
- Edge routing: the node is wired straight to its "Msg Clear" node (no
  ToolNode round trip), matching market/news/fundamentals (#37).
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from tradingagents.graph.conditional_logic import ConditionalLogic
from tradingagents.graph.setup import GraphSetup

pytestmark = pytest.mark.unit


class TestMacroFundamentalsGraphWiring:
    def _create_mock_llms(self):
        return Mock(), Mock()

    def _create_mock_tool_nodes(self):
        # macro_fundamentals has tool_node=None (#132), so it never looks
        # itself up in this dict — same as market/news/fundamentals.
        return {"market": Mock(), "social": Mock(), "news": Mock(), "fundamentals": Mock()}

    def _create_mock_conditional_logic(self):
        return ConditionalLogic(max_debate_rounds=1, max_risk_discuss_rounds=1)

    def _graph_setup(self):
        return GraphSetup(
            *self._create_mock_llms(),
            self._create_mock_tool_nodes(),
            self._create_mock_conditional_logic(),
        )

    def test_macro_fundamentals_node_present_when_selected(self):
        workflow = self._graph_setup().setup_graph(["market", "macro_fundamentals"])
        compiled = workflow.compile()
        assert "Macro Fundamentals Analyst" in compiled.get_graph().nodes
        assert "Msg Clear Macro Fundamentals" in compiled.get_graph().nodes

    def test_macro_fundamentals_absent_with_default_selection(self):
        workflow = self._graph_setup().setup_graph()  # selected_analysts=None -> default
        compiled = workflow.compile()
        assert "Macro Fundamentals Analyst" not in compiled.get_graph().nodes

    def test_default_selected_analysts_graph_unchanged(self):
        """Regression safety: a run with the default selected_analysts produces
        the exact same node set as explicitly passing the pre-#132 four —
        no new node leaks in by default."""
        default_nodes = set(self._graph_setup().setup_graph().compile().get_graph().nodes)
        explicit_nodes = set(
            self._graph_setup()
            .setup_graph(["market", "social", "news", "fundamentals"])
            .compile()
            .get_graph()
            .nodes
        )

        assert default_nodes == explicit_nodes
        assert "Macro Fundamentals Analyst" not in default_nodes
        assert "Msg Clear Macro Fundamentals" not in default_nodes

    def test_macro_fundamentals_has_no_tool_node_edge(self):
        """No ToolNode round trip: the analyst edges straight to its clear node."""
        workflow = self._graph_setup().setup_graph(["macro_fundamentals"])
        compiled = workflow.compile()
        edges = list(compiled.get_graph().edges)

        analyst_edges = [e for e in edges if e[0] == "Macro Fundamentals Analyst"]
        assert len(analyst_edges) == 1
        assert analyst_edges[0][1] == "Msg Clear Macro Fundamentals"

    def test_macro_fundamentals_alone_routes_to_trader(self):
        """selected_analysts=['macro_fundamentals'] alone builds and routes to
        the Trader (research_stage defaults to 'none' in GraphSetup)."""
        workflow = self._graph_setup().setup_graph(["macro_fundamentals"])
        compiled = workflow.compile()
        edges = list(compiled.get_graph().edges)

        clear_edges = [e for e in edges if e[0] == "Msg Clear Macro Fundamentals"]
        assert len(clear_edges) == 1
        assert clear_edges[0][1] == "Trader"

    def test_misspelled_analyst_key_fails_before_graph_setup(self):
        with pytest.raises(ValueError):
            self._graph_setup().setup_graph(["macro_fundamental"])  # missing trailing 's'
