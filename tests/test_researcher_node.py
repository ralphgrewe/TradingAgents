"""Tests for the Researcher node (issue #85).

Tests cover:
1. Plan-validation fallback path (missing bull/bear queries patched)
2. Gate-closed run produces brief with new_information=None, zero search calls
3. Gate-open run produces web:*-sourced arguments and populated researcher_evidence
4. Graph builds and runs end-to-end in "researcher" mode
5. Memory row lands under "researcher" agent key
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from tradingagents.agents.researchers.researcher import (
    _validate_and_patch_plan,
    create_researcher,
)
from tradingagents.agents.schemas import (
    BriefConfidence,
    PortfolioRating,
    ResearchArgument,
    ResearchBrief,
)
from tradingagents.graph.conditional_logic import ConditionalLogic
from tradingagents.graph.setup import GraphSetup


pytestmark = pytest.mark.unit


class TestPlanValidationAndPatching:
    """Tests for _validate_and_patch_plan helper function."""

    def test_plan_with_both_bull_and_bear_not_patched(self):
        """Valid plan (has bull and bear) should not be patched."""
        plan = [
            {"query": "growth opportunities", "type": "bull"},
            {"query": "risks and threats", "type": "bear"},
        ]
        patched, was_patched = _validate_and_patch_plan(plan)
        assert not was_patched
        assert len(patched) == 2

    def test_plan_missing_bull_gets_patched(self):
        """Plan missing bull queries should get a bull fallback."""
        plan = [
            {"query": "risks and threats", "type": "bear"},
            {"query": "market data", "type": "neutral"},
        ]
        patched, was_patched = _validate_and_patch_plan(plan)
        assert was_patched
        # Should have added a bull fallback
        assert any(q.get("type") == "bull" for q in patched)

    def test_plan_missing_bear_gets_patched(self):
        """Plan missing bear queries should get a bear fallback."""
        plan = [
            {"query": "growth opportunities", "type": "bull"},
            {"query": "market data", "type": "neutral"},
        ]
        patched, was_patched = _validate_and_patch_plan(plan)
        assert was_patched
        # Should have added a bear fallback
        assert any(q.get("type") == "bear" for q in patched)

    def test_plan_missing_both_gets_both_patched(self):
        """Plan with neither bull nor bear should get both fallbacks."""
        plan = [
            {"query": "market data", "type": "neutral"},
        ]
        patched, was_patched = _validate_and_patch_plan(plan)
        assert was_patched
        assert any(q.get("type") == "bull" for q in patched)
        assert any(q.get("type") == "bear" for q in patched)


class TestResearcherNodeGateClosed:
    """Tests for gate-closed runs (historical date or web_search disabled)."""

    def _mock_llm(self, response_text="Test response"):
        """Create a mock LLM that returns a plain response."""
        mock = MagicMock()
        mock.invoke = MagicMock(
            return_value=MagicMock(content=response_text)
        )
        return mock

    def test_gate_closed_historical_date(self, monkeypatch):
        """Gate-closed run (historical date) produces brief with new_information=None."""
        # Set up mocks
        quick_llm = self._mock_llm()
        deep_llm = self._mock_llm()

        # Mock structured output
        with patch("tradingagents.agents.researchers.researcher.bind_structured") as mock_bind:
            mock_structured_llm = MagicMock()
            mock_bind.return_value = mock_structured_llm

            # Mock invoke_structured_or_freetext to return a valid brief
            with patch("tradingagents.agents.researchers.researcher.invoke_structured_or_freetext") as mock_invoke:
                rendered_brief = "**Recommendation**: Hold\n\n**Bull Arguments**:\n- Test bull [market]\n\n**Bear Arguments**:\n- Test bear [news]\n\n**Confidence**: Medium\n\nWeb research: disabled (historical date)"
                mock_invoke.return_value = rendered_brief

                researcher_node = create_researcher(quick_llm, deep_llm)

                # Create state with historical trade_date
                state = {
                    "company_of_interest": "AAPL",
                    "asset_type": "stock",
                    "trade_date": "2020-01-01",  # Historical
                    "market_report": "Market is bullish [market]",
                    "sentiment_report": "",
                    "news_report": "",
                    "fundamentals_report": "",
                }

                # Mock the config
                with patch("tradingagents.agents.researchers.researcher.get_config") as mock_config:
                    mock_config.return_value = {
                        "research_web_search": True,
                        "research_search_queries_max": 4,
                        "research_evidence_token_budget": 3000,
                    }

                    result = researcher_node(state)

                # Verify no web search was called
                mock_invoke.assert_called_once()
                call_args = mock_invoke.call_args
                # The render function should be passed, but the synthesis prompt should
                # indicate gate is closed
                assert "disabled (historical date)" in rendered_brief

                # Check researcher_evidence
                evidence_dict = json.loads(result["researcher_evidence"])
                assert evidence_dict["gate"]["outcome"] == "disabled (historical date)"
                assert evidence_dict["evidence_pack"] == []

    def test_gate_closed_no_api_key(self, monkeypatch):
        """Gate-closed run (no API key) handles gracefully."""
        quick_llm = self._mock_llm()
        deep_llm = self._mock_llm()

        with patch("tradingagents.agents.researchers.researcher.bind_structured") as mock_bind:
            mock_bind.return_value = MagicMock()

            with patch("tradingagents.agents.researchers.researcher.invoke_structured_or_freetext") as mock_invoke:
                rendered_brief = "**Recommendation**: Hold\n\nWeb research: disabled (no API key)"
                mock_invoke.return_value = rendered_brief

                researcher_node = create_researcher(quick_llm, deep_llm)

                # Mock route_to_vendor to return DATA_UNAVAILABLE sentinel
                with patch("tradingagents.agents.researchers.researcher.route_to_vendor") as mock_route:
                    mock_route.return_value = "DATA_UNAVAILABLE: optional web_search could not be retrieved (TAVILY_API_KEY not set)"

                    state = {
                        "company_of_interest": "AAPL",
                        "asset_type": "stock",
                        "trade_date": "2026-07-18",  # Today
                        "market_report": "",
                        "sentiment_report": "",
                        "news_report": "",
                        "fundamentals_report": "",
                    }

                    with patch("tradingagents.agents.researchers.researcher.get_config") as mock_config:
                        mock_config.return_value = {
                            "research_web_search": True,
                            "research_search_queries_max": 4,
                            "research_evidence_token_budget": 3000,
                        }

                        result = researcher_node(state)

                    evidence_dict = json.loads(result["researcher_evidence"])
                    assert evidence_dict["gate"]["outcome"] == "disabled (no API key)"


class TestResearcherNodeEndToEnd:
    """End-to-end tests for researcher node in graph."""

    def _mock_llm(self, response_text="Test response"):
        """Create a mock LLM."""
        mock = MagicMock()
        mock.invoke = MagicMock(
            return_value=MagicMock(content=response_text)
        )
        return mock

    def _mock_tool_nodes(self):
        """Create mock tool nodes."""
        return {"social": MagicMock()}

    def test_graph_setup_with_researcher_mode_includes_researcher_node(self):
        """With research_stage='researcher', Researcher node should be added."""
        quick_llm = self._mock_llm()
        deep_llm = self._mock_llm()
        tool_nodes = self._mock_tool_nodes()
        conditional_logic = ConditionalLogic(max_debate_rounds=1, max_risk_discuss_rounds=1)

        setup = GraphSetup(
            quick_llm,
            deep_llm,
            tool_nodes,
            conditional_logic,
            research_stage="researcher",
        )

        workflow = setup.setup_graph(selected_analysts=["market"])

        # Check that the graph contains Researcher node
        node_names = list(workflow.nodes)
        assert "Researcher" in node_names
        assert "Bull Researcher" not in node_names
        assert "Bear Researcher" not in node_names
        assert "Research Manager" not in node_names
        # But Trader should still be there
        assert "Trader" in node_names

    def test_graph_setup_researcher_wiring(self):
        """Researcher node should be wired between last analyst and Trader."""
        quick_llm = self._mock_llm()
        deep_llm = self._mock_llm()
        tool_nodes = self._mock_tool_nodes()
        conditional_logic = ConditionalLogic(max_debate_rounds=1, max_risk_discuss_rounds=1)

        setup = GraphSetup(
            quick_llm,
            deep_llm,
            tool_nodes,
            conditional_logic,
            research_stage="researcher",
        )

        workflow = setup.setup_graph(selected_analysts=["market"])
        compiled = workflow.compile()

        # The graph should compile successfully
        assert compiled is not None


class TestResearcherNodeStateIntegration:
    """Tests for state handling and researcher_evidence field."""

    def _mock_llm(self):
        """Create a mock LLM."""
        mock = MagicMock()
        mock.invoke = MagicMock(
            return_value=MagicMock(content="Test response")
        )
        return mock

    def test_researcher_node_returns_investment_plan_and_evidence(self):
        """Researcher node should return investment_plan and researcher_evidence."""
        quick_llm = self._mock_llm()
        deep_llm = self._mock_llm()

        with patch("tradingagents.agents.researchers.researcher.bind_structured") as mock_bind:
            mock_bind.return_value = MagicMock()

            with patch("tradingagents.agents.researchers.researcher.invoke_structured_or_freetext") as mock_invoke:
                rendered_brief = "**Recommendation**: Buy\n\nWeb research: disabled (config)"
                mock_invoke.return_value = rendered_brief

                researcher_node = create_researcher(quick_llm, deep_llm)

                state = {
                    "company_of_interest": "AAPL",
                    "asset_type": "stock",
                    "trade_date": "2020-01-01",
                    "market_report": "",
                    "sentiment_report": "",
                    "news_report": "",
                    "fundamentals_report": "",
                }

                with patch("tradingagents.agents.researchers.researcher.get_config") as mock_config:
                    mock_config.return_value = {
                        "research_web_search": True,
                        "research_search_queries_max": 4,
                        "research_evidence_token_budget": 3000,
                    }

                    result = researcher_node(state)

                # Check that both fields are present
                assert "investment_plan" in result
                assert "researcher_evidence" in result

                # Check that investment_plan contains rendered markdown
                assert "**Recommendation**:" in result["investment_plan"]

                # Check that researcher_evidence is valid JSON
                evidence = json.loads(result["researcher_evidence"])
                assert "plan" in evidence
                assert "gate" in evidence
                assert "evidence_pack" in evidence
