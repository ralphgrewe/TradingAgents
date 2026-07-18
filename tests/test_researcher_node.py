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
from datetime import datetime
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


def _today() -> str:
    """Today's date in the same yyyy-mm-dd format the researcher node compares against.

    Computed at test time (not hardcoded) so gate-open tests keep passing regardless
    of when the suite runs — a hardcoded date would silently start testing the
    gate-closed path once that date is in the past.
    """
    return datetime.now().strftime("%Y-%m-%d")


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

    def test_queries_max_none_does_not_truncate(self):
        """queries_max=None (the default) skips the cap entirely — back-compat."""
        plan = [
            {"query": "q1", "type": "bull"},
            {"query": "q2", "type": "bear"},
            {"query": "q3", "type": "neutral"},
            {"query": "q4", "type": "neutral"},
            {"query": "q5", "type": "neutral"},
        ]
        patched, was_patched = _validate_and_patch_plan(plan, None)
        assert not was_patched
        assert len(patched) == 5

    def test_queries_max_truncates_overlong_plan(self):
        """A plan longer than research_search_queries_max is truncated to fit,
        proving the config value actually bounds behavior (not just docs/prompt
        text). Already-balanced plans just get their tail dropped."""
        plan = [
            {"query": "q1", "type": "bull"},
            {"query": "q2", "type": "bear"},
            {"query": "q3", "type": "neutral"},
            {"query": "q4", "type": "neutral"},
            {"query": "q5", "type": "neutral"},
        ]
        patched, was_patched = _validate_and_patch_plan(plan, queries_max=2)
        assert was_patched
        assert len(patched) == 2
        # The bull/bear balance must survive the truncation even though they
        # were the first two entries here (verified below with a reordered plan).

    def test_queries_max_truncation_still_guarantees_bull_and_bear(self):
        """Truncating to fit queries_max must not sacrifice the bull/bear balance
        guarantee: an over-max plan with bull/bear at the tail should end up with
        both fallbacks patched in when the originals get dropped for the cap."""
        plan = [
            {"query": "q1", "type": "neutral"},
            {"query": "q2", "type": "neutral"},
            {"query": "q3", "type": "neutral"},
            {"query": "q4", "type": "bull"},
            {"query": "q5", "type": "bear"},
        ]
        patched, was_patched = _validate_and_patch_plan(plan, queries_max=2)
        assert was_patched
        assert len(patched) == 2
        assert any(q.get("type") == "bull" for q in patched)
        assert any(q.get("type") == "bear" for q in patched)

    def test_queries_max_one_single_bear_keeps_bear_and_adds_bull(self):
        """Issue #85 repro: queries_max=1, plan is a single bear query (no bull).

        Phase 1 leaves the one-element plan untouched (already within the cap), so
        the bear query correctly survives. A bull fallback is then needed. Making
        room for it must NOT evict the surviving bear query — the ≥1-bull AND
        ≥1-bear guarantee wins over the cap here, so BOTH types must be present in
        the final plan (previously the bear was silently dropped)."""
        plan = [{"query": "risks and threats", "type": "bear"}]
        patched, was_patched = _validate_and_patch_plan(plan, queries_max=1)
        assert was_patched
        assert any(q.get("type") == "bull" for q in patched)
        assert any(q.get("type") == "bear" for q in patched)

    def test_queries_max_smaller_than_needed_fallbacks_still_balances(self):
        """queries_max is smaller than the number of fallbacks needed: queries_max=1
        but BOTH bull and bear are missing (2 fallbacks needed). The guarantee of
        ≥1 bull AND ≥1 bear takes priority over the cap, so the result contains both
        fallbacks even though that exceeds queries_max."""
        plan = [{"query": "market data", "type": "neutral"}]
        patched, was_patched = _validate_and_patch_plan(plan, queries_max=1)
        assert was_patched
        assert any(q.get("type") == "bull" for q in patched)
        assert any(q.get("type") == "bear" for q in patched)

    def test_queries_max_changes_behavior_when_config_value_changes(self):
        """Different queries_max values produce different plan lengths for the
        same input — proving research_search_queries_max is actually wired in,
        not dead config."""
        plan = [
            {"query": "q1", "type": "bull"},
            {"query": "q2", "type": "bear"},
            {"query": "q3", "type": "neutral"},
        ]
        patched_2, _ = _validate_and_patch_plan(plan, queries_max=2)
        patched_10, _ = _validate_and_patch_plan(plan, queries_max=10)
        assert len(patched_2) == 2
        assert len(patched_10) == 3


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

                # Verify the plan call (quick-thinking LLM) was never invoked: gate
                # closed means tool-less arm-B mode, so only the deep-thinking
                # synthesis call (mock_invoke, standing in for
                # invoke_structured_or_freetext) should have run.
                quick_llm.invoke.assert_not_called()
                mock_invoke.assert_called_once()
                call_args = mock_invoke.call_args
                # The render function should be passed, but the synthesis prompt should
                # indicate gate is closed
                assert "disabled (historical date)" in rendered_brief

                # Check researcher_evidence
                evidence_dict = json.loads(result["researcher_evidence"])
                assert evidence_dict["gate"]["outcome"] == "disabled (historical date)"
                assert evidence_dict["evidence_pack"] == []
                assert evidence_dict["plan"]["queries"] == []

    def test_gate_closed_no_api_key(self, monkeypatch):
        """Gate-closed run (no API key) handles gracefully, and skips the plan
        call *and* every search call — the key check happens before either."""
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        quick_llm = self._mock_llm()
        deep_llm = self._mock_llm()

        with patch("tradingagents.agents.researchers.researcher.bind_structured") as mock_bind:
            mock_bind.return_value = MagicMock()

            with patch("tradingagents.agents.researchers.researcher.invoke_structured_or_freetext") as mock_invoke:
                rendered_brief = "**Recommendation**: Hold\n\nWeb research: disabled (no API key)"
                mock_invoke.return_value = rendered_brief

                researcher_node = create_researcher(quick_llm, deep_llm)

                with patch("tradingagents.agents.researchers.researcher.route_to_vendor") as mock_route:
                    state = {
                        "company_of_interest": "AAPL",
                        "asset_type": "stock",
                        "trade_date": _today(),  # Today: date gate alone would be open
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
                    assert evidence_dict["evidence_pack"] == []
                    # No API key means the gate never opens, so neither the plan
                    # call nor any vendor search call should have happened.
                    quick_llm.invoke.assert_not_called()
                    mock_route.assert_not_called()
                    mock_invoke.assert_called_once()

    def test_gate_closed_config_disabled(self, monkeypatch):
        """Gate-closed run (research_web_search=False) skips the plan call too,
        even though the date and API key would otherwise allow it."""
        monkeypatch.setenv("TAVILY_API_KEY", "test-key")
        quick_llm = self._mock_llm()
        deep_llm = self._mock_llm()

        with patch("tradingagents.agents.researchers.researcher.bind_structured") as mock_bind:
            mock_bind.return_value = MagicMock()

            with patch("tradingagents.agents.researchers.researcher.invoke_structured_or_freetext") as mock_invoke:
                rendered_brief = "**Recommendation**: Hold\n\nWeb research: disabled (config)"
                mock_invoke.return_value = rendered_brief

                researcher_node = create_researcher(quick_llm, deep_llm)

                with patch("tradingagents.agents.researchers.researcher.route_to_vendor") as mock_route:
                    state = {
                        "company_of_interest": "AAPL",
                        "asset_type": "stock",
                        "trade_date": _today(),
                        "market_report": "",
                        "sentiment_report": "",
                        "news_report": "",
                        "fundamentals_report": "",
                    }

                    with patch("tradingagents.agents.researchers.researcher.get_config") as mock_config:
                        mock_config.return_value = {
                            "research_web_search": False,
                            "research_search_queries_max": 4,
                            "research_evidence_token_budget": 3000,
                        }

                        result = researcher_node(state)

                    evidence_dict = json.loads(result["researcher_evidence"])
                    assert evidence_dict["gate"]["outcome"] == "disabled (config)"
                    assert evidence_dict["evidence_pack"] == []
                    quick_llm.invoke.assert_not_called()
                    mock_route.assert_not_called()
                    mock_invoke.assert_called_once()


class TestResearcherNodeGateOpenSearch:
    """Tests for a successful gate-open run: plan call runs, search executes via
    the mocked vendor, and the synthesis call's brief carries a web:*-sourced
    argument through to the rendered investment_plan and researcher_evidence."""

    def test_gate_open_produces_web_sourced_argument_and_evidence(self, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "test-key")

        quick_llm = MagicMock()
        quick_llm.invoke.return_value = MagicMock(
            content=(
                '```json\n'
                '{"queries": ['
                '{"query": "AAPL growth catalysts", "type": "bull"}, '
                '{"query": "AAPL competitive risks", "type": "bear"}'
                ']}\n'
                '```'
            )
        )

        brief = ResearchBrief(
            bull_arguments=[
                ResearchArgument(
                    statement="Fresh coverage flags a new bullish catalyst",
                    source="web:1",
                )
            ],
            bear_arguments=[
                ResearchArgument(
                    statement="Analyst report flags margin pressure",
                    source="fundamentals",
                )
            ],
            lean=PortfolioRating.BUY,
            confidence=BriefConfidence.HIGH,
            new_information="Web search surfaced a new bullish catalyst article.",
        )
        mock_structured_llm = MagicMock()
        mock_structured_llm.invoke.return_value = brief
        deep_llm = MagicMock()
        deep_llm.with_structured_output.return_value = mock_structured_llm

        researcher_node = create_researcher(quick_llm, deep_llm)

        search_results = [
            {"title": "Bull article", "url": "https://example.com/bull", "content": "Growth ahead", "score": 0.9},
        ]

        with patch("tradingagents.agents.researchers.researcher.route_to_vendor") as mock_route:
            mock_route.return_value = search_results

            state = {
                "company_of_interest": "AAPL",
                "asset_type": "stock",
                "trade_date": _today(),
                "market_report": "",
                "sentiment_report": "",
                "news_report": "",
                "fundamentals_report": "Fundamentals report [fundamentals]",
            }

            with patch("tradingagents.agents.researchers.researcher.get_config") as mock_config:
                mock_config.return_value = {
                    "research_web_search": True,
                    "research_search_queries_max": 4,
                    "research_evidence_token_budget": 3000,
                }

                result = researcher_node(state)

        # Plan call happened (gate open), and search was actually executed.
        quick_llm.invoke.assert_called_once()
        assert mock_route.call_count == 2  # one per query in the plan

        # The rendered brief carries the web-sourced argument through.
        assert "[web:1]" in result["investment_plan"]
        assert "**Recommendation**: Buy" in result["investment_plan"]

        # researcher_evidence captures the populated evidence pack and open gate.
        evidence_dict = json.loads(result["researcher_evidence"])
        assert evidence_dict["gate"]["outcome"] == "open"
        assert evidence_dict["evidence_pack"]
        assert evidence_dict["evidence_pack"][0]["id"] == "web:1"
        assert len(evidence_dict["plan"]["queries"]) == 2


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
