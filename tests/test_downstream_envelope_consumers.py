"""Tests for issue #33: downstream consumers of the JSON-envelope analyst reports.

After #30-#32, `market_report`, `fundamentals_report`, and `news_report` are
JSON envelope strings (`skill`/`ticker`/`date`/`signal`/`confidence`/`summary`/
`details`, see skills/SCHEMA.md) rather than markdown; `sentiment_report`
(social analyst) intentionally remains prose. This module covers:

1. Node-level checks that the bull/bear researchers and the three risk
   debators tell the model the market/news/fundamentals reports are
   structured JSON (not prose), and that the trader surfaces the Market
   Analyst's quant `trade_setup` for concrete entry/stop-loss guidance.
2. A graph-level integration test (mocked LLM, no network/API calls) that
   drives the compiled research -> trader -> risk -> portfolio-manager
   sub-graph with envelope-shaped analyst reports end-to-end to a
   `final_trade_decision`, mirroring the shape of the full LangGraph
   pipeline from `tradingagents/graph/setup.py` (analysts themselves are out
   of scope for this issue and already covered at the node level by
   `test_market_analyst.py` / `test_fundamentals_analyst.py` /
   `test_news_analyst.py`).
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from langgraph.graph import END, START, StateGraph

from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager
from tradingagents.agents.managers.research_manager import create_research_manager
from tradingagents.agents.researchers.bear_researcher import create_bear_researcher
from tradingagents.agents.researchers.bull_researcher import create_bull_researcher
from tradingagents.agents.risk_mgmt.aggressive_debator import create_aggressive_debator
from tradingagents.agents.risk_mgmt.conservative_debator import create_conservative_debator
from tradingagents.agents.risk_mgmt.neutral_debator import create_neutral_debator
from tradingagents.agents.schemas import (
    PortfolioDecision,
    PortfolioRating,
    ResearchPlan,
    TraderAction,
    TraderProposal,
)
from tradingagents.agents.trader.trader import _extract_trade_setup, create_trader
from tradingagents.agents.utils.agent_states import AgentState
from tradingagents.graph.conditional_logic import ConditionalLogic

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Shared envelope fixtures
# ---------------------------------------------------------------------------

def _market_envelope(stop_loss=95.0):
    return json.dumps(
        {
            "skill": "market-analyst",
            "ticker": "NVDA",
            "date": "2026-07-06",
            "signal": "BUY",
            "confidence": "HIGH",
            "summary": "BUY (trending_up) — 4 bullish, 1 bearish, 0 missing",
            "details": {
                "as_of": "2026-07-06",
                "close": 100.0,
                "market_regime": "trending_up",
                "indicators": [],
                "convergence": {"confirms": [], "conflicts": [], "missing": []},
                "trade_setup": {
                    "bias": "BUY",
                    "entry_trigger": "Pullback to VWMA (98.0)",
                    "stop_loss": stop_loss,
                    "stop_loss_formula": "close - 1.5 * ATR",
                    "take_profit": 115.0,
                    "risk_reward": "1:2.5",
                },
            },
        },
        indent=2,
    )


def _news_envelope():
    return json.dumps(
        {
            "skill": "financial-news-analyst",
            "ticker": "NVDA",
            "date": "2026-07-06",
            "signal": "BUY",
            "confidence": "MEDIUM",
            "summary": "Positive AI partnership headlines outweigh tariff overhang",
            "details": {
                "articles_analyzed": 14,
                "window_days": 30,
                "conservative": {"rating": "BUY", "confidence": 0.6},
                "risky": {"rating": "BUY", "confidence": 0.8},
            },
        },
        indent=2,
    )


def _fundamentals_envelope():
    return json.dumps(
        {
            "skill": "fundamental-analyst",
            "ticker": "NVDA",
            "date": "2026-07-06",
            "signal": "HOLD",
            "confidence": "MEDIUM",
            "summary": "Fairly valued with steady growth",
            "details": {"value": {"signal": "HOLD"}, "growth": {"signal": "BUY"}},
        },
        indent=2,
    )


_SENTIMENT_PROSE = "Retail sentiment is cautiously optimistic with rising mention volume."


def _envelope_state(**overrides):
    state = {
        "company_of_interest": "NVDA",
        "asset_type": "stock",
        "market_report": _market_envelope(),
        "sentiment_report": _SENTIMENT_PROSE,
        "news_report": _news_envelope(),
        "fundamentals_report": _fundamentals_envelope(),
        "investment_debate_state": {
            "history": "",
            "bull_history": "",
            "bear_history": "",
            "current_response": "",
            "judge_decision": "",
            "count": 0,
        },
    }
    state.update(overrides)
    return state


# ---------------------------------------------------------------------------
# Node-level: bull/bear researchers reference the envelope shape
# ---------------------------------------------------------------------------

class TestResearcherPromptsExplainEnvelopeShape:
    def _run_and_capture(self, factory, state):
        captured = {}
        llm = MagicMock()

        def _invoke(prompt):
            captured["prompt"] = prompt
            return MagicMock(content="Argument text.")

        llm.invoke.side_effect = _invoke
        node = factory(llm)
        node(state)
        return captured["prompt"]

    def test_bull_prompt_mentions_json_envelope_and_embeds_reports(self):
        prompt = self._run_and_capture(create_bull_researcher, _envelope_state())
        assert "JSON envelope" in prompt
        assert '"trade_setup"' in prompt
        assert _SENTIMENT_PROSE in prompt

    def test_bear_prompt_mentions_json_envelope_and_embeds_reports(self):
        prompt = self._run_and_capture(create_bear_researcher, _envelope_state())
        assert "JSON envelope" in prompt
        assert '"trade_setup"' in prompt
        assert _SENTIMENT_PROSE in prompt

    def test_no_placeholder_markers_leak_into_prompt(self):
        """Regression guard for the #31 f-string brace-escaping bug."""
        prompt = self._run_and_capture(create_bull_researcher, _envelope_state())
        assert "PLACEHOLDER" not in prompt


class TestRiskDebatorPromptsExplainEnvelopeShape:
    def _run_and_capture(self, factory, state):
        captured = {}
        llm = MagicMock()

        def _invoke(prompt):
            captured["prompt"] = prompt
            return MagicMock(content="Argument text.")

        llm.invoke.side_effect = _invoke
        node = factory(llm)
        node(state)
        return captured["prompt"]

    def _risk_state(self):
        state = _envelope_state()
        state["risk_debate_state"] = {
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
        state["trader_investment_plan"] = "**Action**: Buy"
        return state

    @pytest.mark.parametrize(
        "factory",
        [create_aggressive_debator, create_conservative_debator, create_neutral_debator],
    )
    def test_prompt_mentions_json_envelope(self, factory):
        prompt = self._run_and_capture(factory, self._risk_state())
        assert "JSON envelope" in prompt
        assert '"trade_setup"' in prompt
        assert _SENTIMENT_PROSE in prompt


# ---------------------------------------------------------------------------
# Node-level: Trader surfaces the quant trade_setup
# ---------------------------------------------------------------------------

class TestTraderTradeSetupExtraction:
    def test_extract_trade_setup_from_valid_envelope(self):
        trade_setup = _extract_trade_setup(_market_envelope(stop_loss=88.5))
        assert trade_setup["stop_loss"] == 88.5
        assert trade_setup["bias"] == "BUY"

    def test_extract_trade_setup_returns_none_for_non_json(self):
        assert _extract_trade_setup("Market looks bullish.") is None

    def test_extract_trade_setup_returns_none_for_missing_report(self):
        assert _extract_trade_setup(None) is None
        assert _extract_trade_setup("") is None

    def test_trader_prompt_includes_trade_setup_when_present(self):
        captured = {}
        llm = MagicMock()
        llm.with_structured_output.side_effect = NotImplementedError("no structured output")

        def _invoke(messages):
            captured["messages"] = messages
            return MagicMock(content="**Action**: Buy")

        llm.invoke.side_effect = _invoke
        trader = create_trader(llm)
        trader(
            {
                "company_of_interest": "NVDA",
                "asset_type": "stock",
                "investment_plan": "**Recommendation**: Buy",
                "market_report": _market_envelope(stop_loss=91.0),
            }
        )
        user_content = captured["messages"][1]["content"]
        assert "Quant trade setup" in user_content
        assert '"stop_loss": 91.0' in user_content

    def test_trader_prompt_omits_trade_setup_when_market_report_absent(self):
        """Backward compatible with pre-#30 callers that never set market_report."""
        captured = {}
        llm = MagicMock()
        llm.with_structured_output.side_effect = NotImplementedError("no structured output")

        def _invoke(messages):
            captured["messages"] = messages
            return MagicMock(content="**Action**: Hold")

        llm.invoke.side_effect = _invoke
        trader = create_trader(llm)
        trader(
            {
                "company_of_interest": "NVDA",
                "asset_type": "stock",
                "investment_plan": "**Recommendation**: Hold",
            }
        )
        user_content = captured["messages"][1]["content"]
        assert "Quant trade setup" not in user_content


# ---------------------------------------------------------------------------
# Graph-level integration test: research -> trader -> risk -> PM
# ---------------------------------------------------------------------------

def _make_mock_llm():
    """A single MagicMock LLM usable by every downstream node.

    Free-text nodes (bull/bear researchers, the three risk debators) call
    ``llm.invoke(prompt).content``. Structured-output nodes (research
    manager, trader, portfolio manager) call
    ``llm.with_structured_output(Schema).invoke(prompt)`` and expect a
    parsed Pydantic instance back.
    """
    llm = MagicMock()
    llm.invoke.side_effect = lambda prompt: MagicMock(content="Debate argument text.")

    def _with_structured_output(schema):
        structured = MagicMock()
        if schema is ResearchPlan:
            structured.invoke.return_value = ResearchPlan(
                recommendation=PortfolioRating.BUY,
                rationale="Bull case wins on AI capex tailwind and technical confirmation.",
                strategic_actions="Build the position gradually over the next two weeks.",
            )
        elif schema is TraderProposal:
            structured.invoke.return_value = TraderProposal(
                action=TraderAction.BUY,
                reasoning="Technical trade setup and news sentiment both align bullish.",
                entry_price=99.0,
                stop_loss=95.0,
                position_sizing="5% of portfolio",
            )
        elif schema is PortfolioDecision:
            structured.invoke.return_value = PortfolioDecision(
                rating=PortfolioRating.BUY,
                executive_summary="Enter with a tight stop at 95.0 and scale in over two weeks.",
                investment_thesis="Debate favored bulls; risk team converged on a measured buy.",
                price_target=115.0,
                time_horizon="3-6 months",
            )
        else:
            raise AssertionError(f"Unexpected structured-output schema: {schema}")
        return structured

    llm.with_structured_output.side_effect = _with_structured_output
    return llm


def _build_downstream_graph(llm, max_debate_rounds=1, max_risk_discuss_rounds=1):
    """Compile the research -> trader -> risk -> PM portion of the real pipeline.

    Mirrors the node wiring in `tradingagents/graph/setup.py`, starting at
    "Bull Researcher" instead of the analyst nodes: this issue is about
    adapting the *consumers* of the analyst envelopes, not the analysts
    themselves (already covered by node-level tests for #30-#32), so the
    graph here is seeded directly with envelope-shaped analyst reports.
    """
    conditional_logic = ConditionalLogic(max_debate_rounds, max_risk_discuss_rounds)
    workflow = StateGraph(AgentState)

    workflow.add_node("Bull Researcher", create_bull_researcher(llm))
    workflow.add_node("Bear Researcher", create_bear_researcher(llm))
    workflow.add_node("Research Manager", create_research_manager(llm))
    workflow.add_node("Trader", create_trader(llm))
    workflow.add_node("Aggressive Analyst", create_aggressive_debator(llm))
    workflow.add_node("Conservative Analyst", create_conservative_debator(llm))
    workflow.add_node("Neutral Analyst", create_neutral_debator(llm))
    workflow.add_node("Portfolio Manager", create_portfolio_manager(llm))

    workflow.add_edge(START, "Bull Researcher")
    workflow.add_conditional_edges(
        "Bull Researcher",
        conditional_logic.should_continue_debate,
        {"Bear Researcher": "Bear Researcher", "Research Manager": "Research Manager"},
    )
    workflow.add_conditional_edges(
        "Bear Researcher",
        conditional_logic.should_continue_debate,
        {"Bull Researcher": "Bull Researcher", "Research Manager": "Research Manager"},
    )
    workflow.add_edge("Research Manager", "Trader")
    workflow.add_edge("Trader", "Aggressive Analyst")
    workflow.add_conditional_edges(
        "Aggressive Analyst",
        conditional_logic.should_continue_risk_analysis,
        {"Conservative Analyst": "Conservative Analyst", "Portfolio Manager": "Portfolio Manager"},
    )
    workflow.add_conditional_edges(
        "Conservative Analyst",
        conditional_logic.should_continue_risk_analysis,
        {"Neutral Analyst": "Neutral Analyst", "Portfolio Manager": "Portfolio Manager"},
    )
    workflow.add_conditional_edges(
        "Neutral Analyst",
        conditional_logic.should_continue_risk_analysis,
        {"Aggressive Analyst": "Aggressive Analyst", "Portfolio Manager": "Portfolio Manager"},
    )
    workflow.add_edge("Portfolio Manager", END)

    return workflow.compile()


class TestFullDownstreamPipelineWithEnvelopeReports:
    def test_pipeline_produces_final_trade_decision(self):
        llm = _make_mock_llm()
        graph = _build_downstream_graph(llm)

        initial_state = _envelope_state()
        initial_state["risk_debate_state"] = {
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
        initial_state["messages"] = []

        final_state = graph.invoke(initial_state)

        # The pipeline reached the end and produced a final decision.
        assert final_state["final_trade_decision"]
        assert "Buy" in final_state["final_trade_decision"]

        # Every intermediate stage ran and produced its artifact.
        assert final_state["investment_plan"]
        assert final_state["trader_investment_plan"]
        assert final_state["investment_debate_state"]["bull_history"]
        assert final_state["investment_debate_state"]["bear_history"]
        assert final_state["risk_debate_state"]["aggressive_history"]
        assert final_state["risk_debate_state"]["conservative_history"]
        assert final_state["risk_debate_state"]["neutral_history"]

        # Envelope-shaped analyst reports flowed through unchanged (nodes
        # only read them, they don't mutate the original report fields).
        assert json.loads(final_state["market_report"])["signal"] == "BUY"
        assert final_state["sentiment_report"] == _SENTIMENT_PROSE

        # The trader's rendered plan reflects the quant trade_setup's stop-loss
        # (note: `trader_structured_data`/`portfolio_structured_data`, while
        # returned by the node functions, are not declared on `AgentState` and
        # so are dropped by LangGraph's schema-restricted state merge — a
        # pre-existing gap unrelated to the envelope migration, out of scope
        # here; asserting on the rendered markdown instead).
        assert "95.0" in final_state["trader_investment_plan"]
