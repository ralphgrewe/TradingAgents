"""Tests for issue #37: market/news/fundamentals no longer route through a
``tool_calls``-checking conditional edge.

Before this fix, `GraphSetup.setup_graph()` wired every analyst — including
`market`/`news`/`fundamentals`, which compute deterministically and never
call tools (#30-#32) — through a conditional edge
(`should_continue_market/news/fundamentals` in `conditional_logic.py`) that
unconditionally read `state["messages"][-1].tool_calls`. Since these three
analysts return `"messages": []`, `last_message` was a stale `HumanMessage`
(no `.tool_calls` attribute), crashing every run with
`AttributeError: 'HumanMessage' object has no attribute 'tool_calls'`.

This module verifies:
1. The compiled graph no longer contains `tools_market`/`tools_news`/
   `tools_fundamentals` nodes, and `setup_graph()` never looks up those keys
   in the `tool_nodes` dict passed to `GraphSetup` (i.e. it works even when
   that dict only has a `"social"` entry).
2. An end-to-end run of the default analyst pipeline — with
   market/news/fundamentals stub nodes returning `"messages": []` and the
   sentiment stub node returning a real `AIMessage`, mirroring the actual
   analysts' behavior — reaches `final_trade_decision` without raising
   `AttributeError`, reproducing the exact bug shape from the issue.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.prebuilt import ToolNode

import tradingagents.graph.setup as graph_setup_module
from tradingagents.graph.conditional_logic import ConditionalLogic
from tradingagents.graph.setup import GraphSetup

pytestmark = pytest.mark.unit


def _dummy_tool(x: str) -> str:
    """Placeholder tool so the "social" ToolNode has something to hold."""
    return x


def _empty_debate_state():
    return {
        "history": "",
        "bull_history": "",
        "bear_history": "",
        "current_response": "",
        "judge_decision": "",
        "count": 0,
    }


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


def _install_stub_node_factories(monkeypatch):
    """Replace every node factory `setup.py` imports with a deterministic stub.

    Keeps this test focused on graph *routing* rather than exercising real
    LLM/tool logic already covered by each node's own test module. The
    analyst stubs mirror the real analysts' message-return shape exactly:
    market/news/fundamentals return no new message (`"messages": []`, the
    behavior that triggers #37), while sentiment returns a real `AIMessage`
    (matching `sentiment_analyst.py`, unaffected by the bug).
    """

    def market_node(state):
        return {"market_report": "market envelope", "messages": []}

    def news_node(state):
        return {"news_report": "news envelope", "messages": []}

    def fundamentals_node(state):
        return {"fundamentals_report": "fundamentals envelope", "messages": []}

    def sentiment_node(state):
        return {
            "sentiment_report": "sentiment prose",
            "messages": [AIMessage(content="sentiment done", tool_calls=[])],
        }

    monkeypatch.setattr(graph_setup_module, "create_market_analyst", lambda llm: market_node)
    monkeypatch.setattr(graph_setup_module, "create_news_analyst", lambda llm: news_node)
    monkeypatch.setattr(
        graph_setup_module, "create_fundamentals_analyst", lambda llm: fundamentals_node
    )
    monkeypatch.setattr(graph_setup_module, "create_sentiment_analyst", lambda llm: sentiment_node)

    def bull_node(state):
        debate = dict(state["investment_debate_state"])
        debate["count"] = debate["count"] + 1
        debate["current_response"] = "Bull: buy the dip"
        debate["bull_history"] = "bull argument"
        return {"investment_debate_state": debate}

    def bear_node(state):
        debate = dict(state["investment_debate_state"])
        debate["count"] = debate["count"] + 1
        debate["current_response"] = "Bear: too risky"
        debate["bear_history"] = "bear argument"
        return {"investment_debate_state": debate}

    def research_manager_node(state):
        return {"investment_plan": "Buy on strength"}

    def trader_node(state):
        return {"trader_investment_plan": "Buy 100 shares"}

    def make_risk_node(speaker, history_key):
        def _node(state):
            risk = dict(state["risk_debate_state"])
            risk["count"] = risk["count"] + 1
            risk["latest_speaker"] = speaker
            risk[history_key] = f"{speaker} argument"
            return {"risk_debate_state": risk}

        return _node

    def portfolio_manager_node(state):
        return {"final_trade_decision": "BUY"}

    monkeypatch.setattr(graph_setup_module, "create_bull_researcher", lambda llm: bull_node)
    monkeypatch.setattr(graph_setup_module, "create_bear_researcher", lambda llm: bear_node)
    monkeypatch.setattr(
        graph_setup_module, "create_research_manager", lambda llm: research_manager_node
    )
    monkeypatch.setattr(graph_setup_module, "create_trader", lambda llm: trader_node)
    monkeypatch.setattr(
        graph_setup_module,
        "create_aggressive_debator",
        lambda llm: make_risk_node("Aggressive", "aggressive_history"),
    )
    monkeypatch.setattr(
        graph_setup_module,
        "create_conservative_debator",
        lambda llm: make_risk_node("Conservative", "conservative_history"),
    )
    monkeypatch.setattr(
        graph_setup_module,
        "create_neutral_debator",
        lambda llm: make_risk_node("Neutral", "neutral_history"),
    )
    monkeypatch.setattr(
        graph_setup_module, "create_portfolio_manager", lambda llm: portfolio_manager_node
    )


def _make_graph_setup(research_stage="debate"):
    """A GraphSetup whose `tool_nodes` dict only has a "social" entry.

    Deliberately omits "market"/"news"/"fundamentals" keys: if `setup_graph()`
    still looked those up (the pre-fix behavior), building the graph would
    raise `KeyError` before we ever get to running it.

    The research_stage parameter controls whether to include the Bull/Bear/
    Research Manager nodes (default "debate" for this test to exercise the
    full pipeline).
    """
    tool_nodes = {"social": ToolNode([_dummy_tool])}
    conditional_logic = ConditionalLogic(max_debate_rounds=1, max_risk_discuss_rounds=1)
    return GraphSetup(
        quick_thinking_llm=None,
        deep_thinking_llm=None,
        tool_nodes=tool_nodes,
        conditional_logic=conditional_logic,
        research_stage=research_stage,
    )


class TestDeadToolNodesRemovedFromGraph:
    def test_market_news_fundamentals_tool_nodes_not_in_graph(self, monkeypatch):
        _install_stub_node_factories(monkeypatch)
        graph_setup = _make_graph_setup()

        workflow = graph_setup.setup_graph(["market", "social", "news", "fundamentals"])

        assert "tools_market" not in workflow.nodes
        assert "tools_news" not in workflow.nodes
        assert "tools_fundamentals" not in workflow.nodes
        # "social" still makes real tool calls, so its ToolNode stays wired.
        assert "tools_social" in workflow.nodes

    def test_graph_compiles_with_only_a_social_tool_node(self, monkeypatch):
        """Building/compiling never does `self.tool_nodes["market"/"news"/"fundamentals"]`."""
        _install_stub_node_factories(monkeypatch)
        graph_setup = _make_graph_setup()

        workflow = graph_setup.setup_graph(["market", "social", "news", "fundamentals"])
        workflow.compile()  # must not raise KeyError


class TestDefaultPipelineNoLongerCrashesOnStaleMessage:
    def test_pipeline_runs_end_to_end_without_attribute_error(self, monkeypatch):
        """Regression test for the exact bug shape in issue #37.

        Reproduces the crash conditions precisely: market (first analyst)
        sees the initial `HumanMessage`, news/fundamentals see the
        "Continue" placeholder left by the previous analyst's Msg Clear
        node — none of these are `AIMessage`s, so the old
        `should_continue_market/news/fundamentals` conditional edges would
        raise `AttributeError: 'HumanMessage' object has no attribute
        'tool_calls'`. With the fix, these three analysts connect straight
        to their Msg Clear node with no conditional edge at all.
        """
        _install_stub_node_factories(monkeypatch)
        graph_setup = _make_graph_setup()

        workflow = graph_setup.setup_graph(["market", "social", "news", "fundamentals"])
        compiled = workflow.compile()

        initial_state = {
            "company_of_interest": "NVDA",
            "trade_date": "2026-07-07",
            "asset_type": "stock",
            "messages": [HumanMessage(content="NVDA")],
            "investment_debate_state": _empty_debate_state(),
            "risk_debate_state": _empty_risk_state(),
        }

        final_state = compiled.invoke(initial_state)

        assert final_state["final_trade_decision"] == "BUY"
        assert final_state["market_report"] == "market envelope"
        assert final_state["news_report"] == "news envelope"
        assert final_state["fundamentals_report"] == "fundamentals envelope"
        assert final_state["sentiment_report"] == "sentiment prose"
        assert final_state["investment_plan"] == "Buy on strength"
        assert final_state["trader_investment_plan"] == "Buy 100 shares"
