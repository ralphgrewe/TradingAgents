"""Trader: turns the Research Manager's investment plan into a concrete transaction proposal."""

from __future__ import annotations

import functools
import json
from typing import Any

from langchain_core.messages import AIMessage

from tradingagents.agents.schemas import TraderProposal, render_trader_proposal
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    format_analyst_reports_section,
    get_language_instruction,
)
from tradingagents.agents.utils.structured import (
    bind_structured,
)


def _extract_trade_setup(market_report: Any) -> dict | None:
    """Pull the Market Analyst's quant ``trade_setup`` out of ``market_report``.

    ``market_report`` is a JSON envelope (``skill``/``ticker``/``date``/``signal``/
    ``confidence``/``summary``/``details``, see skills/SCHEMA.md) since #30. Its
    ``details.trade_setup`` (``bias``/``entry_trigger``/``stop_loss``/
    ``stop_loss_formula``/``take_profit``/``risk_reward``) is the only
    deterministically-computed entry/stop-loss guidance available anywhere in
    the pipeline, so the Trader surfaces it directly rather than re-deriving
    price levels from prose. Returns ``None`` if ``market_report`` isn't a
    parseable envelope or has no trade_setup (e.g. OHLCV was unavailable).
    """
    if not isinstance(market_report, str):
        return None
    try:
        envelope = json.loads(market_report)
    except json.JSONDecodeError:
        return None
    if not isinstance(envelope, dict):
        return None
    return envelope.get("details", {}).get("trade_setup")


def create_trader(llm):
    structured_llm = bind_structured(llm, TraderProposal, "Trader")

    def trader_node(state, name):
        company_name = state["company_of_interest"]
        asset_type = state.get("asset_type", "stock")
        instrument_context = build_instrument_context(company_name, asset_type)
        investment_plan = state["investment_plan"]

        trade_setup = _extract_trade_setup(state.get("market_report"))
        trade_setup_line = ""
        if trade_setup:
            trade_setup_line = (
                "\n\nQuant trade setup from the Market Analyst (JSON, computed "
                "deterministically from technical indicators — prefer its numeric "
                "`stop_loss` for your stop_loss field, and use `entry_trigger`/"
                f"`take_profit`/`risk_reward` to inform entry_price and reasoning): {json.dumps(trade_setup)}"
            )

        # Format analyst reports section with all four envelopes
        market_report = state.get("market_report")
        sentiment_report = state.get("sentiment_report")
        news_report = state.get("news_report")
        fundamentals_report = state.get("fundamentals_report")
        analyst_reports_section = format_analyst_reports_section(
            market_report, sentiment_report, news_report, fundamentals_report, asset_type=asset_type
        )
        reports_line = f"\n\n{analyst_reports_section}" if analyst_reports_section else ""

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a trading agent analyzing market data to make investment decisions. "
                    "Based on your analysis, provide a specific recommendation to buy, sell, or hold. "
                    "Anchor your reasoning in the analysts' reports and the research plan. "
                    "When a quant trade setup is provided below, ground your entry_price and "
                    "stop_loss fields in it instead of guessing numeric levels."
                    + get_language_instruction()
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Based on a comprehensive analysis by a team of analysts, here is an investment "
                    f"plan tailored for {company_name}. {instrument_context} This plan incorporates "
                    f"insights from current technical market trends, macroeconomic indicators, and "
                    f"social media sentiment. Use this plan as a foundation for evaluating your next "
                    f"trading decision.\n\nProposed Investment Plan: {investment_plan}"
                    f"{trade_setup_line}{reports_line}\n\n"
                    f"Leverage these insights to make an informed and strategic decision."
                ),
            },
        ]

        # Try structured output first
        if structured_llm is not None:
            try:
                structured_result = structured_llm.invoke(messages)
                trader_plan = render_trader_proposal(structured_result)
                # Store both the rendered markdown and the structured data
                return {
                    "messages": [AIMessage(content=trader_plan)],
                    "trader_investment_plan": trader_plan,
                    "trader_structured_data": structured_result.dict(),
                    "sender": name,
                }
            except Exception:
                # Fall back to free-text generation
                pass

        # Free-text fallback
        trader_plan = llm.invoke(messages).content
        return {
            "messages": [AIMessage(content=trader_plan)],
            "trader_investment_plan": trader_plan,
            "sender": name,
        }

    return functools.partial(trader_node, name="Trader")
