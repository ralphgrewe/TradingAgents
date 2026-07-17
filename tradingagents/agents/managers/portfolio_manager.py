"""Portfolio Manager: synthesises the risk-analyst debate into the final decision.

Uses LangChain's ``with_structured_output`` so the LLM produces a typed
``PortfolioDecision`` directly, in a single call.  The result is rendered
back to markdown for storage in ``final_trade_decision`` so memory log,
CLI display, and saved reports continue to consume the same shape they do
today.  When a provider does not expose structured output, the agent falls
back gracefully to free-text generation.
"""

from __future__ import annotations

from tradingagents.agents.schemas import PortfolioDecision, render_pm_decision
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    format_analyst_reports_section,
    get_language_instruction,
    is_present_text,
)
from tradingagents.agents.utils.structured import bind_structured


def create_portfolio_manager(llm):
    structured_llm = bind_structured(llm, PortfolioDecision, "Portfolio Manager")

    def portfolio_manager_node(state) -> dict:
        asset_type = state.get("asset_type", "stock")
        instrument_context = build_instrument_context(state["company_of_interest"])

        history = state["risk_debate_state"]["history"]
        risk_debate_state = state["risk_debate_state"]
        research_plan = state["investment_plan"]
        trader_plan = state["trader_investment_plan"]

        past_context = state.get("past_context", "")
        lessons_line = (
            f"- Lessons from prior decisions and outcomes:\n{past_context}\n"
            if past_context
            else ""
        )

        # Format analyst reports section with all four envelopes
        market_report = state.get("market_report")
        sentiment_report = state.get("sentiment_report")
        news_report = state.get("news_report")
        fundamentals_report = state.get("fundamentals_report")
        analyst_reports_section = format_analyst_reports_section(
            market_report, sentiment_report, news_report, fundamentals_report, asset_type=asset_type
        )
        reports_line = f"{analyst_reports_section}\n\n" if analyst_reports_section else ""

        # Format research/trader context lines: omit if empty (in "none" research stage mode)
        context_lines = []
        if is_present_text(research_plan):
            context_lines.append(f"- Research Manager's investment plan: **{research_plan}**")
        if is_present_text(trader_plan):
            context_lines.append(f"- Trader's transaction proposal: **{trader_plan}**")
        research_trader_line = "\n".join(context_lines)
        if research_trader_line:
            research_trader_line += "\n"

        prompt = f"""As the Portfolio Manager, synthesize the risk analysts' debate and deliver the final trading decision.

{instrument_context}

---

**Rating Scale** (use exactly one):
- **Buy**: Strong conviction to enter or add to position
- **Overweight**: Favorable outlook, gradually increase exposure
- **Hold**: Maintain current position, no action needed
- **Underweight**: Reduce exposure, take partial profits
- **Sell**: Exit position or avoid entry

**Context:**
{research_trader_line}{lessons_line}{reports_line}
**Risk Analysts Debate History:**
{history}

---

Be decisive and ground every conclusion in specific evidence from the analysts.{get_language_instruction()}"""

        # Try structured output first
        if structured_llm is not None:
            try:
                structured_result = structured_llm.invoke(prompt)
                final_trade_decision = render_pm_decision(structured_result)
                # Store both the rendered markdown and the structured data
                new_risk_debate_state = {
                    "judge_decision": final_trade_decision,
                    "history": risk_debate_state["history"],
                    "aggressive_history": risk_debate_state["aggressive_history"],
                    "conservative_history": risk_debate_state["conservative_history"],
                    "neutral_history": risk_debate_state["neutral_history"],
                    "latest_speaker": "Judge",
                    "current_aggressive_response": risk_debate_state["current_aggressive_response"],
                    "current_conservative_response": risk_debate_state["current_conservative_response"],
                    "current_neutral_response": risk_debate_state["current_neutral_response"],
                    "count": risk_debate_state["count"],
                }

                return {
                    "risk_debate_state": new_risk_debate_state,
                    "final_trade_decision": final_trade_decision,
                    "portfolio_structured_data": structured_result.dict(),
                }
            except Exception:
                # Fall back to free-text generation
                pass

        # Free-text fallback
        final_trade_decision = llm.invoke(prompt).content
        new_risk_debate_state = {
            "judge_decision": final_trade_decision,
            "history": risk_debate_state["history"],
            "aggressive_history": risk_debate_state["aggressive_history"],
            "conservative_history": risk_debate_state["conservative_history"],
            "neutral_history": risk_debate_state["neutral_history"],
            "latest_speaker": "Judge",
            "current_aggressive_response": risk_debate_state["current_aggressive_response"],
            "current_conservative_response": risk_debate_state["current_conservative_response"],
            "current_neutral_response": risk_debate_state["current_neutral_response"],
            "count": risk_debate_state["count"],
        }

        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": final_trade_decision,
        }

    return portfolio_manager_node
