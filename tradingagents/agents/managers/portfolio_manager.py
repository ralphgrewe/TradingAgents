"""Portfolio Manager: synthesises the risk-analyst debate into the final decision.

Uses LangChain's ``with_structured_output`` so the LLM produces a typed
``PortfolioDecision`` directly, in a single call.  The result is rendered
back to markdown for storage in ``final_trade_decision`` so memory log,
CLI display, and saved reports continue to consume the same shape they do
today.  When a provider does not expose structured output, the agent falls
back gracefully to free-text generation.
"""

from __future__ import annotations

from typing import Any

from tradingagents.agents.schemas import PortfolioDecision, render_pm_decision
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
)
from tradingagents.agents.utils.structured import bind_structured


def _format_analyst_reports_section(
    market_report: Any, sentiment_report: Any, news_report: Any, fundamentals_report: Any
) -> str:
    """Format the four analyst reports with reading instructions for JSON envelopes.

    Returns a formatted section to include in the prompt, or an empty string if all
    reports are missing/empty. Gracefully omits individual reports that are empty.
    """
    # Collect non-empty reports
    reports_to_include = []

    if market_report and isinstance(market_report, str):
        reports_to_include.append(
            f"Market research report (JSON envelope): {market_report}"
        )

    if sentiment_report and isinstance(sentiment_report, str):
        reports_to_include.append(
            f"Social media sentiment report (JSON envelope): {sentiment_report}"
        )

    if news_report and isinstance(news_report, str):
        reports_to_include.append(f"Latest world affairs news (JSON envelope): {news_report}")

    if fundamentals_report and isinstance(fundamentals_report, str):
        reports_to_include.append(
            f"Company fundamentals report (JSON envelope): {fundamentals_report}"
        )

    # If no reports available, return empty
    if not reports_to_include:
        return ""

    # Build the section with reading instructions
    section = """The analyst reports below are structured JSON envelopes
(fields: `signal`, `confidence`, `summary`, `details`), not prose. Read the
`summary` for the headline takeaway and cite specific `details` fields
(e.g. technical indicator values and the `trade_setup`, news headline counts
and the conservative/risky ratings, per-source sentiment directions and key
items, or the fundamentals value/growth sub-signals) as supporting evidence —
do not just restate the raw JSON.

Analyst Reports:
"""
    for report in reports_to_include:
        section += f"{report}\n"

    return section


def create_portfolio_manager(llm):
    structured_llm = bind_structured(llm, PortfolioDecision, "Portfolio Manager")

    def portfolio_manager_node(state) -> dict:
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
        analyst_reports_section = _format_analyst_reports_section(
            market_report, sentiment_report, news_report, fundamentals_report
        )
        reports_line = f"{analyst_reports_section}\n\n" if analyst_reports_section else ""

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
- Research Manager's investment plan: **{research_plan}**
- Trader's transaction proposal: **{trader_plan}**
{lessons_line}{reports_line}
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
