"""Portfolio Manager: synthesises the risk-analyst debate into the final decision.

Uses LangChain's ``with_structured_output`` so the LLM produces a typed
``PortfolioDecision`` directly, in a single call.  The result is rendered
back to markdown for storage in ``final_trade_decision`` so memory log,
CLI display, and saved reports continue to consume the same shape they do
today.  When a provider does not expose structured output, the agent falls
back gracefully to free-text generation.

Every invocation goes through ``run_structured_with_tools`` so the
structured-output contract (tool loop, free-text fallback, schema-repair
retry) lives in exactly one place. ``knowledge_base_enabled`` only decides
whether the strategy-wiki tool and a tool-loop round budget are offered:
when it is False the helper is called with no tools and ``max_rounds=0``,
which reduces it to a single structured call plus the shared fallback.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from tradingagents.agents.managers.exceptions import PortfolioDecisionError
from tradingagents.agents.schemas import PortfolioDecision, render_pm_decision
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    format_analyst_reports_section,
    get_language_instruction,
    is_present_text,
)
from tradingagents.agents.utils.rating import RATINGS_5_TIER
from tradingagents.agents.utils.structured import run_structured_with_tools
from tradingagents.agents.utils.wiki_tools import search_strategy_wiki
from tradingagents.dataflows.config import get_config


def create_portfolio_manager(llm):
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

        # Format analyst reports section with all four core envelopes plus optional macro reports
        market_report = state.get("market_report")
        sentiment_report = state.get("sentiment_report")
        news_report = state.get("news_report")
        fundamentals_report = state.get("fundamentals_report")
        macro_report = state.get("macro_report")
        macro_news_report = state.get("macro_news_report")
        analyst_reports_section = format_analyst_reports_section(
            market_report, sentiment_report, news_report, fundamentals_report,
            asset_type=asset_type, macro_report=macro_report, macro_news_report=macro_news_report
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

        # Format risk-debate history section: omit entirely if empty (risk_stage="none",
        # issue #119) rather than interpolating an empty string, mirroring how
        # research_trader_line above omits the investment-plan section for
        # research_stage="none" (#79).
        risk_debate_history_section = (
            f"**Risk Analysts Debate History:**\n{history}\n"
            if is_present_text(history)
            else ""
        )

        # Determine if wiki tool should be available
        config = get_config()
        knowledge_base_enabled = config.get("knowledge_base_enabled", True)
        knowledge_base_tool_max_rounds = config.get("knowledge_base_tool_max_rounds", 2)

        wiki_availability_note = ""
        if knowledge_base_enabled:
            wiki_availability_note = """
**Available Tools:**
You have access to a strategy knowledge base. Consult it when you want to sanity-check
your decision against established trading principles, risk management frameworks, or
regime-specific approaches (e.g., "Should I use mean reversion in this regime?" or
"What does research say about holding through earnings?")."""

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
{risk_debate_history_section}
---

Be decisive and ground every conclusion in specific evidence from the analysts.{wiki_availability_note}{get_language_instruction()}"""

        # One path, always. `knowledge_base_enabled` selects only *what the LLM
        # is offered* (the wiki tool and a tool-loop budget), never *how the
        # structured call is made*: with the knowledge base off we hand
        # run_structured_with_tools no tools and a zero-round budget, so it
        # skips bind_tools, never enters the loop, and degenerates to the single
        # structured call this branch used to make by hand -- while still
        # inheriting the shared fallback (#152) and schema-repair retry (#153).
        #
        # The previous shape kept a second, hand-rolled structured/except block
        # here for the knowledge-base-off case. Every improvement to the shared
        # helper had to be remembered and re-applied to it, and twice it wasn't
        # (#105's gate bug, then #153's retry, which was dead code for every run
        # with knowledge_base_enabled=False).
        #
        # Note the gate is *not* also conditioned on `structured_llm is not
        # None`: run_structured_with_tools binds tools independently of
        # structured-output support and handles an unusable structured binding
        # internally, so gating on it here would silently skip tool binding (and
        # make wiki_availability_note a lie to the LLM) on providers without
        # structured output (issue #105).
        messages = [HumanMessage(content=prompt)]
        tools = [search_strategy_wiki] if knowledge_base_enabled else []
        max_rounds = knowledge_base_tool_max_rounds if knowledge_base_enabled else 0

        structured_result, fallback_text, message_trace = run_structured_with_tools(
            llm,
            messages,
            tools,
            PortfolioDecision,
            max_rounds=max_rounds,
            agent_name="PortfolioManager",
        )

        # Check structured decision requirement (issue #156) before rendering
        require_structured = config.get("portfolio_manager_require_structured_decision", True)

        # Ticker/model identifiers for PortfolioDecisionError messages, computed once.
        # `state` never carries a "ticker" key (see agent_states.py /
        # propagation.py) — "company_of_interest" is the real field. For the
        # model name, `_llm_type` is useless here: ChatOpenAI (which backs most
        # of this codebase's providers) hard-codes it to "openai-chat"
        # regardless of the configured model. `model_name` (ChatOpenAI/Azure)
        # and `model` (Anthropic/Google/Perplexity) are the actual pydantic
        # fields those langchain chat classes expose for the configured model.
        ticker = state.get("company_of_interest", "unknown")
        model_name = getattr(llm, "model_name", None) or getattr(llm, "model", None) or "unknown"

        # Decide which output to use: structured result or fallback text
        if structured_result is not None:
            portfolio_structured_data = structured_result.dict()

            # Validate rating if required
            if require_structured:
                rating = portfolio_structured_data.get("rating")
                if rating is None or rating not in RATINGS_5_TIER:
                    raise PortfolioDecisionError(
                        f"Portfolio Manager produced invalid rating {rating!r} "
                        f"for {ticker} "
                        f"(model: {model_name}, "
                        f"expected one of {RATINGS_5_TIER}) — aborting ticker"
                    )

            final_trade_decision = render_pm_decision(structured_result)
        else:
            # fallback_text is guaranteed to be non-None when structured_result is None
            final_trade_decision = fallback_text
            portfolio_structured_data = None

            # Structured decision is required but missing
            if require_structured:
                raise PortfolioDecisionError(
                    f"Portfolio Manager produced no structured decision for {ticker} "
                    f"(model: {model_name}, "
                    f"structured_result was None, fallback: free text) — aborting ticker"
                )

        # Update risk_debate_state with the judge decision (same logic for both paths)
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

        result = {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": final_trade_decision,
        }

        # Only include portfolio_structured_data if it's not None
        if portfolio_structured_data is not None:
            result["portfolio_structured_data"] = portfolio_structured_data

        return result

    return portfolio_manager_node
