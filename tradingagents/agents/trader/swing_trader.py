"""Swing Trader: regime-gated hybrid swing trading decisions with numeric setup levels."""

from __future__ import annotations

import functools
import json

from langchain_core.messages import AIMessage

from tradingagents.agents.schemas import SwingDecision, render_swing_decision
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    format_analyst_reports_section,
    get_language_instruction,
    is_present_text,
)
from tradingagents.agents.utils.structured import (
    bind_structured,
)
from tradingagents.dataflows.config import get_config

from .swing_trader_computation import assemble_swing_precompute, resolve_benchmark_ticker


def _build_swing_prompt(
    state: dict,
    precompute: dict,
    config: dict,
) -> list[dict]:
    """
    Build the prompt for the Swing Trader node.

    Encodes the aggressiveness contract (design section 5):
    - HOLD only if no setup meets min R/R or regime gate blocks → must state which
    - With conviction >= conviction_threshold, must act
    - Earnings in window (unless catalyst): avoid/exit
    - Previous decision must not anchor
    - Every non-HOLD trade fully specified

    Args:
        state: LangGraph AgentState
        precompute: output from assemble_swing_precompute()
        config: config dict

    Returns:
        list of message dicts for the LLM
    """
    company_name = state["company_of_interest"]
    asset_type = state.get("asset_type", "stock")
    instrument_context = build_instrument_context(company_name, asset_type)
    investment_plan = state.get("investment_plan", "")
    past_context = state.get("swing_past_context", "")

    # Format analyst reports (JSON envelopes)
    market_report = state.get("market_report")
    sentiment_report = state.get("sentiment_report")
    news_report = state.get("news_report")
    fundamentals_report = state.get("fundamentals_report")
    analyst_reports_section = format_analyst_reports_section(
        market_report, sentiment_report, news_report, fundamentals_report, asset_type=asset_type
    )
    reports_line = f"\n\n{analyst_reports_section}" if analyst_reports_section else ""

    # Format investment plan section
    plan_line = ""
    if is_present_text(investment_plan):
        plan_line = f"\n\nResearch Brief & Investment Plan: {investment_plan}"

    # Format past context (memory injection)
    past_context_line = ""
    if is_present_text(past_context):
        past_context_line = f"\n\nYour past swing decisions on {company_name} and similar setups:\n{past_context}"

    # Format pre-compute inputs
    regime_gate = precompute.get("regime_gate", {})
    trade_setup = precompute.get("trade_setup")
    relative_strength = precompute.get("relative_strength", {})
    earnings_check = precompute.get("earnings_check", {})

    precompute_lines = ["\n\nPre-computed inputs from technical analysis:"]
    precompute_lines.append(f"- Market Regime: {regime_gate.get('regime', 'unknown')}")
    precompute_lines.append(
        f"- Regime gate: pullback LONG allowed={regime_gate.get('allow_pullback_long', False)}, "
        f"pullback SHORT allowed={regime_gate.get('allow_pullback_short', False)}, "
        f"catalyst allowed={regime_gate.get('allow_catalyst', False)}"
    )
    if regime_gate.get("hold_bias"):
        precompute_lines.append(
            "  (Regime argues for a HOLD bias — reduce conviction / stand aside)"
        )

    if trade_setup:
        precompute_lines.append(f"- Candidate Trade Setup (deterministic): {json.dumps(trade_setup)}")

    roc_20d = relative_strength.get("roc_20d")
    benchmark_roc = relative_strength.get("benchmark_roc")
    rs = relative_strength.get("relative_strength")
    if rs is not None:
        precompute_lines.append(
            f"- 20-day Relative Strength vs Benchmark: {rs:.2f}% "
            f"(stock {roc_20d:.2f}% vs benchmark {benchmark_roc:.2f}%)"
        )
    elif roc_20d is not None:
        precompute_lines.append(
            f"- 20-day Relative Strength vs Benchmark: unavailable (benchmark ROC could not "
            f"be fetched; stock 20-day ROC is {roc_20d:.2f}%)"
        )
    else:
        precompute_lines.append("- 20-day Relative Strength vs Benchmark: unavailable")

    if earnings_check.get("status") == "ok":
        earnings_date = earnings_check.get("earnings_date")
        if earnings_check.get("has_earnings_in_window"):
            precompute_lines.append(
                f"- Earnings Date: {earnings_date} (falls within the candidate holding window)"
            )
            if earnings_check.get("should_avoid_non_catalyst"):
                precompute_lines.append(
                    "  (Earnings in window — avoid non-catalyst setups per swing protocol)"
                )
        else:
            precompute_lines.append(f"- Earnings Date: {earnings_date} (outside the candidate holding window)")
    else:
        precompute_lines.append("- Earnings Date: unknown (calendar unavailable)")

    precompute_section = "\n".join(precompute_lines)

    min_r_r = config.get("swing_trader_min_risk_reward", 1.5)
    conviction_threshold = config.get("swing_trader_conviction_threshold", 0.55)
    max_holding = config.get("swing_trader_max_holding_days", 15)

    aggressiveness_contract = (
        f"\n\nAggressiveness Contract:\n"
        f"1. HOLD only if (a) no setup meets risk/reward >= {min_r_r}, "
        f"or (b) the regime gate blocks the playbook. State which.\n"
        f"2. With conviction >= {conviction_threshold}, you must act (BUY or SELL). "
        f"Do not hide in HOLD from conviction uncertainty.\n"
        f"3. Exit/Avoid when earnings falls inside the holding window UNLESS setup_type='catalyst'.\n"
        f"4. Previous swing decision must not anchor today's call. "
        f"Flip BUY→SELL on reversed evidence; churn is expected.\n"
        f"5. Every Buy/Sell trade specifies entry, stop, target. "
        f"Holding period: 1–{max_holding} trading days (you declare it).\n"
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a swing trading analyst for a short-term (3–15 trading day) "
                "strategy. You read daily analyst reports and research briefs, then make "
                "quick directional decisions with numeric entry/stop/target levels. "
                "Your job is to find regime-validated pullback entries or catalyst-driven "
                "moves, with strict risk/reward discipline and earnings awareness. "
                "You are more aggressive than the Portfolio Manager — you trade conviction "
                "over caution, and you flip decisions when the evidence flips."
                + get_language_instruction()
            ),
        },
        {
            "role": "user",
            "content": (
                f"Make a swing trading decision for {company_name}. "
                f"{instrument_context}\n"
                f"You are analyzing a short-term trading opportunity (3–15 days)."
                f"{plan_line}{reports_line}{precompute_section}{aggressiveness_contract}{past_context_line}\n\n"
                f"Emit your decision: action (Buy/Hold/Sell), conviction (0.0–1.0), "
                f"setup type (pullback/catalyst/none), and full trade spec. "
                f"Be specific with numbers; hand-wave nothing."
            ),
        },
    ]

    return messages


def create_swing_trader(llm):
    """Create the Swing Trader node.

    Args:
        llm: LLM client (quick-thinking or deep-thinking)

    Returns:
        callable node function
    """
    structured_llm = bind_structured(llm, SwingDecision, "SwingTrader")

    def swing_trader_node(state, name):
        config = get_config()

        # Run pre-compute step: regime gate, candidate setup, relative
        # strength (fetches the benchmark ROC live), and the earnings-in-
        # window flag (fetches the earnings calendar live). Both vendor
        # calls degrade to None/"unknown" on failure; see
        # swing_trader_computation.fetch_benchmark_roc / fetch_earnings_calendar.
        ticker = state["company_of_interest"]
        trade_date = state.get("trade_date")
        market_report = state.get("market_report")
        holding_period_days = 5  # candidate window for precompute; LLM declares the final value

        benchmark = resolve_benchmark_ticker(ticker, config)

        precompute = assemble_swing_precompute(
            market_report=market_report,
            ticker=ticker,
            trade_date=trade_date,
            holding_period_days=holding_period_days,
            benchmark=benchmark,
        )

        # Build prompt
        messages = _build_swing_prompt(state, precompute, config)

        # Try structured output first
        if structured_llm is not None:
            try:
                structured_result = structured_llm.invoke(messages)

                # Validate holding_period_days against config
                max_holding = config.get("swing_trader_max_holding_days", 15)
                if structured_result.holding_period_days > max_holding:
                    structured_result.holding_period_days = max_holding

                swing_decision = render_swing_decision(structured_result)
                # Extend structured data with precompute regime and gate info (not part of schema)
                structured_data = structured_result.dict()
                regime_gate = precompute.get("regime_gate", {})
                structured_data["regime"] = regime_gate.get("regime", "unknown")
                structured_data["hold_bias"] = regime_gate.get("hold_bias", False)
                return {
                    "messages": [AIMessage(content=swing_decision)],
                    "swing_trade_decision": swing_decision,
                    "swing_structured_data": structured_data,
                    "sender": name,
                }
            except Exception:
                # Fall back to free-text generation
                pass

        # Free-text fallback
        swing_decision = llm.invoke(messages).content
        return {
            "messages": [AIMessage(content=swing_decision)],
            "swing_trade_decision": swing_decision,
            "sender": name,
        }

    return functools.partial(swing_trader_node, name="Swing Trader")
