import json
from typing import Optional

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
)
from tradingagents.dataflows.stockstats_utils import load_ohlcv
from tradingagents.agents.analysts.market_indicators_computation import (
    compute_indicators,
    build_json_envelope,
)


def create_market_analyst(llm):

    def market_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        instrument_context = get_instrument_context_from_state(state)

        # Step 1: Compute indicators deterministically from OHLCV
        signal = None
        confidence = None
        details = None
        summary_from_computation = "Unable to compute indicators"

        try:
            data = load_ohlcv(ticker, current_date)
            if not data.empty:
                # Convert to list of dicts matching yfinance format
                records = data.to_dict('records')
                computation_result = compute_indicators(records, ticker)
                signal = computation_result["signal"]
                confidence = computation_result["confidence"]
                summary_from_computation = computation_result["summary"]
                details = computation_result["details"]
            else:
                details = {
                    "as_of": current_date,
                    "close": None,
                    "market_regime": None,
                    "indicators": [],
                    "convergence": {"confirms": [], "conflicts": [], "missing": []},
                    "trade_setup": None,
                }
        except Exception as e:
            # Graceful degradation: OHLCV unavailable
            details = {
                "as_of": current_date,
                "close": None,
                "market_regime": None,
                "indicators": [],
                "convergence": {
                    "confirms": [],
                    "conflicts": [],
                    "missing": ["All indicators unavailable"]
                },
                "trade_setup": None,
            }
            summary_from_computation = f"Data fetch failed: {str(e)[:50]}"

        # Step 2: LLM writes one-line summary (refine the computed summary if desired)
        system_message = (
            f"""You are a market analyst reviewing pre-computed technical indicator analysis.

The technical analysis has been computed deterministically for {ticker} on {current_date}.
Your job is to write a **single-line summary** that articulates the trading signal:
- Signal: {signal or 'N/A'}
- Confidence: {confidence or 'N/A'}
- Pre-computed summary: {summary_from_computation}

Keep your summary concise (one line, under 100 characters).
You may refine the pre-computed summary for clarity and tone, but it must remain consistent
with the signal ({signal or 'N/A'}) and confidence ({confidence or 'N/A'}).

Do NOT:
- Change the signal or confidence
- Estimate missing indicators
- Call tools or reference external data

Write only the one-line summary, nothing else."""
            + get_language_instruction()
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    system_message,
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        chain = prompt | llm.get_llm()

        try:
            result = chain.invoke({"messages": state["messages"]})
            if result and hasattr(result, 'content') and result.content:
                llm_summary = str(result.content).strip()
            else:
                llm_summary = summary_from_computation
        except Exception:
            llm_summary = summary_from_computation

        # Step 3: Build JSON envelope and serialize to string for market_report field
        envelope_json = build_json_envelope(
            signal=signal,
            confidence=confidence,
            summary=llm_summary,
            details=details,
            ticker=ticker,
            date=current_date,
        )

        return {
            "messages": [],
            "market_report": envelope_json,
        }

    return market_analyst_node
