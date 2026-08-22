import json

from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.analysts.macro_fundamentals_computation import (
    MacroFundamentalsAnalystOutput,
    build_json_envelope,
    derive_signal_and_confidence,
)
from tradingagents.agents.utils.agent_utils import get_language_instruction
from tradingagents.dataflows.interface import route_to_vendor

_DEFAULT_DETAILS = {
    "macro_regime": "NEUTRAL",
    "regime_rationale": "LLM output unparseable; defaulting to a neutral macro read.",
    "drivers": [],
    "conservative": {"rating": "HOLD", "confidence": 0.5},
    "risky": {"rating": "HOLD", "confidence": 0.5},
}


def create_macro_fundamentals_analyst(llm):
    def macro_fundamentals_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        past_context = state.get("macro_past_context", "")

        # Step 1: Fetch the deterministic macro indicator pack (#131) directly —
        # this is a plain Python call, not an LLM tool round trip (the pack is
        # fetched once; there is nothing for the LLM to call tools for).
        pack_result = route_to_vendor("get_macro_pack", current_date)
        pack_unavailable_note = None
        if isinstance(pack_result, str):
            # Whole-pack fetch failure sentinel (e.g. bad date, total outage).
            # Individual series failures are already absorbed inside the pack
            # itself (see macro_pack.py), so this only fires for total failure.
            pack_unavailable_note = pack_result
            pack = None
            indicators = {}
        else:
            pack = pack_result
            indicators = pack.get("indicators", {})

        indicators_json = json.dumps(indicators, indent=2, sort_keys=True)

        past_context_block = ""
        if past_context:
            past_context_block = (
                f"\n\nYour past macro fundamentals calls on {ticker}:\n{past_context}"
            )

        unavailable_block = (
            f"\n\nNote: the indicator pack could not be fetched ({pack_unavailable_note}). "
            "Treat all indicators as unavailable; do not fabricate values."
            if pack_unavailable_note
            else ""
        )

        system_message = f"""You are a macro fundamentals analyst for {ticker} on {current_date}.

You are given a deterministic, Python-computed snapshot of the macro indicator
universe (policy rates, Treasury yields, inflation, growth, labor, dollar index,
VIX, consumer sentiment, gold; CoT marked unavailable). Each entry already
carries latest_value, prior_value, delta, direction, and a trailing z-score —
do NOT recompute, restate as new arithmetic, or second-guess these numbers;
only read and interpret them.

Macro indicator pack (curr_date={current_date}):
{indicators_json}{unavailable_block}

Your task:
1. Characterize the overall macro regime as one of RISK_ON, RISK_OFF, NEUTRAL,
   or MIXED, with a short rationale (<=240 chars).
2. Identify up to 5 indicators that most drove this read (drivers), each with
   the indicator alias and a short interpretation of its latest/direction/
   z-score (<=160 chars each).
3. Provide two action proposals for how this macro backdrop conditions a
   decision on {ticker}:
   - conservative: rating (BUY/HOLD/SELL) + confidence (0.0-1.0) assuming cautious risk tolerance
   - risky: rating (BUY/HOLD/SELL) + confidence (0.0-1.0) assuming aggressive risk tolerance

Return ONLY a valid JSON object matching this structure (no markdown, no prose):
{{
  "macro_regime": "RISK_ON|RISK_OFF|NEUTRAL|MIXED",
  "regime_rationale": "<...>",
  "drivers": [{{"indicator": "<alias>", "reading": "<...>"}}, ...],
  "conservative": {{"rating": "BUY|HOLD|SELL", "confidence": <0.0-1.0>}},
  "risky": {{"rating": "BUY|HOLD|SELL", "confidence": <0.0-1.0>}}
}}

Do NOT:
- Compute or restate deltas, z-scores, or ratios beyond what is given
- Estimate missing/unavailable indicator values
- Adjust confidences based on past context or memory{past_context_block}""" + get_language_instruction()

        # Use a concrete SystemMessage (not a ("system", text) template tuple):
        # the indicator pack JSON and JSON-schema example both contain literal
        # `{`/`}` characters that LangChain's template parser would otherwise
        # misread as format placeholders (mirrors fundamentals_analyst.py).
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=system_message),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        chain = prompt | llm
        result = chain.invoke({"messages": state["messages"]})

        # Step 2: Parse the LLM's JSON response
        signal = None
        confidence = None
        details = None

        if result and hasattr(result, "content") and result.content:
            llm_response = str(result.content).strip()
            try:
                parsed = json.loads(llm_response)
                MacroFundamentalsAnalystOutput(**parsed)
                details = parsed
                signal, confidence = derive_signal_and_confidence(details)
            except (json.JSONDecodeError, ValueError):
                # Graceful fallback: could not parse LLM's JSON response
                pass

        # Step 3: Build default details if parsing failed
        if details is None:
            details = dict(_DEFAULT_DETAILS)
            if signal is None:
                signal, confidence = derive_signal_and_confidence(details)

        # Step 4: Deterministic one-line summary (no second LLM call — the
        # regime + rationale are already exactly what a summary would restate).
        rationale = (details.get("regime_rationale") or "").strip()
        regime = details.get("macro_regime", "NEUTRAL")
        summary = f"Macro regime: {regime}" + (f" — {rationale}" if rationale else "")

        # Step 5: Build JSON envelope
        envelope_json = build_json_envelope(
            signal=signal,
            confidence=confidence,
            summary=summary,
            details=details,
            ticker=ticker,
            date=current_date,
            pack=pack,
        )

        return {
            "messages": [],
            "macro_report": envelope_json,
        }

    return macro_fundamentals_analyst_node
