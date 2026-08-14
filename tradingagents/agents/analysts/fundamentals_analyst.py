import json

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.analysts.fundamentals_computation import (
    build_json_envelope,
    compute,
)
from tradingagents.agents.utils.agent_utils import (
    get_language_instruction,
)


def create_fundamentals_analyst(llm):
    def fundamentals_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]

        # Step 1: Compute ratio tables deterministically
        signal = None
        confidence = None
        details = None
        summary_from_computation = "Unable to compute fundamental ratios"

        try:
            ratio_data = compute(ticker)
            details = {
                "context": ratio_data.get("context"),
                "annual": ratio_data.get("annual"),
                "insider_sentiment": ratio_data.get("insider_sentiment"),
                "forecast": ratio_data.get("forecast"),
                "value": {"signal": None, "confidence": None, "data_confidence": None, "key_ratios": []},
                "growth": {"signal": None, "confidence": None, "data_confidence": None, "key_ratios": []},
            }
            summary_from_computation = "Ratios computed; awaiting value/growth evaluation"
        except Exception as e:
            # Graceful degradation: ratio computation unavailable
            details = {
                "context": {},
                "annual": {},
                "insider_sentiment": "NEUTRAL",
                "forecast": {},
                "value": {"signal": None, "confidence": None, "data_confidence": None, "key_ratios": []},
                "growth": {"signal": None, "confidence": None, "data_confidence": None, "key_ratios": []},
            }
            summary_from_computation = f"Data fetch failed: {str(e)[:50]}"

        # Step 2: LLM evaluates value/growth from pre-computed ratios
        # Build full system message with context (using string format, not template variables)
        ratio_context = json.dumps(details.get("annual", {}), indent=2)
        context_info = json.dumps(details.get("context", {}), indent=2)

        system_message = f"""You are a fundamental analyst evaluating {ticker} on {current_date}.

You have been provided a pre-computed table of financial ratios for the past 3 years.
Your job is to evaluate two dimensions:

1. **Value Investing (Graham/Buffett):** Is the company below intrinsic value?
   - Look at: PE, PB, PS, EV/EBITDA ratios; FCF yield; debt/equity; ROE
   - Output: signal (BUY/HOLD/SELL), confidence (HIGH/MEDIUM/LOW), data_confidence, key_ratios

2. **Growth Investing (Fisher/Lynch):** Is the company growing with strong execution?
   - Look at: revenue growth, margin trends, FCF generation, PEG ratio
   - Output: signal (BUY/HOLD/SELL), confidence (HIGH/MEDIUM/LOW), data_confidence, key_ratios

For each, provide:
- signal: BUY (attractive), HOLD (fairly valued), SELL (overvalued)
- confidence: HIGH (strong conviction), MEDIUM (moderate), LOW (uncertain)
- data_confidence: HIGH/MEDIUM/LOW (quality of underlying financial data)
- key_ratios: list of field names from the annual table that most drove this signal

Return ONLY a JSON object (no markdown, no prose):
```json
{{"value": {{"signal": "BUY|HOLD|SELL", "confidence": "HIGH|MEDIUM|LOW", "data_confidence": "HIGH|MEDIUM|LOW", "key_ratios": []}}, "growth": {{"signal": "BUY|HOLD|SELL", "confidence": "HIGH|MEDIUM|LOW", "data_confidence": "HIGH|MEDIUM|LOW", "key_ratios": []}}}}
```

Do NOT call any tools. Do NOT invent data. Evaluate only from the ratio table provided.

Ratio Table (3 most recent years):
{ratio_context}

Insider Sentiment: {details.get('insider_sentiment')}

Context: {context_info}"""

        system_message += get_language_instruction()

        # Use a simple message structure to avoid LangChain template parsing issues
        from langchain_core.messages import SystemMessage

        prompt_messages = [
            SystemMessage(content=system_message),
            MessagesPlaceholder(variable_name="messages"),
        ]

        prompt = ChatPromptTemplate.from_messages(prompt_messages)

        chain = prompt | llm

        try:
            result = chain.invoke({"messages": state["messages"]})
            if result and hasattr(result, 'content') and result.content:
                llm_response = str(result.content).strip()
                # Try to parse JSON from LLM response
                try:
                    parsed = json.loads(llm_response)
                    if "value" in parsed and "growth" in parsed:
                        details["value"] = parsed["value"]
                        details["growth"] = parsed["growth"]
                except json.JSONDecodeError:
                    pass
        except Exception:
            pass

        # Step 3: Derive top-level signal and confidence
        value_signal = details.get("value", {}).get("signal")
        growth_signal = details.get("growth", {}).get("signal")
        value_conf = details.get("value", {}).get("confidence")
        growth_conf = details.get("growth", {}).get("confidence")

        # Signal: if both agree, use that signal; else HOLD
        signal = value_signal if value_signal == growth_signal else "HOLD"

        # Confidence: if they agree, take higher; if disagree, take lower
        conf_map = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}

        if value_signal == growth_signal:
            confidence = max(value_conf, growth_conf, key=lambda x: conf_map.get(str(x).upper(), 0))
        else:
            confidence = min(value_conf, growth_conf, key=lambda x: conf_map.get(str(x).upper(), 0))

        # Step 4: LLM writes one-line summary
        summary_system = f"""Write a single-line summary of the fundamental analysis for {ticker}:
- Value signal: {value_signal or 'N/A'}, confidence: {value_conf or 'N/A'}
- Growth signal: {growth_signal or 'N/A'}, confidence: {growth_conf or 'N/A'}
- Top-level signal: {signal or 'N/A'}, confidence: {confidence or 'N/A'}

Example: "Below intrinsic value but margins compressing — value BUY, growth HOLD"

Write only the one-line summary, nothing else."""

        summary_system += get_language_instruction()

        summary_messages = [
            SystemMessage(content=summary_system),
            MessagesPlaceholder(variable_name="messages"),
        ]

        summary_prompt_obj = ChatPromptTemplate.from_messages(summary_messages)

        summary_chain = summary_prompt_obj | llm

        try:
            summary_result = summary_chain.invoke({"messages": state["messages"]})
            if summary_result and hasattr(summary_result, 'content') and summary_result.content:
                llm_summary = str(summary_result.content).strip()
            else:
                llm_summary = summary_from_computation
        except Exception:
            llm_summary = summary_from_computation

        # Step 5: Build JSON envelope
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
            "fundamentals_report": envelope_json,
        }

    return fundamentals_analyst_node
