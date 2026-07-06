import json
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.utils.agent_utils import (
    get_global_news,
    get_language_instruction,
    get_news,
    get_instrument_context_from_state,
)
from tradingagents.agents.analysts.news_computation import (
    NewsAnalystOutput,
    build_json_envelope,
    derive_signal_and_confidence,
)


def create_news_analyst(llm):
    def news_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        instrument_context = get_instrument_context_from_state(state)

        # Step 1: Fetch news using existing tools via agent-with-tools loop
        # The agent will call get_news and get_global_news as needed and aggregate results
        tools = [
            get_news,
            get_global_news,
        ]

        # Step 2: LLM fetches news and scores articles internally, outputting structured JSON
        # Build prompt instructing the LLM to:
        # 1. Fetch news articles (via tools)
        # 2. Score each article for fundamentals impact and sentiment impact
        # 3. Aggregate into top items per quadrant
        # 4. Provide conservative and risky ratings
        system_message = f"""You are a financial news analyst for {ticker} on {current_date}.

Your task:
1. Fetch recent news articles using the available tools (get_news for ticker-specific, get_global_news for macroeconomic)
2. For each article, internally score:
   - **Fundamentals impact**: POSITIVE/NEUTRAL/NEGATIVE with confidence 0.0-1.0
   - **Market sentiment**: POSITIVE/NEUTRAL/NEGATIVE with confidence 0.0-1.0
3. Aggregate the results and identify:
   - Top 3 strongest positive/negative headlines per dimension (fundamentals, sentiment)
   - Count of articles per polarity with average confidence
4. Provide two action proposals:
   - **conservative**: rating (BUY/HOLD/SELL) + confidence (0.0-1.0) assuming cautious risk tolerance
   - **risky**: rating (BUY/HOLD/SELL) + confidence (0.0-1.0) assuming aggressive risk tolerance

Return ONLY a valid JSON object matching this structure (no markdown, no prose):
{{
  "articles_analyzed": <number>,
  "window_days": <number>,
  "top_positive_fundamentals": ["<headline ≤120 chars>", ...],
  "top_negative_fundamentals": ["<headline ≤120 chars>", ...],
  "top_positive_sentiment": ["<headline ≤120 chars>", ...],
  "top_negative_sentiment": ["<headline ≤120 chars>", ...],
  "fundamentals_counts": {{"positive": {{"count": <n>, "avg_confidence": <0.0-1.0>}}, "neutral": {{"count": <n>, "avg_confidence": <0.0-1.0>}}, "negative": {{"count": <n>, "avg_confidence": <0.0-1.0>}}}},
  "sentiment_counts": {{"positive": {{"count": <n>, "avg_confidence": <0.0-1.0>}}, "neutral": {{"count": <n>, "avg_confidence": <0.0-1.0>}}, "negative": {{"count": <n>, "avg_confidence": <0.0-1.0>}}}},
  "conservative": {{"rating": "BUY|HOLD|SELL", "confidence": <0.0-1.0>}},
  "risky": {{"rating": "BUY|HOLD|SELL", "confidence": <0.0-1.0>}}
}}

Do NOT:
- Include per-article scores or full headlines in the output
- Return markdown tables or narrative text
- Estimate missing data; only use fetched articles
- Adjust confidences based on past context or memory

Aim for 10-20 articles spanning the last 30 days.""" + get_language_instruction()

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " You have access to the following tools: {tool_names}.\n{system_message}"
                    "For your reference, the current date is {current_date}. {instrument_context}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        chain = prompt | llm.bind_tools(tools)
        result = chain.invoke(state["messages"])

        # Step 3: Parse LLM's JSON response
        signal = None
        confidence = None
        details = None
        llm_summary = None

        if result and hasattr(result, 'content') and result.content:
            llm_response = str(result.content).strip()
            try:
                # Try to parse the LLM's JSON output
                parsed = json.loads(llm_response)
                # Validate the structure matches our schema
                llm_output = NewsAnalystOutput(**parsed)

                # Extract the details payload for the envelope
                details = parsed

                # Derive signal and confidence from conservative/risky ratings
                signal, confidence = derive_signal_and_confidence(details)
            except (json.JSONDecodeError, ValueError) as e:
                # Graceful fallback: could not parse LLM's JSON response
                pass

        # Step 4: Build default details if parsing failed
        if details is None:
            details = {
                "articles_analyzed": 0,
                "window_days": 30,
                "top_positive_fundamentals": [],
                "top_negative_fundamentals": [],
                "top_positive_sentiment": [],
                "top_negative_sentiment": [],
                "fundamentals_counts": {
                    "positive": {"count": 0, "avg_confidence": 0.0},
                    "neutral": {"count": 0, "avg_confidence": 0.0},
                    "negative": {"count": 0, "avg_confidence": 0.0},
                },
                "sentiment_counts": {
                    "positive": {"count": 0, "avg_confidence": 0.0},
                    "neutral": {"count": 0, "avg_confidence": 0.0},
                    "negative": {"count": 0, "avg_confidence": 0.0},
                },
                "conservative": {"rating": "HOLD", "confidence": 0.5},
                "risky": {"rating": "HOLD", "confidence": 0.5},
            }
            if signal is None:
                signal, confidence = derive_signal_and_confidence(details)

        # Step 5: LLM writes one-line summary (unless we already have one from parsed output)
        if signal and confidence:
            # Ask LLM to write a summary consistent with the signal/confidence
            summary_system = f"""Write a single-line summary of the news analysis for {ticker}:
- Signal: {signal}, confidence: {confidence}
- Articles analyzed: {details.get('articles_analyzed', 0)}
- Top positive fundamentals: {details.get('top_positive_fundamentals', [])[:1]}
- Top negative fundamentals: {details.get('top_negative_fundamentals', [])[:1]}

Example: "Q1 beat plus AI partnership headlines outweigh tariff overhang — conservative BUY"

Write only the one-line summary, nothing else."""
            summary_system += get_language_instruction()

            summary_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", summary_system),
                    MessagesPlaceholder(variable_name="messages"),
                ]
            )

            summary_chain = summary_prompt | llm.get_llm()

            try:
                summary_result = summary_chain.invoke({"messages": state["messages"]})
                if summary_result and hasattr(summary_result, 'content') and summary_result.content:
                    llm_summary = str(summary_result.content).strip()
            except Exception:
                pass

        if not llm_summary:
            llm_summary = f"Analysis of {details.get('articles_analyzed', 0)} articles from the past {details.get('window_days', 30)} days"

        # Step 6: Build JSON envelope
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
            "news_report": envelope_json,
        }

    return news_analyst_node
