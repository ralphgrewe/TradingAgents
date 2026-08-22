import json

from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.analysts.macro_news_computation import (
    MacroNewsAnalystOutput,
    build_json_envelope,
    compute_category_sentiment_scores,
    derive_signal_and_confidence,
)
from tradingagents.agents.utils.agent_utils import get_language_instruction
from tradingagents.dataflows.macro_news_pack import build_macro_news_pack

_DEFAULT_DETAILS = {
    "articles_analyzed": 0,
    "categories_with_articles": [],
    "category_sentiments": [],
    "conservative": {"rating": "HOLD", "confidence": 0.5},
    "risky": {"rating": "HOLD", "confidence": 0.5},
}


def create_macro_news_analyst(llm):
    def macro_news_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        past_context = state.get("macro_news_past_context", "")

        # Step 1: Fetch the deterministic macro news pack (#133) directly —
        # this is a plain Python call, not an LLM tool round trip (the pack is
        # fetched once; there is nothing for the LLM to call tools for).
        pack_result = build_macro_news_pack(current_date)
        pack_unavailable_note = None
        if not isinstance(pack_result, dict):
            # Unexpected result type.
            pack_unavailable_note = "Could not build macro news pack"
            articles_by_category = {}
            total_articles = 0
        else:
            gate_outcome = pack_result.get("gate", {}).get("outcome", "unknown")
            if gate_outcome != "enabled":
                # Historical date or other gate closure.
                pack_unavailable_note = gate_outcome
                articles_by_category = {}
                total_articles = 0
            else:
                articles_by_category = pack_result.get("categories", {})
                total_articles = pack_result.get("article_count", 0)

        # Step 1b: Gate — skip the LLM call entirely when there is nothing to
        # analyze (gate not enabled / pack fetch failed / zero articles),
        # mirroring researcher.py's gate-then-maybe-call structure: a closed
        # gate must never reach the LLM at all, so a non-compliant response
        # can't fabricate a signal from an empty set. Build a neutral envelope
        # directly in Python instead.
        no_articles_available = pack_unavailable_note is not None or total_articles == 0

        if no_articles_available:
            details = dict(_DEFAULT_DETAILS)
            signal, confidence = derive_signal_and_confidence(details)

            if pack_unavailable_note:
                summary = f"Macro news unavailable ({pack_unavailable_note})"
            else:
                summary = "No macro news available for this date"

            envelope_json = build_json_envelope(
                signal=signal,
                confidence=confidence,
                summary=summary,
                details=details,
                ticker=ticker,
                date=current_date,
            )

            return {
                "messages": [],
                "macro_news_report": envelope_json,
            }

        # Render articles for the prompt (gate open, articles present).
        lines = ["Available macro news, organized by category:"]
        for category, articles in articles_by_category.items():
            lines.append(f"\n**{category.replace('_', ' ').title()}** ({len(articles)} articles):")
            for article in articles[:3]:  # Show max 3 per category in prompt
                lines.append(f"- {article.get('title', 'No title')}")
                summary_text = article.get("summary", "").strip()
                if summary_text:
                    lines.append(f"  {summary_text[:150]}")
        articles_block = "\n".join(lines)

        past_context_block = ""
        if past_context:
            past_context_block = (
                f"\n\nYour past macro news calls on {ticker}:\n{past_context}"
            )

        system_message = f"""You are a macro news sentiment analyst for {ticker} on {current_date}.

You are given macro headlines, organized by category: monetary_policy, inflation_prices,
labor_market, growth_output, markets_volatility, geopolitical_trade. Each article has
a title and summary; the category assignment has already been done deterministically
in Python.

{articles_block}

Your task:
1. For each category that has articles, classify each article's sentiment:
   - Count articles as bullish (positive for stocks), bearish (negative), or neutral
   - Pick up to 2 key headlines per category
   Do not compute an aggregate sentiment score yourself — Python derives it
   from your bullish/bearish/neutral counts.
2. Provide two action proposals for how this macro news backdrop conditions a
   decision on {ticker}:
   - conservative: rating (BUY/HOLD/SELL) + confidence (0.0-1.0) assuming cautious risk tolerance
   - risky: rating (BUY/HOLD/SELL) + confidence (0.0-1.0) assuming aggressive risk tolerance

Return ONLY a valid JSON object matching this structure (no markdown, no prose):
{{
  "articles_analyzed": <total number of articles>,
  "categories_with_articles": ["monetary_policy", "inflation_prices", ...],
  "category_sentiments": [
    {{
      "category": "monetary_policy",
      "bullish_count": <n>,
      "bearish_count": <n>,
      "neutral_count": <n>,
      "top_articles": ["headline ≤100 chars", ...]
    }},
    ...
  ],
  "conservative": {{"rating": "BUY|HOLD|SELL", "confidence": <0.0-1.0>}},
  "risky": {{"rating": "BUY|HOLD|SELL", "confidence": <0.0-1.0>}}
}}

Do NOT:
- Compute or restate raw article counts beyond what is shown
- Compute a sentiment_score yourself (Python computes it from your counts)
- Estimate missing/unavailable news
- Adjust confidences based on past context or memory{past_context_block}""" + get_language_instruction()

        # Use a concrete SystemMessage (not a ("system", text) template tuple):
        # the article pack and JSON-schema example both contain literal `{`/`}`
        # characters that LangChain's template parser would otherwise misread
        # as format placeholders (mirrors macro_fundamentals_analyst.py).
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
                MacroNewsAnalystOutput(**parsed)
                details = parsed
                # Python-compute each category's sentiment_score from the
                # LLM's bullish/bearish/neutral counts — never trust an
                # LLM-claimed sentiment_score as the source of truth.
                compute_category_sentiment_scores(details)
                signal, confidence = derive_signal_and_confidence(details)
            except (json.JSONDecodeError, ValueError):
                # Graceful fallback: could not parse LLM's JSON response
                pass

        # Step 3: Build default details if parsing failed
        if details is None:
            details = dict(_DEFAULT_DETAILS)
            if signal is None:
                signal, confidence = derive_signal_and_confidence(details)

        # Step 4: Deterministic one-line summary (no second LLM call).
        # Summarize the dominant sentiment across categories with articles.
        sentiments = details.get("category_sentiments", [])
        if sentiments:
            avg_sentiment = sum(s.get("sentiment_score", 0) for s in sentiments) / len(sentiments)
            sentiment_label = "Bullish" if avg_sentiment > 0.2 else ("Bearish" if avg_sentiment < -0.2 else "Neutral")
            summary = f"Macro news: {sentiment_label} ({total_articles} articles, {len(sentiments)} categories)"
        else:
            summary = f"Macro news analyzed ({total_articles} articles)"

        # Step 5: Build JSON envelope
        envelope_json = build_json_envelope(
            signal=signal,
            confidence=confidence,
            summary=summary,
            details=details,
            ticker=ticker,
            date=current_date,
        )

        return {
            "messages": [],
            "macro_news_report": envelope_json,
        }

    return macro_news_analyst_node
