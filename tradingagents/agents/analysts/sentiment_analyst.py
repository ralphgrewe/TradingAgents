"""Sentiment analyst — multi-source sentiment analysis for a target ticker.

Previously named ``social_media_analyst``. Renamed and redesigned because
the old version had a prompt that demanded social-media analysis but the
only tool available was Yahoo Finance news — which led LLMs to fabricate
Reddit/X/StockTwits content under prompt pressure (verified live).

The redesigned agent pre-fetches three complementary data sources before
the LLM is invoked and injects them into the prompt as structured blocks:

  1. News headlines     — Yahoo Finance (institutional framing)
  2. StockTwits messages — retail-trader posts indexed by cashtag, with
                           user-labeled Bullish/Bearish sentiment tags
  3. Reddit posts        — r/wallstreetbets, r/stocks, r/investing

The agent does not use tool-calling; the data is in the prompt from
turn 0.

Since issue #71, the sentiment report is a JSON envelope (per
``skills/SCHEMA.md``), matching the market/news/fundamentals analysts:
Python parses the pre-fetched blocks into count/availability fields
(``sentiment_computation.py``), the LLM provides a structured per-source
directional read plus a cross-source synthesis, and Python derives the
top-level ``signal``/``confidence`` from that structured output. A second,
small LLM call writes the one-line ``summary`` consistent with the derived
signal/confidence.

See: https://github.com/TauricResearch/TradingAgents/issues/557
"""

import logging
from datetime import datetime, timedelta

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.analysts.sentiment_computation import (
    SentimentAnalystOutput,
    build_details,
    build_json_envelope,
    build_sources_skeleton,
    derive_signal_and_confidence,
)
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
    get_news,
)
from tradingagents.agents.utils.structured import run_structured_with_tools
from tradingagents.dataflows.reddit import fetch_reddit_posts
from tradingagents.dataflows.stocktwits import fetch_stocktwits_messages

logger = logging.getLogger(__name__)


def _seven_days_back(trade_date: str) -> str:
    return (datetime.strptime(trade_date, "%Y-%m-%d") - timedelta(days=7)).strftime("%Y-%m-%d")


def create_sentiment_analyst(llm):
    """Create a sentiment analyst node for the trading graph.

    Pre-fetches news + StockTwits + Reddit data, injects them into the
    prompt as structured blocks, and produces a JSON-envelope sentiment
    report (``sentiment_report``) from a structured LLM call plus
    Python-derived signal/confidence.
    """

    def sentiment_analyst_node(state):
        ticker = state["company_of_interest"]
        end_date = state["trade_date"]
        start_date = _seven_days_back(end_date)
        instrument_context = build_instrument_context(ticker)

        # Pre-fetch all three sources. Each fetcher degrades gracefully and
        # returns a string (no exceptions surface from here), so the LLM
        # always sees something — either real data or a clear placeholder.
        news_block = get_news.func(ticker, start_date, end_date)
        stocktwits_block = fetch_stocktwits_messages(ticker, limit=30)
        reddit_block = fetch_reddit_posts(ticker)

        # Step 1: Python-side count/availability computation from the
        # already-fetched blocks (never trust the LLM to count).
        sources_skeleton = build_sources_skeleton(news_block, stocktwits_block, reddit_block)

        # Default fallback details: Python-computed counts, null directions.
        # Used verbatim if the LLM call/parse/validation fails below.
        details = build_details(start_date, end_date, sources_skeleton, llm_output=None)

        system_message = _build_system_message(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            news_block=news_block,
            stocktwits_block=stocktwits_block,
            reddit_block=reddit_block,
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "{system_message}\n"
                    "For your reference, the current date is {current_date}. {instrument_context}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(current_date=end_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        # Build prompt for structured output. The sentiment analyst does not
        # use tool-calling; the data is already in the prompt. We invoke the
        # structured LLM through run_structured_with_tools with tools=[] and
        # max_rounds=0 to inherit the shared recovery ladder (schema-repair
        # retry #153, text extraction #162) without a tool loop.
        formatted_prompt = prompt.format_prompt(messages=state["messages"])

        # Convert formatted prompt to HumanMessage for run_structured_with_tools
        # (it expects a list of BaseMessage objects, not a PromptValue).
        messages = [HumanMessage(content=formatted_prompt.to_string())]

        # Step 2: Run the structured-output call through the shared ladder.
        # With tools=[] and max_rounds=0, this degenerates to a single
        # structured call plus the shared fallback/retry/extraction logic.
        #
        # run_structured_with_tools's own docstring documents a "true double
        # failure" mode: the structured call fails/is unsupported *and* the
        # free-text fallback llm.invoke also raises (e.g. a provider outage
        # hitting both calls) -- which propagates uncaught out of the helper.
        # Unlike the Portfolio Manager (which is designed to hard-fail on
        # structured-output failure per #156, via PortfolioDecisionError),
        # the sentiment analyst must never abort the ticker -- it is one of
        # several analyst inputs, not the final decision -- so that
        # exception is caught here and folded into the same "total failure"
        # path as a plain ``structured_result is None`` return: fall through
        # to the Python-only skeleton and let the run continue.
        try:
            structured_result, fallback_text, _message_trace = run_structured_with_tools(
                llm,
                messages,
                tools=[],
                response_model=SentimentAnalystOutput,
                max_rounds=0,
                agent_name="SentimentAnalyst",
            )
        except Exception as exc:
            logger.warning(
                "SentimentAnalyst: structured-output ladder raised an uncaught "
                "exception (%s); using Python-only skeleton with null directions",
                exc,
            )
            structured_result = None

        # Merge structured result (if any) with the Python-computed skeleton
        if structured_result is not None:
            details = build_details(start_date, end_date, sources_skeleton, llm_output=structured_result)
        else:
            # structured_result is None: either the structured call failed entirely
            # (and was logged by run_structured_with_tools, or above if the ladder
            # itself raised) or text extraction was attempted (and logged if it
            # succeeded). Keep the Python-only fallback `details` from above. Logged
            # at WARNING (not DEBUG) since this is a real degradation -- the
            # sentiment stage is producing a null-signal envelope for this run.
            logger.warning(
                "SentimentAnalyst: structured-output parsing failed completely; "
                "using Python-only skeleton with null directions"
            )

        signal, confidence = derive_signal_and_confidence(details)

        # Step 3: A separate, small LLM call writes the one-line summary
        # consistent with the derived signal/confidence (same pattern as
        # the news/market analysts).
        llm_summary = _write_summary(llm, state, ticker, signal, confidence, details)

        envelope_json = build_json_envelope(
            signal=signal,
            confidence=confidence,
            summary=llm_summary,
            details=details,
            ticker=ticker,
            date=end_date,
        )

        return {
            "messages": [],
            "sentiment_report": envelope_json,
        }

    return sentiment_analyst_node


def _write_summary(llm, state, ticker, signal, confidence, details) -> str:
    """Ask the LLM for a one-line summary consistent with the derived
    signal/confidence; fall back to a generic Python-built line on failure.
    """
    overall_direction = details.get("overall_direction")
    caveats = details.get("data_quality", {}).get("caveats", [])
    sources_available = details.get("data_quality", {}).get("sources_available", 0)

    fallback_summary = (
        f"Sentiment read from {sources_available} of 3 sources"
        + (f"; overall direction {overall_direction}" if overall_direction else "")
        + (f" ({'; '.join(caveats)})" if caveats else "")
    )

    if not signal or not confidence:
        return fallback_summary

    summary_system = f"""Write a single-line summary of the sentiment analysis for {ticker}:
- Signal: {signal}, confidence: {confidence}
- Overall direction: {overall_direction}
- Sources available: {sources_available} of 3
- Data quality caveats: {caveats}

Example: "Retail chatter and Reddit engagement lean bullish despite muted news coverage — cautious BUY"

Write only the one-line summary, nothing else.""" + get_language_instruction()

    summary_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", summary_system),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    summary_chain = summary_prompt | llm

    try:
        summary_result = summary_chain.invoke({"messages": state["messages"]})
        if summary_result and hasattr(summary_result, "content") and summary_result.content:
            return str(summary_result.content).strip()
    except Exception:
        pass

    return fallback_summary


def _build_system_message(
    *,
    ticker: str,
    start_date: str,
    end_date: str,
    news_block: str,
    stocktwits_block: str,
    reddit_block: str,
) -> str:
    """Assemble the sentiment-analyst system message with structured data blocks."""
    return f"""You are a financial market sentiment analyst. Your task is to analyze sentiment for {ticker} covering the period from {start_date} to {end_date}, drawing on three complementary data sources that have already been collected for you.

## Data sources (pre-fetched, in this prompt)

### News headlines — Yahoo Finance, past 7 days
Institutional framing. Fact-driven, slower-moving signal.

<start_of_news>
{news_block}
<end_of_news>

### StockTwits messages — retail-trader social platform indexed by cashtag
Fast-moving signal. Each message carries a user-labeled sentiment tag (Bullish / Bearish / no-label) plus the message body.

<start_of_stocktwits>
{stocktwits_block}
<end_of_stocktwits>

### Reddit posts — r/wallstreetbets, r/stocks, r/investing (past 7 days)
Community discussion. Engagement signal via upvote score and comment count. Subreddit character matters (r/wallstreetbets is often contrarian/exuberant; r/stocks more measured; r/investing longer-term).

<start_of_reddit>
{reddit_block}
<end_of_reddit>

## How to analyze this data (best practices)

1. **Read the StockTwits Bullish/Bearish ratio as a leading retail-sentiment signal.** A 70/30 bullish/bearish split is moderately bullish; ≥90/10 may indicate over-extension and contrarian risk; 50/50 is uncertainty. Sample size matters — base rates on the actual message count, not percentages alone.

2. **Look for cross-source divergences.** If news framing is bearish but StockTwits is overwhelmingly bullish, that mismatch is itself a signal — it can mean retail is leaning into a thesis the news flow hasn't caught up to (or vice versa, that retail is chasing while institutions are cautious).

3. **Weight Reddit posts by engagement.** A 400-upvote / 200-comment thread reflects community attention; a 3-upvote post is noise. Read the body excerpts for context — the title alone often misleads.

4. **Distinguish opinion from event.** A news headline ("Nvidia announces $500M Corning deal") is an event; a StockTwits post ("buying NVDA, this is going to moon") is opinion. Both are inputs but should be weighted differently in your conclusions.

5. **Identify recurring narrative themes.** What topic keeps coming up across sources? That's the dominant narrative driving current sentiment.

6. **Be honest about data limits.** If StockTwits returned only a handful of messages, or one or more sources returned an "<unavailable>" placeholder / empty result, set that source's direction and confidence to null — do not guess. If the sources are silent on a given subreddit, say so via divergences/narratives rather than fabricating a read.

7. **Identify catalysts and risks** that emerge across sources — news of upcoming earnings, product launches, competitive threats, macro headlines, etc.

8. **Past sentiment is not predictive.** Your read is signal for the trader to weigh alongside fundamentals and technicals, not a price call.

## Output

Return ONLY a valid JSON object matching this structure (no markdown, no prose, no code fences):
{{
  "news": {{"direction": "POSITIVE|NEUTRAL|NEGATIVE or null", "confidence": <0.0-1.0 or null>, "key_items": ["<=120 chars", ...] (up to 3)}},
  "stocktwits": {{"direction": "POSITIVE|NEUTRAL|NEGATIVE or null", "confidence": <0.0-1.0 or null>, "key_items": ["<=120 chars", ...] (up to 3)}},
  "reddit": {{"direction": "POSITIVE|NEUTRAL|NEGATIVE or null", "confidence": <0.0-1.0 or null>, "key_items": ["<=120 chars", ...] (up to 3)}},
  "overall_direction": "BULLISH|BEARISH|NEUTRAL|MIXED",
  "divergences": ["<=160 chars", ...] (up to 3),
  "narratives": ["...", ...] (up to 3),
  "catalysts": ["...", ...] (up to 3),
  "risks": ["...", ...] (up to 3)
}}

For a source with no usable data (an "<unavailable>" placeholder or an empty result above), set that source's "direction" and "confidence" to null and "key_items" to an empty list.

The JSON keys and the enum values above (POSITIVE/NEUTRAL/NEGATIVE, BULLISH/BEARISH/NEUTRAL/MIXED) must stay exactly as written, in English, regardless of output language. Only the free-text fields (key_items, divergences, narratives, catalysts, risks) follow the language instruction below, if any.

Do NOT:
- Include markdown tables, headings, or narrative prose in your response
- Wrap the JSON in a code fence
- Fabricate data for a source that returned no results
- Adjust your read based on past context or memory
- Compute or state counts/ratios yourself — those are already computed for you elsewhere; focus on interpreting direction and evidence
{get_language_instruction()}"""


# ---------------------------------------------------------------------------
# Backwards-compatibility shim
# ---------------------------------------------------------------------------
def create_social_media_analyst(llm):
    """Deprecated alias for :func:`create_sentiment_analyst`.

    Kept so existing code that imports ``create_social_media_analyst``
    continues to work.

    .. deprecated::
        Import :func:`create_sentiment_analyst` directly instead.
    """
    import warnings
    warnings.warn(
        "create_social_media_analyst is deprecated and will be removed in a "
        "future version. Use create_sentiment_analyst instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return create_sentiment_analyst(llm)
