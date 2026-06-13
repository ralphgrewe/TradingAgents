import pandas as pd
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_indicators,
    get_language_instruction,
    get_stock_data,
)
from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.stockstats_utils import load_ohlcv
from stockstats import wrap

def compute_signal_context(df: pd.DataFrame) -> str:
    latest = df.iloc[-1]
    prev   = df.iloc[-2]

    atr        = latest['atr']
    rsi        = latest['rsi']
    macdh      = latest['macdh']
    close      = latest['close']
    boll_ub    = latest['boll_ub']
    boll_lb    = latest['boll_lb']
    sma50      = latest['close_50_sma']
    ema10      = latest['close_10_ema']

    bb_pct     = (close - boll_lb) / (boll_ub - boll_lb)   # 0=lower, 1=upper
    sl_1x_atr  = round(close - 1.0 * atr, 2)
    sl_2x_atr  = round(close - 2.0 * atr, 2)
    sl_25x_atr = round(close - 2.5 * atr, 2)
    rsi_dist_ob = round(70 - rsi, 2)   # positive = distance to overbought
    rsi_dist_os = round(rsi - 30, 2)   # positive = distance to oversold

    def trend(now, before):
        if now > before * 1.001: return "Rising"
        if now < before * 0.999: return "Falling"
        return "Flat"

    return f"""
## Pre-Computed Signal Context (do not recalculate — use these values directly)
| Metric                         | Value      |
|-------------------------------|------------|
| Latest close                  | {close}    |
| ATR                           | {atr}      |
| Stop-loss @ 1.0× ATR          | {sl_1x_atr}|
| Stop-loss @ 2.0× ATR          | {sl_2x_atr}|
| Stop-loss @ 2.5× ATR          | {sl_25x_atr}|
| RSI distance to overbought (70)| {rsi_dist_ob}|
| RSI distance to oversold (30) | {rsi_dist_os}|
| Bollinger %B (0=LB, 1=UB)     | {bb_pct:.2f}|
| MACDH trend                   | {trend(macdh, prev['macdh'])}|
| 10 EMA trend                  | {trend(ema10, prev['close_10_ema'])}|
| 50 SMA trend                  | {trend(sma50, prev['close_50_sma'])}|
| Price vs 50 SMA               | {'Above' if close > sma50 else 'Below'}|
"""

def create_market_analyst(llm):

    def market_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        asset_type = state.get("asset_type", "stock")
        instrument_context = build_instrument_context(ticker, asset_type)

        tools = [
            get_stock_data,
            get_indicators,
        ]

        # Fetch data and compute signal context
        try:
            data = load_ohlcv(ticker, current_date)
            if not data.empty:
                df = wrap(data)
                # Trigger calculation of all needed indicators
                df['atr']
                df['rsi']
                df['macdh']
                df['boll_ub']
                df['boll_lb']
                df['close_50_sma']
                df['close_10_ema']
                signal_context = compute_signal_context(df)
            else:
                signal_context = ""
        except Exception as e:
            signal_context = ""

        system_message = ("""You are a quantitative trading analyst. Analyze the provided indicator data for {ticker} ({date_range}) and produce a structured report in **exactly** the following format — no deviations:

        ---
        ## 1. Market Context (2–3 sentences)
        Brief narrative on the overall trend and regime (trending/ranging/volatile).

        ## 2. Indicator Readings
        For each selected indicator, one line:
        - **[Indicator Name]**: Current value = X | Trend = [Rising/Falling/Flat] | Signal = [Bullish/Bearish/Neutral]

        ## 3. Convergence & Conflicts
        List indicators that confirm each other (convergence) and any contradictions. Max 5 bullet points.

        ## 4. Trade Setup
        - **Bias**: BUY / SELL / HOLD
        - **Entry trigger**: [exact condition, e.g., price closes above 34.25 with macdh > 0 and rising]
        - **Stop-loss**: [ATR × multiplier, state multiplier explicitly, e.g., entry − 2.5 × ATR]
        - **Take-profit**: [exact price level or condition]
        - **Risk/Reward ratio**: [numeric, e.g., 1:2.3]

        ## 5. Indicator Summary
        Output this section as a **pure JSON array** — no markdown, no code fences, no explanation. Start with `[` and end with `]`.

        Each object must have exactly these keys:
        {
        "indicator": "<tool_name>",
        "value": <number>,
        "trend": "Rising" | "Falling" | "Flat",
        "signal": "Bullish" | "Bearish" | "Neutral",
        "role": "<one sentence>"
        }

        Example (do not copy values — use real data):
        [
        {"indicator": "close_50_sma", "value": 28.42, "trend": "Rising", "signal": "Bullish", "role": "Confirms medium-term uptrend and acts as dynamic support."},
        {"indicator": "rsi", "value": 60.03, "trend": "Flat", "signal": "Neutral", "role": "No overbought/oversold extreme; trend remains healthy."}
        ]
        ...

                                                    

        ---
        ## Available Indicators
        Select up to 6. Use the **exact** tool_name for get_indicators calls — any deviation will cause the tool call to fail.

        | tool_name      | Category        | Description                                      | Anti-redundancy rule                          |
        |----------------|-----------------|--------------------------------------------------|-----------------------------------------------|
        | close_50_sma   | Moving Average  | Medium-term trend & dynamic support/resistance   |                                               |
        | close_200_sma  | Moving Average  | Long-term trend benchmark, golden/death cross    | Avoid with close_50_sma unless cross matters  |
        | close_10_ema   | Moving Average  | Short-term momentum & entry timing               |                                               |
        | macd           | MACD            | EMA-difference momentum line                     | Pick at most 1 of: macd, macds, macdh         |
        | macds          | MACD            | Signal line (EMA of macd)                        | Pick at most 1 of: macd, macds, macdh         |
        | macdh          | MACD            | Histogram = macd − macds; preferred default      | Pick at most 1 of: macd, macds, macdh         |
        | rsi            | Momentum        | Overbought/oversold via 70/30 thresholds         |                                               |
        | boll           | Volatility      | Bollinger Middle (20 SMA baseline)               | Pick at most 2 of: boll, boll_ub, boll_lb     |
        | boll_ub        | Volatility      | Bollinger Upper Band (~2σ above middle)          | Pick at most 2 of: boll, boll_ub, boll_lb     |
        | boll_lb        | Volatility      | Bollinger Lower Band (~2σ below middle)          | Pick at most 2 of: boll, boll_ub, boll_lb     |
        | atr            | Volatility      | Volatility measure for stop-loss sizing          |                                               |
        | vwma           | Volume          | Volume-weighted moving average                   |                                               |

        {signal_context}

                          
        ## Selection Rules
        1. Call get_stock_data first, then get_indicators with chosen tool_names.
        2. Select indicators that cover diverse categories — avoid two indicators from the same category unless they serve clearly different roles.
        3. From the MACD family, select **at most one**. Prefer macdh — it encodes direction, strength, and divergence in a single value. Only use macd or macds if a crossover signal is the explicit focus.
        4. From Bollinger Bands, select **at most two** (e.g., boll_ub + boll_lb for range trading, or boll + boll_ub for breakout detection).
        5. Always include atr when a Trade Setup (Section 4) is required.
        6. Do not repeat information across report sections.
        7. All numeric values in Section 4 (Trade Setup) must be taken from the Pre-Computed Signal Context table above — do not recalculate stop-loss or trend directions yourself.
        """    
        + get_language_instruction()
        )
        system_message = system_message.format(signal_context=signal_context, ticker=ticker, date_range=current_date)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
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

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "market_report": report,
        }

    return market_analyst_node
