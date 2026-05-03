---
name: financial-news-analyst
description: "Financial news and macro analysis for the last 7 days (or user‑defined period). Use when asked to summarize recent market‑moving news, provide a trading‑oriented macro update, highlight key risks and catalysts, or produce a JSON‑formatted report with BUY/HOLD/SELL assessments. Covers: central‑bank policy, inflation and growth data, equity/FX/commodity moves, geopolitical and regulatory events, and structured JSON summaries of key events and positions."
---

# Financial News Analyst Skill (English)

## Skill Name
`financial_news_analyst`

## Skill Description
The **Financial News Analyst** skill automatically collects, analyzes, and summarizes recent financial and macroeconomic news from the last 7 days (or another user‑defined period) and turns it into a concise, trading‑oriented report.

Instead of receiving pre‑pasted news text, the skill uses web search or financial‑news plugins/APIs to fetch relevant articles and data from major financial sources. The output includes:
- A structured report in English.
- A JSON‑formatted events table (`key_events`).
- A BUY/HOLD/SELL assessment with confidence level (`assessments` or `overall_assessment`).

---

## When to Use This Skill
This skill should be used **whenever the user wants an up‑to‑date, macro‑ and market‑focused analysis based on recent financial news**, for example:

- When the user asks for:
  - “What’s the latest macro update for the last [X] days?”
  - “Summarize the most important market‑moving news of the past week.”
  - “Give me a trading‑oriented overview of recent central‑bank and economic data.”
  - “What are the key risks and opportunities in global markets right now?”

- When the user is interested in:
  - Interest‑rate and central‑bank policy.
  - Equity, FX, or commodity‑market moves.
  - Geopolitical or regulatory events that could impact markets.

Do **not** use this skill when:
- The user only wants general explanations of financial concepts (e.g., “Explain quantitative easing”).
- The user requests historical backtests or quantitative strategy code without news context.
- The User explicitly asks for a simple Q&A or non‑news‑based analysis (e.g., “Tell me about Company X” without time‑frame or macro context).

The skill should be triggered **only when the request explicitly or implicitly involves recent financial or macro news and market implications**.

---

## Input
The skill receives:
- Optional time range (e.g., “last 7 days”, “last 30 days”).
- Optional focus (e.g., “rates only”, “equity markets”, “EM macro”, “Energy sector”).

If no constraints are given, the skill defaults to:
- Time range: **last 7 days**.
- Focus: **global macro and financial markets**.

---

## Behavior and Responsibilities
The skill should:

1. **Fetch financial and macro news**
   - Use a web search or financial‑news plugin (e.g., a function like `search_query(query, days=7)`) to retrieve relevant articles and data.
   - Focus on:
     - Central‑bank decisions and communications.
     - Inflation, growth, employment, and fiscal data.
     - Major equity, FX, and commodity‑market moves.
     - Geopolitical or regulatory events that could impact markets.
   - Prioritize reputable sources (see “Preferred Sources” below).
   - Call the search function multiple times with different queries (e.g., “central bank policy last 7 days”, “inflation data last week”, “equity markets last 7 days”).

2. **Analyze and structure the information**
   - Summarize the most relevant developments in a clear, concise way suitable for traders and portfolio managers.
   - Avoid speculative predictions; base conclusions on reported facts, data releases, and official statements.
   - Highlight:
     - Emerging trends.
     - Shifts in sentiment or volatility.
     - Upcoming catalysts or risks.

3. **Generate a structured output**
   - Provide a well‑structured report in English with clear sections (see “Output Structure”).
   - At the end, include:
     - A JSON object `key_events` listing the most important events.
     - A JSON object `assessments` with BUY/HOLD/SELL recommendations and confidence levels.

---

## Output Structure

### 1. Report Body (Natural Language)
The report should be written in English and contain the following sections:

- **Global Macro Recap**  
  - 2–4 paragraphs summarizing the most important macro developments over the specified period.

- **Equity Markets**  
  - Overview of major indices and large‑cap / sector‑relevant moves.

- **FX & Rates**  
  - Key FX moves and relevant central‑bank or rate‑policy news.

- **Commodities & Sectors**  
  - Movement in oil, gold, and other relevant commodities; remarks on sensitive sectors.

- **Key Risks & Catalysts**  
  - Brief list of upcoming data releases, political events, or structural risks.

---

### 2. JSON‑Formatted Events Table
At the end of the report, the skill must output a **JSON object** (not Markdown) with a list of key events. Example:

```json
{
  "key_events": [
    {
      "event": "Fed leaves policy rate unchanged but signals slower hiking pace",
      "source_domain": "reuters.com",
      "date": "2026-05-01",
      "impact": "Neutral-to-moderately-bullish for equities, mildly-bearish for USD",
      "relevant_for": "US equities, USD, rates"
    },
    {
      "event": "Eurozone inflation cools faster than expected",
      "source_domain": "ft.com",
      "date": "2026-04-30",
      "impact": "Bullish for Euro bonds, slightly-bearish for EUR FX",
      "relevant_for": "EUR, European bonds"
    },
    {
      "event": "Oil prices rise on supply concerns from Middle East tensions",
      "source_domain": "cnbc.com",
      "date": "2026-04-29",
      "impact": "Bullish for oil, bullish for energy equities",
      "relevant_for": "Oil, energy stocks"
    }
  ]
}
```

Fields in `key_events`:
- `event`: short, clear description of the event.
- `source_domain`: domain of the source (e.g., `bloomberg.com`, `reuters.com`).
- `date`: string in ISO format `YYYY-MM-DD`.
- `impact`: qualitative assessment (e.g., “Bullish”, “Bearish”, “Neutral”, with nuance).
- `relevant_for`: asset class or region (e.g., “US equities”, “EUR”, “Oil”).

---

### 3. BUY/HOLD/SELL Assessment with Confidence
At the very end, the skill must append a **BUY/HOLD/SELL** assessment with a **confidence level**, in JSON format.

For a single overall view:

```json
{
  "overall_assessment": {
    "asset_class": "global_equities",
    "action": "BUY",
    "confidence": "High",
    "reason": "Latest macro data shows inflation is easing without triggering a hard landing; central banks are pausing rate hikes, supporting valuations."
  }
}
```

For multiple asset classes or regions:

```json
{
  "assessments": [
    {
      "asset_class": "global_equities",
      "action": "BUY",
      "confidence": "High",
      "reason": "Macro backdrop is supportive; valuations remain reasonable."
    },
    {
      "asset_class": "US_rates",
      "action": "HOLD",
      "confidence": "Medium",
      "reason": "Policy is on hold; more data needed before clear directional move."
    },
    {
      "asset_class": "oil",
      "action": "SELL",
      "confidence": "Medium",
      "reason": "Supply concerns are overblown; demand is slowing."
    }
  ]
}
```

Allowed values:
- `asset_class`: clear label (e.g., `global_equities`, `EURUSD`, `oil`, `US_treasuries`).
- `action`: exactly one of `"BUY"`, `"HOLD"`, `"SELL"`.
- `confidence`: exactly one of `"High"`, `"Medium"`, `"Low"`.
- `reason`: 1–2 concise sentences explaining the rationale based on the news and data.

---

## Integration with Your Plugin / Tool API
The skill is expected to use an existing web‑search or news‑fetching function, for example:

- `search_query(query: str, days: int = 7)`  
  - Fetches financial news articles matching the query over the last `days` days.
  - The skill may call this function multiple times with different queries (e.g., “central bank policy last 7 days”, “inflation data last week”, “equity markets last 7 days”).

Example usage pattern:
1. Define a list of search queries based on macro and asset‑class focus.
2. Call `search_query(query, days=7)` for each relevant theme.
3. Parse and filter the results to prioritize:
   - Major central‑bank announcements.
   - High‑impact economic data releases.
   - Significant equity, FX, or commodity moves.
4. Build the report and JSON outputs from these filtered results.

---

## Preferred Sources for Web Search
The skill should focus searches on reputable financial‑news domains, such as:

- `bloomberg.com`
- `reuters.com`
- `ft.com`
- `cnbc.com`
- `marketwatch.com`
- `wsj.com`
- `finance.yahoo.com`
- `investing.com`

If your plugin or web‑search layer supports domain‑specific filters (e.g., via `site:` parameters or a domain‑restriction flag), the skill should use them to narrow results to these domains where possible.

---

## Example Skill Workflow

1. **User query:**  
   - “Give me a macro update for the last 7 days”

2. **Skill actions:**
   - Call `search_query("central bank policy last 7 days", days=7)`.
   - Call `search_query("inflation data last week", days=7)`.
   - Call `search_query("major equity markets moves last 7 days", days=7)`.
   - Filter results by domain (e.g., `bloomberg.com`, `reuters.com`) if the plugin allows it.

3. **Skill generates:**
   - A structured report in English (sections as above).
   - A JSON object `key_events` with the most important events.
   - A JSON object `assessments` with BUY/HOLD/SELL recommendations and confidence levels.

4. **Final output structure (simplified):**
   - Natural language report.  
   - A code block with JSON for `key_events`.  
   - A code block with JSON for `assessments` (or `overall_assessment` if only one asset class is relevant).