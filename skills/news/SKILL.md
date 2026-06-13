---
name: financial-news-analyst
description: "Call this skill when the user asks for a structured news sentiment analysis for a stock, company or symbol"
---

# Financial News Analyst

Produce a structured news + sentiment analysis for `<TICKER>`. Output is the shared JSON envelope from `../SCHEMA.md`. **No narrative, no article-by-article summary, no markdown to chat.** Final chat message is one line per the schema's chat-output rule.

## Step 1 — Fetch news

Use WebSearch / WebFetch and `yfinance_get_ticker_news` to retrieve recent articles. Prioritize reputable financial sources (Bloomberg, Reuters, FT, WSJ, CNBC, Barron's, MarketWatch, company IR pages). Remove duplicates. Aim for 10–20 distinct articles spanning the last 30 days.

## Step 2 — Per-article scoring

For each article, internally score:
- **Fundamentals impact** ∈ `{POSITIVE, NEUTRAL, NEGATIVE}` with confidence 0.0–1.0.
- **Market sentiment** ∈ `{POSITIVE, NEUTRAL, NEGATIVE}` with confidence 0.0–1.0.

Do not output the per-article table. Aggregate into the counts below.

## Step 3 — Aggregate top items

Select up to 3 strongest items per quadrant:
- top positive fundamentals, top negative fundamentals
- top positive sentiment, top negative sentiment

Each is a short string (≤ 120 chars), no source URL, no scores.

## Step 4 — Action proposals

Produce a `conservative` rating and a `risky` rating, each `{ "rating": "BUY|HOLD|SELL", "confidence": 0.0..1.0 }`.

## Step 5 — Derive top-level envelope fields

- `signal` = `conservative.rating`.
- `confidence`:
  - `HIGH` if mean of `conservative.confidence` and `risky.confidence` > 0.7
  - `MEDIUM` if > 0.4
  - else `LOW`
- `summary`: one line, e.g. `"Q1 beat plus AI partnership headlines outweigh tariff overhang — conservative BUY"`.

## Step 6 — Write artifact

Write the envelope to `C:\Users\ralph\Documents\Claude\Projects\trading-skills\runs\<YYYYMMDD>\<TICKER>\financial-news-analyst.json` via the `Write` tool.

## `details` payload

```json
{
  "articles_analyzed": 0,
  "window_days": 30,
  "top_positive_fundamentals": ["<≤120 chars>", "..."],
  "top_negative_fundamentals": ["..."],
  "top_positive_sentiment":    ["..."],
  "top_negative_sentiment":    ["..."],
  "fundamentals_counts": {
    "positive": { "count": 0, "avg_confidence": 0.0 },
    "neutral":  { "count": 0, "avg_confidence": 0.0 },
    "negative": { "count": 0, "avg_confidence": 0.0 }
  },
  "sentiment_counts": {
    "positive": { "count": 0, "avg_confidence": 0.0 },
    "neutral":  { "count": 0, "avg_confidence": 0.0 },
    "negative": { "count": 0, "avg_confidence": 0.0 }
  },
  "conservative": { "rating": "BUY|HOLD|SELL", "confidence": 0.0 },
  "risky":        { "rating": "BUY|HOLD|SELL", "confidence": 0.0 }
}
```

## Step 7 — Chat output

Exactly one line:

```
<TICKER> financial-news-analyst: <signal> (<confidence>) → [financial-news-analyst.json](computer://C:\Users\ralph\Documents\Claude\Projects\trading-skills\runs\<YYYYMMDD>\<TICKER>\financial-news-analyst.json)
```
