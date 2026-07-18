---
name: financial-news-analyst
description: "Call this skill when the user asks for a structured news sentiment analysis for a stock, company or symbol"
---

# Financial News Analyst

Produce a structured news + sentiment analysis for `<TICKER>`. Output is the shared JSON envelope from `../SCHEMA.md`. **No narrative, no article-by-article summary, no markdown to chat.** Final chat message is one line per the schema's chat-output rule.

This skill is wired to the shared SQLite memory core (`tradingagents/memory/`, exposed via the
`memory_*` MCP tools in `mcp_server.py`) following the pattern validated by the quant skill
(memory-system pilot, issue #8) — mechanical repetition, no new design (issue #10). Unlike the
quant skill, whose `signal`/`confidence` come from a deterministic script that never sees past
context, this skill's `conservative`/`risky` ratings come from your own reading of the fetched
articles (Steps 1–4) — there is no script to structurally shield them from Past Context. The
determinism guarantee here is therefore procedural, not structural: complete Steps 1–4 using only
the article evidence, **before** consulting Past Context (loaded in Step 0). Past Context may only
be woven into the `summary` sentence in Step 5 — it must never change a per-article score, the
aggregate counts, or the `conservative`/`risky` ratings (and therefore never changes `signal`/
`confidence`, which are derived from those ratings).

## Step 0 — Memory: resolve pending + load past context (mandatory, before Step 1)

Agent id for all memory calls in this skill: `"financial-news-analyst"` (matches the `name:` in
this file's frontmatter).

1. Call `memory_resolve_pending` with `agent="financial-news-analyst"`, `ticker=<TICKER>` to fill
   in `forward_return`/`lesson` on any of this skill's own past decisions for `<TICKER>` whose
   horizon has elapsed. Ignore the return value (a list of resolved row ids, possibly empty) — it
   is a side effect, not an input to this run.
2. Call `memory_get_past_context` with `agent="financial-news-analyst"`, `ticker=<TICKER>` to
   retrieve prior resolved lessons (same-ticker + cross-ticker) as a markdown block.
3. Carry that markdown forward as **"Past Context"** and inject it only in Step 5 when writing
   `summary`: if it contains prior lessons (i.e. it is not just the "no prior lessons yet"
   placeholder), let it inform the tone/emphasis of `summary` (e.g. flag if a similar setup
   previously missed). It must not change `conservative`, `risky`, `signal`, `confidence`, or any
   other `details` field — those come from Steps 1–4, verbatim, reasoned about before Past Context
   is even read.

If either call errors (returns a string starting with `"ERROR:"`), proceed without past context —
do not abort the run over a memory-core failure.

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

- `signal` = `conservative.rating` — fixed by Step 4, before Past Context (Step 0) is consulted;
  never adjusted using Past Context or anything else.
- `confidence` — also fixed by Step 4's ratings, never adjusted using Past Context:
  - `HIGH` if mean of `conservative.confidence` and `risky.confidence` > 0.7
  - `MEDIUM` if > 0.4
  - else `LOW`
- `summary`: one line, e.g. `"Q1 beat plus AI partnership headlines outweigh tariff overhang — conservative BUY"`. This is the one place Past Context (Step 0.3) may be woven in — a clause noting a relevant prior lesson, still consistent with the ratings already fixed above.

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

## Step 7 — Memory: store decision

Call `memory_store_decision` (agent id `"financial-news-analyst"`, same as Step 0) with:

| Argument | Value |
|---|---|
| `agent` | `"financial-news-analyst"` |
| `ticker` | `<TICKER>` |
| `date` | `<YYYY-MM-DD>` — the envelope's `date` |
| `signal` | the envelope's `signal` (verbatim) |
| `confidence` | the envelope's `confidence` mapped to a numeric score via `news_memory.py`'s `confidence_to_score` (`HIGH -> 1.0`, `MEDIUM -> 0.6`, `LOW -> 0.3` — the same convention `skills/quant/compute_indicators.py` and `skills/trader/score_trader.py` use) |
| `key_drivers` | `news_memory.py`'s `build_key_drivers(details)` — `{"top_positive_fundamentals", "top_negative_fundamentals", "top_positive_sentiment", "top_negative_sentiment", "conservative", "risky"}` taken verbatim from the envelope's `details` |
| `thesis` | the envelope's `summary` |

`news_memory.py` exposes `confidence_to_score` and `build_key_drivers` as pure functions of the
already-written envelope's `details` — compute them yourself (either by importing those two
functions from `skills/news/news_memory.py` if running in the same Python process, or by
reproducing the trivial mapping/reshaping shown above) before calling `memory_store_decision`.

This call is idempotent on `(agent, ticker, date)` — a duplicate call for a ticker/date already
recorded today is a harmless no-op, so it is safe to call unconditionally every run. Do not let a
memory-core error (a string starting with `"ERROR:"`) block writing the envelope or the chat
output — the envelope file (Step 6) is already on disk by this point; log/ignore the error and
continue.

## Step 8 — Chat output

Exactly one line:

```
<TICKER> financial-news-analyst: <signal> (<confidence>) → [financial-news-analyst.json](computer://C:\Users\ralph\Documents\Claude\Projects\trading-skills\runs\<YYYYMMDD>\<TICKER>\financial-news-analyst.json)
```
