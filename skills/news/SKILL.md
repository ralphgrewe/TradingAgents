---
name: financial-news-analyst
description: "Call this skill when the user asks for a structured news sentiment analysis for a stock, company or symbol"
---

Delivering a structured news sentiment analysis for a symbol follows steps and the output is given in a structured format. Apply the following rules to the analysis:
- Collect news first
- For each news researched, evaluate the fundamentals and sentiment contained

## Step 1: Fetch company news
- Use a web search or financial‑news plugin to retrieve relevant articles and data.
- Prioritize reputable sources (see “Preferred Sources” below).
- Remove duplicates

## Step 2: Analyze and structure the information
- Evaluate the fundamentals as POSITIVE, NEUTRAL, NEGATIVE and the sentiment as POSITIVE, NEUTRAL, NEGATIVE for each news article. Give the confidence for the fundamentals and sentiment analysis based on the article.
- Across all articles, summarize up to 3 top positive and negative fundamentals and sentiment news in a suitable depth to give traders and portfolio managers a base for decision

## Step 3: Generate a structures output
- Output a json structure summarizing the analysis in the following way, counting the categories and giving the confidence from 0.0 (very inconfident) to 1.0 (very confident)
{
  "fundamentals": {
    "positive": {
      "count": 0,
      "avg_confidence": 0.0
    },
    "neutral": {
      "count": 0,
      "avg_confidence": 0.0
    },
    "negative": {
      "count": 0,
      "avg_confidence": 0.0
    }
  },
  "sentiment": {
    "positive": {
      "count": 0,
      "avg_confidence": 0.0
    },
    "neutral": {
      "count": 0,
      "avg_confidence": 0.0
    },
    "negative": {
      "count": 0,
      "avg_confidence": 0.0
    }
  }
}

### Step 4: BUY/HOLD/SELL action proposal with confidence
Give an action proposal for a conservative and risky buyer with a confidence from 0.0 (very inconfident) to 1.0 (very confident)in the following json structure:
{
  "conservatice": {
    "rating": "BUY, HOLD or SELL",
    "confidence": 0.0
  },
  "risky": {
    "rating": "BUY, HOLD or SELL".
    "confidence": 0.0
  }
}
