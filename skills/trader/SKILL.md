---
name: trader
description: "Call this skill when asked for a trading decision. Synthesize fundamental, news, and quant analyst outputs into a final BUY, HOLD, or SELL decision for a given stock ticker."
---

# Trader

## Role
You orchestrate the three analyst skills, synthesize their JSON outputs, decide BUY/HOLD/SELL, write a JSON envelope, and produce a PDF report. You do **not** rerun their analyses — you read their files.

The shared output contract is in `../SCHEMA.md`. Read it before proceeding.

> **Execution environment — never use computer control.** All work runs through the file tools
> and the **bash shell sandbox** (Linux). Never request desktop/computer access, the desktop
> Terminal, or a browser app — these skills do not need them. If a shell command fails (e.g. a
> "file not found" from a `C:\...` path), the fix is to re-run it with the **Linux mount path**,
> not to escalate to computer-use.

---

## Step 0 — Dispatch analyst subagents (mandatory)

For ticker `<TICKER>` and today's date `<YYYYMMDD>`, **always** dispatch all three analyst skills as subagents via the `Agent` tool with explicit model overrides. Do **not** ask the user for permission between steps. Do **not** print their narrative output — each subagent writes its own JSON file.

**Dispatch all three in a single parallel batch** (one `Agent` tool call per subagent, all issued together in the same message). Skip a dispatch only if its JSON file for today already exists.

Base path: `C:\Users\ralph\Documents\Claude\Projects\trading-skills\runs\<YYYYMMDD>\<TICKER>\`

| # | Skill | Model | Output file |
|---|---|---|---|
| 1 | `fundamental-analyst` | **sonnet** | `fundamental-analyst.json` |
| 2 | `financial-news-analyst` | **sonnet** | `financial-news-analyst.json` |
| 3 | `quant-indicator-analyst` | **haiku** | `quant-indicator-analyst.json` |

**Agent prompt template** (fill in TICKER, YYYYMMDD, and skill name):
```
Run the <skill-name> skill for ticker <TICKER> on date <YYYYMMDD>.
The skill writes its output to:
C:\Users\ralph\Documents\Claude\Projects\trading-skills\runs\<YYYYMMDD>\<TICKER>\<skill-name>.json
Verify the file exists before finishing.
```

After the batch returns, `Read` each of the three output files to confirm they exist. If any file is missing, re-invoke that single analyst once. If it still fails, abort with a single error line in chat.

---

## Step 1 — Load analyst envelopes

Use `Read` on the three JSON files above. Validate each has the envelope fields `skill`, `ticker`, `date`, `signal`, `confidence`, `details`. If any field is missing or the file is unreadable, re-invoke that sub-skill once. If it still fails, abort with a single error line in chat and do not write a trader envelope.

---

## Steps 2–5 — Score (run script)

Run `score_trader.py` in the shell to perform signal extraction, weighted scoring,
conflict checks, and confidence derivation:

> **Shell path note (important).** The shell is a Linux sandbox, **not** Windows. Do not pass
> `C:\...` paths to bash — they fail, and a failed script run must **not** be worked around with
> computer-use / "use my computer". The project folder
> `C:\Users\ralph\Documents\Claude\Projects\trading-skills` is mounted in the sandbox at the
> Linux path given in your environment (e.g. `/sessions/<id>/mnt/trading-skills`). `cd` into that
> mount and run the script with paths relative to it.

```bash
cd <trading-skills-mount>   # Linux mount of the project folder (see environment)
python TradingAgents/skills/trader/score_trader.py runs/<YYYYMMDD>/<TICKER>
```

The script reads the three analyst JSONs from the run directory and prints JSON with:
`signals`, `composite_score`, `thresholds`, `analyst_aggregates`, `conflicts`,
`signal`, `confidence`.

Parse the output — these fields go directly into the trader envelope (Step 6).

**After the script:** dispatch a **Haiku subagent** via the `Agent` tool (model: **haiku**)
to write `rationale` and `risk_note`. Build the prompt by substituting the actual values:

```
For ticker <TICKER>, write two fields based on the scoring results below.

Analyst aggregates: <fundamental|news|quant aggregate signals>
Composite score: <S>  Thresholds: buy ≥ 0.15, sell ≤ -0.15
Signal: <SIGNAL>  Confidence: <CONFIDENCE>
Conflicts: <conflicts list or "none">
Fundamental summary: <details.summary from fundamental-analyst.json>
News summary: <details.summary from financial-news-analyst.json>
Quant summary: <details.summary from quant-indicator-analyst.json>

Respond with ONLY this JSON (no other text):
{
  "rationale": "<3–5 sentences: key drivers, agreement/conflict, notable risks>",
  "risk_note": "<one sentence: primary risk>"
}
```

Parse the subagent's JSON response to extract `rationale` and `risk_note`.

---

## Step 6 — Write trader envelope

Write to `C:\Users\ralph\Documents\Claude\Projects\trading-skills\runs\<YYYYMMDD>\<TICKER>\trader.json` via the `Write` tool.

### `details` payload

```json
{
  "signals": [
    {
      "id": "F1", "source": "fundamental", "raw": "BUY|HOLD|SELL",
      "signal_weight": 0.0, "analyst_weight": 0.35, "weighted_score": 0.0
    }
  ],
  "composite_score": 0.0,
  "thresholds": { "buy": 0.15, "sell": -0.15 },
  "analyst_aggregates": {
    "fundamental": "BUY|HOLD|SELL",
    "news":        "BUY|HOLD|SELL",
    "quant":       "BUY|HOLD|SELL"
  },
  "conflicts": ["<short string or empty>"],
  "rationale": "<3–5 sentence narrative — explain key drivers, agreement/conflict, and notable risks>",
  "risk_note": "<one sentence on primary risk>",
  "inputs": {
    "fundamental_json": "computer://...fundamental-analyst.json",
    "news_json":        "computer://...financial-news-analyst.json",
    "quant_json":       "computer://...quant-indicator-analyst.json"
  },
  "report_pdf": "computer://C:\\Users\\ralph\\Documents\\Claude\\Projects\\trading-skills\\runs\\<YYYYMMDD>\\<TICKER>\\<TICKER>_trading_report_<YYYYMMDD>.pdf"
}
```

`signals` must contain all 8 entries (F1, F2, F3, N1, N2, N3, N4, Q1, Q2 — note: the ids are F1..F3, N1..N4, Q1..Q2; that is 9 rows).

---

## Step 7 — Build PDF report

Invoke the `pdf` skill via the `Skill` tool to build the report. Provide it with:

**Output path**: `C:\Users\ralph\Documents\Claude\Projects\trading-skills\runs\<YYYYMMDD>\<TICKER>\<TICKER>_trading_report_<YYYYMMDD>.pdf`

**Content (in this exact order):**

1. **Cover**
   - Title: `Trading Analysis Report — <TICKER>`
   - Date: `<YYYY-MM-DD>`
   - Subtitle: `Decision: <SIGNAL>   |   Conviction: <CONFIDENCE>   |   Composite: <S>`

2. **1. Fundamental Analysis** — pretty-printed summary of `fundamental-analyst.json`:
   - Top-line: signal, confidence, value & growth signals, insider sentiment.
   - A table of the latest annual valuation, profitability, balance sheet, cashflow rows.
   - Forecast table.

3. **2. News & Sentiment Analysis** — from `financial-news-analyst.json`:
   - Top-line: signal, confidence, conservative & risky ratings.
   - Top positive / negative fundamentals and sentiment items as bulleted lists.
   - Fundamentals + sentiment counts table.

4. **3. Quantitative / Technical Analysis** — from `quant-indicator-analyst.json`:
   - Top-line: signal, confidence, market regime, close.
   - Indicators table (indicator | value | prev | trend | signal | role).
   - Convergence (confirms / conflicts / missing).
   - Trade setup (bias, entry, stop, take-profit, risk-reward).

5. **4. Trader Decision** — from the trader envelope just written:
   - Signal extraction & scoring table (id | source | raw | signal_weight | analyst_weight | weighted_score) with a Total row showing composite `S`.
   - Conflict check result.
   - Decision rationale (3–5 sentences from `details.rationale`).
   - Risk note.

**Do not embed raw JSON in the PDF.** The PDF is the human-readable report; the four `.json`
files in the run directory are the canonical machine-readable copies. Render only the
pretty-printed summaries and tables above — no monospace envelope dumps.

If the `pdf` skill cannot be invoked or fails, fall back to `reportlab` inline:
```
pip install reportlab --break-system-packages
```
…using `SimpleDocTemplate` (letter, 0.75" margins) and `getSampleStyleSheet()`.

---

## Step 8 — Chat output

Exactly one line:

```
<TICKER> trader: <signal> (<confidence>, composite <S>) → [report.pdf](computer://C:\Users\ralph\Documents\Claude\Projects\trading-skills\runs\<YYYYMMDD>\<TICKER>\<TICKER>_trading_report_<YYYYMMDD>.pdf)
```

Do not summarise the report content. Do not echo sub-skill outputs. The PDF and the four JSON files are the deliverable.
