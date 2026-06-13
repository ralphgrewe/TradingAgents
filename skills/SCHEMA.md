# Trading Agents — Output Schema (shared contract)

All trading-agents skills produce a single JSON object that follows the envelope below. This is the contract between skills so the trader and the (future) portfolio-manager can read each other's outputs deterministically — without parsing chat or markdown.

## Envelope

Every skill writes exactly one JSON object with this shape:

```json
{
  "skill": "<skill-name>",
  "ticker": "<TICKER>",
  "date": "YYYY-MM-DD",
  "signal": "BUY | HOLD | SELL",
  "confidence": "HIGH | MEDIUM | LOW",
  "summary": "<one-line human-readable verdict>",
  "details": { /* skill-specific payload */ }
}
```

The top-level `signal` and `confidence` are **always present** and are the only fields a generic consumer needs. Per-skill detail payloads are documented in each `SKILL.md`.

## File location

Each skill MUST write its envelope to:

```
C:\Users\ralph\Documents\Claude\Projects\trading-skills\runs\<YYYYMMDD>\<TICKER>\<skill-name>.json
```

- `<YYYYMMDD>` is today's date (use `date +%Y%m%d` via bash if unsure).
- `<TICKER>` is uppercase (e.g. `AAPL`).
- `<skill-name>` matches the `name:` in the SKILL.md frontmatter (`fundamental-analyst`, `financial-news-analyst`, `quant-indicator-analyst`, `trader`).
- Use the `Write` tool — it creates parent directories automatically.

Example: `C:\Users\ralph\Documents\Claude\Projects\trading-skills\runs\20260517\AAPL\fundamental-analyst.json`

## Chat output rule

Skills MUST NOT print the JSON envelope, raw data, narrative analysis, or any long markdown to chat. After writing the file, the final chat message from the skill is exactly one line:

```
<TICKER> <skill-name>: <signal> (<confidence>) → [<skill-name>.json](computer://<absolute-path>)
```

Anything else (debug, errors, conflicts) is fine in chat **only** if the skill failed to produce a valid envelope file.

## Cross-skill conventions

- `signal` ∈ `{BUY, HOLD, SELL}` — uppercase.
- `confidence` ∈ `{HIGH, MEDIUM, LOW}` — uppercase. Numeric 0.0–1.0 confidences from sub-analyses live inside `details`.
- `date` is ISO 8601 (`YYYY-MM-DD`).
- Missing/unavailable values use `null`, not `"N/A"` or empty string.
- All numbers are JSON numbers, not strings.
