# TradingAgents as Claude Skills — Implementation Overview

This document summarizes the **TradingAgents** paper (arXiv 2412.20138v7) and the
agent prompts in `TradingAgents/tradingagents/agents/`, as a basis for
re-implementing the framework as Claude Cowork skills. It is the reference for
building the individual skills under `TradingAgents/skills/`.

---

## 1. What TradingAgents is

A multi-agent LLM framework that simulates a real trading firm. For a given
ticker and trade date it runs a fixed pipeline of specialized agents that
gather data, debate it, decide, risk-check, and emit a final Buy/Sell/Hold
decision. The paper's claim is that role specialization + structured
communication + agentic debate beats single-agent and rule-based baselines on
cumulative return, Sharpe ratio, and max drawdown.

Two design principles drive the whole system and should drive the skill
implementation:

- **Structured communication over free chat.** Agents do *not* pass long
  message histories around. Each analyst writes a self-contained **report**
  into a shared state; downstream agents read the reports they need. This
  avoids the "telephone effect" where context degrades over long runs.
- **Natural-language debate only where it adds value.** Debate is used in
  exactly two places — the bull/bear research debate and the three-way risk
  debate — and each is bounded to a fixed number of rounds and closed by a
  facilitator/manager agent that writes a structured verdict.

## 2. The five-stage pipeline

```
I.  ANALYST TEAM   → 4 analysts, run in sequence, each writes one report
II. RESEARCH TEAM  → Bull vs Bear debate (N rounds) → Research Manager verdict
III. TRADER        → turns the research plan into a concrete trade proposal
IV. RISK TEAM      → Aggressive / Conservative / Neutral debate (N rounds)
V.  FUND MANAGER   → Portfolio Manager writes the final decision
```

Orchestration details from `graph/setup.py` and `graph/conditional_logic.py`:

- Analysts run **sequentially**, not in parallel (market → social → news →
  fundamentals by default). Each analyst loops with its tools until it has
  enough data, then emits its report.
- Research debate: bull and bear alternate. Ends when
  `count >= 2 * max_debate_rounds` (default `max_debate_rounds = 1` → one
  bull + one bear turn), then Research Manager runs.
- Risk debate: aggressive → conservative → neutral, repeating. Ends when
  `count >= 3 * max_risk_discuss_rounds` (default → one turn each), then
  Portfolio Manager runs.
- After the analysts, message history is cleared ("Msg Clear" nodes) — only
  the reports survive into the next stage.

## 3. Shared state (the "communication protocol")

From `agents/utils/agent_states.py`. Every skill reads from and writes to a
shared state. The load-bearing fields:

| Field | Written by | Consumed by |
|-------|-----------|-------------|
| `company_of_interest`, `trade_date` | run input | all |
| `market_report` | Market/Technical Analyst | researchers, risk team |
| `sentiment_report` | Social Media Analyst | researchers, risk team |
| `news_report` | News Analyst | researchers, risk team |
| `fundamentals_report` | Fundamentals Analyst | researchers, risk team |
| `investment_debate_state` | Bull, Bear, Research Manager | Trader |
| `investment_plan` | Research Manager | Trader, Portfolio Manager |
| `trader_investment_plan` | Trader | Risk team, Portfolio Manager |
| `risk_debate_state` | risk debators, Portfolio Manager | Portfolio Manager |
| `final_trade_decision` | Portfolio Manager | output / memory log |
| `past_context` | memory log at run start | Portfolio Manager |

`investment_debate_state` and `risk_debate_state` are small structured records
holding `history`, per-speaker histories, `current_response`/`latest_speaker`,
`count`, and the manager's `judge_decision`.

## 4. The agents → skills

Each agent below becomes one skill (`skills/<name>/SKILL.md`). For each:
**goal**, **inputs**, **tools/data**, **output**, **prompt essence**.

### 4.1 Analyst Team (4 skills)

All four share the same shape: gather data with tools, write one comprehensive
report ending with a summary table (JSON or Markdown). They are independent —
no technical data in the fundamentals skill, no fundamentals in the news skill.

**Fundamentals Analyst** (`skills/fundamental/` — exists)
- Goal: assess intrinsic value from financial statements, earnings, profile,
  insider transactions; flag under/overvalued.
- Tools in original: `get_fundamentals`, `get_balance_sheet`, `get_cashflow`,
  `get_income_statement`. Note: `get_insider_transactions` was imported in the
  original but never added to the tools list — the skill adds it.
- Output: `fundamentals_report` — structured tables + JSON verdict (no prose;
  downstream agents consume structured data directly).
- Structure: 5-year/4-quarter metric table, 12-month insider transactions table
  with net-sentiment summary, 2-year forward prediction, separate Value
  (Graham/Buffett) and Growth (Fisher/Lynch) evaluations, JSON verdict with
  `insider_sentiment`, `value_signal`, `growth_signal`, and per-field
  `confidence` / `fundamentals_confidence` fields.

**Social Media / Sentiment Analyst** (`skills/` — to build)
- Goal: analyze social posts, sentiment scores, recent company-specific news;
  gauge short-term collective investor behavior.
- Tools in original: `get_news` (company-scoped).
- Output: `sentiment_report` — prose + Markdown summary table.

**News Analyst** (`skills/news/` — exists)
- Goal: macro + world-affairs + company news relevant to trading.
- Tools in original: `get_news` (targeted), `get_global_news` (macro).
- Output: `news_report` — prose + a summary table.
- The existing skill formalizes this: per-article fundamentals tag
  (POSITIVE/NEUTRAL/NEGATIVE) + sentiment tag + confidence, then aggregated
  JSON counts and a conservative-vs-risky Buy/Hold/Sell proposal. Keep that.

**Technical / Market / Quant Analyst** (`skills/quant/` — exists)
- Goal: select and read technical indicators, call market regime, produce an
  ATR-based trade setup.
- Tools in original: `get_stock_data`, `get_indicators`. Indicator values are
  **pre-computed in Python** and handed to the model in a "Signal Context"
  table — the model interprets, it does not calculate.
- Output: `market_report` — fixed 5-section format (Market Context, Indicator
  Readings, Convergence & Conflicts, Trade Setup, Indicator Summary as pure
  JSON array). The existing `skills/quant/SKILL.md` already captures this
  exactly — use it as the template for the other analyst skills' rigor.
- Indicator selection rules: ≤6 indicators, diverse categories, ≤1 from the
  MACD family (prefer `macdh`), ≤2 Bollinger bands, always include `atr` when
  a trade setup is required.

### 4.2 Research Team (3 skills)

**Bull Researcher**
- Goal: build the strongest evidence-based bull case; rebut the bear.
- Inputs: all four analyst reports + debate `history` + last bear argument.
- Output: appends a "Bull Analyst: …" turn to `investment_debate_state`.
- Prompt essence: focus on growth potential, competitive advantages, positive
  indicators; engage *conversationally* with the bear's specific points rather
  than listing data.

**Bear Researcher**
- Mirror of the bull: risks/challenges, competitive weaknesses, negative
  indicators; rebut the bull's over-optimistic assumptions. Same inputs,
  appends "Bear Analyst: …".

**Research Manager** (debate facilitator)
- Goal: judge the bull/bear debate and produce an actionable investment plan.
- Inputs: full debate `history`.
- Output: `investment_plan` — **structured** (see §5): a 5-tier rating
  (Buy/Overweight/Hold/Underweight/Sell), a conversational rationale, and
  concrete strategic actions incl. position-sizing guidance.
- Prompt essence: commit to a clear stance when the strongest arguments
  warrant it; reserve Hold for genuinely balanced evidence.

### 4.3 Trader (1 skill)

**Trader**
- Goal: turn the Research Manager's plan into a concrete transaction proposal.
- Inputs: `investment_plan` (+ analyst reports as context).
- Output: `trader_investment_plan` — **structured**: action (Buy/Hold/Sell),
  2–4 sentence reasoning, optional entry price, stop-loss, position sizing.
- Prompt essence: anchor every conclusion in the analysts' reports and the
  research plan; be decisive.

### 4.4 Risk Management Team (4 skills)

Three debators evaluate the trader's decision from fixed risk stances, then the
Portfolio Manager closes it out.

**Aggressive (Risky) Debator** — champion high-reward/high-risk; argue the
caution of the others misses opportunity. Inputs: trader decision + 4 reports +
risk `history` + last conservative & neutral turns. Output conversational, no
special formatting; appends to `risk_debate_state`.

**Conservative (Safe) Debator** — protect assets, minimize volatility; expose
where the trade creates undue exposure. Same input/output shape.

**Neutral Debator** — balanced view; challenge both other stances as too
optimistic or too cautious. Same input/output shape.

**Portfolio Manager / Fund Manager** (facilitator)
- Goal: synthesize the risk debate into the final decision.
- Inputs: risk debate `history`, `investment_plan`, `trader_investment_plan`,
  and `past_context` (prior-decision lessons from the memory log, if any).
- Output: `final_trade_decision` — **structured**: 5-tier rating, executive
  summary (entry, sizing, risk levels, horizon), investment thesis grounded in
  the debate, optional price target and time horizon.

## 5. Structured output schemas

From `agents/schemas.py`. The three decision agents emit typed output, then
render it back to a fixed Markdown shape so display/memory/reports stay stable.
Skills should reproduce these shapes exactly (field names become the model's
output instructions).

- **Rating scales:** `PortfolioRating` = Buy / Overweight / Hold / Underweight /
  Sell (Research Manager + Portfolio Manager). `TraderAction` = Buy / Hold /
  Sell (Trader only — sizing nuance is deferred to the PM).
- **ResearchPlan:** `recommendation`, `rationale`, `strategic_actions`.
- **TraderProposal:** `action`, `reasoning`, `entry_price?`, `stop_loss?`,
  `position_sizing?`.
- **PortfolioDecision:** `rating`, `executive_summary`, `investment_thesis`,
  `price_target?`, `time_horizon?`.

The rendered Markdown always carries a `**Rating**: X` (or `**Action**: X`)
header so a deterministic parser can extract the signal — no extra LLM call.

## 6. Reflection & memory

From `graph/reflection.py` and `default_config.py`. After a decision's outcome
is known, a `Reflector` writes a terse 2–4 sentence prose lesson (was the
directional call right vs. alpha, which thesis part held/failed, one concrete
lesson). These are stored in a memory log and re-injected as `past_context`
into the Portfolio Manager on future runs of the same or related tickers. A
skill-based implementation should keep this loop: a lightweight reflection step
plus a memory file the Portfolio Manager skill reads.

## 7. Backbone model strategy

The paper splits work between **quick-thinking** models (data retrieval, tool
calls, summarization) and **deep-thinking** models (analysis, debate judging,
decision-making). In `setup.py`, analysts/researchers/trader/risk debators use
the quick model; Research Manager and Portfolio Manager use the deep model.
For skills, this maps to: keep analyst/debator skills tight and tool-driven;
allow the manager skills more reasoning room.

## 8. Implementation lessons (from `LEARNINGS.md`)

- Runs were **non-deterministic** across identical inputs. Mitigations: lower
  temperature; **pre-compute numeric values in Python** and pass results in
  the prompt (the quant skill already does this); force explicit decision +
  confidence output to stabilize borderline cases.
- **Do not over-shorten prompts** — cutting critical context tokens made the
  system *less* stable. Instead reduce redundancy and force structured (JSON)
  output to cut output tokens without losing context.
- Naive LLM-converted skills were too complex and triggered unreliably; keep
  each skill's scope strict and its trigger description precise.

## 9. Build order recommendation

1. Finish the three existing analyst skills (`fundamental`, `news`, `quant`)
   and add the **social/sentiment** analyst — they are independent and produce
   the four reports everything else depends on.
2. Bull + Bear researcher skills, then Research Manager (introduces the
   structured `ResearchPlan` output and debate handling).
3. Trader skill (`TraderProposal`).
4. Three risk debator skills + Portfolio Manager (`PortfolioDecision`).
5. Reflection + memory-log skill, and an orchestration layer that runs the
   five stages in order, manages debate-round counts, and clears intermediate
   message history between stages.

## 10. Source references

- Paper: `2412.20138v7.pdf` — §3 role specialization, §4 agent workflow &
  communication protocol, §5–6 experiments.
- Agent prompts: `TradingAgents/tradingagents/agents/{analysts,researchers,managers,trader,risk_mgmt}/`
- Orchestration: `TradingAgents/tradingagents/graph/{setup,conditional_logic,reflection,signal_processing}.py`
- State & schemas: `TradingAgents/tradingagents/agents/utils/agent_states.py`,
  `TradingAgents/tradingagents/agents/schemas.py`
- Config: `TradingAgents/tradingagents/default_config.py`
- Existing skills: `TradingAgents/skills/{fundamental,news,quant}/SKILL.md`
