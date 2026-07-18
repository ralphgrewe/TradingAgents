"""Researcher node: plan–execute–synthesize pipeline for bull/bear research with web search.

Part of research_stage="researcher" mode (#85). Single-shot from the graph's perspective —
exactly two LLM calls (plan + synthesis) *when the gate is open*, one LLM call (synthesis
only) when it's closed — no tool loop, no conditional edges. Everything between the calls
is deterministic Python.

Pipeline:
1. **Gate check** (no LLM, evaluated first): date gate (`is_web_search_allowed`) ∧
   `research_web_search` config ∧ `TAVILY_API_KEY` present. If closed, skip straight to
   synthesis with an empty evidence pack and the appropriate "disabled (...)" reason —
   no plan call, no search calls at all (tool-less arm-B mode: synthesis only).
2. **Plan call** (quick-thinking LLM, gate-open only): input is the four analyst envelopes +
   instrument context; output is a small JSON list of ≤`research_search_queries_max` queries,
   each labeled `bull`/`bear`/`neutral`. Python validates ≥1 bull-seeking and ≥1 bear-seeking
   query and enforces the `research_search_queries_max` cap; on violation, patches in
   template fallbacks (truncating original queries first, if needed, to keep the mandatory
   fallbacks within the cap).
3. **Execute** (no LLM, gate-open only): runs the validated queries via the `web_search`
   vendor and builds the evidence pack (budget `research_evidence_token_budget`).
4. **Synthesis call** (deep-thinking LLM, structured output): envelopes + evidence pack + rubric
   → `ResearchBrief`. Prompt rules (per #78 §8): any argument that cannot be source-tagged must
   be dropped; numeric claims only quoted from tagged sources; must not re-summarize envelopes
   without tagging as envelope-sourced.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime

from pydantic import BaseModel, Field

from tradingagents.agents.schemas import ResearchBrief, render_research_brief
from tradingagents.agents.utils.agent_utils import (
    ANALYST_REPORTS_READING_INSTRUCTIONS,
    build_instrument_context,
    format_analyst_reports_section,
    get_language_instruction,
    is_present_text,
)
from tradingagents.agents.utils.structured import (
    bind_structured,
    invoke_structured_or_freetext,
)
from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.interface import route_to_vendor
from tradingagents.dataflows.tavily_search import (
    build_evidence_pack,
    is_web_search_allowed,
)
from tradingagents.dataflows.errors import VendorError

logger = logging.getLogger(__name__)


class ResearchQuery(BaseModel):
    """A single web search query for the researcher."""

    query: str = Field(description="The search query string")
    type: str = Field(description="Query type: 'bull', 'bear', or 'neutral'")


class ResearchPlan(BaseModel):
    """Output from the plan call: a list of search queries to execute."""

    queries: list[ResearchQuery] = Field(
        min_length=1,
        max_length=10,
        description="List of search queries, each labeled as bull, bear, or neutral",
    )


def _validate_and_patch_plan(
    plan_queries: list[dict], queries_max: int | None = None
) -> tuple[list[dict], bool]:
    """Validate the plan has ≥1 bull-seeking and ≥1 bear-seeking query, and that it
    does not exceed ``queries_max`` (the configured ``research_search_queries_max``).

    Two-phase, in this order:
    1. Truncate to ``queries_max`` first (if set) — a plan longer than the cap has
       its tail dropped.
    2. *Then* check bull/bear balance on the truncated result (not the pre-truncation
       list) — truncation can itself remove the only bull or bear query, so checking
       balance before truncating would miss that. If either is still missing, patch
       in a template fallback, making room for the mandatory fallback(s) *without*
       evicting the originals that already supply the type(s) present.

    Making room (phase 2 detail): the fallbacks cover only the MISSING type(s). The
    first surviving query of each PRESENT type is load-bearing — it is the reason the
    guarantee holds for that side — and must never be dropped to fit the cap. Only
    non-load-bearing filler (neutral queries, or extra bull/bear beyond the first of
    each present type) may be evicted. This closes issue #85, where a blind prefix
    cut (``original[:queries_max - len(fallbacks)]``) could discard the lone real
    bull-or-bear query that had correctly survived phase 1, silently violating the
    invariant this function exists to guarantee.

    Priority when the cap can't hold everything: when ``queries_max`` is too small to
    fit both the required fallbacks and the load-bearing originals (e.g.
    ``queries_max=1`` with one fallback needed, or ``queries_max=1`` with BOTH types
    missing so two fallbacks are needed), the ≥1-bull AND ≥1-bear guarantee wins over
    strictly respecting ``queries_max`` — the returned plan may exceed the cap so both
    types are always represented.

    ``queries_max=None`` skips the cap entirely (back-compat for callers that don't
    have a config value handy). Returns (patched_queries, was_patched).
    """
    original = list(plan_queries)
    was_patched = False

    # Phase 1: cap an over-long raw plan by dropping its tail.
    if queries_max is not None and len(original) > queries_max:
        original = original[:queries_max]
        was_patched = True

    has_bull = any(q.get("type") == "bull" for q in original)
    has_bear = any(q.get("type") == "bear" for q in original)

    fallbacks = []
    if not has_bull:
        fallbacks.append({"query": "bull case catalysts opportunities upside", "type": "bull"})
    if not has_bear:
        fallbacks.append({"query": "risks short thesis downside bearish", "type": "bear"})

    patched = original
    if fallbacks:
        was_patched = True
        if queries_max is None:
            # No cap: just append the mandatory fallbacks.
            patched = original + fallbacks
        else:
            # Phase 2: make room for the fallbacks while preserving the first
            # query of each already-present type (load-bearing — never evicted).
            must_keep: set[int] = set()
            for present_type, present in (("bull", has_bull), ("bear", has_bear)):
                if present:
                    for i, q in enumerate(original):
                        if q.get("type") == present_type:
                            must_keep.add(i)
                            break

            # Budget for surviving originals: leave room for the fallbacks, but
            # never drop a load-bearing query even if that overruns the cap (the
            # guarantee takes priority — see the docstring).
            budget = max(queries_max - len(fallbacks), len(must_keep))
            keep_indices = set(must_keep)
            for i in range(len(original)):
                if len(keep_indices) >= budget:
                    break
                keep_indices.add(i)

            kept = [original[i] for i in sorted(keep_indices)]
            patched = kept + fallbacks

    if was_patched:
        logger.warning(
            "Plan validation failed or exceeded cap (has_bull=%s, has_bear=%s, "
            "queries_max=%s); patching/truncating",
            has_bull, has_bear, queries_max,
        )
    return patched, was_patched


def create_researcher(quick_thinking_llm, deep_thinking_llm):
    """Create the Researcher node function.

    Args:
        quick_thinking_llm: LLM for the plan call
        deep_thinking_llm: LLM for the synthesis call
    """
    structured_brief_llm = bind_structured(deep_thinking_llm, ResearchBrief, "Researcher")

    def researcher_node(state) -> dict:
        config = get_config()
        asset_type = state.get("asset_type", "stock")
        company_name = state["company_of_interest"]
        trade_date = state["trade_date"]
        today = datetime.now().strftime("%Y-%m-%d")

        # Build instrument context and analyst reports
        instrument_context = build_instrument_context(company_name)
        market_report = state.get("market_report", "")
        sentiment_report = state.get("sentiment_report", "")
        news_report = state.get("news_report", "")
        fundamentals_report = state.get("fundamentals_report", "")
        analyst_reports_section = format_analyst_reports_section(
            market_report, sentiment_report, news_report, fundamentals_report, asset_type=asset_type
        )
        reports_line = f"{analyst_reports_section}\n\n" if analyst_reports_section else ""

        # ===== STEP 1: GATE CHECK (no LLM, evaluated BEFORE the plan call) =====
        # Gate = date gate ∧ research_web_search config ∧ API key present. Checked
        # first so a closed gate skips the plan call entirely (tool-less arm-B
        # mode: synthesis only) — not just the search execution. Priority order
        # matches the metadata line: date, then config, then key presence.
        web_search_enabled = config.get("research_web_search", True)
        date_is_today = is_web_search_allowed(trade_date, today)
        api_key_present = bool(os.environ.get("TAVILY_API_KEY"))
        queries_max = config.get("research_search_queries_max", 4)

        if not date_is_today:
            gate_outcome = "disabled (historical date)"
            web_search_status = "disabled (historical date)"
            gate_open = False
            logger.info("Web search disabled: trade_date=%s is not today", trade_date)
        elif not web_search_enabled:
            gate_outcome = "disabled (config)"
            web_search_status = "disabled (config)"
            gate_open = False
            logger.info("Web search disabled: research_web_search=False in config")
        elif not api_key_present:
            gate_outcome = "disabled (no API key)"
            web_search_status = "disabled (no API key)"
            gate_open = False
            logger.info("Web search disabled: TAVILY_API_KEY not set")
        else:
            gate_outcome = None  # resolved after execution below
            web_search_status = "enabled"
            gate_open = True

        plan_queries: list[dict] = []
        was_patched = False
        evidence_pack: list[dict] = []

        if gate_open:
            # ===== STEP 2: PLAN CALL (quick-thinking LLM) — gate-open only =====
            plan_prompt = f"""As the Researcher, design a targeted web search plan to strengthen the bull and bear cases.

{instrument_context}

---

{reports_line}**Task**: Generate a list of up to {queries_max} web search queries that will surface evidence for:
- The bullish case (catalysts, growth opportunities, positive signals)
- The bearish case (risks, competitive threats, headwinds)
- Optional neutral queries for baseline/comparison data

Include at least one query targeting the bullish case and at least one targeting the bearish case.
Each query should be specific and focused. Avoid vague or overlapping searches.{get_language_instruction()}"""

            try:
                plan_response = quick_thinking_llm.invoke(plan_prompt)
                plan_text = plan_response.content if hasattr(plan_response, "content") else str(plan_response)

                # Extract JSON from the response (wrapped in ```json...```  or direct JSON)
                if "```json" in plan_text:
                    json_start = plan_text.index("```json") + 7
                    json_end = plan_text.index("```", json_start)
                    plan_json_str = plan_text[json_start:json_end].strip()
                elif "```" in plan_text:
                    json_start = plan_text.index("```") + 3
                    json_end = plan_text.index("```", json_start)
                    plan_json_str = plan_text[json_start:json_end].strip()
                else:
                    plan_json_str = plan_text

                plan_data = json.loads(plan_json_str)
                plan_queries = plan_data.get("queries", [])
            except Exception as e:
                logger.warning("Failed to parse plan response: %s. Using empty plan.", e)
                plan_queries = []

            # Validate and patch the plan (bull/bear balance + queries_max cap)
            plan_queries, was_patched = _validate_and_patch_plan(plan_queries, queries_max)

            # ===== STEP 3: EXECUTE (deterministic Python, no LLM) =====
            try:
                query_results_list = []
                for query_item in plan_queries:
                    query_str = query_item.get("query", "")
                    if not query_str:
                        continue
                    try:
                        results = route_to_vendor("get_web_search_results", query_str)
                        # Check if results is the DATA_UNAVAILABLE sentinel string
                        if isinstance(results, str) and results.startswith("DATA_UNAVAILABLE"):
                            logger.warning("Web search unavailable for query '%s': %s", query_str, results)
                            gate_outcome = "disabled (no API key)"
                            web_search_status = "disabled (no API key)"
                            evidence_pack = []
                            break
                        query_results_list.append(results if isinstance(results, list) else [])
                    except VendorError as e:
                        logger.warning("Web search failed for query '%s': %s", query_str, e)
                        gate_outcome = f"disabled (search error: {type(e).__name__})"
                        web_search_status = "disabled (vendor error)"
                        evidence_pack = []
                        break

                if gate_outcome is None:  # All queries succeeded
                    # Build evidence pack from results
                    token_budget = config.get("research_evidence_token_budget", 3000)
                    # Collect already-cited URLs from news_report if present
                    already_cited = set()
                    # (simplified: would extract URLs from news_report in production)
                    evidence_pack = build_evidence_pack(query_results_list, already_cited, token_budget)
                    gate_outcome = "open"
                    web_search_status = "enabled"

            except Exception as e:
                logger.warning("Web search execution failed: %s", e)
                gate_outcome = f"error ({type(e).__name__})"
                web_search_status = "disabled (vendor error)"
                evidence_pack = []
        else:
            logger.info(
                "Web search gate closed (%s); skipping plan call (tool-less arm-B mode).",
                gate_outcome,
            )

        # Build researcher_evidence metadata for full-state log
        researcher_evidence_dict = {
            "plan": {
                "queries": plan_queries,
                "was_patched": was_patched,
            },
            "gate": {
                "date_is_today": date_is_today,
                "web_search_enabled": web_search_enabled,
                "outcome": gate_outcome,
            },
            "evidence_pack": evidence_pack,
        }
        researcher_evidence_json = json.dumps(researcher_evidence_dict, indent=2)

        # ===== STEP 3: SYNTHESIS CALL (deep-thinking LLM, structured output) =====
        # Build evidence pack text for the prompt
        evidence_text = ""
        if evidence_pack:
            evidence_text = "**Web Research Evidence:**\n"
            for item in evidence_pack:
                evidence_text += f"\n[{item['id']}] {item['title']}\n"
                evidence_text += f"URL: {item['url']}\n"
                evidence_text += f"Content: {item['content']}\n"
                evidence_text += f"Score: {item.get('score', 'N/A')}\n"

        synthesis_prompt = f"""As the Researcher, synthesize the analyst reports and web research evidence into a balanced investment brief.

{instrument_context}

---

{reports_line}{ANALYST_REPORTS_READING_INSTRUCTIONS}

Analyst Reports:
{f"- Market research report (JSON envelope): {market_report}" if market_report else ""}
{f"- Social media sentiment report (JSON envelope): {sentiment_report}" if sentiment_report else ""}
{f"- Latest world affairs news (JSON envelope): {news_report}" if news_report else ""}
{f"- Fundamentals report (JSON envelope): {fundamentals_report}" if fundamentals_report else ""}

---

{evidence_text}

---

**Your Task:**
1. Distill **1–5 bullish arguments** (each source-tagged to an analyst envelope or web evidence ID)
2. Distill **1–5 bearish arguments** (each source-tagged)
3. Pick your lean: Buy / Overweight / Hold / Underweight / Sell
4. Rate confidence: High / Medium / Low
5. Summarize any new information from web research (1–2 sentences), or None if none

**Stability Rules** (per design §8):
- Any argument you cannot source-tag must be dropped
- Numeric claims may ONLY be quoted from tagged sources, never computed
- Do NOT re-summarize envelope contents without tagging them as envelope-sourced
- If web research was disabled, set new_information to None

{get_language_instruction()}"""

        # Create a render function that includes web research metadata
        def render_brief_with_metadata(brief: ResearchBrief) -> str:
            result_count = len(evidence_pack) if evidence_pack else None
            return render_research_brief(brief, web_search_status, result_count)

        investment_plan = invoke_structured_or_freetext(
            structured_brief_llm,
            deep_thinking_llm,
            synthesis_prompt,
            render_brief_with_metadata,
            "Researcher",
        )

        return {
            "investment_plan": investment_plan,
            "researcher_evidence": researcher_evidence_json,
        }

    return researcher_node
