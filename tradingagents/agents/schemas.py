"""Pydantic schemas used by agents that produce structured output.

The framework's primary artifact is still prose: each agent's natural-language
reasoning is what users read in the saved markdown reports and what the
downstream agents read as context.  Structured output is layered onto the
three decision-making agents (Research Manager, Trader, Portfolio Manager)
so that:

- Their outputs follow consistent section headers across runs and providers
- Each provider's native structured-output mode is used (json_schema for
  OpenAI/xAI, response_schema for Gemini, tool-use for Anthropic)
- Schema field descriptions become the model's output instructions, freeing
  the prompt body to focus on context and the rating-scale guidance
- A render helper turns the parsed Pydantic instance back into the same
  markdown shape the rest of the system already consumes, so display,
  memory log, and saved reports keep working unchanged
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Shared rating types
# ---------------------------------------------------------------------------


class PortfolioRating(str, Enum):
    """5-tier rating used by the Research Manager and Portfolio Manager."""

    BUY = "Buy"
    OVERWEIGHT = "Overweight"
    HOLD = "Hold"
    UNDERWEIGHT = "Underweight"
    SELL = "Sell"


class BriefConfidence(str, Enum):
    """Confidence level for research brief recommendation."""

    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class TraderAction(str, Enum):
    """3-tier transaction direction used by the Trader.

    The Trader's job is to translate the Research Manager's investment plan
    into a concrete transaction proposal: should the desk execute a Buy, a
    Sell, or sit on Hold this round.  Position sizing and the nuanced
    Overweight / Underweight calls happen later at the Portfolio Manager.
    """

    BUY = "Buy"
    HOLD = "Hold"
    SELL = "Sell"


# ---------------------------------------------------------------------------
# Research Manager
# ---------------------------------------------------------------------------


class ResearchArgument(BaseModel):
    """A single argument in a bull or bear case.

    Used by ResearchBrief to structure the bull/bear debate outcomes.
    """

    statement: str = Field(
        description=(
            "One short sentence, 30 words or fewer, with no hedging filler. "
            "State the finding or argument as a fact."
        ),
    )
    source: str = Field(
        description=(
            'Exactly one of "market", "sentiment", "news", "fundamentals", '
            'or a web evidence ID like "web:3". Indicates where the claim comes from.'
        ),
    )


class ResearchBrief(BaseModel):
    """Structured investment research output from the Researcher node.

    Captures the bull and bear arguments from research, the lean direction,
    confidence level, and any new information discovered via web research.
    Renders to markdown that lands directly in `investment_plan` without
    downstream prompt or parsing changes.
    """

    bull_arguments: list[ResearchArgument] = Field(
        min_length=1,
        max_length=5,
        description="1–5 arguments supporting a bullish view (source-tagged).",
    )
    bear_arguments: list[ResearchArgument] = Field(
        min_length=1,
        max_length=5,
        description="1–5 arguments supporting a bearish view (source-tagged).",
    )
    lean: PortfolioRating = Field(
        description=(
            "The overall recommendation: Buy / Overweight / Hold / Underweight / Sell. "
            "Derived from the weight and strength of bull/bear arguments."
        ),
    )
    confidence: BriefConfidence = Field(
        description=(
            "Confidence in the recommendation: High / Medium / Low, based on the "
            "consistency and clarity of the research evidence."
        ),
    )
    new_information: str | None = Field(
        default=None,
        description=(
            "1–2 sentences on what web research added to the analysis. "
            "None if web research was disabled or no new information was found."
        ),
    )


def render_research_brief(
    brief: ResearchBrief,
    web_research_status: str,
    web_research_result_count: int | None = None,
) -> str:
    """Render a ResearchBrief to markdown for storage and downstream prompt context.

    The output preserves the label-first `**Recommendation**: {lean}` format so
    parse_rating() in the downstream agents can extract the rating without parsing
    the entire markdown structure.

    Args:
        brief: The ResearchBrief to render.
        web_research_status: One of "enabled", "disabled (historical date)",
            "disabled (no API key)", or "disabled (config)".
        web_research_result_count: Number of web results if web research was enabled.
    """
    parts = [
        f"**Recommendation**: {brief.lean.value}",
        "",
        "**Bull Arguments**:",
    ]

    for arg in brief.bull_arguments:
        parts.append(f"- {arg.statement} [{arg.source}]")

    parts.extend([
        "",
        "**Bear Arguments**:",
    ])

    for arg in brief.bear_arguments:
        parts.append(f"- {arg.statement} [{arg.source}]")

    parts.extend([
        "",
        f"**Confidence**: {brief.confidence.value}",
    ])

    if brief.new_information:
        parts.extend([
            "",
            f"**New Information**: {brief.new_information}",
        ])

    # Build web research metadata line
    if web_research_status == "enabled":
        web_meta = f"Web research: enabled ({web_research_result_count} results)"
    else:
        web_meta = f"Web research: {web_research_status}"

    parts.extend([
        "",
        web_meta,
    ])

    return "\n".join(parts)


class ResearchPlan(BaseModel):
    """Structured investment plan produced by the Research Manager.

    Hand-off to the Trader: the recommendation pins the directional view,
    the rationale captures which side of the bull/bear debate carried the
    argument, and the strategic actions translate that into concrete
    instructions the trader can execute against.
    """

    recommendation: PortfolioRating = Field(
        description=(
            "The investment recommendation. Exactly one of Buy / Overweight / "
            "Hold / Underweight / Sell. Reserve Hold for situations where the "
            "evidence on both sides is genuinely balanced; otherwise commit to "
            "the side with the stronger arguments."
        ),
    )
    rationale: str = Field(
        description=(
            "Conversational summary of the key points from both sides of the "
            "debate, ending with which arguments led to the recommendation. "
            "Speak naturally, as if to a teammate."
        ),
    )
    strategic_actions: str = Field(
        description=(
            "Concrete steps for the trader to implement the recommendation, "
            "including position sizing guidance consistent with the rating."
        ),
    )


def render_research_plan(plan: ResearchPlan) -> str:
    """Render a ResearchPlan to markdown for storage and the trader's prompt context."""
    return "\n".join([
        f"**Recommendation**: {plan.recommendation.value}",
        "",
        f"**Rationale**: {plan.rationale}",
        "",
        f"**Strategic Actions**: {plan.strategic_actions}",
    ])


# ---------------------------------------------------------------------------
# Trader
# ---------------------------------------------------------------------------


class TraderProposal(BaseModel):
    """Structured transaction proposal produced by the Trader.

    The trader reads the Research Manager's investment plan and the analyst
    reports, then turns them into a concrete transaction: what action to
    take, the reasoning that justifies it, and the practical levels for
    entry, stop-loss, and sizing.
    """

    action: TraderAction = Field(
        description="The transaction direction. Exactly one of Buy / Hold / Sell.",
    )
    reasoning: str = Field(
        description=(
            "The case for this action, anchored in the analysts' reports and "
            "the research plan. Two to four sentences."
        ),
    )
    entry_price: float | None = Field(
        default=None,
        description="Optional entry price target in the instrument's quote currency.",
    )
    stop_loss: float | None = Field(
        default=None,
        description="Optional stop-loss price in the instrument's quote currency.",
    )
    position_sizing: str | None = Field(
        default=None,
        description="Optional sizing guidance, e.g. '5% of portfolio'.",
    )


def render_trader_proposal(proposal: TraderProposal) -> str:
    """Render a TraderProposal to markdown."""
    parts = [
        f"**Action**: {proposal.action.value}",
        "",
        f"**Reasoning**: {proposal.reasoning}",
    ]
    if proposal.entry_price is not None:
        parts.extend(["", f"**Entry Price**: {proposal.entry_price}"])
    if proposal.stop_loss is not None:
        parts.extend(["", f"**Stop Loss**: {proposal.stop_loss}"])
    if proposal.position_sizing:
        parts.extend(["", f"**Position Sizing**: {proposal.position_sizing}"])

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Portfolio Manager
# ---------------------------------------------------------------------------


class PortfolioDecision(BaseModel):
    """Structured output produced by the Portfolio Manager.

    The model fills every field as part of its primary LLM call; no separate
    extraction pass is required. Field descriptions double as the model's
    output instructions, so the prompt body only needs to convey context and
    the rating-scale guidance.
    """

    rating: PortfolioRating = Field(
        description=(
            "The final position rating. Exactly one of Buy / Overweight / Hold / "
            "Underweight / Sell, picked based on the analysts' debate."
        ),
    )
    executive_summary: str = Field(
        description=(
            "A concise action plan covering entry strategy, position sizing, "
            "key risk levels, and time horizon. Two to four sentences."
        ),
    )
    investment_thesis: str = Field(
        description=(
            "Detailed reasoning anchored in specific evidence from the analysts' "
            "debate. If prior lessons are referenced in the prompt context, "
            "incorporate them; otherwise rely solely on the current analysis."
        ),
    )
    price_target: float | None = Field(
        default=None,
        description="Optional target price in the instrument's quote currency.",
    )
    time_horizon: str | None = Field(
        default=None,
        description="Optional recommended holding period, e.g. '3-6 months'.",
    )


def render_pm_decision(decision: PortfolioDecision) -> str:
    """Render a PortfolioDecision back to the markdown shape the rest of the system expects.

    Memory log, CLI display, and saved report files all read this markdown,
    so the rendered output preserves the exact section headers (``**Rating**``,
    ``**Executive Summary**``, ``**Investment Thesis**``) that downstream
    parsers and the report writers already handle.
    """
    parts = [
        f"**Rating**: {decision.rating.value}",
        "",
        f"**Executive Summary**: {decision.executive_summary}",
        "",
        f"**Investment Thesis**: {decision.investment_thesis}",
    ]
    if decision.price_target is not None:
        parts.extend(["", f"**Price Target**: {decision.price_target}"])
    if decision.time_horizon:
        parts.extend(["", f"**Time Horizon**: {decision.time_horizon}"])
    return "\n".join(parts)
