import functools
import logging
from collections.abc import Mapping
from typing import Any

import yfinance as yf
from langchain_core.messages import HumanMessage, RemoveMessage

# Import tools from separate utility files. Re-exported (not used directly in
# this module) for the many agent modules that import tool functions from
# agent_utils rather than the specialized modules directly.
from tradingagents.agents.utils.core_stock_tools import get_stock_data as get_stock_data
from tradingagents.agents.utils.fundamental_data_tools import (
    get_balance_sheet as get_balance_sheet,
    get_cashflow as get_cashflow,
    get_fundamentals as get_fundamentals,
    get_income_statement as get_income_statement,
)
from tradingagents.agents.utils.market_data_validation_tools import (
    get_verified_market_snapshot as get_verified_market_snapshot,
)
from tradingagents.agents.utils.news_data_tools import (
    get_global_news as get_global_news,
    get_insider_transactions as get_insider_transactions,
    get_news as get_news,
)
from tradingagents.agents.utils.technical_indicators_tools import (
    get_indicators as get_indicators,
)

logger = logging.getLogger(__name__)


ANALYST_REPORTS_READING_INSTRUCTIONS = (
    "The analyst reports below are structured JSON envelopes\n"
    "(fields: `signal`, `confidence`, `summary`, `details`), not prose. Read the\n"
    "`summary` for the headline takeaway and cite specific `details` fields\n"
    "(e.g. technical indicator values and the `trade_setup`, news headline counts\n"
    "and the conservative/risky ratings, per-source sentiment directions and key\n"
    "items, or the fundamentals value/growth sub-signals) as supporting evidence —\n"
    "do not just restate the raw JSON."
)
"""Shared envelope-framing paragraph for prompts that inject analyst JSON reports.

Reused by the bull/bear researchers, the trader, and the portfolio manager
(issue #77) so the reading-instructions text lives in one place instead of
being hand-copied at each call site.
"""


def fundamentals_report_label(asset_type: str) -> str:
    """Return the asset-type-aware label for the fundamentals report.

    Crypto assets may not have company fundamentals available, so the label
    flags that explicitly rather than implying a stock-style report always
    exists.
    """
    return (
        "Company fundamentals report"
        if asset_type == "stock"
        else "Asset fundamentals report (may be unavailable for crypto)"
    )


def format_analyst_reports_section(
    market_report: Any,
    sentiment_report: Any,
    news_report: Any,
    fundamentals_report: Any,
    asset_type: str = "stock",
) -> str:
    """Format the four analyst reports with reading instructions for JSON envelopes.

    Shared by the trader and portfolio manager prompts (issue #77) — both need
    to inject all four analyst envelopes with the same JSON-reading-instructions
    framing the researchers use (:data:`ANALYST_REPORTS_READING_INSTRUCTIONS`).
    Returns an empty string if all reports are missing/empty so callers can
    omit the section entirely; individual missing/non-string reports are
    silently dropped rather than interpolated as empty text.
    """
    reports_to_include = []

    if market_report and isinstance(market_report, str):
        reports_to_include.append(f"Market research report (JSON envelope): {market_report}")

    if sentiment_report and isinstance(sentiment_report, str):
        reports_to_include.append(
            f"Social media sentiment report (JSON envelope): {sentiment_report}"
        )

    if news_report and isinstance(news_report, str):
        reports_to_include.append(f"Latest world affairs news (JSON envelope): {news_report}")

    if fundamentals_report and isinstance(fundamentals_report, str):
        label = fundamentals_report_label(asset_type)
        reports_to_include.append(f"{label} (JSON envelope): {fundamentals_report}")

    if not reports_to_include:
        return ""

    section = f"{ANALYST_REPORTS_READING_INSTRUCTIONS}\n\nAnalyst Reports:\n"
    for report in reports_to_include:
        section += f"{report}\n"

    return section


def get_language_instruction() -> str:
    """Return a prompt instruction for the configured output language.

    Returns empty string when English (default), so no extra tokens are used.
    Applied to every agent whose output reaches the saved report —
    analysts, researchers, debaters, research manager, trader, and
    portfolio manager — so a non-English run produces a fully localized
    report rather than a mix of languages.
    """
    from tradingagents.dataflows.config import get_config
    lang = get_config().get("output_language", "English")
    if lang.strip().lower() == "english":
        return ""
    return f" Write your entire response in {lang}."


def _clean_identity_value(value: Any) -> str | None:
    """Return a trimmed string, or None for empty / placeholder-ish values."""
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    if not cleaned or cleaned.lower() in {"none", "n/a", "nan", "null"}:
        return None
    return cleaned


@functools.lru_cache(maxsize=256)
def resolve_instrument_identity(ticker: str) -> dict:
    """Resolve deterministic identity metadata (company name, sector, …) for a ticker.

    This exists to stop the pipeline from hallucinating a *different* company
    when a chart pattern suggests a different industry than the real one:
    without a ground-truth name, the market analyst could pattern-match the
    price action to a narrative and invent an identity (e.g. reading `TOTDY`
    as "TotalEnergies") that then cascades into every downstream report.

    Best-effort by design: if yfinance is unavailable, rate-limited, or
    doesn't recognise the ticker, we return ``{}`` and the caller falls back
    to ticker-only context rather than failing before analysis starts.
    Cached so the lookup happens at most once per ticker per process.

    The symbol is normalized first (e.g. ``XAUUSD`` -> ``GC=F``) so identity
    resolves for the same instrument the price path actually fetches (#983).
    """
    from tradingagents.dataflows.symbol_utils import normalize_symbol

    try:
        info = yf.Ticker(normalize_symbol(ticker)).info or {}
    except Exception as exc:  # noqa: BLE001 — fail open, never block the run
        logger.debug("Could not resolve instrument identity for %s: %s", ticker, exc)
        return {}

    identity: dict[str, str] = {}
    company_name = _clean_identity_value(info.get("longName")) or _clean_identity_value(
        info.get("shortName")
    )
    if company_name:
        identity["company_name"] = company_name
    for source_key, target_key in (
        ("sector", "sector"),
        ("industry", "industry"),
        ("exchange", "exchange"),
        ("quoteType", "quote_type"),
    ):
        value = _clean_identity_value(info.get(source_key))
        if value:
            identity[target_key] = value
    return identity


def build_instrument_context(
    ticker: str,
    asset_type: str = "stock",
    identity: Mapping[str, str] | None = None,
) -> str:
    """Describe the exact instrument so agents preserve identity and ticker.

    When ``identity`` is provided (resolved via
    :func:`resolve_instrument_identity`), the company name and business
    classification are injected so agents anchor to the real company rather
    than pattern-matching the price chart to a wrong one.
    """
    is_crypto = asset_type == "crypto"
    instrument_label = "asset" if is_crypto else "instrument"
    context = (
        f"The {instrument_label} to analyze is `{ticker}`. "
        "Use this exact ticker in every tool call, report, and recommendation, "
        "preserving any exchange suffix (e.g. `.TO`, `.L`, `.HK`, `.T`, `-USD`)."
    )

    details = []
    if identity:
        name = identity.get("company_name") or identity.get("name")
        if name:
            details.append(f"{'Name' if is_crypto else 'Company'}: {name}")
        sector, industry = identity.get("sector"), identity.get("industry")
        if sector and industry:
            details.append(f"Business classification: {sector} / {industry}")
        elif sector:
            details.append(f"Sector: {sector}")
        elif industry:
            details.append(f"Industry: {industry}")
        if identity.get("exchange"):
            details.append(f"Exchange: {identity['exchange']}")

    if details:
        context += (
            f" Resolved identity: {'; '.join(details)}. "
            "Do not substitute a different company or ticker unless a tool "
            "result explicitly disproves this resolved identity."
        )

    if is_crypto:
        context += (
            " Treat it as a crypto asset rather than a company, and do not "
            "assume company fundamentals are available."
        )
    return context


def get_instrument_context_from_state(state: Mapping[str, Any]) -> str:
    """Return the (instrument-identity-grounded) context for the current run.

    Prefers a context already resolved and stored on the state (the shape a
    future graph-level, resolve-once-per-run wiring would populate — see
    upstream's ``resolve_instrument_identity`` commit d7b40a2 — so this
    function is forward-compatible once that lands for every agent). Until
    then, resolves identity directly here: ``resolve_instrument_identity`` is
    cached per ticker and fail-open, so this never blocks or fails the
    analyst even without network access.
    """
    context = state.get("instrument_context")
    if isinstance(context, str) and context.strip():
        return context
    ticker = str(state["company_of_interest"])
    asset_type = state.get("asset_type", "stock")
    identity = resolve_instrument_identity(ticker)
    return build_instrument_context(ticker, asset_type, identity)


def create_msg_delete():
    def delete_messages(state):
        """Clear messages and add placeholder for Anthropic compatibility"""
        messages = state["messages"]

        # Remove all messages
        removal_operations = [RemoveMessage(id=m.id) for m in messages]

        # Add a minimal placeholder message
        placeholder = HumanMessage(content="Continue")

        return {"messages": removal_operations + [placeholder]}

    return delete_messages



