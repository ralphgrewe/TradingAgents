"""Deterministic macro news preparation layer (issue #133).

Backs the ``get_macro_news`` tool (``tradingagents/agents/utils/news_data_tools.py``)
with the Python-side prep the design spec (and ``LEARNINGS.md``) calls for:
dedup, category tagging, recency ordering, and per-category caps happen here
in plain Python — no LLM is involved anywhere in this module.

Source (per the decision comment on issue #133, made in place of the #128/#129
research tickets which were not run this round): reuse the existing
``news_data`` vendor category (yfinance/alpha_vantage) via ``route_to_vendor``,
rather than adding a new RSS vendor. Fallback is entirely ``route_to_vendor``'s
existing comma-separated vendor-chain support (``data_vendors="yfinance,
alpha_vantage"``) — no custom fallback chain lives here. The vendor-level
functions (``get_global_news_articles_yfinance``,
``get_global_news_articles``) return ``{"vendor": ..., "articles": [...]}``
so the pack can record which vendor actually served the request.

Historical-date handling: yfinance's/Alpha Vantage's global-news search is a
live/recent-window feed, not a true historical archive (confirmed in
``yfinance_news.py``: date filtering happens client-side over whatever the
live search currently surfaces). So this module mirrors the researcher's
date-gate convention (``tradingagents/agents/researchers/researcher.py``,
documented in ``CLAUDE.md``): for a non-today ``curr_date`` the gate outcome
is ``"disabled (historical date)"`` and no articles are fetched or returned
— never today's news silently mislabeled as the historical date's news.

Category taxonomy (fixed, per the decision comment): ``monetary_policy``,
``inflation_prices``, ``labor_market``, ``growth_output``,
``markets_volatility``, ``geopolitical_trade``. Tagging is a case-insensitive
keyword match against title+summary; the first matching category (in
taxonomy order) wins. Articles matching no keyword fall into
``uncategorized`` so nothing is silently dropped.
"""
from __future__ import annotations

import re
from datetime import datetime

from .config import get_config
from .interface import route_to_vendor
from .tavily_search import is_web_search_allowed

# Fixed category taxonomy (decision comment on issue #133). Order matters:
# an article is tagged with the first category whose keywords match.
CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "monetary_policy": [
        "federal reserve", "fomc", "fed", "fed rate", "interest rate",
        "rate hike", "rate cut", "rate decision", "central bank",
        "monetary policy", "quantitative easing", "powell", "ecb",
        "bank of england", "boj", "bank of japan",
    ],
    "inflation_prices": [
        "inflation", "cpi", "consumer price", "ppi", "producer price",
        "deflation", "price index", "cost of living",
    ],
    "labor_market": [
        "jobs report", "unemployment", "payroll", "labor market",
        "jobless claims", "nonfarm", "hiring", "layoffs",
    ],
    "growth_output": [
        "gdp", "economic growth", "recession", "economic outlook",
        "industrial production", "manufacturing", "pmi",
    ],
    "markets_volatility": [
        "s&p 500", "stock market", "wall street", "volatility", "vix",
        "sell-off", "selloff", "rally", "equities", "bond yield",
        "treasury yield", "stocks fall", "stocks rise",
    ],
    "geopolitical_trade": [
        "geopolitical", "trade war", "tariff", "sanctions", "war",
        "conflict", "opec", "oil supply", "energy crisis", "supply chain",
    ],
}

UNCATEGORIZED = "uncategorized"

# Category display order for rendering (taxonomy order, uncategorized last).
_CATEGORY_ORDER = [*CATEGORY_KEYWORDS.keys(), UNCATEGORIZED]

DEFAULT_CATEGORY_CAP = 3


def _keyword_pattern(keyword: str) -> re.Pattern:
    """Compile a case-insensitive, word-boundary regex for one keyword.

    Word boundaries prevent short keywords (e.g. "war", "fed") from matching
    inside unrelated words (e.g. "award", "federal" is a separate, longer
    keyword so this doesn't shadow it).
    """
    return re.compile(r"\b" + re.escape(keyword) + r"\b", re.IGNORECASE)


_CATEGORY_PATTERNS: dict[str, list[re.Pattern]] = {
    category: [_keyword_pattern(kw) for kw in keywords]
    for category, keywords in CATEGORY_KEYWORDS.items()
}


def _tag_category(title: str, summary: str) -> str:
    """Case-insensitive, word-boundary keyword match against title+summary.

    Returns the first matching category in taxonomy order, or
    ``UNCATEGORIZED`` when no keyword matches.
    """
    haystack = f"{title} {summary}"
    for category, patterns in _CATEGORY_PATTERNS.items():
        if any(pattern.search(haystack) for pattern in patterns):
            return category
    return UNCATEGORIZED


def _dedup_key(article: dict) -> tuple:
    """Dedup identity: prefer a non-empty URL, else (title, publisher)."""
    link = (article.get("link") or "").strip().lower()
    if link:
        return ("url", link)
    title = (article.get("title") or "").strip().lower()
    publisher = (article.get("publisher") or "").strip().lower()
    return ("title_publisher", title, publisher)


def _dedup(articles: list[dict]) -> list[dict]:
    """Dedup by (title, publisher) / URL, keeping the first occurrence."""
    seen: set[tuple] = set()
    deduped = []
    for article in articles:
        key = _dedup_key(article)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(article)
    return deduped


def _sort_key(article: dict):
    """Recency-descending sort key; undated articles sort last."""
    pub_date = article.get("pub_date")
    # datetime sorts fine among themselves; None is pushed to the end by
    # sorting on a (has_date, date) tuple rather than comparing None to a
    # datetime (which raises).
    return (pub_date is not None, pub_date or datetime.min)


def _iso(pub_date) -> str | None:
    return pub_date.isoformat() if pub_date is not None else None


def build_macro_news_pack(
    curr_date: str,
    today: str | None = None,
    look_back_days: int | None = None,
    category_cap: int | None = None,
) -> dict:
    """Build the deterministic macro news pack for ``curr_date``.

    No LLM call happens here. Fetches via the configured ``news_data``
    vendor (``route_to_vendor("get_macro_news", ...)``, itself routed
    through the existing yfinance/alpha_vantage vendor chain), then dedups,
    category-tags, recency-orders, and caps the result in Python.

    Args:
        curr_date: The as-of date, ``yyyy-mm-dd``.
        today: ISO date string used for the historical-date gate check; only
            overridden in tests, ``None`` uses the real current date.
        look_back_days: Lookback window in days; ``None`` uses
            ``global_news_lookback_days`` from config.
        category_cap: Max articles kept per category; ``None`` uses
            ``macro_news_category_cap`` from config.

    Returns:
        A dict with ``curr_date``, ``gate`` (``date_is_today`` +
        ``outcome``), ``vendor`` (which vendor served the request, or
        ``None`` when the gate is closed), ``look_back_days``,
        ``category_cap``, ``article_count``, and ``categories`` (a
        ``{category: [article, ...]}`` mapping in taxonomy order; each
        article has ``title``, ``summary``, ``publisher``, ``link``, and
        ``pub_date`` as an ISO string or ``None``).
    """
    config = get_config()
    if look_back_days is None:
        look_back_days = config["global_news_lookback_days"]
    if category_cap is None:
        category_cap = config.get("macro_news_category_cap", DEFAULT_CATEGORY_CAP)
    # Resolved explicitly (not left as None) so both vendor implementations
    # see a concrete int — the alpha_vantage vendor function's default only
    # applies when the kwarg is omitted, not when None is passed positionally.
    article_limit = config["global_news_article_limit"]

    date_is_today = is_web_search_allowed(curr_date, today)

    if not date_is_today:
        return {
            "curr_date": curr_date,
            "gate": {"date_is_today": False, "outcome": "disabled (historical date)"},
            "vendor": None,
            "look_back_days": look_back_days,
            "category_cap": category_cap,
            "article_count": 0,
            "categories": {},
        }

    result = route_to_vendor("get_macro_news", curr_date, look_back_days, article_limit)

    # Optional-category degradation sentinel (route_to_vendor's own contract)
    # or an unexpected non-dict result: treat as "no articles", vendor unknown.
    if not isinstance(result, dict):
        vendor = None
        raw_articles: list[dict] = []
    else:
        vendor = result.get("vendor")
        raw_articles = result.get("articles") or []

    deduped = _dedup(raw_articles)
    for article in deduped:
        article["category"] = _tag_category(article.get("title", ""), article.get("summary", ""))
    deduped.sort(key=_sort_key, reverse=True)

    categories: dict[str, list[dict]] = {cat: [] for cat in _CATEGORY_ORDER}
    for article in deduped:
        bucket = categories[article["category"]]
        if len(bucket) >= category_cap:
            continue
        bucket.append({
            "title": article.get("title", "No title"),
            "summary": article.get("summary", ""),
            "publisher": article.get("publisher", "Unknown"),
            "link": article.get("link", ""),
            "pub_date": _iso(article.get("pub_date")),
        })

    # Drop empty categories from the output so the render layer doesn't have
    # to special-case them.
    categories = {cat: articles for cat, articles in categories.items() if articles}
    article_count = sum(len(articles) for articles in categories.values())

    return {
        "curr_date": curr_date,
        "gate": {"date_is_today": True, "outcome": "enabled"},
        "vendor": vendor,
        "look_back_days": look_back_days,
        "category_cap": category_cap,
        "article_count": article_count,
        "categories": categories,
    }


_CATEGORY_DISPLAY_NAMES = {
    "monetary_policy": "Monetary Policy",
    "inflation_prices": "Inflation & Prices",
    "labor_market": "Labor Market",
    "growth_output": "Growth & Output",
    "markets_volatility": "Markets & Volatility",
    "geopolitical_trade": "Geopolitical & Trade",
    UNCATEGORIZED: "Other Macro News",
}


def render_macro_news_pack(pack: dict) -> str:
    """Render a ``build_macro_news_pack`` result into an LLM-ready report string."""
    curr_date = pack["curr_date"]
    outcome = pack["gate"]["outcome"]
    vendor = pack.get("vendor")
    vendor_note = f" (source: {vendor})" if vendor else ""

    header = f"## Macro News, as of {curr_date}{vendor_note}\nMacro news search: {outcome}\n"

    if outcome != "enabled":
        return (
            f"{header}\n"
            "Historical dates are not supported by the underlying live/recent-window "
            "news feed, so no macro news is returned for this date to avoid mislabeling "
            "today's news as historical."
        )

    categories = pack.get("categories") or {}
    if not categories:
        return f"{header}\nNo macro news found for {curr_date}."

    lines = [header]
    for cat in _CATEGORY_ORDER:
        articles = categories.get(cat)
        if not articles:
            continue
        lines.append(f"\n### {_CATEGORY_DISPLAY_NAMES.get(cat, cat)}")
        for article in articles:
            lines.append(f"- **{article['title']}** (source: {article['publisher']})")
            if article.get("summary"):
                lines.append(f"  {article['summary']}")
            if article.get("link"):
                lines.append(f"  Link: {article['link']}")

    return "\n".join(lines) + "\n"
