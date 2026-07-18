"""Tavily-based web search dataflow.

Provides news-weighted web search for the Researcher node using the Tavily API.
"""

import logging
import os
from datetime import datetime

import requests

from .errors import VendorError, VendorNotConfiguredError

logger = logging.getLogger(__name__)

# Tavily API configuration
TAVILY_BASE_URL = "https://api.tavily.com/search"
TAVILY_TOPIC = "news"
TAVILY_MAX_RESULTS = 3


class TavilySearchError(VendorError):
    """A Tavily request failed or its response could not be parsed.

    Distinct from ``VendorNotConfiguredError``, which is reserved for the
    "no API key" case. Keeping these separate lets callers (the Researcher
    node, follow-up issue #85) tell "search disabled" apart from "search
    degraded" without string-matching the exception message.
    """


def get_web_search_results(query: str) -> list[dict] | str:
    """Search the web using Tavily API for news-weighted results.

    Args:
        query: Search query string.

    Returns:
        List of search results with keys: title, url, content, score.

    Raises:
        VendorNotConfiguredError: TAVILY_API_KEY is not set.
        TavilySearchError: the HTTP request failed or the response could not
            be parsed.

    Each result dict contains:
        - title (str): Article title
        - url (str): Article URL
        - content (str): Article snippet
        - score (float): Relevance score (0-1)
    """
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        error_msg = (
            "TAVILY_API_KEY environment variable not set. "
            "Web search is unavailable (disabled, no API key)."
        )
        logger.warning(error_msg)
        raise VendorNotConfiguredError(error_msg)

    payload = {
        "api_key": api_key,
        "query": query,
        "topic": TAVILY_TOPIC,
        "max_results": TAVILY_MAX_RESULTS,
        "include_raw_content": False,  # snippets only, no raw_content
    }

    try:
        response = requests.post(TAVILY_BASE_URL, json=payload, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        error_msg = f"Tavily API request failed: {e}"
        logger.warning(error_msg)
        raise TavilySearchError(error_msg) from e

    try:
        data = response.json()

        # Extract results from Tavily response
        results = []
        for result in data.get("results", []):
            results.append({
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "content": result.get("content", ""),  # snippet, not raw_content
                "score": result.get("score", 0.0),
            })

        return results

    except TavilySearchError:
        raise
    except Exception as e:
        error_msg = f"Error parsing Tavily response: {e}"
        logger.warning(error_msg)
        raise TavilySearchError(error_msg) from e


def build_evidence_pack(
    query_results_list: list[list[dict]],
    already_cited_urls: set[str],
    token_budget: int = 3000,
) -> list[dict]:
    """Build a deduplicated, sorted, truncated evidence pack from search results.

    Takes per-query result lists, dedupes by URL (against already-cited URLs and
    across queries), sorts deterministically (score desc, then URL), truncates to
    token budget (chars/4 heuristic), and assigns stable IDs (web:1, web:2, ...).

    Args:
        query_results_list: List of result lists, one per query.
        already_cited_urls: Set of URLs already cited (e.g., from news envelope).
        token_budget: Maximum token count for the assembled pack (default 3000).
                     Uses chars/4 as the heuristic for token conversion.

    Returns:
        List of evidence dicts with keys: id, title, url, content, score.
    """
    # Flatten, then sort deterministically (score descending, then URL
    # ascending) *before* deduping. Sorting first guarantees that when the
    # same URL appears in multiple queries with different scores, the
    # highest-scoring instance is the one kept — the dedup outcome must not
    # depend on which query happens to be iterated first.
    all_results = [result for result_list in query_results_list for result in result_list]
    all_results.sort(key=lambda r: (-r.get("score", 0.0), r.get("url", "")))

    seen_urls = set(already_cited_urls)
    unique_results = []

    for result in all_results:
        url = result.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_results.append(result)

    # Truncate to token budget (chars/4 heuristic)
    assembled_pack = []
    total_chars = 0
    max_chars = token_budget * 4

    for result in unique_results:
        # Estimate character count for this result
        result_chars = (
            len(result.get("title", ""))
            + len(result.get("url", ""))
            + len(result.get("content", ""))
        )

        if total_chars + result_chars > max_chars:
            break

        assembled_pack.append(result)
        total_chars += result_chars

    # Assign stable IDs
    evidence_pack = []
    for i, result in enumerate(assembled_pack, start=1):
        evidence_pack.append({
            "id": f"web:{i}",
            "title": result.get("title", ""),
            "url": result.get("url", ""),
            "content": result.get("content", ""),
            "score": result.get("score", 0.0),
        })

    return evidence_pack


def is_web_search_allowed(trade_date: str, today: str | None = None) -> bool:
    """Check if web search is allowed for the given trade_date.

    Web search is only allowed when trade_date == today (live runs).
    For historical/backtest dates, web search is never allowed (hard gate).

    Args:
        trade_date: ISO date string (yyyy-mm-dd) of the trade date.
        today: ISO date string (yyyy-mm-dd) of today's date. If None, uses current date.

    Returns:
        True if trade_date == today, False otherwise.
    """
    if today is None:
        today = datetime.now().strftime("%Y-%m-%d")

    # ISO string comparison is safe since they're in yyyy-mm-dd format
    return trade_date == today
