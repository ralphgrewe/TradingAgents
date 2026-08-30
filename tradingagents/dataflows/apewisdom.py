"""ApeWisdom public API fetcher for retail/4chan engagement metrics.

ApeWisdom (https://apewisdom.io/api/) exposes aggregated Reddit + 4chan /biz
engagement metrics -- mentions, upvotes, and rank -- for US-listed tickers.
No API key required.

Endpoint and request-budget strategy (issue #167, fixed in design review of
commit 7304291): the original implementation requested
``https://www.apewisdom.io/api/filter/gme_dd/{ticker}``, which returns HTTP
200 but serves the site's HTML homepage (``content-type: text/html``), not
JSON -- confirmed live with ``curl``, it never returned real data for any
ticker. There is no per-ticker filter endpoint. The real, working API is
paginated-list-only:

    https://apewisdom.io/api/v1.0/filter/all-stocks/page/{n}

returning ``{"count", "pages", "current_page", "results": [{"ticker",
"mentions", "upvotes", "rank_24h_ago", ...}, ...]}`` -- verified live
2026-08-30, 707-763 tickers across 8 pages. Since there is no per-ticker
query, this module implements the second of issue #167's two explicitly
allowed strategies: fetch the full list once, build a
``{ticker: {mentions, upvotes, rank_24h_ago}}`` lookup, and cache it
in-process (module-level, thread-safe) so a batch run over many tickers
issues a bounded number of requests total (~8, the page count) rather than
8 per ticker. ``_MAX_PAGES`` is a hard safety cap independent of whatever
the API's own ``pages`` field reports, and a page that fails to fetch is
skipped rather than aborting the whole snapshot -- a partial snapshot
(missing the long tail of low-mention tickers on the last page or two) is
still useful.

Date interaction: ApeWisdom has no historical query and no per-trade-date
dimension server-side -- every page always reflects "now". This module
therefore does NOT key the cache by ``trade_date``: doing so would not
produce more accurate historical data (none exists), only a redundant
identical refetch per date. A historical ``trade_date`` run still receives
today's live snapshot; callers should treat the ApeWisdom read as "current
retail engagement," not a value specific to the analyzed date -- similar in
spirit to how the researcher stage's live web search only runs when
``trade_date == today`` (see CLAUDE.md), except ApeWisdom offers no
equivalent gate to fall back on, so the limitation is simply documented
here rather than faked. To keep a long-lived process (e.g. the MCP server,
which can stay up across days of unrelated runs) from serving indefinitely
stale data, the cache also carries a TTL (``_CACHE_TTL_SECONDS``); this is
a staleness bound, not a per-date cache key.

US-only coverage is handled explicitly: exchange-suffixed symbols (e.g.
``ALFEN.AS``, ``SAP.DE``) have zero coverage on ApeWisdom and short-circuit
to the unavailable placeholder with a reason naming the coverage limit,
via ``tradingagents.dataflows.symbol_utils.has_non_us_exchange_suffix`` --
consistent with the rest of the repo, which does symbol handling at the
point of use rather than duplicating a suffix table locally.

The public function is deliberately self-contained: short timeout,
graceful degradation on any HTTP or parse failure, and a string return type
so the calling agent gets a uniform interface regardless of whether the
network call succeeded -- returns a string, never raises.
"""

from __future__ import annotations

import http.client
import json
import logging
import threading
import time
from urllib.request import Request, urlopen

from tradingagents.dataflows.symbol_utils import has_non_us_exchange_suffix

logger = logging.getLogger(__name__)

_LIST_API = "https://apewisdom.io/api/v1.0/filter/all-stocks/page/{page}"
_UA = "tradingagents/0.2 (+https://github.com/TauricResearch/TradingAgents)"

# Hard cap on pages fetched per snapshot, independent of what the API's own
# "pages" field reports (currently 8) -- defends against a runaway loop if
# the field is malformed or the site's universe grows unexpectedly.
_MAX_PAGES = 20

# How long a fetched snapshot is served before the next call triggers a
# refetch. ApeWisdom has no date dimension to key on (see module docstring);
# this is a simple staleness bound for long-lived processes, not a
# per-trade-date cache.
_CACHE_TTL_SECONDS = 3600.0

_cache_lock = threading.Lock()
_cache: dict[str, dict] | None = None
_cache_fetched_at: float = 0.0


def _fetch_page(page: int, timeout: float) -> dict | None:
    """Fetch one page of the all-stocks list. Returns None on any failure."""
    url = _LIST_API.format(page=page)
    req = Request(url, headers={"User-Agent": _UA, "Accept": "application/json"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
    except (OSError, http.client.HTTPException, json.JSONDecodeError) as exc:
        # OSError covers URLError/TimeoutError/connection resets; HTTPException
        # covers chunked-transfer errors (IncompleteRead/BadStatusLine, #1024).
        logger.warning("ApeWisdom page %d fetch failed: %s", page, exc)
        return None
    if not isinstance(data, dict):
        logger.warning("ApeWisdom page %d: unexpected response shape (not an object)", page)
        return None
    return data


def _merge_results(lookup: dict[str, dict], results) -> None:
    """Merge one page's ``results`` list into the running ticker lookup."""
    if not isinstance(results, list):
        return
    for item in results:
        if not isinstance(item, dict):
            continue
        ticker = item.get("ticker")
        if not isinstance(ticker, str) or not ticker:
            continue
        lookup[ticker.upper()] = {
            "mentions": item.get("mentions", 0) or 0,
            "upvotes": item.get("upvotes", 0) or 0,
            # rank_24h_ago may be absent or null for a newly-appearing
            # ticker; keep it as None rather than treating it as a parse
            # failure.
            "rank_24h_ago": item.get("rank_24h_ago"),
        }


def _fetch_full_snapshot(timeout: float) -> dict[str, dict] | None:
    """Fetch every page of the all-stocks list and build a ticker lookup.

    Returns None only if the first page fails -- without it there is no
    ``pages`` count to iterate and nothing to return. A later page failing
    is not fatal: the partial snapshot built from the pages that succeeded
    is returned as-is (see module docstring).
    """
    first = _fetch_page(1, timeout)
    if first is None:
        return None

    lookup: dict[str, dict] = {}
    _merge_results(lookup, first.get("results"))

    try:
        total_pages = int(first.get("pages", 1))
    except (TypeError, ValueError):
        total_pages = 1
    total_pages = max(1, min(total_pages, _MAX_PAGES))

    for page in range(2, total_pages + 1):
        data = _fetch_page(page, timeout)
        if data is None:
            continue
        _merge_results(lookup, data.get("results"))

    return lookup


def _get_snapshot(timeout: float) -> dict[str, dict] | None:
    """Return the cached ticker snapshot, fetching/refreshing it if stale.

    One full-list fetch (up to ``_MAX_PAGES`` requests) per cache lifetime,
    shared across every ticker calling this module in the same process --
    this is what keeps a batch run's request count bounded (~8 requests
    total, not 8-per-ticker) per issue #167's acceptance criteria.
    """
    global _cache, _cache_fetched_at
    with _cache_lock:
        now = time.monotonic()
        if _cache is not None and (now - _cache_fetched_at) < _CACHE_TTL_SECONDS:
            return _cache
        snapshot = _fetch_full_snapshot(timeout)
        if snapshot is None:
            # Fetch failed entirely (first page unreachable/unparseable);
            # keep serving whatever stale cache we have (if any) rather
            # than blanking out every ticker for the rest of the run.
            return _cache
        _cache = snapshot
        _cache_fetched_at = now
        return _cache


def _reset_cache_for_tests() -> None:
    """Clear the module-level snapshot cache. Test-only; not part of the
    public API."""
    global _cache, _cache_fetched_at
    with _cache_lock:
        _cache = None
        _cache_fetched_at = 0.0


def fetch_apewisdom_mentions(ticker: str, timeout: float = 10.0) -> str:
    """Fetch retail/4chan engagement metrics for ``ticker`` from ApeWisdom.

    Returns a placeholder string when the ticker is non-US
    (exchange-suffixed), the snapshot fetch fails, or the ticker has no
    mentions in the current snapshot -- the caller never has to
    special-case None or exceptions.

    ApeWisdom aggregates discussions from ~12 subreddits plus 4chan /biz,
    returning mention counts and upvotes for US-listed tickers. Coverage is
    checked via ``symbol_utils.has_non_us_exchange_suffix`` before touching
    the network; the actual data comes from a process-wide cached snapshot
    (see module docstring for the pagination/caching strategy).
    """
    if has_non_us_exchange_suffix(ticker):
        return "<apewisdom unavailable: non-US listing (exchange suffix detected)>"

    ticker_u = ticker.upper()
    snapshot = _get_snapshot(timeout)
    if snapshot is None:
        return "<apewisdom unavailable: snapshot fetch failed>"

    entry = snapshot.get(ticker_u)
    if entry is None:
        # Not present in the all-stocks list: either genuinely untracked or
        # below whatever mention floor ApeWisdom applies to the list. The
        # API never lists a ticker with a $0 count explicitly, so this is
        # the same "available with zero" outcome as an explicit zero below.
        return f"<no ApeWisdom mentions found for ${ticker_u}>"

    mentions = entry.get("mentions", 0) or 0
    upvotes = entry.get("upvotes", 0) or 0
    rank_24h_ago = entry.get("rank_24h_ago")

    if mentions == 0:
        return f"<no ApeWisdom mentions found for ${ticker_u}>"

    summary = f"ApeWisdom (Reddit + 4chan /biz aggregate): {mentions} mentions"
    if upvotes > 0:
        summary += f", {upvotes} upvotes"
    if rank_24h_ago is not None:
        summary += f", rank 24h ago: #{rank_24h_ago}"

    return summary
