"""ApeWisdom public API fetcher for retail/4chan engagement metrics.

ApeWisdom (https://apewisdom.io/api/) exposes aggregated Reddit + 4chan /biz
engagement metrics — mentions, upvotes, and rank — for US-listed tickers. No
API key required; keyless endpoint verified as of 2026-08-30. Coverage is
US-only (~763 tickers across 8 pages); exchange-suffixed symbols (e.g.
ALFEN.AS) have zero coverage and short-circuit to the unavailable placeholder
with a reason naming the coverage limit, consistent with the architectural
note in issue #158.

The function is deliberately self-contained: short timeout, graceful
degradation on any HTTP or parse failure, US-only coverage check at the point
of use (symbol_utils.py, consistent with the rest of the repo — normalization
never happens at storage time), and a string return type so the calling agent
gets a uniform interface regardless of whether the network call succeeded.

Each request fetches a single ticker. The endpoint is paginated (8 pages for
the full list), but per-ticker queries are more efficient and do not require
caching across trades.
"""

from __future__ import annotations

import http.client
import json
import logging
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

_API = "https://www.apewisdom.io/api/filter/gme_dd/{ticker}"
_UA = "tradingagents/0.2 (+https://github.com/TauricResearch/TradingAgents)"


def _has_exchange_suffix(ticker: str) -> bool:
    """True if ticker carries an exchange suffix (e.g., ALFEN.AS, ACHR.DE).

    Exchange suffixes indicate non-US listings that are not covered by ApeWisdom.
    """
    if "." not in ticker:
        return False
    # Check for common European and international exchange suffixes.
    # Format is usually TICKER.EXCH (e.g., ALFEN.AS, ASML.AS, SAP.DE, NOKIA.HE).
    parts = ticker.rsplit(".", 1)
    if len(parts) == 2:
        suffix = parts[1].upper()
        # Common exchange suffixes for non-US markets.
        non_us_suffixes = {
            "AS",   # Amsterdam Stock Exchange
            "DE",   # Deutsche Börse (Frankfurt)
            "T",    # Tokyo Stock Exchange
            "AX",   # Australian Securities Exchange
            "PA",   # Euronext Paris
            "BA",   # Bolsa de Madrid
            "BR",   # Brussels Stock Exchange
            "DB",   # Borsa Italiana (Milan)
            "SW",   # SIX Swiss Exchange
            "TA",   # Tel Aviv Stock Exchange
            "L",    # London Stock Exchange (LSE)
            "HK",   # Hong Kong Stock Exchange
            "SG",   # Singapore Exchange
            "NZ",   # NZX (New Zealand)
            "TO",   # Toronto Stock Exchange
            "V",    # TSX Venture Exchange
            "HE",   # Helsinki Stock Exchange
            "CO",   # Copenhagen Stock Exchange
            "OL",   # Oslo Stock Exchange
            "ST",   # Stockholm Stock Exchange
            "VX",   # SIX Swiss Exchange (Virt-X)
            "MC",   # Euronext Brussels
            "WR",   # Warsaw Stock Exchange
            "PR",   # Prague Stock Exchange
        }
        return suffix in non_us_suffixes
    return False


def fetch_apewisdom_mentions(ticker: str, timeout: float = 10.0) -> str:
    """Fetch retail/4chan engagement metrics for ``ticker`` from ApeWisdom.

    Returns a placeholder string when the endpoint is unreachable, the
    symbol has no mentions, the ticker is non-US (exchange-suffixed), or the
    response shape is unexpected — the caller never has to special-case None
    or exceptions.

    ApeWisdom aggregates discussions from ~12 subreddits plus 4chan /biz,
    returning mention counts and upvotes. Coverage is US-only; non-US tickers
    (those carrying exchange suffixes like .AS, .DE, .T) are detected and
    return an unavailable placeholder without issuing a network request.
    """
    # Check for exchange suffix (non-US listing) before issuing a request.
    if _has_exchange_suffix(ticker):
        return "<apewisdom unavailable: non-US listing (exchange suffix detected)>"

    url = _API.format(ticker=ticker.upper())
    req = Request(url, headers={"User-Agent": _UA, "Accept": "application/json"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
    except (OSError, http.client.HTTPException, json.JSONDecodeError) as exc:
        # OSError covers URLError/TimeoutError/connection resets; HTTPException
        # covers chunked-transfer errors (IncompleteRead/BadStatusLine, #1024).
        logger.warning("ApeWisdom fetch failed for %s: %s", ticker, exc)
        return f"<apewisdom unavailable: {type(exc).__name__}>"

    # ApeWisdom returns a list of objects. Parse the aggregated counts.
    if not isinstance(data, list):
        logger.warning("ApeWisdom: unexpected response shape for %s (not a list)", ticker)
        return "<apewisdom unavailable: unexpected response shape>"

    if not data:
        # Empty list: ticker not in ApeWisdom's database (non-US or not tracked).
        return f"<no ApeWisdom mentions found for ${ticker.upper()}>"

    # Aggregate mentions and upvotes from all entries (typically there's one,
    # but the endpoint structure allows for multiple).
    total_mentions = 0
    total_upvotes = 0
    rank_24h_ago = None

    for item in data:
        if not isinstance(item, dict):
            continue
        # Extract counts; rank_24h_ago may be absent or null for new tickers.
        total_mentions += item.get("mentions", 0) or 0
        total_upvotes += item.get("upvotes", 0) or 0
        # Capture rank_24h_ago from the first item that has it (usually present).
        if rank_24h_ago is None and item.get("rank_24h_ago") is not None:
            rank_24h_ago = item.get("rank_24h_ago")

    if total_mentions == 0:
        # Ticker is in ApeWisdom but has zero mentions: available with zero,
        # distinct from unavailable (which uses a placeholder).
        return f"<no ApeWisdom mentions found for ${ticker.upper()}>"

    # Format the output for injection into the prompt.
    summary = f"ApeWisdom (Reddit + 4chan /biz aggregate): {total_mentions} mentions"
    if total_upvotes > 0:
        summary += f", {total_upvotes} upvotes"
    if rank_24h_ago is not None:
        summary += f", rank 24h ago: #{rank_24h_ago}"

    return summary
