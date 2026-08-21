import json
from datetime import datetime

from .alpha_vantage_common import _make_api_request, format_datetime_for_api


def get_news(ticker, start_date, end_date) -> dict[str, str] | str:
    """Returns live and historical market news & sentiment data from premier news outlets worldwide.

    Covers stocks, cryptocurrencies, forex, and topics like fiscal policy, mergers & acquisitions, IPOs.

    Args:
        ticker: Stock symbol for news articles.
        start_date: Start date for news search.
        end_date: End date for news search.

    Returns:
        Dictionary containing news sentiment data or JSON string.
    """

    params = {
        "tickers": ticker,
        "time_from": format_datetime_for_api(start_date),
        "time_to": format_datetime_for_api(end_date),
    }

    return _make_api_request("NEWS_SENTIMENT", params)

def get_global_news(curr_date, look_back_days: int = 7, limit: int = 50) -> dict[str, str] | str:
    """Returns global market news & sentiment data without ticker-specific filtering.

    Covers broad market topics like financial markets, economy, and more.

    Args:
        curr_date: Current date in yyyy-mm-dd format.
        look_back_days: Number of days to look back (default 7).
        limit: Maximum number of articles (default 50).

    Returns:
        Dictionary containing global news sentiment data or JSON string.
    """
    from datetime import datetime, timedelta

    # Calculate start date
    curr_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    start_dt = curr_dt - timedelta(days=look_back_days)
    start_date = start_dt.strftime("%Y-%m-%d")

    params = {
        "topics": "financial_markets,economy_macro,economy_monetary",
        "time_from": format_datetime_for_api(start_date),
        "time_to": format_datetime_for_api(curr_date),
        "limit": str(limit),
    }

    return _make_api_request("NEWS_SENTIMENT", params)


def _parse_time_published(raw: str | None) -> datetime | None:
    """Parse Alpha Vantage's ``time_published`` (``YYYYMMDDTHHMMSS``) format."""
    if not raw:
        return None
    try:
        return datetime.strptime(raw, "%Y%m%dT%H%M%S")
    except ValueError:
        return None


def get_global_news_articles(
    curr_date: str,
    look_back_days: int = 7,
    limit: int = 50,
) -> dict:
    """Retrieve global/macro news as structured articles (Alpha Vantage vendor).

    Same underlying ``NEWS_SENTIMENT`` request as ``get_global_news`` (issue
    #133's macro news pack reuses the existing global-news vendor path, per
    the decision comment on issue #133) but returns structured data instead
    of the raw API payload, so a downstream deterministic prep layer
    (dedup / category-tag / cap) can operate on it in Python.

    Returns:
        ``{"vendor": "alpha_vantage", "articles": [article, ...]}`` where
        each article has ``title``, ``summary``, ``publisher``, ``link``,
        and ``pub_date`` (a ``datetime`` or ``None``).
    """
    raw = get_global_news(curr_date, look_back_days=look_back_days, limit=limit)

    articles: list[dict] = []
    if isinstance(raw, str):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {}
    elif isinstance(raw, dict):
        payload = raw
    else:
        payload = {}

    for item in payload.get("feed", []) or []:
        articles.append({
            "title": item.get("title", "No title"),
            "summary": item.get("summary", ""),
            "publisher": item.get("source", "Unknown"),
            "link": item.get("url", ""),
            "pub_date": _parse_time_published(item.get("time_published")),
        })

    return {"vendor": "alpha_vantage", "articles": articles}


def get_insider_transactions(symbol: str) -> dict[str, str] | str:
    """Returns latest and historical insider transactions by key stakeholders.

    Covers transactions by founders, executives, board members, etc.

    Args:
        symbol: Ticker symbol. Example: "IBM".

    Returns:
        Dictionary containing insider transaction data or JSON string.
    """

    params = {
        "symbol": symbol,
    }

    return _make_api_request("INSIDER_TRANSACTIONS", params)
