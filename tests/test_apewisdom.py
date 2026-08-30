"""Tests for the ApeWisdom fetcher (issue #167).

Covers the real request shape (the paginated all-stocks list, not the
non-functional per-ticker URL the prior commit used), the process-wide
snapshot cache that bounds a batch run's request count, the
symbol_utils-backed US-only coverage check, and graceful degradation on
network/parse failure. All HTTP is mocked here -- the real endpoint was
verified live with ``curl`` as part of implementing this fix (see the
module docstring in ``tradingagents/dataflows/apewisdom.py`` and the issue
#167 discussion for the verification commands), which is what caught the
prior commit's non-functional URL in the first place.
"""

from __future__ import annotations

import http.client
import json
from unittest.mock import patch

import pytest

from tradingagents.dataflows import apewisdom


def _page(count, pages, current_page, results):
    return json.dumps(
        {"count": count, "pages": pages, "current_page": current_page, "results": results}
    ).encode()


class _Resp:
    def __init__(self, payload: bytes):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return self._payload


class _RaisingResp:
    def __init__(self, exc):
        self._exc = exc

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        raise self._exc


@pytest.fixture(autouse=True)
def _clear_cache():
    """Every test gets a clean snapshot cache -- the cache is module-level
    and shared across calls by design, so tests must not leak into each
    other."""
    apewisdom._reset_cache_for_tests()
    yield
    apewisdom._reset_cache_for_tests()


@pytest.mark.unit
class TestPaginationAndCache:
    def test_fetches_all_pages_and_builds_lookup(self):
        page1 = _page(
            2, 2, 1,
            [
                {"rank": 1, "ticker": "GME", "mentions": 100, "upvotes": 500, "rank_24h_ago": 2},
                {"rank": 2, "ticker": "AMC", "mentions": 50, "upvotes": 200, "rank_24h_ago": None},
            ],
        )
        page2 = _page(
            2, 2, 2,
            [{"rank": 3, "ticker": "NVDA", "mentions": 10, "upvotes": 20, "rank_24h_ago": 5}],
        )
        responses = [_Resp(page1), _Resp(page2)]
        with patch.object(apewisdom, "urlopen", side_effect=lambda *a, **k: responses.pop(0)):
            out = apewisdom.fetch_apewisdom_mentions("GME")
        assert "100 mentions" in out
        assert "500 upvotes" in out
        assert "rank 24h ago: #2" in out

    def test_snapshot_shared_across_tickers_bounds_request_count(self):
        page1 = _page(
            2, 2, 1,
            [{"rank": 1, "ticker": "GME", "mentions": 100, "upvotes": 500, "rank_24h_ago": 2}],
        )
        page2 = _page(
            2, 2, 2,
            [{"rank": 2, "ticker": "AMC", "mentions": 50, "upvotes": 200, "rank_24h_ago": None}],
        )
        responses = [_Resp(page1), _Resp(page2)]
        call_count = {"n": 0}

        def _urlopen(*a, **k):
            call_count["n"] += 1
            return responses.pop(0)

        with patch.object(apewisdom, "urlopen", side_effect=_urlopen):
            first = apewisdom.fetch_apewisdom_mentions("GME")
            second = apewisdom.fetch_apewisdom_mentions("AMC")

        # Two tickers, but only one full-list fetch (2 pages) -- the second
        # call must reuse the cached snapshot rather than refetching.
        assert call_count["n"] == 2
        assert "100 mentions" in first
        assert "50 mentions" in second

    def test_rank_24h_ago_absent_is_none_not_a_failure(self):
        page1 = _page(
            1, 1, 1,
            [{"rank": 1, "ticker": "NEWCO", "mentions": 5, "upvotes": 0}],
        )
        with patch.object(apewisdom, "urlopen", return_value=_Resp(page1)):
            out = apewisdom.fetch_apewisdom_mentions("NEWCO")
        assert "5 mentions" in out
        assert "rank 24h ago" not in out

    def test_pages_field_bounds_iteration_and_is_capped(self):
        # pages reported as something absurd must be capped by _MAX_PAGES,
        # not trusted outright.
        first = _page(1, 999999, 1, [{"ticker": "X", "mentions": 1, "upvotes": 1}])
        calls = {"n": 0}

        def _urlopen(*a, **k):
            calls["n"] += 1
            return _Resp(first if calls["n"] == 1 else _page(1, 999999, calls["n"], []))

        with patch.object(apewisdom, "urlopen", side_effect=_urlopen):
            apewisdom.fetch_apewisdom_mentions("X")

        assert calls["n"] == apewisdom._MAX_PAGES

    def test_partial_page_failure_keeps_other_pages(self):
        page1 = _page(
            2, 2, 1,
            [{"rank": 1, "ticker": "GME", "mentions": 100, "upvotes": 500}],
        )
        responses = [_Resp(page1), _RaisingResp(TimeoutError("slow"))]
        with patch.object(apewisdom, "urlopen", side_effect=lambda *a, **k: responses.pop(0)):
            out = apewisdom.fetch_apewisdom_mentions("GME")
        assert "100 mentions" in out

    def test_first_page_failure_returns_unavailable_placeholder(self):
        with patch.object(apewisdom, "urlopen", return_value=_RaisingResp(TimeoutError("slow"))):
            out = apewisdom.fetch_apewisdom_mentions("GME")
        assert out.startswith("<apewisdom unavailable")

    @pytest.mark.parametrize(
        "exc",
        [
            http.client.IncompleteRead(b""),
            json.JSONDecodeError("bad", "<html>", 0),
            TimeoutError("slow"),
        ],
    )
    def test_transport_and_parse_errors_degrade_to_placeholder(self, exc):
        with patch.object(apewisdom, "urlopen", return_value=_RaisingResp(exc)):
            out = apewisdom.fetch_apewisdom_mentions("GME")
        assert out.startswith("<apewisdom unavailable")

    def test_ticker_not_in_snapshot_is_available_with_zero_placeholder(self):
        page1 = _page(1, 1, 1, [{"ticker": "GME", "mentions": 100, "upvotes": 500}])
        with patch.object(apewisdom, "urlopen", return_value=_Resp(page1)):
            out = apewisdom.fetch_apewisdom_mentions("ZZZZ")
        assert out == "<no ApeWisdom mentions found for $ZZZZ>"

    def test_zero_mentions_entry_uses_same_placeholder(self):
        page1 = _page(1, 1, 1, [{"ticker": "QUIET", "mentions": 0, "upvotes": 0}])
        with patch.object(apewisdom, "urlopen", return_value=_Resp(page1)):
            out = apewisdom.fetch_apewisdom_mentions("QUIET")
        assert out == "<no ApeWisdom mentions found for $QUIET>"


@pytest.mark.unit
class TestUsOnlyCoverage:
    def test_non_us_suffix_short_circuits_without_network_call(self):
        with patch.object(apewisdom, "urlopen") as mock_urlopen:
            out = apewisdom.fetch_apewisdom_mentions("ALFEN.AS")
        mock_urlopen.assert_not_called()
        assert out == "<apewisdom unavailable: non-US listing (exchange suffix detected)>"

    def test_us_ticker_issues_network_call(self):
        page1 = _page(1, 1, 1, [{"ticker": "AAPL", "mentions": 3, "upvotes": 1}])
        with patch.object(apewisdom, "urlopen", return_value=_Resp(page1)) as mock_urlopen:
            apewisdom.fetch_apewisdom_mentions("AAPL")
        mock_urlopen.assert_called()


@pytest.mark.unit
class TestCacheTtl:
    def test_stale_cache_triggers_refetch(self):
        page_v1 = _page(1, 1, 1, [{"ticker": "GME", "mentions": 1, "upvotes": 1}])
        page_v2 = _page(1, 1, 1, [{"ticker": "GME", "mentions": 999, "upvotes": 1}])
        responses = [_Resp(page_v1), _Resp(page_v2)]

        with patch.object(apewisdom, "urlopen", side_effect=lambda *a, **k: responses.pop(0)):
            first = apewisdom.fetch_apewisdom_mentions("GME")
            # Simulate the cache going stale without sleeping in the test.
            apewisdom._cache_fetched_at -= apewisdom._CACHE_TTL_SECONDS + 1
            second = apewisdom.fetch_apewisdom_mentions("GME")

        assert "1 mentions" in first
        assert "999 mentions" in second
