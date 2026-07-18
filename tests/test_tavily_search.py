"""Unit tests for Tavily web search dataflow.

Tests cover search parsing, evidence-pack builder (dedupe/sort/truncate/id-assignment),
gate true/false cases, and missing-key path with mocked HTTP.
"""

import unittest
from unittest import mock
from datetime import datetime, timedelta

import pytest

from tradingagents.dataflows.tavily_search import (
    get_web_search_results,
    build_evidence_pack,
    is_web_search_allowed,
    TavilySearchError,
)
from tradingagents.dataflows.errors import VendorNotConfiguredError


@pytest.mark.unit
class TavilySearchTests(unittest.TestCase):
    """Test Tavily API integration."""

    def test_search_requires_api_key(self):
        """Missing TAVILY_API_KEY raises VendorNotConfiguredError."""
        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(VendorNotConfiguredError) as ctx:
                get_web_search_results("test query")
            self.assertIn("TAVILY_API_KEY", str(ctx.exception))

    def test_search_parses_results_from_api(self):
        """Search successfully parses Tavily API response."""
        mock_response = {
            "results": [
                {
                    "title": "Article 1",
                    "url": "https://example.com/1",
                    "content": "Snippet 1",
                    "score": 0.95,
                },
                {
                    "title": "Article 2",
                    "url": "https://example.com/2",
                    "content": "Snippet 2",
                    "score": 0.85,
                },
            ]
        }

        with mock.patch.dict("os.environ", {"TAVILY_API_KEY": "test_key"}):
            with mock.patch("tradingagents.dataflows.tavily_search.requests.post") as mock_post:
                mock_post.return_value.json.return_value = mock_response
                mock_post.return_value.raise_for_status.return_value = None

                results = get_web_search_results("test query")

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["title"], "Article 1")
        self.assertEqual(results[0]["url"], "https://example.com/1")
        self.assertEqual(results[0]["content"], "Snippet 1")
        self.assertEqual(results[0]["score"], 0.95)

    def test_search_handles_api_request_failure(self):
        """API request failure raises TavilySearchError, not VendorNotConfiguredError."""
        import requests
        with mock.patch.dict("os.environ", {"TAVILY_API_KEY": "test_key"}):
            with mock.patch("tradingagents.dataflows.tavily_search.requests.post") as mock_post:
                mock_post.side_effect = requests.exceptions.RequestException("Network error")

                with self.assertRaises(TavilySearchError) as ctx:
                    get_web_search_results("test query")
                self.assertIn("Tavily API request failed", str(ctx.exception))
                self.assertNotIsInstance(ctx.exception, VendorNotConfiguredError)

    def test_search_handles_json_parse_error(self):
        """JSON parsing error raises TavilySearchError, not VendorNotConfiguredError."""
        with mock.patch.dict("os.environ", {"TAVILY_API_KEY": "test_key"}):
            with mock.patch("tradingagents.dataflows.tavily_search.requests.post") as mock_post:
                mock_post.return_value.json.side_effect = ValueError("Invalid JSON")
                mock_post.return_value.raise_for_status.return_value = None

                with self.assertRaises(TavilySearchError) as ctx:
                    get_web_search_results("test query")
                self.assertIn("Error parsing Tavily response", str(ctx.exception))
                self.assertNotIsInstance(ctx.exception, VendorNotConfiguredError)

    def test_missing_key_and_http_failure_are_distinct_exception_types(self):
        """Missing-API-key and HTTP/JSON failures raise distinguishable exception types.

        This is the crux of design-review finding #2: callers must be able to
        tell "disabled, no API key" apart from a generic degraded-search
        outcome by exception type, not by string-matching the message.
        """
        import requests

        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(VendorNotConfiguredError) as key_ctx:
                get_web_search_results("test query")
        self.assertNotIsInstance(key_ctx.exception, TavilySearchError)

        with mock.patch.dict("os.environ", {"TAVILY_API_KEY": "test_key"}):
            with mock.patch("tradingagents.dataflows.tavily_search.requests.post") as mock_post:
                mock_post.side_effect = requests.exceptions.RequestException("boom")
                with self.assertRaises(TavilySearchError) as http_ctx:
                    get_web_search_results("test query")
        self.assertNotIsInstance(http_ctx.exception, VendorNotConfiguredError)

    def test_search_handles_missing_fields(self):
        """Missing fields in response are handled gracefully."""
        mock_response = {
            "results": [
                {
                    "title": "Article 1",
                    # url missing
                    "content": "Snippet 1",
                    "score": 0.95,
                },
            ]
        }

        with mock.patch.dict("os.environ", {"TAVILY_API_KEY": "test_key"}):
            with mock.patch("tradingagents.dataflows.tavily_search.requests.post") as mock_post:
                mock_post.return_value.json.return_value = mock_response
                mock_post.return_value.raise_for_status.return_value = None

                results = get_web_search_results("test query")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["url"], "")


@pytest.mark.unit
class EvidencePackBuilderTests(unittest.TestCase):
    """Test evidence-pack builder logic."""

    def test_empty_input_returns_empty_pack(self):
        """Empty query results return empty pack."""
        pack = build_evidence_pack([], set())
        self.assertEqual(pack, [])

    def test_single_query_single_result(self):
        """Single result is included and assigned ID."""
        results = [[
            {"title": "Article", "url": "https://example.com", "content": "Text", "score": 0.9}
        ]]
        pack = build_evidence_pack(results, set())

        self.assertEqual(len(pack), 1)
        self.assertEqual(pack[0]["id"], "web:1")
        self.assertEqual(pack[0]["title"], "Article")
        self.assertEqual(pack[0]["score"], 0.9)

    def test_dedupes_by_url_across_queries(self):
        """Duplicate URLs across queries are removed."""
        url = "https://example.com/article"
        results = [
            [
                {"title": "Article A", "url": url, "content": "Text A", "score": 0.95}
            ],
            [
                {"title": "Article A", "url": url, "content": "Text A", "score": 0.90}  # duplicate
            ],
        ]
        pack = build_evidence_pack(results, set())

        self.assertEqual(len(pack), 1)
        # First one (higher score) should be kept
        self.assertEqual(pack[0]["title"], "Article A")

    def test_dedupe_keeps_highest_scoring_duplicate_regardless_of_query_order(self):
        """Dedup keeps the highest-scoring duplicate even when it is NOT seen first.

        Design-review finding #3: dedup must sort (score desc, then URL) before
        deduping, so the outcome doesn't depend on which query happens to be
        iterated first. Here the *lower*-scoring duplicate is listed first.
        """
        url = "https://example.com/article"
        results = [
            [
                {"title": "Lower score, seen first", "url": url, "content": "Text", "score": 0.50}
            ],
            [
                {"title": "Higher score, seen second", "url": url, "content": "Text", "score": 0.95}
            ],
        ]
        pack = build_evidence_pack(results, set())

        self.assertEqual(len(pack), 1)
        self.assertEqual(pack[0]["title"], "Higher score, seen second")
        self.assertEqual(pack[0]["score"], 0.95)

    def test_dedupes_against_already_cited_urls(self):
        """URLs already cited are excluded from pack."""
        already_cited = {"https://example.com/article1"}
        results = [[
            {"title": "Article 1", "url": "https://example.com/article1", "content": "Text", "score": 0.9},
            {"title": "Article 2", "url": "https://example.com/article2", "content": "Text", "score": 0.85},
        ]]
        pack = build_evidence_pack(results, already_cited)

        self.assertEqual(len(pack), 1)
        self.assertEqual(pack[0]["title"], "Article 2")

    def test_sorts_by_score_descending(self):
        """Results are sorted by score (descending)."""
        results = [[
            {"title": "Low", "url": "https://example.com/3", "content": "Text", "score": 0.7},
            {"title": "High", "url": "https://example.com/1", "content": "Text", "score": 0.95},
            {"title": "Mid", "url": "https://example.com/2", "content": "Text", "score": 0.85},
        ]]
        pack = build_evidence_pack(results, set())

        self.assertEqual(len(pack), 3)
        self.assertEqual(pack[0]["title"], "High")
        self.assertEqual(pack[1]["title"], "Mid")
        self.assertEqual(pack[2]["title"], "Low")

    def test_sorts_by_url_ascending_when_score_tied(self):
        """When scores are tied, sort by URL ascending."""
        results = [[
            {"title": "B", "url": "https://example.com/b", "content": "Text", "score": 0.9},
            {"title": "A", "url": "https://example.com/a", "content": "Text", "score": 0.9},
            {"title": "C", "url": "https://example.com/c", "content": "Text", "score": 0.9},
        ]]
        pack = build_evidence_pack(results, set())

        self.assertEqual(len(pack), 3)
        self.assertEqual(pack[0]["url"], "https://example.com/a")
        self.assertEqual(pack[1]["url"], "https://example.com/b")
        self.assertEqual(pack[2]["url"], "https://example.com/c")

    def test_truncates_to_token_budget(self):
        """Pack is truncated to token budget (chars/4)."""
        # Each result is ~100 chars, so at 200 tokens (800 chars), should fit ~8 results
        results = [[{
            "title": f"Article {i}",
            "url": f"https://example.com/{i}",
            "content": "x" * 50,  # ~50 chars
            "score": 1.0 - (i * 0.01),  # descending scores
        } for i in range(20)]]

        pack = build_evidence_pack(results, set(), token_budget=200)

        # Should be truncated
        self.assertLess(len(pack), 20)
        # All included results should fit in budget
        total_chars = sum(
            len(r["title"]) + len(r["url"]) + len(r["content"])
            for r in pack
        )
        self.assertLessEqual(total_chars, 200 * 4)

    def test_assigns_sequential_ids(self):
        """Results are assigned sequential IDs (web:1, web:2, ...)."""
        results = [[
            {"title": f"Article {i}", "url": f"https://example.com/{i}", "content": "Text", "score": 0.9}
            for i in range(3)
        ]]
        pack = build_evidence_pack(results, set())

        self.assertEqual(len(pack), 3)
        self.assertEqual(pack[0]["id"], "web:1")
        self.assertEqual(pack[1]["id"], "web:2")
        self.assertEqual(pack[2]["id"], "web:3")

    def test_complex_scenario(self):
        """Complex scenario with multiple queries, dedupe, sort, truncate."""
        already_cited = {"https://example.com/old"}
        results = [
            [
                {"title": "A", "url": "https://example.com/a", "content": "x" * 100, "score": 0.9},
                {"title": "Dup", "url": "https://example.com/old", "content": "x" * 100, "score": 0.95},
            ],
            [
                {"title": "B", "url": "https://example.com/b", "content": "x" * 100, "score": 0.85},
                {"title": "A2", "url": "https://example.com/a", "content": "x" * 100, "score": 0.88},  # dup
            ],
        ]

        pack = build_evidence_pack(results, already_cited, token_budget=600)

        # Should have deduplicated and sorted: A (0.9), B (0.85), Dup removed, A2 removed
        self.assertEqual(len(pack), 2)
        self.assertEqual(pack[0]["id"], "web:1")
        self.assertEqual(pack[0]["title"], "A")
        self.assertEqual(pack[1]["id"], "web:2")
        self.assertEqual(pack[1]["title"], "B")


@pytest.mark.unit
class WebSearchLeakageGateTests(unittest.TestCase):
    """Test web-search leakage hard gate."""

    def test_allows_search_on_live_date(self):
        """Web search allowed when trade_date == today."""
        today_str = datetime.now().strftime("%Y-%m-%d")
        self.assertTrue(is_web_search_allowed(today_str, today_str))

    def test_blocks_search_on_past_date(self):
        """Web search blocked for past dates."""
        today_str = datetime.now().strftime("%Y-%m-%d")
        past_str = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        self.assertFalse(is_web_search_allowed(past_str, today_str))

    def test_blocks_search_on_future_date(self):
        """Web search blocked for future dates."""
        today_str = datetime.now().strftime("%Y-%m-%d")
        future_str = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        self.assertFalse(is_web_search_allowed(future_str, today_str))

    def test_uses_current_date_when_today_not_provided(self):
        """When today is not provided, uses current date."""
        today_str = datetime.now().strftime("%Y-%m-%d")
        # Should allow today
        self.assertTrue(is_web_search_allowed(today_str))
        # Should block yesterday
        past_str = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        self.assertFalse(is_web_search_allowed(past_str))

    def test_iso_string_comparison(self):
        """Date comparison uses ISO string format (yyyy-mm-dd)."""
        # These are explicitly formatted ISO strings
        trade_date = "2026-07-17"
        today = "2026-07-17"
        self.assertTrue(is_web_search_allowed(trade_date, today))

        # Off by one day should fail
        self.assertFalse(is_web_search_allowed("2026-07-16", "2026-07-17"))
        self.assertFalse(is_web_search_allowed("2026-07-18", "2026-07-17"))


if __name__ == "__main__":
    unittest.main()
