"""Tests for the deterministic macro news prep layer (issue #133).

Covers tradingagents/dataflows/macro_news_pack.py (dedup, category tagging,
recency ordering, per-category caps, the historical-date gate) and its
vendor-routing wiring in interface.py. All vendor access is mocked; no
network calls happen.
"""
import copy
import unittest
from datetime import datetime
from unittest import mock

import pytest

import tradingagents.default_config as default_config
from tradingagents.agents.utils.news_data_tools import get_macro_news
from tradingagents.dataflows import config as config_module, interface
from tradingagents.dataflows.alpha_vantage_common import AlphaVantageNotConfiguredError
from tradingagents.dataflows.config import set_config
from tradingagents.dataflows.errors import VendorNotConfiguredError
from tradingagents.dataflows.macro_news_pack import (
    CATEGORY_KEYWORDS,
    build_macro_news_pack,
    render_macro_news_pack,
)


def _reset_config():
    config_module._config = copy.deepcopy(default_config.DEFAULT_CONFIG)


def _article(title, publisher="Reuters", link="", summary="", pub_date=None):
    return {
        "title": title,
        "summary": summary,
        "publisher": publisher,
        "link": link,
        "pub_date": pub_date,
    }


@pytest.mark.unit
class RoutingWiringTests(unittest.TestCase):
    """get_macro_news is registered like every other news_data tool."""

    def setUp(self):
        _reset_config()

    def tearDown(self):
        _reset_config()

    def test_get_macro_news_is_in_news_data_category(self):
        self.assertEqual(interface.get_category_for_method("get_macro_news"), "news_data")

    def test_get_macro_news_has_yfinance_and_alpha_vantage_impls(self):
        impls = interface.VENDOR_METHODS["get_macro_news"]
        self.assertIn("yfinance", impls)
        self.assertIn("alpha_vantage", impls)


@pytest.mark.unit
class HistoricalDateGateTests(unittest.TestCase):
    """Mirrors the researcher's date-gate convention (CLAUDE.md)."""

    def setUp(self):
        _reset_config()

    def tearDown(self):
        _reset_config()

    def test_non_today_date_disables_and_skips_vendor_call(self):
        vendor_impl = mock.Mock(side_effect=AssertionError("vendor must not be called"))
        with mock.patch.dict(
            interface.VENDOR_METHODS, {"get_macro_news": {"yfinance": vendor_impl}}, clear=False
        ):
            pack = build_macro_news_pack("2025-01-01", today="2026-08-21")

        self.assertEqual(pack["gate"]["outcome"], "disabled (historical date)")
        self.assertFalse(pack["gate"]["date_is_today"])
        self.assertIsNone(pack["vendor"])
        self.assertEqual(pack["categories"], {})
        vendor_impl.assert_not_called()

    def test_disabled_gate_render_does_not_leak_todays_news(self):
        pack = build_macro_news_pack("2025-01-01", today="2026-08-21")
        out = render_macro_news_pack(pack)
        self.assertIn("disabled (historical date)", out)
        self.assertNotIn("###", out)  # no article sections rendered

    def test_today_opens_the_gate(self):
        raw = {"vendor": "yfinance", "articles": []}
        with mock.patch.dict(
            interface.VENDOR_METHODS,
            {"get_macro_news": {"yfinance": lambda *a, **k: raw}},
            clear=False,
        ):
            pack = build_macro_news_pack("2026-08-21", today="2026-08-21")
        self.assertEqual(pack["gate"]["outcome"], "enabled")
        self.assertTrue(pack["gate"]["date_is_today"])


@pytest.mark.unit
class HappyPathTests(unittest.TestCase):
    def setUp(self):
        _reset_config()

    def tearDown(self):
        _reset_config()

    def test_dedup_category_tag_recency_and_cap(self):
        set_config({"macro_news_category_cap": 2})
        articles = [
            _article("Fed hikes interest rate again", link="u1",
                     pub_date=datetime(2026, 8, 20)),
            _article("Federal Reserve signals more rate hikes", link="u2",
                     pub_date=datetime(2026, 8, 21)),
            _article("Central bank monetary policy meeting today", link="u3",
                     pub_date=datetime(2026, 8, 19)),
            _article("CPI inflation report beats expectations", link="u4",
                     pub_date=datetime(2026, 8, 21)),
            # Duplicate of the first article (same title+publisher, diff URL) -> dropped.
            _article("Fed hikes interest rate again", link="u5",
                     pub_date=datetime(2026, 8, 20)),
        ]
        raw = {"vendor": "yfinance", "articles": articles}
        with mock.patch.dict(
            interface.VENDOR_METHODS,
            {"get_macro_news": {"yfinance": lambda *a, **k: raw}},
            clear=False,
        ):
            pack = build_macro_news_pack("2026-08-21", today="2026-08-21")

        self.assertEqual(pack["vendor"], "yfinance")
        # monetary_policy had 3 distinct articles (dup dropped), capped to 2,
        # newest first.
        monetary = pack["categories"]["monetary_policy"]
        self.assertEqual(len(monetary), 2)
        self.assertEqual(monetary[0]["title"], "Federal Reserve signals more rate hikes")
        self.assertEqual(monetary[1]["title"], "Fed hikes interest rate again")
        self.assertEqual(pack["categories"]["inflation_prices"][0]["title"],
                          "CPI inflation report beats expectations")
        # Total article count: 2 (capped monetary_policy) + 1 (inflation) = 3.
        self.assertEqual(pack["article_count"], 3)

    def test_uncategorized_bucket_used_when_no_keyword_matches(self):
        raw = {"vendor": "yfinance", "articles": [
            _article("Local bakery wins award", link="u1", pub_date=datetime(2026, 8, 21)),
        ]}
        with mock.patch.dict(
            interface.VENDOR_METHODS,
            {"get_macro_news": {"yfinance": lambda *a, **k: raw}},
            clear=False,
        ):
            pack = build_macro_news_pack("2026-08-21", today="2026-08-21")
        self.assertIn("uncategorized", pack["categories"])
        self.assertEqual(pack["categories"]["uncategorized"][0]["title"], "Local bakery wins award")

    def test_render_groups_by_category_and_notes_vendor(self):
        set_config({"data_vendors": {"news_data": "alpha_vantage"}})
        raw = {"vendor": "alpha_vantage", "articles": [
            _article("GDP growth beats forecasts", publisher="Bloomberg", link="u1",
                     summary="Economic growth accelerated.", pub_date=datetime(2026, 8, 21)),
        ]}
        with mock.patch.dict(
            interface.VENDOR_METHODS,
            {"get_macro_news": {"alpha_vantage": lambda *a, **k: raw}},
            clear=False,
        ):
            pack = build_macro_news_pack("2026-08-21", today="2026-08-21")
        out = render_macro_news_pack(pack)
        self.assertIn("source: alpha_vantage", out)
        self.assertIn("Growth & Output", out)
        self.assertIn("GDP growth beats forecasts", out)
        self.assertIn("Bloomberg", out)


@pytest.mark.unit
class EmptyResultTests(unittest.TestCase):
    def setUp(self):
        _reset_config()

    def tearDown(self):
        _reset_config()

    def test_empty_article_list_renders_informative_message(self):
        raw = {"vendor": "yfinance", "articles": []}
        with mock.patch.dict(
            interface.VENDOR_METHODS,
            {"get_macro_news": {"yfinance": lambda *a, **k: raw}},
            clear=False,
        ):
            pack = build_macro_news_pack("2026-08-21", today="2026-08-21")
        self.assertEqual(pack["categories"], {})
        self.assertEqual(pack["article_count"], 0)
        out = render_macro_news_pack(pack)
        self.assertIn("No macro news found", out)


@pytest.mark.unit
class FallbackAndErrorTests(unittest.TestCase):
    """route_to_vendor's existing comma-separated fallback; no custom chain here."""

    def setUp(self):
        _reset_config()

    def tearDown(self):
        _reset_config()

    def test_primary_unavailable_falls_back_to_next_configured_vendor(self):
        set_config({"data_vendors": {"news_data": "yfinance,alpha_vantage"}})

        def _unconfigured(*a, **k):
            raise VendorNotConfiguredError("yfinance not configured")

        av_result = {"vendor": "alpha_vantage", "articles": [
            _article("Oil supply disruption sparks fears", link="u1",
                     pub_date=datetime(2026, 8, 21)),
        ]}
        with mock.patch.dict(
            interface.VENDOR_METHODS,
            {"get_macro_news": {"yfinance": _unconfigured, "alpha_vantage": lambda *a, **k: av_result}},
            clear=False,
        ):
            pack = build_macro_news_pack("2026-08-21", today="2026-08-21")

        self.assertEqual(pack["vendor"], "alpha_vantage")
        self.assertIn("geopolitical_trade", pack["categories"])

    def test_missing_api_key_surfaces_clean_message_not_a_crash(self):
        # news_data is a core (non-optional) category with a single configured
        # vendor here, so the VendorNotConfiguredError propagates instead of
        # being swallowed -- and its message is human-readable, not a traceback.
        set_config({"data_vendors": {"news_data": "alpha_vantage"}})

        def _unconfigured(*a, **k):
            raise AlphaVantageNotConfiguredError("ALPHA_VANTAGE_API_KEY environment variable is not set.")

        with mock.patch.dict(
            interface.VENDOR_METHODS,
            {"get_macro_news": {"alpha_vantage": _unconfigured}},
            clear=False,
        ), self.assertRaises(VendorNotConfiguredError) as ctx:
            build_macro_news_pack("2026-08-21", today="2026-08-21")

        self.assertIn("ALPHA_VANTAGE_API_KEY", str(ctx.exception))


@pytest.mark.unit
class CategoryTaggingTests(unittest.TestCase):
    """Direct coverage of the fixed keyword taxonomy (decision comment on #133)."""

    def test_taxonomy_has_the_six_decided_categories(self):
        self.assertEqual(
            set(CATEGORY_KEYWORDS.keys()),
            {
                "monetary_policy", "inflation_prices", "labor_market",
                "growth_output", "markets_volatility", "geopolitical_trade",
            },
        )

    def test_case_insensitive_matching(self):
        from tradingagents.dataflows.macro_news_pack import _tag_category
        self.assertEqual(_tag_category("FEDERAL RESERVE HOLDS RATES", ""), "monetary_policy")
        self.assertEqual(_tag_category("unemployment claims fall", ""), "labor_market")
        self.assertEqual(_tag_category("Tariff war escalates", ""), "geopolitical_trade")

    def test_matches_against_summary_too(self):
        from tradingagents.dataflows.macro_news_pack import _tag_category
        self.assertEqual(
            _tag_category("Markets react", "Stocks fall amid vix spike"), "markets_volatility"
        )


@pytest.mark.unit
class ToolWrapperTests(unittest.TestCase):
    """The get_macro_news @tool is a thin wrapper over build+render."""

    def setUp(self):
        _reset_config()

    def tearDown(self):
        _reset_config()

    def test_tool_renders_the_built_pack(self):
        raw = {"vendor": "yfinance", "articles": [
            _article("Fed signals rate pause", link="u1", pub_date=datetime(2026, 8, 21)),
        ]}
        with mock.patch.dict(
            interface.VENDOR_METHODS,
            {"get_macro_news": {"yfinance": lambda *a, **k: raw}},
            clear=False,
        ):
            out = get_macro_news.func(curr_date="2026-08-21")
        self.assertIn("Monetary Policy", out)
        self.assertIn("Fed signals rate pause", out)

    def test_tool_reflects_historical_date_gate(self):
        out = get_macro_news.func(curr_date="2020-01-01")
        self.assertIn("disabled (historical date)", out)


if __name__ == "__main__":
    unittest.main()
