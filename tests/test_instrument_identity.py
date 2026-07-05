"""Tests for deterministic instrument-identity resolution.

Ported/adapted from upstream TradingAgents (d7b40a2, "fix(graph): resolve
instrument identity to stop wrong-company hallucination") as part of issue
#18: grafting anti-hallucination grounding into this fork's market analyst.

Unlike upstream, this fork does not (yet) resolve identity once at the graph
level and stash it on ``state["instrument_context"]`` — that would touch
nearly every agent/prompt/graph file, which is out of scope for the
market-analyst-only reconciliation in #18. Instead
``get_instrument_context_from_state`` resolves identity lazily (and still
only once per ticker per process, via ``resolve_instrument_identity``'s
``lru_cache``) when the state doesn't already carry a precomputed context —
so it stays forward-compatible with a future graph-level wiring pass while
giving the market analyst real grounding today.
"""

import unittest
from unittest.mock import patch

import pytest

from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_instrument_context_from_state,
    resolve_instrument_identity,
)


@pytest.mark.unit
class ResolveInstrumentIdentityTests(unittest.TestCase):
    def setUp(self):
        resolve_instrument_identity.cache_clear()

    def test_resolves_company_metadata_from_yfinance(self):
        with patch("tradingagents.agents.utils.agent_utils.yf.Ticker") as mock:
            mock.return_value.info = {
                "longName": "TOTO LTD.",
                "shortName": "TOTO",
                "sector": "Industrials",
                "industry": "Building Products & Equipment",
                "exchange": "PNK",
                "quoteType": "EQUITY",
            }
            identity = resolve_instrument_identity("totdy")
        mock.assert_called_once_with("TOTDY")
        self.assertEqual(identity["company_name"], "TOTO LTD.")
        self.assertEqual(identity["sector"], "Industrials")
        self.assertEqual(identity["industry"], "Building Products & Equipment")
        self.assertEqual(identity["exchange"], "PNK")

    def test_falls_back_to_short_name(self):
        with patch("tradingagents.agents.utils.agent_utils.yf.Ticker") as mock:
            mock.return_value.info = {"shortName": "TOTO", "sector": "Industrials"}
            identity = resolve_instrument_identity("TOTDY")
        self.assertEqual(identity["company_name"], "TOTO")

    def test_skips_placeholder_values(self):
        with patch("tradingagents.agents.utils.agent_utils.yf.Ticker") as mock:
            mock.return_value.info = {"longName": "  ", "sector": "None", "industry": "n/a"}
            identity = resolve_instrument_identity("TOTDY")
        self.assertEqual(identity, {})

    def test_fails_open_on_exception(self):
        with patch(
            "tradingagents.agents.utils.agent_utils.yf.Ticker",
            side_effect=RuntimeError("rate limited"),
        ):
            self.assertEqual(resolve_instrument_identity("TOTDY"), {})

    def test_result_is_cached(self):
        with patch("tradingagents.agents.utils.agent_utils.yf.Ticker") as mock:
            mock.return_value.info = {"longName": "TOTO LTD."}
            first = resolve_instrument_identity("TOTDY")
            second = resolve_instrument_identity("TOTDY")
        mock.assert_called_once()  # second call served from cache
        self.assertEqual(first, second)


@pytest.mark.unit
class BuildInstrumentContextTests(unittest.TestCase):
    def test_mentions_exact_symbol_without_identity(self):
        context = build_instrument_context("7203.T")
        self.assertIn("7203.T", context)
        self.assertIn("exchange suffix", context)
        self.assertNotIn("Resolved identity", context)

    def test_injects_resolved_identity(self):
        context = build_instrument_context(
            "TOTDY", "stock",
            {
                "company_name": "TOTO LTD.",
                "sector": "Industrials",
                "industry": "Building Products & Equipment",
                "exchange": "PNK",
            },
        )
        self.assertIn("Company: TOTO LTD.", context)
        self.assertIn("Industrials / Building Products & Equipment", context)
        self.assertIn("Exchange: PNK", context)
        self.assertIn("Do not substitute a different company", context)

    def test_crypto_uses_name_label_and_keeps_hint(self):
        context = build_instrument_context(
            "BTC-USD", "crypto", {"company_name": "Bitcoin USD"}
        )
        self.assertIn("Name: Bitcoin USD", context)
        self.assertIn("crypto asset rather than a company", context)

    def test_backward_compatible_without_identity_arg(self):
        # Existing callers (news/fundamentals/sentiment analysts, trader,
        # managers) pass only ticker/asset_type positionally — must still work.
        context = build_instrument_context("NVDA", "stock")
        self.assertIn("NVDA", context)


@pytest.mark.unit
class GetInstrumentContextFromStateTests(unittest.TestCase):
    def setUp(self):
        resolve_instrument_identity.cache_clear()

    def test_prefers_precomputed_context(self):
        with patch("tradingagents.agents.utils.agent_utils.yf.Ticker") as mock:
            state = {"company_of_interest": "TOTDY", "instrument_context": "PRECOMPUTED"}
            self.assertEqual(get_instrument_context_from_state(state), "PRECOMPUTED")
        mock.assert_not_called()

    def test_resolves_identity_when_not_precomputed(self):
        with patch("tradingagents.agents.utils.agent_utils.yf.Ticker") as mock:
            mock.return_value.info = {"longName": "Ecopetrol", "sector": "Energy"}
            context = get_instrument_context_from_state(
                {"company_of_interest": "EC", "asset_type": "stock"}
            )
        mock.assert_called_once_with("EC")
        self.assertIn("EC", context)
        self.assertIn("Ecopetrol", context)

    def test_fails_open_when_lookup_errors(self):
        with patch(
            "tradingagents.agents.utils.agent_utils.yf.Ticker",
            side_effect=RuntimeError("network down"),
        ):
            context = get_instrument_context_from_state(
                {"company_of_interest": "NVDA", "asset_type": "stock"}
            )
        self.assertIn("NVDA", context)

    def test_respects_asset_type(self):
        with patch("tradingagents.agents.utils.agent_utils.yf.Ticker") as mock:
            mock.return_value.info = {}
            context = get_instrument_context_from_state(
                {"company_of_interest": "BTC-USD", "asset_type": "crypto"}
            )
        self.assertIn("crypto asset", context)


if __name__ == "__main__":
    unittest.main()
