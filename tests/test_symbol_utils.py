"""Tests for symbol normalization and the no-data routing sentinel.

Merges the fork's original scoped-port tests (issue #4, upstream commit
7c8fe2f) with upstream's superset coverage (commit 1ff3f07 and later) after
adopting upstream's full ``symbol_utils`` module (issue #20).
"""

import unittest

import pytest

from tradingagents.dataflows.symbol_utils import (
    NoMarketDataError,
    has_non_us_exchange_suffix,
    is_non_equity_symbol,
    is_yahoo_safe,
    normalize_symbol,
)


@pytest.mark.unit
class TestNormalizeSymbol(unittest.TestCase):
    def test_plain_equities_unchanged(self):
        for sym in ("AAPL", "MSFT", "TSM", "BRK.B", "0700.HK", "^GSPC", "GC=F"):
            self.assertEqual(normalize_symbol(sym), sym)

    def test_lowercases_are_upper(self):
        self.assertEqual(normalize_symbol("aapl"), "AAPL")
        self.assertEqual(normalize_symbol("  msft  "), "MSFT")

    def test_metal_aliases_map_to_futures(self):
        self.assertEqual(normalize_symbol("XAUUSD"), "GC=F")
        self.assertEqual(normalize_symbol("XAUUSD+"), "GC=F")   # broker CFD suffix
        self.assertEqual(normalize_symbol("xauusd+"), "GC=F")
        self.assertEqual(normalize_symbol("GOLD"), "GC=F")
        self.assertEqual(normalize_symbol("XAGUSD"), "SI=F")

    def test_energy_and_index_aliases(self):
        self.assertEqual(normalize_symbol("USOIL"), "CL=F")
        self.assertEqual(normalize_symbol("SPX500"), "^GSPC")
        self.assertEqual(normalize_symbol("NAS100"), "^NDX")
        self.assertEqual(normalize_symbol("US30"), "^DJI")

    def test_forex_pairs_get_x_suffix(self):
        self.assertEqual(normalize_symbol("EURUSD"), "EURUSD=X")
        self.assertEqual(normalize_symbol("GBPJPY"), "GBPJPY=X")
        self.assertEqual(normalize_symbol("eurusd"), "EURUSD=X")

    def test_crypto_pairs_get_dash_usd(self):
        self.assertEqual(normalize_symbol("BTCUSD"), "BTC-USD")
        self.assertEqual(normalize_symbol("ETHUSD"), "ETH-USD")
        self.assertEqual(normalize_symbol("ethusd"), "ETH-USD")  # lowercase input

    def test_crypto_pairs_accept_usdt_and_usdc_quotes(self):
        # Yahoo only lists <BASE>-USD, so stablecoin-quoted broker symbols
        # (dashed or not) must resolve to the same -USD pair (#982).
        self.assertEqual(normalize_symbol("BTCUSDT"), "BTC-USD")
        self.assertEqual(normalize_symbol("BTC-USDT"), "BTC-USD")
        self.assertEqual(normalize_symbol("BTC-USDC"), "BTC-USD")

    def test_six_letter_non_currency_left_alone(self):
        # GOOGLE-style 6-letter tickers that aren't two currency codes
        # must not be mangled into a fake forex pair.
        self.assertEqual(normalize_symbol("ABCDEF"), "ABCDEF")

    def test_empty_input_passthrough(self):
        self.assertEqual(normalize_symbol(""), "")

    def test_non_string_passthrough(self):
        self.assertIsNone(normalize_symbol(None))


@pytest.mark.unit
class TestNoMarketDataError(unittest.TestCase):
    def test_message_includes_resolution(self):
        err = NoMarketDataError("XAUUSD+", "GC=F", "no rows")
        self.assertIn("XAUUSD+", str(err))
        self.assertIn("GC=F", str(err))
        self.assertEqual(err.symbol, "XAUUSD+")
        self.assertEqual(err.canonical, "GC=F")

    def test_canonical_defaults_to_symbol(self):
        err = NoMarketDataError("FOOBAR")
        self.assertEqual(err.canonical, "FOOBAR")


@pytest.mark.unit
class TestIsNonEquitySymbol(unittest.TestCase):
    def test_hyphenated_equity_share_classes_are_not_non_equity(self):
        # Regression for the design-review finding on issue #90: a bare
        # "-" character check misclassified these as non-equity.
        for sym in ("BRK-B", "BF-B"):
            self.assertFalse(is_non_equity_symbol(sym))

    def test_plain_equities_are_not_non_equity(self):
        for sym in ("AAPL", "MSFT", "TSM", "BRK.B", "0700.HK"):
            self.assertFalse(is_non_equity_symbol(sym))

    def test_index_symbols_are_non_equity(self):
        for sym in ("^GSPC", "^NDX", "^DJI", "^GDAXI", "^FTSE"):
            self.assertTrue(is_non_equity_symbol(sym))

    def test_futures_and_commodities_are_non_equity(self):
        for sym in ("GC=F", "SI=F", "CL=F", "BZ=F", "NG=F"):
            self.assertTrue(is_non_equity_symbol(sym))

    def test_forex_pairs_are_non_equity(self):
        for sym in ("EURUSD=X", "GBPJPY=X"):
            self.assertTrue(is_non_equity_symbol(sym))

    def test_crypto_pairs_are_non_equity(self):
        for sym in ("BTC-USD", "ETH-USD"):
            self.assertTrue(is_non_equity_symbol(sym))

    def test_works_on_normalize_symbol_output(self):
        # The intended call pattern: classify the canonical symbol produced
        # by normalize_symbol, e.g. for a broker/forex ticker like XAUUSD.
        self.assertTrue(is_non_equity_symbol(normalize_symbol("XAUUSD")))
        self.assertTrue(is_non_equity_symbol(normalize_symbol("EURUSD")))
        self.assertTrue(is_non_equity_symbol(normalize_symbol("BTCUSD")))
        self.assertTrue(is_non_equity_symbol(normalize_symbol("SPX500")))
        self.assertFalse(is_non_equity_symbol(normalize_symbol("AAPL")))
        self.assertFalse(is_non_equity_symbol(normalize_symbol("BRK-B")))

    def test_empty_and_non_string_inputs(self):
        self.assertFalse(is_non_equity_symbol(""))
        self.assertFalse(is_non_equity_symbol(None))


@pytest.mark.unit
class TestHasNonUsExchangeSuffix(unittest.TestCase):
    def test_non_us_suffixes_detected(self):
        for sym in ("ALFEN.AS", "SAP.DE", "0700.HK", "NOKIA.HE", "ASML.AS"):
            self.assertTrue(has_non_us_exchange_suffix(sym))

    def test_lowercase_suffix_detected(self):
        self.assertTrue(has_non_us_exchange_suffix("sap.de"))

    def test_us_tickers_not_flagged(self):
        for sym in ("AAPL", "MSFT", "SPCX", "ACHR", "CRWV"):
            self.assertFalse(has_non_us_exchange_suffix(sym))

    def test_hyphenated_share_class_not_flagged(self):
        # "." doesn't appear here, but guard against confusing "-" with a
        # suffix separator.
        self.assertFalse(has_non_us_exchange_suffix("BRK-B"))

    def test_dotted_share_class_without_suffix_not_flagged(self):
        # "BRK.B" has a "." but "B" is not a recognized exchange suffix.
        self.assertFalse(has_non_us_exchange_suffix("BRK.B"))

    def test_empty_and_non_string_inputs(self):
        self.assertFalse(has_non_us_exchange_suffix(""))
        self.assertFalse(has_non_us_exchange_suffix(None))

    def test_trailing_dot_no_suffix(self):
        self.assertFalse(has_non_us_exchange_suffix("AAPL."))


@pytest.mark.unit
class TestIsYahooSafe(unittest.TestCase):
    def test_accepts_structural_chars(self):
        for sym in ("AAPL", "GC=F", "^GSPC", "BRK.B", "BTC-USD"):
            self.assertTrue(is_yahoo_safe(sym))

    def test_rejects_slash_and_space(self):
        for sym in ("a/b", "AA PL", ""):
            self.assertFalse(is_yahoo_safe(sym))


if __name__ == "__main__":
    unittest.main()
