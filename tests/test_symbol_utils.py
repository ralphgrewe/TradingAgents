"""Unit tests for tradingagents.dataflows.symbol_utils.normalize_symbol.

Scoped port of upstream 7c8fe2f for issue #4 — this module only backs
``TradingAgentsGraph._fetch_returns`` in this fork (see test_memory_log.py's
TestDeferredReflection.test_fetch_returns_normalizes_broker_symbol for the
integration path). Covers the mapping rules directly.
"""
import pytest

from tradingagents.dataflows.symbol_utils import normalize_symbol


@pytest.mark.unit
class TestNormalizeSymbol:
    def test_metal_alias(self):
        assert normalize_symbol("XAUUSD") == "GC=F"
        assert normalize_symbol("xauusd") == "GC=F"

    def test_metal_alias_strips_broker_cfd_suffix(self):
        assert normalize_symbol("XAUUSD+") == "GC=F"

    def test_energy_alias(self):
        assert normalize_symbol("WTI") == "CL=F"

    def test_index_alias(self):
        assert normalize_symbol("SPX500") == "^GSPC"

    def test_crypto_pair(self):
        assert normalize_symbol("BTCUSD") == "BTC-USD"
        assert normalize_symbol("ethusd") == "ETH-USD"

    def test_forex_pair(self):
        assert normalize_symbol("EURUSD") == "EURUSD=X"

    def test_plain_equity_unchanged_but_uppercased(self):
        assert normalize_symbol("nvda") == "NVDA"

    def test_already_canonical_yahoo_symbol_unchanged(self):
        assert normalize_symbol("GC=F") == "GC=F"
        assert normalize_symbol("^GSPC") == "^GSPC"

    def test_non_string_passthrough(self):
        assert normalize_symbol(None) is None

    def test_empty_string_passthrough(self):
        assert normalize_symbol("") == ""
