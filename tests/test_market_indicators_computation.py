"""
Unit tests for market_indicators_computation.py

Tests the deterministic indicator computation module with fixed OHLCV fixtures.
"""

import json
from datetime import datetime, timedelta

import pytest

from tradingagents.agents.analysts.market_indicators_computation import (
    compute_indicators,
    build_json_envelope,
    confidence_to_score,
    build_key_drivers,
    v,
    trend,
    interpret_signal,
)


@pytest.fixture
def sample_ohlcv_data():
    """Generate a simple fixture of OHLCV data (100 rows)."""
    base_date = datetime(2024, 1, 1)
    records = []
    price = 100.0
    for i in range(100):
        date = base_date + timedelta(days=i)
        # Gentle uptrend with noise
        price += 0.1 + (0.05 if i % 3 == 0 else -0.02)
        records.append({
            "Date": date.strftime("%Y-%m-%d"),
            "Open": price - 0.5,
            "High": price + 0.75,
            "Low": price - 0.75,
            "Close": price,
            "Volume": 1000000 + (i * 1000),
        })
    return records


@pytest.fixture
def sample_ohlcv_bearish():
    """Generate fixture with downtrend."""
    base_date = datetime(2024, 1, 1)
    records = []
    price = 100.0
    for i in range(100):
        date = base_date + timedelta(days=i)
        # Gentle downtrend
        price -= 0.1 + (0.05 if i % 3 == 0 else 0.02)
        records.append({
            "Date": date.strftime("%Y-%m-%d"),
            "Open": price + 0.5,
            "High": price + 0.75,
            "Low": price - 0.75,
            "Close": price,
            "Volume": 1000000,
        })
    return records


class TestHelperFunctions:
    """Test standalone helper functions."""

    def test_v_converts_float(self):
        """v() should convert valid numbers."""
        assert v(100.5) == 100.5
        assert v(0.0) == 0.0

    def test_v_returns_none_for_nan(self):
        """v() should return None for NaN."""
        import math
        assert v(float('nan')) is None
        assert v(float('inf')) is None
        assert v(None) is None

    def test_v_rounds_to_4_decimals(self):
        """v() should round to 4 decimal places."""
        assert v(100.123456) == 100.1235

    def test_trend_rising(self):
        """trend() should detect rising."""
        assert trend(100.2, 100.0) == "Rising"

    def test_trend_falling(self):
        """trend() should detect falling."""
        assert trend(99.8, 100.0) == "Falling"

    def test_trend_flat(self):
        """trend() should detect flat."""
        assert trend(100.05, 100.0) == "Flat"

    def test_trend_with_none(self):
        """trend() should return Flat when given None."""
        assert trend(None, 100.0) == "Flat"
        assert trend(100.0, None) == "Flat"

    def test_interpret_signal_sma_bullish(self):
        """interpret_signal should detect bullish SMA-50."""
        # Close above SMA
        sig = interpret_signal("sma_50", 100.0, 100.0, 101.0)
        assert sig == "Bullish"

    def test_interpret_signal_sma_bearish(self):
        """interpret_signal should detect bearish SMA-50."""
        # Close below SMA
        sig = interpret_signal("sma_50", 100.0, 100.0, 99.0)
        assert sig == "Bearish"

    def test_interpret_signal_rsi_overbought(self):
        """interpret_signal should detect RSI overbought."""
        sig = interpret_signal("rsi", 75.0, 70.0, None)
        assert sig == "Bearish"

    def test_interpret_signal_rsi_oversold(self):
        """interpret_signal should detect RSI oversold."""
        sig = interpret_signal("rsi", 25.0, 30.0, None)
        assert sig == "Bullish"

    def test_confidence_to_score_high(self):
        """confidence_to_score should map HIGH -> 1.0."""
        assert confidence_to_score("HIGH") == 1.0

    def test_confidence_to_score_medium(self):
        """confidence_to_score should map MEDIUM -> 0.6."""
        assert confidence_to_score("MEDIUM") == 0.6

    def test_confidence_to_score_low(self):
        """confidence_to_score should map LOW -> 0.3."""
        assert confidence_to_score("LOW") == 0.3

    def test_confidence_to_score_unknown(self):
        """confidence_to_score should default to 0.3 for unknown."""
        assert confidence_to_score("UNKNOWN") == 0.3


class TestComputeIndicators:
    """Test the main compute_indicators function."""

    def test_compute_indicators_empty(self):
        """compute_indicators should handle empty records gracefully."""
        result = compute_indicators([], "AAPL")
        assert result["signal"] is None
        assert result["confidence"] is None
        assert result["summary"] == "Insufficient data"

    def test_compute_indicators_single_record(self):
        """compute_indicators should handle single record gracefully."""
        records = [{
            "Date": "2024-01-01",
            "Open": 100.0,
            "High": 101.0,
            "Low": 99.0,
            "Close": 100.5,
            "Volume": 1000000,
        }]
        result = compute_indicators(records, "AAPL")
        assert result["signal"] is None
        assert result["confidence"] is None

    def test_compute_indicators_basic_uptrend(self, sample_ohlcv_data):
        """compute_indicators should detect uptrend."""
        result = compute_indicators(sample_ohlcv_data, "AAPL")

        assert "signal" in result
        assert "confidence" in result
        assert "details" in result
        assert "summary" in result

        # Uptrend should result in BUY or HOLD
        assert result["signal"] in ("BUY", "HOLD", "SELL")
        assert result["confidence"] in ("HIGH", "MEDIUM", "LOW")

    def test_compute_indicators_basic_downtrend(self, sample_ohlcv_bearish):
        """compute_indicators should detect downtrend."""
        result = compute_indicators(sample_ohlcv_bearish, "AAPL")

        assert result["signal"] in ("BUY", "HOLD", "SELL")
        assert result["confidence"] in ("HIGH", "MEDIUM", "LOW")

    def test_compute_indicators_deterministic(self, sample_ohlcv_data):
        """compute_indicators should be deterministic."""
        result1 = compute_indicators(sample_ohlcv_data, "AAPL")
        result2 = compute_indicators(sample_ohlcv_data, "AAPL")

        # signal and confidence should be identical
        assert result1["signal"] == result2["signal"]
        assert result1["confidence"] == result2["confidence"]

        # details should be identical
        assert result1["details"] == result2["details"]

    def test_compute_indicators_details_structure(self, sample_ohlcv_data):
        """compute_indicators details should have correct structure."""
        result = compute_indicators(sample_ohlcv_data, "AAPL")
        details = result["details"]

        assert "as_of" in details
        assert "close" in details
        assert "market_regime" in details
        assert "indicators" in details
        assert "convergence" in details
        assert "trade_setup" in details

        # Indicators list should have 7 indicators
        assert len(details["indicators"]) == 7

        # Each indicator should have required fields
        for ind in details["indicators"]:
            assert "indicator" in ind
            assert "value" in ind
            assert "prev" in ind
            assert "trend" in ind
            assert "signal" in ind
            assert "role" in ind

    def test_compute_indicators_convergence(self, sample_ohlcv_data):
        """compute_indicators convergence should list confirms/conflicts/missing."""
        result = compute_indicators(sample_ohlcv_data, "AAPL")
        conv = result["details"]["convergence"]

        assert "confirms" in conv
        assert "conflicts" in conv
        assert "missing" in conv

        assert isinstance(conv["confirms"], list)
        assert isinstance(conv["conflicts"], list)
        assert isinstance(conv["missing"], list)

    def test_compute_indicators_trade_setup(self, sample_ohlcv_data):
        """compute_indicators trade_setup should have required fields."""
        result = compute_indicators(sample_ohlcv_data, "AAPL")
        ts = result["details"]["trade_setup"]

        if ts:
            assert "bias" in ts
            assert "entry_trigger" in ts
            assert ts["bias"] in ("BUY", "SELL", "HOLD")

    def test_compute_indicators_market_regime(self, sample_ohlcv_data):
        """compute_indicators should identify market regime."""
        result = compute_indicators(sample_ohlcv_data, "AAPL")
        regime = result["details"]["market_regime"]

        assert regime in ("trending_up", "trending_down", "ranging", "volatile")


class TestBuildJsonEnvelope:
    """Test the JSON envelope builder."""

    def test_build_json_envelope_structure(self):
        """build_json_envelope should create valid JSON."""
        details = {"test": "data"}
        envelope_str = build_json_envelope(
            signal="BUY",
            confidence="HIGH",
            summary="Test summary",
            details=details,
            ticker="AAPL",
            date="2024-01-15",
        )

        # Should be valid JSON
        envelope = json.loads(envelope_str)

        assert envelope["skill"] == "market-analyst"
        assert envelope["ticker"] == "AAPL"
        assert envelope["date"] == "2024-01-15"
        assert envelope["signal"] == "BUY"
        assert envelope["confidence"] == "HIGH"
        assert envelope["summary"] == "Test summary"
        assert envelope["details"] == details

    def test_build_json_envelope_with_nulls(self):
        """build_json_envelope should handle null values."""
        details = None
        envelope_str = build_json_envelope(
            signal=None,
            confidence=None,
            summary="No data",
            details={},
            ticker="AAPL",
            date="2024-01-15",
        )

        envelope = json.loads(envelope_str)
        assert envelope["signal"] is None
        assert envelope["confidence"] is None


class TestBuildKeyDrivers:
    """Test the key_drivers builder for memory integration."""

    def test_build_key_drivers(self):
        """build_key_drivers should extract market regime and trade setup."""
        details = {
            "market_regime": "trending_up",
            "convergence": {
                "confirms": ["sma_50", "macdh"],
                "conflicts": [],
                "missing": [],
            },
            "trade_setup": {
                "bias": "BUY",
                "entry_trigger": "above SMA",
            },
        }

        drivers = build_key_drivers(details)

        assert drivers["market_regime"] == "trending_up"
        assert drivers["confirms"] == ["sma_50", "macdh"]
        assert drivers["conflicts"] == []
        assert drivers["trade_setup"]["bias"] == "BUY"
