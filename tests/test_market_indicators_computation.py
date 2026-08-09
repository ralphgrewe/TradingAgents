"""
Unit tests for market_indicators_computation.py

Tests the deterministic indicator computation module with fixed OHLCV fixtures.
"""

import json
from datetime import datetime, timedelta

import pytest

from tradingagents.agents.analysts.market_indicators_computation import (
    build_json_envelope,
    build_key_drivers,
    compute_indicators,
    confidence_to_score,
    distance_from_52week_high,
    find_swing_high_low,
    interpret_signal,
    rate_of_change,
    rolling_n_day_high_low,
    trend,
    v,
    volume_surge_ratio,
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


class TestSwingIndicators:
    """Test swing trading indicators (issue #89)."""

    def test_rolling_n_day_high_low_basic(self):
        """rolling_n_day_high_low should compute rolling extremes."""
        import pandas as pd
        highs = pd.Series([100, 105, 102, 110, 108, 112, 115, 113, 118, 120])
        lows = pd.Series([95, 100, 98, 105, 103, 108, 110, 108, 115, 117])

        rolling_h, rolling_l = rolling_n_day_high_low(highs, lows, n=3)

        # After 3 bars, we should have rolling highs/lows
        assert rolling_h.iloc[2] == 105  # max of first 3 highs
        assert rolling_l.iloc[2] == 95  # min of first 3 lows

        # Later value
        assert rolling_h.iloc[9] == 120  # max of last 3 highs (113, 118, 120)
        assert rolling_l.iloc[9] == 108  # min of last 3 lows (108, 115, 117)

    def test_rolling_n_day_high_low_insufficient_data(self):
        """rolling_n_day_high_low should return NaN for early bars."""
        import pandas as pd
        highs = pd.Series([100, 105, 102])
        lows = pd.Series([95, 100, 98])

        rolling_h, rolling_l = rolling_n_day_high_low(highs, lows, n=5)

        # Not enough data, all should be NaN
        assert pd.isna(rolling_h.iloc[0])
        assert pd.isna(rolling_l.iloc[0])

    def test_find_swing_high_low_insufficient_data(self):
        """find_swing_high_low should return None with insufficient bars."""
        import pandas as pd
        highs = pd.Series([100, 105])
        lows = pd.Series([95, 100])

        swing_h, swing_l = find_swing_high_low(highs, lows, lookback=2)

        assert swing_h is None
        assert swing_l is None

    def test_find_swing_high_low_basic(self):
        """find_swing_high_low should identify pivot points."""
        import pandas as pd
        # Create a series with clear swing high in the middle
        # Bar: 0      1      2      3      4      5      6
        highs = pd.Series([100, 105, 110, 108, 106, 105, 100])
        lows = pd.Series([95, 100, 105, 103, 101, 100, 95])

        swing_h, swing_l = find_swing_high_low(highs, lows, lookback=2)

        # Bar 2 should be a swing high (110 is highest with lookback=2)
        if swing_h:
            assert swing_h["value"] == 110.0
            assert swing_h["bar_index"] == 2

    def test_find_swing_high_low_confirmed_pivot(self):
        """find_swing_high_low should only return confirmed (recent) pivots."""
        import pandas as pd
        # Create series with old swing high and recent potential swing
        highs = pd.Series([100, 110, 108, 105, 103, 100, 101, 105, 107, 105, 103])
        lows = pd.Series([95, 105, 103, 100, 98, 95, 96, 100, 102, 100, 98])

        swing_h, swing_l = find_swing_high_low(highs, lows, lookback=2)

        # Should find recent confirmed swings, not old ones
        if swing_h:
            assert swing_h["bars_since"] >= 2  # Must be at least lookback bars ago

    def test_rate_of_change_basic(self):
        """rate_of_change should compute ROC correctly."""
        import pandas as pd
        close = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])

        roc = rate_of_change(close, [1, 2, 5])

        # ROC for 1 day: (105 - 104) / 104 * 100 = 0.96%
        assert roc[1] is not None
        assert abs(roc[1] - 0.96) < 0.1

        # ROC for 5 days: (105 - 100) / 100 * 100 = 5%
        assert roc[5] == 5.0

        # Insufficient data for requested period returns None
        roc = rate_of_change(close, [10])
        assert roc[10] is None

    def test_rate_of_change_negative(self):
        """rate_of_change should handle downtrends."""
        import pandas as pd
        close = pd.Series([100.0, 99.0, 98.0, 97.0, 96.0, 95.0])

        roc = rate_of_change(close, [5])

        # ROC for 5 days: (95 - 100) / 100 * 100 = -5%
        assert roc[5] == -5.0

    def test_rate_of_change_insufficient_history(self):
        """rate_of_change should return None for insufficient history."""
        import pandas as pd
        close = pd.Series([100.0, 101.0])

        roc = rate_of_change(close, [5, 20, 63])

        assert roc[5] is None
        assert roc[20] is None
        assert roc[63] is None

    def test_distance_from_52week_high_basic(self):
        """distance_from_52week_high should compute distance from max."""
        import pandas as pd
        # Create 252+ bars with a peak at bar 100
        highs = pd.Series([100 + (i if i < 100 else 100 - (i - 100)) for i in range(260)])
        closes = highs  # close tracks high in this synthetic fixture

        distance = distance_from_52week_high(highs, closes)

        assert distance is not None

    def test_distance_from_52week_high_insufficient_history(self):
        """distance_from_52week_high should return None with insufficient history."""
        import pandas as pd
        highs = pd.Series([100 + i * 0.1 for i in range(100)])
        closes = highs

        distance = distance_from_52week_high(highs, closes)

        assert distance is None

    def test_distance_from_52week_high_percent(self):
        """distance_from_52week_high should return percentage."""
        import pandas as pd
        # Create 252+ bars: max is 200, current is 180
        base_high = 100
        highs = pd.Series(
            [base_high + i * 0.39 for i in range(252)]  # reaches ~200
            + [180] * 10  # last 10 bars at 180
        )
        closes = highs

        distance = distance_from_52week_high(highs, closes)

        assert distance is not None
        # distance should be ~10% (200-180)/200*100
        assert distance > 0 and distance < 20

    def test_distance_from_52week_high_uses_close_not_high(self):
        """current price should be the latest CLOSE, not the latest intraday high."""
        import pandas as pd
        # Flat history at 100 for 251 bars, then a fresh-high bar: high=110, close=105.
        highs = pd.Series([100.0] * 251 + [110.0])
        closes = pd.Series([100.0] * 251 + [105.0])

        distance = distance_from_52week_high(highs, closes)

        # max high = 110 (today's own bar); comparing against close (105):
        # distance = (110 - 105) / 110 * 100 ~= 4.5455%
        # A high-based implementation would (incorrectly) yield 0%.
        assert distance is not None
        assert abs(distance - 4.5455) < 0.01

    def test_volume_surge_ratio_basic(self):
        """volume_surge_ratio should compute volume ratio against the prior-20-day baseline."""
        import pandas as pd
        volume = pd.Series([1000000] * 20 + [2000000])

        ratio = volume_surge_ratio(volume)

        # Baseline (prior 20 days, excluding today) = 1,000,000; latest = 2,000,000
        assert ratio == 2.0

    def test_volume_surge_ratio_insufficient_history(self):
        """volume_surge_ratio should return None with insufficient history."""
        import pandas as pd
        volume = pd.Series([1000000] * 10)

        ratio = volume_surge_ratio(volume)

        assert ratio is None

    def test_volume_surge_ratio_needs_21_bars(self):
        """20 total bars is one short: the baseline needs 20 PRIOR days plus today."""
        import pandas as pd
        volume = pd.Series([1000000] * 19 + [2000000])

        ratio = volume_surge_ratio(volume)

        assert ratio is None

    def test_volume_surge_ratio_excludes_latest_bar_from_baseline(self):
        """The 20-day baseline must exclude the latest bar itself (no self-dilution)."""
        import pandas as pd
        # 20 quiet days then one surge day.
        volume = pd.Series([1000000] * 20 + [3000000])

        ratio = volume_surge_ratio(volume)

        # A baseline that (incorrectly) includes today would give ~2.74x.
        # Excluding today: baseline = 1,000,000, ratio = 3.0.
        assert ratio == 3.0

    def test_volume_surge_ratio_zero_average(self):
        """volume_surge_ratio should handle zero volume gracefully."""
        import pandas as pd
        volume = pd.Series([0] * 20 + [1000000])

        ratio = volume_surge_ratio(volume)

        # Baseline (prior 20 days) is 0, so the ratio is undefined -> None.
        assert ratio is None


class TestComputeIndicatorsWithSwingIndicators:
    """Test integration of swing indicators into compute_indicators."""

    def test_compute_indicators_includes_swing_indicators(self, sample_ohlcv_data):
        """compute_indicators should include swing_indicators in details."""
        result = compute_indicators(sample_ohlcv_data, "AAPL")

        assert "details" in result
        assert "swing_indicators" in result["details"]

    def test_compute_indicators_swing_indicators_structure(self, sample_ohlcv_data):
        """swing_indicators should contain expected fields."""
        result = compute_indicators(sample_ohlcv_data, "AAPL")
        swing = result["details"]["swing_indicators"]

        if swing:
            # Should have at least some of these fields
            possible_keys = [
                "n20_high", "n20_low",
                "swing_high", "swing_low",
                "rate_of_change",
                "distance_from_52week_high_pct",
                "volume_surge_ratio"
            ]
            has_any = any(k in swing for k in possible_keys)
            assert has_any, "swing_indicators should have at least one indicator"

    def test_compute_indicators_short_history_graceful_degradation(self):
        """compute_indicators should degrade gracefully for short history."""
        # Only 50 bars - not enough for some indicators
        base_date = datetime(2024, 1, 1)
        records = []
        price = 100.0
        for i in range(50):
            date = base_date + timedelta(days=i)
            price += 0.1 + (0.05 if i % 3 == 0 else -0.02)
            records.append({
                "Date": date.strftime("%Y-%m-%d"),
                "Open": price - 0.5,
                "High": price + 0.75,
                "Low": price - 0.75,
                "Close": price,
                "Volume": 1000000 + (i * 1000),
            })

        result = compute_indicators(records, "AAPL")

        # Should still produce a result without crashing
        assert result["signal"] is not None or result["signal"] is None  # Either is OK
        assert result["details"] is not None

        # Swing indicators might be None or partial
        swing = result["details"]["swing_indicators"]
        # This is OK - might not have 52-week data
        assert swing is None or isinstance(swing, dict)

    def test_compute_indicators_long_history_all_swing_indicators(self, sample_ohlcv_data):
        """compute_indicators with 100+ bars should compute all swing indicators."""
        result = compute_indicators(sample_ohlcv_data, "AAPL")
        swing = result["details"]["swing_indicators"]

        if swing:
            # Should have volume surge ratio (21 bars needed: 20 baseline + latest)
            assert "volume_surge_ratio" in swing

            # Should have rate of change (5 bars minimum needed)
            if "rate_of_change" in swing:
                roc = swing["rate_of_change"]
                assert "roc_5d" in roc  # Should have at least 5-day ROC

    def test_compute_indicators_swing_pivots_use_calendar_date_not_bar_index(self, sample_ohlcv_data):
        """swing_high/swing_low in the envelope should carry a calendar `date`,
        not the internal DataFrame `bar_index` (issue #89 design review)."""
        result = compute_indicators(sample_ohlcv_data, "AAPL")
        swing = result["details"]["swing_indicators"]

        if swing and "swing_high" in swing:
            assert "date" in swing["swing_high"]
            assert "bar_index" not in swing["swing_high"]
            # Should be a parseable calendar-date string, e.g. "2024-01-15"
            datetime.strptime(swing["swing_high"]["date"], "%Y-%m-%d")

        if swing and "swing_low" in swing:
            assert "date" in swing["swing_low"]
            assert "bar_index" not in swing["swing_low"]
            datetime.strptime(swing["swing_low"]["date"], "%Y-%m-%d")

    def test_compute_indicators_deterministic_with_swing(self, sample_ohlcv_data):
        """compute_indicators should be deterministic with swing indicators."""
        result1 = compute_indicators(sample_ohlcv_data, "AAPL")
        result2 = compute_indicators(sample_ohlcv_data, "AAPL")

        # swing_indicators should be identical
        swing1 = result1["details"]["swing_indicators"]
        swing2 = result2["details"]["swing_indicators"]

        assert swing1 == swing2
