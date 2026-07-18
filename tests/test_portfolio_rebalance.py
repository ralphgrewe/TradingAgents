"""Unit tests for the deterministic portfolio allocation engine.

Covers the pure-function core ported from skills/portfolio-manager/rebalance.py:
raw_weight (both styles, incl. HOLD=exit vs HOLD=keep and the SELL trim
rule), normalise (cap + re-normalisation), compute_allocation end-to-end,
and the rating -> style-table signal mapping.
"""

import pytest

from tradingagents.portfolio.rebalance import (
    RATING_TO_SIGNAL,
    STYLE_PARAMS,
    compute_allocation,
    normalise,
    rating_to_signal,
    raw_weight,
)


class TestRawWeightAggressive:
    """Aggressive: HOLD/SELL both mean 'exit' (raw weight 0), regardless of current weight."""

    def test_buy_high(self):
        assert raw_weight("aggressive", "BUY", "HIGH", current_weight=0.5, slot=0.19) == pytest.approx(0.38)

    def test_buy_medium(self):
        assert raw_weight("aggressive", "BUY", "MEDIUM", current_weight=0.5, slot=0.19) == pytest.approx(0.19)

    def test_buy_low(self):
        assert raw_weight("aggressive", "BUY", "LOW", current_weight=0.5, slot=0.19) == pytest.approx(0.095)

    def test_hold_is_exit_regardless_of_current_weight(self):
        assert raw_weight("aggressive", "HOLD", "MEDIUM", current_weight=0.9, slot=0.19) == 0.0

    def test_sell_is_exit_regardless_of_confidence(self):
        assert raw_weight("aggressive", "SELL", "HIGH", current_weight=0.9, slot=0.19) == 0.0
        assert raw_weight("aggressive", "SELL", "LOW", current_weight=0.9, slot=0.19) == 0.0


class TestRawWeightConservative:
    """Conservative: HOLD keeps current weight; SELL trims 50% unless HIGH (full exit)."""

    def test_buy_high(self):
        assert raw_weight("conservative", "BUY", "HIGH", current_weight=0.0, slot=0.085) == pytest.approx(0.085)

    def test_buy_medium(self):
        assert raw_weight("conservative", "BUY", "MEDIUM", current_weight=0.0, slot=0.085) == pytest.approx(0.06375)

    def test_buy_low(self):
        assert raw_weight("conservative", "BUY", "LOW", current_weight=0.0, slot=0.085) == pytest.approx(0.034)

    def test_hold_keeps_current_weight(self):
        assert raw_weight("conservative", "HOLD", "MEDIUM", current_weight=0.12, slot=0.085) == pytest.approx(0.12)

    def test_sell_high_is_full_exit(self):
        assert raw_weight("conservative", "SELL", "HIGH", current_weight=0.12, slot=0.085) == 0.0

    def test_sell_medium_trims_half(self):
        assert raw_weight("conservative", "SELL", "MEDIUM", current_weight=0.12, slot=0.085) == pytest.approx(0.06)

    def test_sell_low_trims_half(self):
        assert raw_weight("conservative", "SELL", "LOW", current_weight=0.12, slot=0.085) == pytest.approx(0.06)


class TestNormalise:
    def test_scales_to_target_when_no_cap_fires(self):
        weights = {"A": 0.1, "B": 0.1}
        # Cap set high enough (0.6) that scaling to the target never triggers
        # the clamp/re-normalise branch -- isolates the plain scaling step.
        result = normalise(weights, cash_reserve=0.05, position_cap=0.6)
        assert result["A"] == pytest.approx(0.475)
        assert result["B"] == pytest.approx(0.475)

    def test_zero_total_returns_all_zero(self):
        weights = {"A": 0.0, "B": 0.0}
        result = normalise(weights, cash_reserve=0.05, position_cap=0.35)
        assert result == {"A": 0.0, "B": 0.0}

    def test_cap_clamps_and_renormalises(self):
        # Single dominant position: scaled weight would exceed the cap, so it's
        # clamped and re-normalised (re-normalising a single value against
        # itself is a no-op past the cap, so it settles exactly at the cap).
        weights = {"A": 0.38, "B": 0.0}
        result = normalise(weights, cash_reserve=0.05, position_cap=0.35)
        assert result["A"] == pytest.approx(0.35)
        assert result["B"] == 0.0


class TestRatingToSignal:
    def test_all_five_tiers_mapped(self):
        assert rating_to_signal("Buy") == {"signal": "BUY", "confidence": "HIGH"}
        assert rating_to_signal("Overweight") == {"signal": "BUY", "confidence": "MEDIUM"}
        assert rating_to_signal("Hold") == {"signal": "HOLD", "confidence": "MEDIUM"}
        assert rating_to_signal("Underweight") == {"signal": "SELL", "confidence": "MEDIUM"}
        assert rating_to_signal("Sell") == {"signal": "SELL", "confidence": "HIGH"}

    def test_table_covers_exactly_five_tiers(self):
        assert set(RATING_TO_SIGNAL) == {"Buy", "Overweight", "Hold", "Underweight", "Sell"}

    def test_unknown_rating_falls_back_to_hold_medium(self):
        assert rating_to_signal("???") == {"signal": "HOLD", "confidence": "MEDIUM"}


class TestComputeAllocationAggressive:
    def test_buy_high_and_hold_exit(self):
        universe = ["AAA", "BBB"]
        signals = {
            "AAA": {"signal": "BUY", "confidence": "HIGH"},
            "BBB": {"signal": "HOLD", "confidence": "MEDIUM"},
        }
        prices = {"AAA": 100.0, "BBB": 50.0}
        pre_equity = 10_000.0
        pre_positions = {"BBB": {"shares": 10, "price": 50.0, "market_value": 500.0}}

        allocation = compute_allocation(
            "aggressive", universe, signals, prices, pre_equity, pre_positions
        )

        # AAA's raw weight (2x slot = 0.38) dominates and hits the 35% cap.
        assert allocation["AAA"]["target_weight"] == pytest.approx(0.35)
        assert allocation["AAA"]["target_shares"] == 35
        assert allocation["AAA"]["current_shares"] == 0
        assert allocation["AAA"]["delta"] == 35

        # BBB is HOLD -> exit under the aggressive style, existing 10 shares sold.
        assert allocation["BBB"]["target_weight"] == 0.0
        assert allocation["BBB"]["target_shares"] == 0
        assert allocation["BBB"]["current_shares"] == 10
        assert allocation["BBB"]["delta"] == -10

    def test_min_trade_size_suppresses_tiny_deltas(self):
        universe = ["AAA"]
        signals = {"AAA": {"signal": "BUY", "confidence": "HIGH"}}
        prices = {"AAA": 100_000.0}  # huge price -> target_shares rounds to same as current
        pre_equity = 10_000.0
        pre_positions = {"AAA": {"shares": 0, "price": 100_000.0, "market_value": 0.0}}

        allocation = compute_allocation(
            "aggressive", universe, signals, prices, pre_equity, pre_positions
        )
        # target dollars ~3500, price 100000 -> target_shares = 0, current = 0 -> delta 0
        assert allocation["AAA"]["delta"] == 0

    def test_missing_price_is_skipped(self):
        universe = ["AAA", "BBB"]
        signals = {
            "AAA": {"signal": "BUY", "confidence": "HIGH"},
            "BBB": {"signal": "BUY", "confidence": "LOW"},
        }
        prices = {"AAA": 100.0}  # BBB has no price
        allocation = compute_allocation(
            "aggressive", universe, signals, prices, 10_000.0, {}
        )
        assert "BBB" not in allocation
        assert "AAA" in allocation


class TestComputeAllocationConservative:
    def test_hold_keeps_and_sell_trims(self):
        universe = ["CCC", "DDD"]
        signals = {
            "CCC": {"signal": "HOLD", "confidence": "MEDIUM"},
            "DDD": {"signal": "SELL", "confidence": "MEDIUM"},
        }
        prices = {"CCC": 20.0, "DDD": 30.0}
        pre_equity = 10_000.0
        pre_positions = {
            "CCC": {"shares": 100, "price": 20.0, "market_value": 2_000.0},
            "DDD": {"shares": 50, "price": 30.0, "market_value": 1_500.0},
        }

        allocation = compute_allocation(
            "conservative", universe, signals, prices, pre_equity, pre_positions
        )

        # Both raw weights (0.20 keep, 0.075 trim of 0.15) exceed the 15% cap once
        # scaled to the 85% target, so both clamp to the cap.
        assert allocation["CCC"]["target_weight"] == pytest.approx(0.15)
        assert allocation["DDD"]["target_weight"] == pytest.approx(0.15)
        # CCC's capped target (0.15 * 10000 / 20 = 75 shares) is below its current
        # 100 shares -> sell 25 (min trade size 2, so the delta survives).
        assert allocation["CCC"]["delta"] == -25

    def test_sell_high_is_full_exit(self):
        universe = ["EEE"]
        signals = {"EEE": {"signal": "SELL", "confidence": "HIGH"}}
        prices = {"EEE": 10.0}
        pre_equity = 10_000.0
        pre_positions = {"EEE": {"shares": 40, "price": 10.0, "market_value": 400.0}}

        allocation = compute_allocation(
            "conservative", universe, signals, prices, pre_equity, pre_positions
        )
        assert allocation["EEE"]["target_weight"] == 0.0
        assert allocation["EEE"]["target_shares"] == 0
        assert allocation["EEE"]["delta"] == -40


def test_style_params_match_reference_skill():
    assert STYLE_PARAMS["aggressive"] == {
        "max_positions": 5,
        "cash_reserve": 0.05,
        "slot": 0.19,
        "position_cap": 0.35,
        "min_trade_size": 1,
    }
    assert STYLE_PARAMS["conservative"] == {
        "max_positions": 10,
        "cash_reserve": 0.15,
        "slot": 0.085,
        "position_cap": 0.15,
        "min_trade_size": 2,
    }
