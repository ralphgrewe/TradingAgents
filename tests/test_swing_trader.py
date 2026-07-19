"""Tests for Swing Trader node and related components (issue #92).

Tests cover:
1. SwingDecision schema validation
2. Swing trader pre-compute module
3. Graph wiring with swing_trader_enabled config
"""

from __future__ import annotations

import json
import pytest
from pydantic import ValidationError

from tradingagents.agents.schemas import (
    SwingAction,
    SwingDecision,
    render_swing_decision,
)
from tradingagents.agents.trader.swing_trader_computation import (
    assess_regime_gate,
    assemble_swing_precompute,
    check_earnings_in_window,
    compute_relative_strength,
    extract_trade_setup,
)

pytestmark = pytest.mark.unit


# ── SwingDecision Schema Tests ───────────────────────────────────────────────


class TestSwingDecisionSchema:
    """Tests for SwingDecision Pydantic schema."""

    def test_valid_buy_decision(self):
        """Test creating a valid BUY decision with all required fields."""
        decision = SwingDecision(
            action=SwingAction.BUY,
            conviction=0.75,
            holding_period_days=5,
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=110.0,
            exit_conditions="Time stop after 5 days or if RSI drops below 30",
            setup_type="pullback",
            key_drivers=[
                "Price above 20-day MA [market]",
                "RSI oversold at 25 [market]",
            ],
            thesis="Pullback entry in uptrend. Expect mean reversion to 20-day MA.",
        )
        assert decision.action == SwingAction.BUY
        assert decision.conviction == 0.75
        assert decision.entry_price == 100.0

    def test_valid_sell_decision(self):
        """Test creating a valid SELL decision."""
        decision = SwingDecision(
            action=SwingAction.SELL,
            conviction=0.60,
            holding_period_days=3,
            entry_price=50.0,
            stop_loss=52.0,
            take_profit=45.0,
            exit_conditions="Stop at 52 or after 3 days",
            setup_type="pullback",
            key_drivers=["Bearish divergence on daily [market]"],
            thesis="Short setup in downtrend.",
        )
        assert decision.action == SwingAction.SELL

    def test_valid_hold_decision(self):
        """Test creating a valid HOLD decision without price fields."""
        decision = SwingDecision(
            action=SwingAction.HOLD,
            conviction=0.50,
            holding_period_days=1,
            entry_price=None,
            stop_loss=None,
            take_profit=None,
            exit_conditions="Regime volatility blocks pullback entries; catalyst setup unlikely.",
            setup_type="none",
            key_drivers=["Market regime volatile [market]"],
            thesis="No setup meets R/R threshold today.",
        )
        assert decision.action == SwingAction.HOLD
        assert decision.entry_price is None

    def test_buy_without_entry_price_rejected(self):
        """Test that BUY without entry_price is rejected."""
        with pytest.raises(ValidationError):
            SwingDecision(
                action=SwingAction.BUY,
                conviction=0.75,
                holding_period_days=5,
                entry_price=None,
                stop_loss=95.0,
                take_profit=110.0,
                exit_conditions="Time stop",
                setup_type="pullback",
                key_drivers=["Test [market]"],
                thesis="Test.",
            )

    def test_sell_without_stop_loss_rejected(self):
        """Test that SELL without stop_loss is rejected."""
        with pytest.raises(ValidationError):
            SwingDecision(
                action=SwingAction.SELL,
                conviction=0.60,
                holding_period_days=3,
                entry_price=50.0,
                stop_loss=None,
                take_profit=45.0,
                exit_conditions="Stop at target",
                setup_type="pullback",
                key_drivers=["Test [market]"],
                thesis="Test.",
            )

    def test_buy_without_take_profit_rejected(self):
        """Test that BUY without take_profit is rejected."""
        with pytest.raises(ValidationError):
            SwingDecision(
                action=SwingAction.BUY,
                conviction=0.75,
                holding_period_days=5,
                entry_price=100.0,
                stop_loss=95.0,
                take_profit=None,
                exit_conditions="Time stop",
                setup_type="pullback",
                key_drivers=["Test [market]"],
                thesis="Test.",
            )

    def test_conviction_range_validation(self):
        """Test that conviction must be between 0.0 and 1.0."""
        with pytest.raises(ValidationError):
            SwingDecision(
                action=SwingAction.HOLD,
                conviction=1.5,  # > 1.0
                holding_period_days=1,
                exit_conditions="No setup",
                setup_type="none",
                key_drivers=["Test [market]"],
                thesis="Test.",
            )

        with pytest.raises(ValidationError):
            SwingDecision(
                action=SwingAction.HOLD,
                conviction=-0.1,  # < 0.0
                holding_period_days=1,
                exit_conditions="No setup",
                setup_type="none",
                key_drivers=["Test [market]"],
                thesis="Test.",
            )

    def test_negative_price_rejected(self):
        """Test that negative prices are rejected."""
        with pytest.raises(ValidationError):
            SwingDecision(
                action=SwingAction.BUY,
                conviction=0.75,
                holding_period_days=5,
                entry_price=-100.0,  # negative
                stop_loss=95.0,
                take_profit=110.0,
                exit_conditions="Time stop",
                setup_type="pullback",
                key_drivers=["Test [market]"],
                thesis="Test.",
            )

    def test_holding_period_minimum(self):
        """Test that holding_period_days must be >= 1."""
        with pytest.raises(ValidationError):
            SwingDecision(
                action=SwingAction.HOLD,
                conviction=0.50,
                holding_period_days=0,  # < 1
                exit_conditions="No setup",
                setup_type="none",
                key_drivers=["Test [market]"],
                thesis="Test.",
            )


# ── Swing Trader Computation Tests ──────────────────────────────────────────


class TestRegimeGateAssessment:
    """Tests for assess_regime_gate() function."""

    def test_trending_up_regime(self):
        """Test that trending_up regime allows pullback entries."""
        market_report = {
            "details": {"market_regime": "trending_up"},
            "confidence": "0.85",
        }
        result = assess_regime_gate(market_report)
        assert result["regime"] == "trending_up"
        assert result["regime_allows_pullback"] is True
        assert result["regime_allows_catalyst"] is True

    def test_trending_down_regime(self):
        """Test that trending_down regime blocks pullback, allows catalyst."""
        market_report = {
            "details": {"market_regime": "trending_down"},
            "confidence": "0.80",
        }
        result = assess_regime_gate(market_report)
        assert result["regime"] == "trending_down"
        assert result["regime_allows_pullback"] is False
        assert result["regime_allows_catalyst"] is True

    def test_ranging_regime(self):
        """Test that ranging regime blocks pullback, allows catalyst."""
        market_report = {
            "details": {"market_regime": "ranging"},
            "confidence": "0.65",
        }
        result = assess_regime_gate(market_report)
        assert result["regime"] == "ranging"
        assert result["regime_allows_pullback"] is False
        assert result["regime_allows_catalyst"] is True

    def test_volatile_regime(self):
        """Test that volatile regime blocks both."""
        market_report = {
            "details": {"market_regime": "volatile"},
            "confidence": "0.50",
        }
        result = assess_regime_gate(market_report)
        assert result["regime"] == "volatile"
        assert result["regime_allows_pullback"] is False
        assert result["regime_allows_catalyst"] is False

    def test_invalid_json(self):
        """Test that invalid JSON returns unknown regime."""
        result = assess_regime_gate("not json")
        assert result["regime"] == "unknown"
        assert result["regime_allows_pullback"] is False
        assert result["regime_allows_catalyst"] is False

    def test_none_input(self):
        """Test that None input returns unknown regime."""
        result = assess_regime_gate(None)
        assert result["regime"] == "unknown"


class TestTradeSetupExtraction:
    """Tests for extract_trade_setup() function."""

    def test_extract_valid_trade_setup(self):
        """Test extracting trade_setup from market report."""
        market_report = {
            "details": {
                "trade_setup": {
                    "bias": "bullish",
                    "entry_trigger": 100.5,
                    "stop_loss": 95.0,
                    "take_profit": 110.0,
                    "risk_reward": 2.0,
                }
            }
        }
        result = extract_trade_setup(market_report)
        assert result is not None
        assert result["bias"] == "bullish"
        assert result["entry_trigger"] == 100.5

    def test_missing_trade_setup(self):
        """Test that missing trade_setup returns None."""
        market_report = {"details": {}}
        result = extract_trade_setup(market_report)
        assert result is None

    def test_invalid_json(self):
        """Test that invalid JSON returns None."""
        result = extract_trade_setup("not json")
        assert result is None


class TestRelativeStrengthComputation:
    """Tests for compute_relative_strength() function."""

    def test_with_both_roc_values(self):
        """Test RS calculation when both stock and benchmark ROC are available."""
        market_report = {
            "details": {
                "roc": {"20d": 5.0}
            }
        }
        result = compute_relative_strength(market_report, benchmark_roc=2.0)
        assert result["roc_20d"] == 5.0
        assert result["benchmark_roc"] == 2.0
        assert result["relative_strength"] == 3.0  # 5.0 - 2.0

    def test_with_missing_stock_roc(self):
        """Test RS calculation when stock ROC is missing."""
        market_report = {"details": {"roc": {}}}
        result = compute_relative_strength(market_report, benchmark_roc=2.0)
        assert result["roc_20d"] is None
        assert result["relative_strength"] is None

    def test_with_missing_benchmark_roc(self):
        """Test RS calculation when benchmark ROC is None."""
        market_report = {
            "details": {
                "roc": {"20d": 5.0}
            }
        }
        result = compute_relative_strength(market_report, benchmark_roc=None)
        assert result["roc_20d"] == 5.0
        assert result["benchmark_roc"] is None
        assert result["relative_strength"] is None


class TestEarningsInWindowCheck:
    """Tests for check_earnings_in_window() function."""

    def test_earnings_in_window_pullback(self):
        """Test that earnings in window is detected for pullback setup."""
        earnings = {
            "earnings_date": "2026-07-25",
            "days_to_earnings": 3,
        }
        result = check_earnings_in_window(earnings, holding_period_days=5, setup_type="pullback")
        assert result["has_earnings_in_window"] is True
        assert result["should_avoid_non_catalyst"] is True

    def test_earnings_in_window_catalyst(self):
        """Test that catalyst setup is not blocked by earnings in window."""
        earnings = {
            "earnings_date": "2026-07-25",
            "days_to_earnings": 3,
        }
        result = check_earnings_in_window(earnings, holding_period_days=5, setup_type="catalyst")
        assert result["has_earnings_in_window"] is True
        assert result["should_avoid_non_catalyst"] is False

    def test_earnings_outside_window(self):
        """Test that earnings outside window is not flagged."""
        earnings = {
            "earnings_date": "2026-08-10",
            "days_to_earnings": 20,
        }
        result = check_earnings_in_window(earnings, holding_period_days=5, setup_type="pullback")
        assert result["has_earnings_in_window"] is False
        assert result["should_avoid_non_catalyst"] is False

    def test_none_earnings(self):
        """Test with None earnings_calendar."""
        result = check_earnings_in_window(None, holding_period_days=5, setup_type="pullback")
        assert result["has_earnings_in_window"] is False


class TestAssembleSwingPrecompute:
    """Tests for assemble_swing_precompute() function."""

    def test_full_precompute(self):
        """Test assembling all pre-compute inputs."""
        market_report = {
            "details": {
                "market_regime": "trending_up",
                "roc": {"20d": 5.0},
                "trade_setup": {
                    "bias": "bullish",
                    "entry_trigger": 100.5,
                    "stop_loss": 95.0,
                    "take_profit": 110.0,
                    "risk_reward": 2.0,
                }
            },
            "confidence": "0.85",
        }
        earnings = {
            "earnings_date": "2026-07-25",
            "days_to_earnings": 10,
        }
        result = assemble_swing_precompute(
            market_report=market_report,
            earnings_calendar=earnings,
            holding_period_days=5,
            benchmark_roc=2.0,
        )

        assert "regime_gate" in result
        assert "trade_setup" in result
        assert "relative_strength" in result
        assert "earnings_check" in result

        assert result["regime_gate"]["regime"] == "trending_up"
        assert result["relative_strength"]["relative_strength"] == 3.0


# ── Renderer Tests ──────────────────────────────────────────────────────────


class TestSwingDecisionRenderer:
    """Tests for render_swing_decision() function."""

    def test_render_buy_decision(self):
        """Test rendering a BUY decision to markdown."""
        decision = SwingDecision(
            action=SwingAction.BUY,
            conviction=0.75,
            holding_period_days=5,
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=110.0,
            exit_conditions="Time stop after 5 days",
            setup_type="pullback",
            key_drivers=["Price above 20-day MA [market]"],
            thesis="Pullback entry in uptrend.",
        )
        rendered = render_swing_decision(decision)
        assert "**Action**: Buy" in rendered
        assert "**Conviction**: 0.75" in rendered
        assert "**Entry Price**: 100.0" in rendered
        assert "**Stop Loss**: 95.0" in rendered
        assert "**Take Profit**: 110.0" in rendered
        assert "**Setup Type**: pullback" in rendered

    def test_render_hold_decision(self):
        """Test rendering a HOLD decision (no price fields)."""
        decision = SwingDecision(
            action=SwingAction.HOLD,
            conviction=0.50,
            holding_period_days=1,
            entry_price=None,
            stop_loss=None,
            take_profit=None,
            exit_conditions="Regime blocks entry",
            setup_type="none",
            key_drivers=["Volatile regime [market]"],
            thesis="No setup today.",
        )
        rendered = render_swing_decision(decision)
        assert "**Action**: Hold" in rendered
        assert "**Entry Price**:" not in rendered
        assert "**Stop Loss**:" not in rendered
        assert "**Take Profit**:" not in rendered
        assert "**Setup Type**: none" in rendered
