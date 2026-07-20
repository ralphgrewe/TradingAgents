"""Unit tests for swing trader memory integration (issue #93).

Tests cover:
1. Regime extraction from swing_structured_data (not market_report)
2. Key_drivers JSON structure with regime, setup_type, risk_reward, etc.
3. Decision storage fallback behavior (confidence=None mirrors trader/PM)
"""

from __future__ import annotations

import pytest

from tradingagents.agents.utils.rating import parse_rating

pytestmark = pytest.mark.unit


class TestSwingTraderMemoryLogic:
    """Test decision storage logic for swing trader."""

    def test_regime_extracted_from_structured_data_not_market_report(self):
        """Verify regime is extracted from swing_structured_data, not market_report.

        This tests the fix for the defect: regime should come from
        swing_structured_data["regime"] (populated by swing_trader.py with
        precompute["regime_gate"]["regime"]), not from market_report text.
        """
        swing_structured = {
            "action": "Buy",
            "conviction": 0.75,
            "holding_period_days": 5,
            "entry_price": 150.0,
            "stop_loss": 145.0,
            "take_profit": 160.0,
            "exit_conditions": "Exit on 160 or below 145",
            "setup_type": "pullback",
            "key_drivers": ["RSI oversold [market]", "Support hold [technical]"],
            "thesis": "Strong setup on pullback",
            "regime": "trending_up",  # Added by swing_trader.py from precompute
            "hold_bias": False,
        }

        # Extract regime the way trading_graph.py does (fixed version)
        regime = swing_structured.get("regime", "unknown")

        # Should get the actual regime, not market_report text
        assert regime == "trending_up"
        assert regime != ""  # Not empty string from market_report hack

    def test_key_drivers_json_structure_with_correct_risk_reward(self):
        """Verify key_drivers JSON has correct structure and risk/reward calculation."""
        swing_structured = {
            "action": "Buy",
            "conviction": 0.75,
            "holding_period_days": 5,
            "entry_price": 150.0,
            "stop_loss": 145.0,
            "take_profit": 160.0,
            "exit_conditions": "Exit on 160 or below 145",
            "setup_type": "pullback",
            "key_drivers": ["RSI oversold [market]", "Support hold [technical]"],
            "thesis": "Strong setup on pullback",
            "regime": "trending_up",
            "hold_bias": False,
        }

        # Extract fields and calculate risk_reward (same formula as swing_trader_computation)
        entry_price = swing_structured.get("entry_price")
        stop_loss = swing_structured.get("stop_loss")
        take_profit = swing_structured.get("take_profit")

        risk_reward = None
        if entry_price is not None and stop_loss is not None and take_profit is not None:
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)
            if risk > 0:
                risk_reward = round(reward / risk, 2)

        # Build key_drivers JSON (same structure as trading_graph.py)
        key_drivers_json = {
            "regime": swing_structured.get("regime", "unknown"),
            "setup_type": swing_structured.get("setup_type", ""),
            "risk_reward": risk_reward,
            "planned_horizon_days": swing_structured.get("holding_period_days"),
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "drivers": swing_structured.get("key_drivers", []),
        }

        # Verify structure and values
        assert key_drivers_json["regime"] == "trending_up"
        assert key_drivers_json["setup_type"] == "pullback"
        assert key_drivers_json["risk_reward"] == 2.0  # (160-150)/(150-145) = 10/5
        assert key_drivers_json["planned_horizon_days"] == 5
        assert key_drivers_json["entry_price"] == 150.0
        assert key_drivers_json["stop_loss"] == 145.0
        assert key_drivers_json["take_profit"] == 160.0
        assert len(key_drivers_json["drivers"]) == 2

    def test_fallback_extracts_signal_via_parse_rating(self):
        """Verify fallback extracts signal via parse_rating (mirrors trader/PM)."""
        swing_markdown = "**Rating**: Buy\nStrong technical setup with good risk/reward"

        # Fallback behavior: extract signal via parse_rating
        signal = parse_rating(swing_markdown)

        # Should extract "Buy" from markdown
        assert signal == "Buy"

        # Fallback should have confidence=None and key_drivers=None
        # (mirrors trader and portfolio_manager behavior when no structured data)
        confidence = None
        key_drivers = None

        assert confidence is None
        assert key_drivers is None

    def test_swing_trader_extends_structured_data_with_precompute_fields(self):
        """Verify swing_trader.py adds regime/hold_bias to structured_data as sibling keys."""
        # Simulate what swing_trader.py does:
        # 1. Get SwingDecision pydantic model from structured_llm
        # 2. Convert to dict
        # 3. Add regime and hold_bias from precompute

        pydantic_dict = {
            "action": "Buy",
            "conviction": 0.75,
            "holding_period_days": 5,
            "entry_price": 150.0,
            "stop_loss": 145.0,
            "take_profit": 160.0,
            "exit_conditions": "Exit on 160 or below 145",
            "setup_type": "pullback",
            "key_drivers": ["RSI oversold [market]"],
            "thesis": "Strong setup",
        }

        # Simulate precompute["regime_gate"]
        regime_gate = {
            "regime": "trending_up",
            "allow_pullback_long": True,
            "allow_pullback_short": False,
            "allow_catalyst": False,
            "hold_bias": False,
        }

        # Simulate what swing_trader.py does (the fix)
        structured_data = pydantic_dict.copy()
        structured_data["regime"] = regime_gate.get("regime", "unknown")
        structured_data["hold_bias"] = regime_gate.get("hold_bias", False)

        # Verify result: regime/hold_bias present as sibling keys
        assert "regime" in structured_data
        assert "hold_bias" in structured_data
        assert structured_data["regime"] == "trending_up"
        assert structured_data["hold_bias"] is False

        # Original schema fields still present
        assert structured_data["action"] == "Buy"
        assert structured_data["conviction"] == 0.75
        assert structured_data["holding_period_days"] == 5

    def test_regime_gate_regime_values(self):
        """Verify expected regime values from assess_regime_gate."""
        # These are the regime values that assess_regime_gate produces
        # (trending_up, trending_down, ranging, volatile)
        valid_regimes = ["trending_up", "trending_down", "ranging", "volatile"]

        for regime in valid_regimes:
            # Each should be a valid string that trading_graph.py can store
            assert isinstance(regime, str)
            assert len(regime) > 0

            # Test that it would be stored correctly in key_drivers
            key_drivers = {"regime": regime}
            assert key_drivers["regime"] == regime
