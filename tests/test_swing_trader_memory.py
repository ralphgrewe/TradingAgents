"""Unit tests for swing trader memory integration (issue #93).

Tests cover:
1. Regime extraction from swing_structured_data (not market_report)
2. Key_drivers JSON structure with regime, setup_type, risk_reward, etc.
3. Decision storage fallback behavior (confidence=None mirrors trader/PM)
4. Integration-style coverage of the REAL `trading_graph.py`/`swing_trader.py`
   code path (design-review follow-up, see issue #93 comment): the classes
   below in section 4 drive `TradingAgentsGraph._run_graph` through the same
   `FakeMemoryMCPClient` harness `test_memory_core_integration.py` uses for
   research_manager/trader/portfolio_manager, so a real regression (typo'd
   dict key, dropped `regime` field, broken risk/reward formula, wrong
   fallback value) would actually fail these tests — unlike section 1-3's
   tests, which only assert hand-copied logic against itself.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tests.test_memory_core_integration import FakeMemoryMCPClient, _make_final_state
from tradingagents.agents.utils.rating import parse_rating
from tradingagents.graph.trading_graph import TradingAgentsGraph

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


# ---------------------------------------------------------------------------
# Section 4: integration-style tests through the REAL _run_graph code path.
# ---------------------------------------------------------------------------

def _mock_swing_graph(tmp_path, final_state, fake_client, *, swing_trader_enabled=True):
    """Build a MagicMock TradingAgentsGraph, same shape as
    test_memory_core_integration.py's ``_mock_graph`` helpers, with
    ``swing_trader_enabled`` toggled in config."""
    mock_graph = MagicMock()
    mock_graph.config = {
        "results_dir": str(tmp_path),
        "checkpoint_enabled": False,
        "research_stage": "none",
        "swing_trader_enabled": swing_trader_enabled,
    }
    mock_graph.memory_log = MagicMock()
    mock_graph.log_states_dict = {}
    mock_graph.debug = False
    mock_graph.signal_processor = MagicMock()
    mock_graph.signal_processor.process_signal.return_value = "Buy"
    mock_graph.propagator = MagicMock()
    mock_graph.propagator.create_initial_state.return_value = final_state
    mock_graph.propagator.get_graph_args.return_value = {}
    mock_graph.graph = MagicMock()
    mock_graph.graph.invoke.return_value = final_state
    mock_graph._memory_client = fake_client
    return mock_graph


def _make_swing_final_state(
    *,
    structured=True,
    action="Buy",
    conviction=0.75,
    thesis="Strong pullback setup",
    holding_period_days=5,
    entry_price=150.0,
    stop_loss=145.0,
    take_profit=160.0,
    setup_type="pullback",
    regime="trending_up",
    hold_bias=False,
    drivers=None,
):
    """Build a final_state dict with swing_structured_data / swing_trade_decision
    populated the way the real swing_trader.py node populates them (or, when
    ``structured=False``, only the free-text fallback field, mirroring the
    parseable-fallback path)."""
    final_state = _make_final_state()
    markdown = f"**Rating**: {action}\n{thesis}"
    final_state["swing_trade_decision"] = markdown
    if structured:
        final_state["swing_structured_data"] = {
            "action": action,
            "conviction": conviction,
            "holding_period_days": holding_period_days,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "exit_conditions": "Exit at target or below stop",
            "setup_type": setup_type,
            "key_drivers": drivers if drivers is not None else [
                "RSI oversold [market]", "Support hold [technical]",
            ],
            "thesis": thesis,
            # Sibling keys added by swing_trader.py from precompute["regime_gate"].
            "regime": regime,
            "hold_bias": hold_bias,
        }
    else:
        final_state["swing_structured_data"] = None
    return final_state


class TestSwingTraderPastContextIntegration:
    """Read side: swing_past_context is fetched via the real MemoryMCPClient
    call in trading_graph.py's _run_graph, only when swing_trader_enabled."""

    def test_get_past_context_called_when_enabled(self, tmp_path):
        final_state = _make_swing_final_state()
        fake_client = FakeMemoryMCPClient()
        mock_graph = _mock_swing_graph(tmp_path, final_state, fake_client, swing_trader_enabled=True)

        TradingAgentsGraph._run_graph(mock_graph, "NVDA", "2026-01-10", asset_type="stock")

        assert fake_client.get_past_context_calls == [
            {"agent": "swing_trader", "ticker": "NVDA", "db_path": None}
        ]

    def test_get_past_context_not_called_when_disabled(self, tmp_path):
        final_state = _make_swing_final_state()
        fake_client = FakeMemoryMCPClient()
        mock_graph = _mock_swing_graph(tmp_path, final_state, fake_client, swing_trader_enabled=False)

        TradingAgentsGraph._run_graph(mock_graph, "NVDA", "2026-01-10", asset_type="stock")

        assert fake_client.get_past_context_calls == []


class TestSwingTraderStoreDecisionIntegration:
    """Write side: store_decision is called through the real _run_graph code
    path with the exact field mapping specified in issue #93."""

    def test_not_called_when_swing_disabled(self, tmp_path):
        final_state = _make_swing_final_state()
        fake_client = FakeMemoryMCPClient()
        mock_graph = _mock_swing_graph(tmp_path, final_state, fake_client, swing_trader_enabled=False)

        TradingAgentsGraph._run_graph(mock_graph, "NVDA", "2026-01-10", asset_type="stock")

        assert not any(c["agent"] == "swing_trader" for c in fake_client.store_calls)

    def test_structured_field_mapping(self, tmp_path):
        """Structured-data-present case: exact field mapping from the design,
        including the regime-sourcing fix and the risk_reward formula."""
        final_state = _make_swing_final_state(
            action="Buy",
            conviction=0.75,
            thesis="Strong pullback setup",
            holding_period_days=5,
            entry_price=150.0,
            stop_loss=145.0,
            take_profit=160.0,
            setup_type="pullback",
            regime="trending_up",
            drivers=["RSI oversold [market]", "Support hold [technical]"],
        )
        fake_client = FakeMemoryMCPClient()
        mock_graph = _mock_swing_graph(tmp_path, final_state, fake_client, swing_trader_enabled=True)

        TradingAgentsGraph._run_graph(mock_graph, "NVDA", "2026-01-10", asset_type="stock")

        swing_calls = [c for c in fake_client.store_calls if c["agent"] == "swing_trader"]
        assert len(swing_calls) == 1
        call = swing_calls[0]

        assert call["ticker"] == "NVDA"
        assert call["date"] == "2026-01-10"
        assert call["signal"] == "Buy"
        assert call["confidence"] == 0.75  # numeric conviction, never None here
        assert call["thesis"] == "Strong pullback setup"
        assert call["horizon_days"] == 5  # #91's parameter, from holding_period_days

        key_drivers = call["key_drivers"]
        assert key_drivers == {
            "regime": "trending_up",
            "setup_type": "pullback",
            "risk_reward": 2.0,  # (160-150)/(150-145) = 10/5
            "planned_horizon_days": 5,
            "entry_price": 150.0,
            "stop_loss": 145.0,
            "take_profit": 160.0,
            "drivers": ["RSI oversold [market]", "Support hold [technical]"],
        }

    def test_thesis_truncated_to_500_chars(self, tmp_path):
        long_thesis = "x" * 1000
        final_state = _make_swing_final_state(thesis=long_thesis)
        fake_client = FakeMemoryMCPClient()
        mock_graph = _mock_swing_graph(tmp_path, final_state, fake_client, swing_trader_enabled=True)

        TradingAgentsGraph._run_graph(mock_graph, "NVDA", "2026-01-10", asset_type="stock")

        call = next(c for c in fake_client.store_calls if c["agent"] == "swing_trader")
        assert len(call["thesis"]) == 500

    def test_fallback_when_no_structured_data(self, tmp_path):
        """When swing_structured_data is None (unparseable structured output),
        store_decision degrades the same way trader/portfolio_manager do:
        signal parsed from markdown via parse_rating, confidence=None,
        key_drivers=None."""
        final_state = _make_swing_final_state(
            structured=False, action="Sell", thesis="Fallback thesis text",
        )
        fake_client = FakeMemoryMCPClient()
        mock_graph = _mock_swing_graph(tmp_path, final_state, fake_client, swing_trader_enabled=True)

        TradingAgentsGraph._run_graph(mock_graph, "NVDA", "2026-01-10", asset_type="stock")

        swing_calls = [c for c in fake_client.store_calls if c["agent"] == "swing_trader"]
        assert len(swing_calls) == 1
        call = swing_calls[0]

        assert call["confidence"] is None
        assert call["key_drivers"] is None
        assert call["horizon_days"] is None
        assert call["signal"] == parse_rating(final_state["swing_trade_decision"])
        assert call["signal"] == "Sell"
        assert call["thesis"] == final_state["swing_trade_decision"][:500]
