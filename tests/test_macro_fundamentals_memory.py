"""Integration-style tests for macro fundamentals analyst memory wiring (issue #132).

Drives the REAL `TradingAgentsGraph._run_graph` code path through the same
`FakeMemoryMCPClient` harness `test_memory_core_integration.py` uses for
research_manager/trader/portfolio_manager and `test_swing_trader_memory.py`
uses for swing_trader, so a real regression (gate always/never firing, wrong
field mapping, wrong truncation) actually fails these tests.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from tests.test_memory_core_integration import FakeMemoryMCPClient, _make_final_state
from tradingagents.graph.trading_graph import TradingAgentsGraph

pytestmark = pytest.mark.unit


def _macro_envelope(
    signal="BUY",
    macro_regime="RISK_ON",
    cons_conf=0.6,
    risky_conf=0.8,
    drivers=None,
    summary="Macro regime: RISK_ON — easing cycle underway.",
):
    return json.dumps({
        "skill": "macro-fundamentals-analyst",
        "ticker": "NVDA",
        "date": "2026-01-10",
        "signal": signal,
        "confidence": "HIGH",
        "summary": summary,
        "details": {
            "macro_regime": macro_regime,
            "regime_rationale": "easing cycle underway.",
            "drivers": drivers if drivers is not None else [
                {"indicator": "fed_funds_rate", "reading": "down"},
            ],
            "conservative": {"rating": signal, "confidence": cons_conf},
            "risky": {"rating": signal, "confidence": risky_conf},
        },
    })


def _mock_graph(tmp_path, final_state, fake_client, *, selected_analysts):
    mock_graph = MagicMock()
    mock_graph.config = {
        "results_dir": str(tmp_path),
        "checkpoint_enabled": False,
        "research_stage": "none",
        "swing_trader_enabled": False,
    }
    mock_graph.selected_analysts = selected_analysts
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


class TestMacroFundamentalsPastContextIntegration:
    """Read side: macro_past_context is fetched only when macro_fundamentals
    is in selected_analysts."""

    def test_get_past_context_called_when_selected(self, tmp_path):
        final_state = _make_final_state()
        final_state["macro_report"] = _macro_envelope()
        fake_client = FakeMemoryMCPClient()
        mock_graph = _mock_graph(
            tmp_path, final_state, fake_client, selected_analysts=["market", "macro_fundamentals"]
        )

        TradingAgentsGraph._run_graph(mock_graph, "NVDA", "2026-01-10", asset_type="stock")

        assert fake_client.get_past_context_calls == [
            {"agent": "macro_fundamentals", "ticker": "NVDA", "db_path": None}
        ]

    def test_get_past_context_not_called_when_not_selected(self, tmp_path):
        final_state = _make_final_state()
        fake_client = FakeMemoryMCPClient()
        mock_graph = _mock_graph(
            tmp_path, final_state, fake_client, selected_analysts=["market", "social", "news", "fundamentals"]
        )

        TradingAgentsGraph._run_graph(mock_graph, "NVDA", "2026-01-10", asset_type="stock")

        assert fake_client.get_past_context_calls == []


class TestMacroFundamentalsStoreDecisionIntegration:
    """Write side: store_decision is called only when selected, under the
    run's own ticker, with the field mapping specified in issue #132."""

    def test_not_called_when_not_selected(self, tmp_path):
        final_state = _make_final_state()
        fake_client = FakeMemoryMCPClient()
        mock_graph = _mock_graph(
            tmp_path, final_state, fake_client, selected_analysts=["market", "social", "news", "fundamentals"]
        )

        TradingAgentsGraph._run_graph(mock_graph, "NVDA", "2026-01-10", asset_type="stock")

        assert not any(c["agent"] == "macro_fundamentals" for c in fake_client.store_calls)

    def test_stored_under_run_ticker_with_field_mapping(self, tmp_path):
        final_state = _make_final_state()
        final_state["macro_report"] = _macro_envelope(
            signal="BUY", cons_conf=0.6, risky_conf=0.8,
            drivers=[{"indicator": "fed_funds_rate", "reading": "down"}],
            summary="Macro regime: RISK_ON — easing cycle underway.",
        )
        fake_client = FakeMemoryMCPClient()
        mock_graph = _mock_graph(
            tmp_path, final_state, fake_client, selected_analysts=["macro_fundamentals"]
        )

        TradingAgentsGraph._run_graph(mock_graph, "NVDA", "2026-01-10", asset_type="stock")

        macro_calls = [c for c in fake_client.store_calls if c["agent"] == "macro_fundamentals"]
        assert len(macro_calls) == 1
        call = macro_calls[0]

        assert call["ticker"] == "NVDA"  # the run's own ticker, not a synthetic one
        assert call["date"] == "2026-01-10"
        assert call["signal"] == "BUY"
        assert call["confidence"] == pytest.approx(0.7)  # mean(0.6, 0.8)
        assert call["key_drivers"] == [{"indicator": "fed_funds_rate", "reading": "down"}]
        assert call["thesis"] == "Macro regime: RISK_ON — easing cycle underway."

    def test_thesis_truncated_to_500_chars(self, tmp_path):
        final_state = _make_final_state()
        final_state["macro_report"] = _macro_envelope(summary="x" * 1000)
        fake_client = FakeMemoryMCPClient()
        mock_graph = _mock_graph(
            tmp_path, final_state, fake_client, selected_analysts=["macro_fundamentals"]
        )

        TradingAgentsGraph._run_graph(mock_graph, "NVDA", "2026-01-10", asset_type="stock")

        call = next(c for c in fake_client.store_calls if c["agent"] == "macro_fundamentals")
        assert len(call["thesis"]) == 500

    def test_fallback_when_envelope_unparseable(self, tmp_path):
        """When macro_report isn't valid JSON (shouldn't normally happen — the
        node always emits a valid envelope — but the storage code must degrade
        the same way trader/PM do rather than raising)."""
        final_state = _make_final_state()
        final_state["macro_report"] = "**Rating**: Sell\nNot a JSON envelope."
        fake_client = FakeMemoryMCPClient()
        mock_graph = _mock_graph(
            tmp_path, final_state, fake_client, selected_analysts=["macro_fundamentals"]
        )

        TradingAgentsGraph._run_graph(mock_graph, "NVDA", "2026-01-10", asset_type="stock")

        call = next(c for c in fake_client.store_calls if c["agent"] == "macro_fundamentals")
        assert call["signal"] == "Sell"
        assert call["confidence"] is None
        assert call["key_drivers"] is None

    def test_coexists_with_trader_and_portfolio_manager_calls(self, tmp_path):
        """macro_fundamentals's store_decision is additive — trader/PM calls
        still happen exactly as before."""
        final_state = _make_final_state()
        final_state["macro_report"] = _macro_envelope()
        fake_client = FakeMemoryMCPClient()
        mock_graph = _mock_graph(
            tmp_path, final_state, fake_client, selected_analysts=["macro_fundamentals"]
        )

        TradingAgentsGraph._run_graph(mock_graph, "NVDA", "2026-01-10", asset_type="stock")

        agents = [c["agent"] for c in fake_client.store_calls]
        assert agents == ["trader", "portfolio_manager", "macro_fundamentals"]
