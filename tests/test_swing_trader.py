"""Tests for Swing Trader node and related components (issue #92, design-review fix-forward).

Tests cover:
1. SwingDecision schema validation (unchanged; design review verified correct)
2. Swing trader pre-compute module, exercised against the REAL #89 envelope
   shape (details.swing_indicators.*) built via
   market_indicators_computation.compute_indicators on synthetic OHLCV,
   rather than the fabricated `details.roc.20d` shape the original tests used.
3. The earnings-calendar (#90) and benchmark-ROC vendor calls, mocked.
4. Node-level behavior: holding-period clamp, structured/free-text fallback,
   and that the prompt carries the computed values + aggressiveness contract.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from tradingagents.agents.analysts.market_indicators_computation import (
    build_json_envelope,
    compute_indicators,
)
from tradingagents.agents.schemas import (
    SwingAction,
    SwingDecision,
    render_swing_decision,
)
from tradingagents.agents.trader.swing_trader import create_swing_trader
from tradingagents.agents.trader.swing_trader_computation import (
    assemble_swing_precompute,
    assess_regime_gate,
    check_earnings_in_window,
    compute_relative_strength,
    extract_trade_setup,
    fetch_benchmark_roc,
    fetch_earnings_calendar,
    resolve_benchmark_ticker,
)

pytestmark = pytest.mark.unit


# ── Synthetic OHLCV helpers (mirrors tests/test_market_indicators_computation.py) ──


def _make_ohlcv(n: int, trend: str = "up", start_price: float = 100.0) -> list[dict]:
    """Build n rows of synthetic OHLCV records in yfinance-like dict shape."""
    base_date = datetime(2024, 1, 1)
    records = []
    price = start_price
    step = 0.4 if trend == "up" else -0.4
    for i in range(n):
        date = base_date + timedelta(days=i)
        price += step + (0.1 if i % 3 == 0 else -0.05) * (1 if trend == "up" else -1)
        records.append(
            {
                "Date": date.strftime("%Y-%m-%d"),
                "Open": price - 0.5,
                "High": price + 0.75,
                "Low": price - 0.75,
                "Close": price,
                "Volume": 1_000_000 + (i * 1000),
            }
        )
    return records


def _envelope_from_records(records: list[dict], ticker: str = "TEST", date: str = "2024-06-01") -> dict:
    """Run the real compute_indicators pipeline and return the parsed envelope dict
    (the same shape state["market_report"] carries once json.loads'd)."""
    result = compute_indicators(records, ticker)
    envelope_json = build_json_envelope(
        signal=result["signal"],
        confidence=result["confidence"],
        summary=result["summary"],
        details=result["details"],
        ticker=ticker,
        date=date,
    )
    return json.loads(envelope_json)


# ── SwingDecision Schema Tests (unchanged; verified correct by design review) ──


class TestSwingDecisionSchema:
    """Tests for SwingDecision Pydantic schema."""

    def test_valid_buy_decision(self):
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
        with pytest.raises(ValidationError):
            SwingDecision(
                action=SwingAction.HOLD,
                conviction=1.5,
                holding_period_days=1,
                exit_conditions="No setup",
                setup_type="none",
                key_drivers=["Test [market]"],
                thesis="Test.",
            )
        with pytest.raises(ValidationError):
            SwingDecision(
                action=SwingAction.HOLD,
                conviction=-0.1,
                holding_period_days=1,
                exit_conditions="No setup",
                setup_type="none",
                key_drivers=["Test [market]"],
                thesis="Test.",
            )

    def test_negative_price_rejected(self):
        with pytest.raises(ValidationError):
            SwingDecision(
                action=SwingAction.BUY,
                conviction=0.75,
                holding_period_days=5,
                entry_price=-100.0,
                stop_loss=95.0,
                take_profit=110.0,
                exit_conditions="Time stop",
                setup_type="pullback",
                key_drivers=["Test [market]"],
                thesis="Test.",
            )

    def test_holding_period_minimum(self):
        with pytest.raises(ValidationError):
            SwingDecision(
                action=SwingAction.HOLD,
                conviction=0.50,
                holding_period_days=0,
                exit_conditions="No setup",
                setup_type="none",
                key_drivers=["Test [market]"],
                thesis="Test.",
            )


# ── Regime gate: all four branches pinned individually ─────────────────────


class TestRegimeGateAssessment:
    """Tests for assess_regime_gate(), pinning all four design branches distinctly."""

    def test_trending_up_allows_pullback_long_only(self):
        market_report = {"details": {"market_regime": "trending_up"}, "confidence": "HIGH"}
        result = assess_regime_gate(market_report)
        assert result["regime"] == "trending_up"
        assert result["allow_pullback_long"] is True
        assert result["allow_pullback_short"] is False
        assert result["allow_catalyst"] is True
        assert result["hold_bias"] is False
        assert result["confidence"] == pytest.approx(1.0)

    def test_trending_down_allows_pullback_short_not_long(self):
        """trending_down must be a distinct branch from ranging: it clears the
        SHORT pullback direction, which ranging never does."""
        market_report = {"details": {"market_regime": "trending_down"}, "confidence": "MEDIUM"}
        result = assess_regime_gate(market_report)
        assert result["regime"] == "trending_down"
        assert result["allow_pullback_long"] is False
        assert result["allow_pullback_short"] is True
        assert result["allow_catalyst"] is True
        assert result["hold_bias"] is False
        assert result["confidence"] == pytest.approx(0.6)

    def test_ranging_allows_catalyst_only(self):
        market_report = {"details": {"market_regime": "ranging"}, "confidence": "LOW"}
        result = assess_regime_gate(market_report)
        assert result["regime"] == "ranging"
        assert result["allow_pullback_long"] is False
        assert result["allow_pullback_short"] is False
        assert result["allow_catalyst"] is True
        assert result["hold_bias"] is False
        assert result["confidence"] == pytest.approx(0.3)

    def test_volatile_is_hold_biased_and_blocks_everything(self):
        market_report = {"details": {"market_regime": "volatile"}, "confidence": "LOW"}
        result = assess_regime_gate(market_report)
        assert result["regime"] == "volatile"
        assert result["allow_pullback_long"] is False
        assert result["allow_pullback_short"] is False
        assert result["allow_catalyst"] is False
        assert result["hold_bias"] is True

    def test_trending_down_and_ranging_are_distinguishable(self):
        down = assess_regime_gate({"details": {"market_regime": "trending_down"}})
        ranging = assess_regime_gate({"details": {"market_regime": "ranging"}})
        assert down["allow_pullback_short"] != ranging["allow_pullback_short"]

    def test_invalid_json(self):
        result = assess_regime_gate("not json")
        assert result["regime"] == "unknown"
        assert result["allow_pullback_long"] is False
        assert result["allow_pullback_short"] is False
        assert result["allow_catalyst"] is False
        assert result["hold_bias"] is False

    def test_none_input(self):
        result = assess_regime_gate(None)
        assert result["regime"] == "unknown"

    def test_unrecognized_regime_string(self):
        result = assess_regime_gate({"details": {"market_regime": "sideways_choppy"}})
        assert result["regime"] == "sideways_choppy"
        assert result["allow_pullback_long"] is False
        assert result["allow_pullback_short"] is False
        assert result["allow_catalyst"] is False
        assert result["hold_bias"] is False


# ── Trade setup extraction against the REAL envelope shape ──────────────────


class TestTradeSetupExtraction:
    """extract_trade_setup() against real compute_indicators() output."""

    def test_uptrend_uses_analyst_stop_loss(self):
        """A clear uptrend produces a BUY bias with the analyst's own
        ATR-anchored stop_loss already populated; no fallback needed."""
        envelope = _envelope_from_records(_make_ohlcv(100, trend="up"))
        result = extract_trade_setup(envelope)
        assert result is not None
        assert result["bias"] == "BUY"
        assert result["stop_loss"] is not None
        assert result["stop_loss_source"] == "trade_setup"
        assert result["take_profit"] is not None
        assert result["risk_reward"] is not None
        assert result["risk_reward"] > 0

    def test_downtrend_uses_analyst_stop_loss(self):
        envelope = _envelope_from_records(_make_ohlcv(100, trend="down"))
        result = extract_trade_setup(envelope)
        assert result is not None
        assert result["bias"] == "SELL"
        assert result["stop_loss"] is not None
        assert result["stop_loss_source"] == "trade_setup"

    def test_atr_fallback_when_trade_setup_stop_missing(self):
        """When trade_setup.stop_loss is None (e.g. HOLD bias at analyst
        compute time) but ATR + close are available, fall back to a direct
        ATR-based stop, not silently dropping the level."""
        details = {
            "close": 100.0,
            "trade_setup": {
                "bias": "BUY",
                "entry_trigger": "Pullback to VWMA",
                "stop_loss": None,
                "take_profit": 115.0,
            },
            "indicators": [{"indicator": "atr", "value": 2.0}],
            "swing_indicators": None,
        }
        result = extract_trade_setup({"details": details})
        assert result["stop_loss"] == round(100.0 - 1.5 * 2.0, 2)
        assert result["stop_loss_source"] == "atr_fallback"

    def test_atr_fallback_for_short_bias(self):
        details = {
            "close": 100.0,
            "trade_setup": {"bias": "SELL", "stop_loss": None, "take_profit": 85.0},
            "indicators": [{"indicator": "atr", "value": 2.0}],
            "swing_indicators": None,
        }
        result = extract_trade_setup({"details": details})
        assert result["stop_loss"] == round(100.0 + 1.5 * 2.0, 2)
        assert result["stop_loss_source"] == "atr_fallback"

    def test_swing_level_fallback_when_atr_also_missing(self):
        """When neither trade_setup.stop_loss nor ATR are available, fall
        back to the structural swing low (long) / swing high (short)."""
        details = {
            "close": 100.0,
            "trade_setup": {"bias": "BUY", "stop_loss": None, "take_profit": 115.0},
            "indicators": [{"indicator": "atr", "value": None}],
            "swing_indicators": {"swing_low": {"value": 92.5, "date": "2024-05-01", "bars_since": 3}},
        }
        result = extract_trade_setup({"details": details})
        assert result["stop_loss"] == 92.5
        assert result["stop_loss_source"] == "swing_level_fallback"

    def test_swing_level_fallback_short_uses_swing_high(self):
        details = {
            "close": 100.0,
            "trade_setup": {"bias": "SELL", "stop_loss": None, "take_profit": 85.0},
            "indicators": [],
            "swing_indicators": {"swing_high": {"value": 107.0, "date": "2024-05-01", "bars_since": 3}},
        }
        result = extract_trade_setup({"details": details})
        assert result["stop_loss"] == 107.0
        assert result["stop_loss_source"] == "swing_level_fallback"

    def test_no_fallback_available_stop_loss_is_none(self):
        details = {
            "close": 100.0,
            "trade_setup": {"bias": "BUY", "stop_loss": None, "take_profit": 115.0},
            "indicators": [],
            "swing_indicators": None,
        }
        result = extract_trade_setup({"details": details})
        assert result["stop_loss"] is None
        assert result["stop_loss_source"] is None

    def test_hold_bias_no_setup(self):
        details = {"close": 100.0, "trade_setup": {"bias": "HOLD"}, "indicators": [], "swing_indicators": None}
        result = extract_trade_setup({"details": details})
        assert result["bias"] == "HOLD"
        assert result["stop_loss"] is None

    def test_missing_trade_setup_defaults_to_hold(self):
        result = extract_trade_setup({"details": {}})
        assert result is not None
        assert result["bias"] == "HOLD"

    def test_invalid_json_returns_none(self):
        assert extract_trade_setup("not json") is None

    def test_none_input_returns_none(self):
        assert extract_trade_setup(None) is None


# ── Relative strength against the REAL swing_indicators.rate_of_change path ──


class TestRelativeStrengthComputation:
    def test_with_both_roc_values_from_real_envelope(self):
        envelope = _envelope_from_records(_make_ohlcv(100, trend="up"))
        stock_roc = envelope["details"]["swing_indicators"]["rate_of_change"]["roc_20d"]
        result = compute_relative_strength(envelope, benchmark_roc=2.0)
        assert result["roc_20d"] == stock_roc
        assert result["benchmark_roc"] == 2.0
        assert result["relative_strength"] == round(stock_roc - 2.0, 2)

    def test_with_missing_benchmark_roc(self):
        envelope = _envelope_from_records(_make_ohlcv(100, trend="up"))
        result = compute_relative_strength(envelope, benchmark_roc=None)
        assert result["roc_20d"] is not None
        assert result["benchmark_roc"] is None
        assert result["relative_strength"] is None

    def test_short_history_degrades_to_none(self):
        """Explicit short-history degradation test against the REAL envelope
        shape: with < 21 bars, swing_indicators.rate_of_change has no roc_20d
        key at all (compute_indicators only adds non-None entries), so this
        must degrade to None rather than KeyError or a fabricated 0."""
        envelope = _envelope_from_records(_make_ohlcv(10, trend="up"))
        # 10 bars is enough for roc_5d but not roc_20d (needs >= 21 bars) —
        # compute_indicators only adds non-None entries, so the roc_20d key
        # is simply absent rather than present-and-None.
        assert "roc_20d" not in (envelope["details"]["swing_indicators"] or {}).get("rate_of_change", {})
        result = compute_relative_strength(envelope, benchmark_roc=2.0)
        assert result["roc_20d"] is None
        assert result["relative_strength"] is None

    def test_invalid_json_degrades_to_none(self):
        result = compute_relative_strength("not json", benchmark_roc=2.0)
        assert result["roc_20d"] is None
        assert result["relative_strength"] is None


# ── Short-history degradation across the whole pre-compute stack ───────────


class TestShortHistoryDegradation:
    """A real, short-history envelope (< 21 bars) must degrade every
    pre-compute output gracefully — never raise, never fabricate a value."""

    def test_regime_gate_and_trade_setup_degrade_gracefully(self):
        envelope = _envelope_from_records(_make_ohlcv(3, trend="up"))
        details = envelope["details"]

        # With only 3 bars, every swing/momentum indicator is None and
        # swing_indicators collapses to None entirely.
        assert details["swing_indicators"] is None

        gate = assess_regime_gate(envelope)
        assert gate["regime"] in ("ranging", "trending_up", "trending_down", "volatile", "unknown")

        setup = extract_trade_setup(envelope)
        assert setup is not None
        # No ATR (insufficient window) and no swing levels: stop_loss must be
        # None rather than a crash or a fabricated number.
        if setup["bias"] in ("BUY", "SELL"):
            assert setup["stop_loss"] is None or isinstance(setup["stop_loss"], float)

        rs = compute_relative_strength(envelope, benchmark_roc=1.0)
        assert rs["roc_20d"] is None
        assert rs["relative_strength"] is None


# ── Earnings calendar text parsing + fetch (mocked vendor call) ─────────────


class TestEarningsCalendarFetchAndParse:
    def test_successful_calendar_parses_date_and_days(self):
        raw = (
            "# Earnings Calendar for AAPL\n"
            "# Current date: 2026-07-19\n\n"
            "Next Earnings Date: 2026-07-25\n"
            "Days Until Next Earnings: 6\n"
            "Most Recent Past Earnings Date: 2026-04-20"
        )
        with patch(
            "tradingagents.dataflows.interface.route_to_vendor", return_value=raw
        ) as mock_route:
            result = fetch_earnings_calendar("AAPL", "2026-07-19")
        mock_route.assert_called_once_with("get_earnings_calendar", "AAPL", "2026-07-19")
        assert result == {"status": "ok", "earnings_date": "2026-07-25", "days_to_earnings": 6}

    def test_no_data_available_degrades_to_unknown(self):
        with patch(
            "tradingagents.dataflows.interface.route_to_vendor",
            return_value="NO_DATA_AVAILABLE: no usable data",
        ):
            result = fetch_earnings_calendar("FAKE", "2026-07-19")
        assert result == {"status": "unknown", "earnings_date": None, "days_to_earnings": None}

    def test_non_equity_symbol_degrades_to_unknown(self):
        with patch(
            "tradingagents.dataflows.interface.route_to_vendor",
            return_value="NO_EARNINGS_CALENDAR_AVAILABLE: XAUUSD is a commodity",
        ):
            result = fetch_earnings_calendar("XAUUSD", "2026-07-19")
        assert result["status"] == "unknown"

    def test_vendor_error_string_degrades_to_unknown(self):
        with patch(
            "tradingagents.dataflows.interface.route_to_vendor",
            return_value="Error retrieving earnings calendar for AAPL: Network error",
        ):
            result = fetch_earnings_calendar("AAPL", "2026-07-19")
        assert result["status"] == "unknown"

    def test_raised_exception_never_crashes_never_propagates(self):
        """Network-shaped failures (a raised exception, not just a sentinel
        string) must degrade to unknown, never crash the node."""
        with patch(
            "tradingagents.dataflows.interface.route_to_vendor",
            side_effect=ConnectionError("network unreachable"),
        ):
            result = fetch_earnings_calendar("AAPL", "2026-07-19")
        assert result == {"status": "unknown", "earnings_date": None, "days_to_earnings": None}

    def test_no_scheduled_earnings_degrades_to_unknown(self):
        raw = (
            "# Earnings Calendar for AAPL\n\n"
            "Next Earnings Date: Not scheduled (or no upcoming dates known)\n"
            "Days Until Next Earnings: Unknown"
        )
        with patch("tradingagents.dataflows.interface.route_to_vendor", return_value=raw):
            result = fetch_earnings_calendar("AAPL", "2026-07-19")
        assert result["status"] == "unknown"


# ── Earnings-in-window check (trading-day window via numpy.busday_offset) ──


class TestEarningsInWindowCheck:
    def test_earnings_in_window_pullback_should_avoid(self):
        earnings = {"status": "ok", "earnings_date": "2026-07-24", "days_to_earnings": 3}
        result = check_earnings_in_window(
            earnings, trade_date="2026-07-19", holding_period_days=5, setup_type="pullback"
        )
        assert result["has_earnings_in_window"] is True
        assert result["should_avoid_non_catalyst"] is True

    def test_earnings_in_window_catalyst_not_blocked(self):
        earnings = {"status": "ok", "earnings_date": "2026-07-24", "days_to_earnings": 3}
        result = check_earnings_in_window(
            earnings, trade_date="2026-07-19", holding_period_days=5, setup_type="catalyst"
        )
        assert result["has_earnings_in_window"] is True
        assert result["should_avoid_non_catalyst"] is False

    def test_earnings_outside_window_not_flagged(self):
        earnings = {"status": "ok", "earnings_date": "2026-09-01", "days_to_earnings": 40}
        result = check_earnings_in_window(
            earnings, trade_date="2026-07-19", holding_period_days=5, setup_type="pullback"
        )
        assert result["has_earnings_in_window"] is False
        assert result["should_avoid_non_catalyst"] is False

    def test_unknown_status_degrades_safely(self):
        earnings = {"status": "unknown", "earnings_date": None, "days_to_earnings": None}
        result = check_earnings_in_window(
            earnings, trade_date="2026-07-19", holding_period_days=5, setup_type="pullback"
        )
        assert result["status"] == "unknown"
        assert result["has_earnings_in_window"] is False

    def test_none_earnings_info_degrades_safely(self):
        result = check_earnings_in_window(
            None, trade_date="2026-07-19", holding_period_days=5, setup_type="pullback"
        )
        assert result["status"] == "unknown"
        assert result["has_earnings_in_window"] is False


# ── Benchmark ROC (mocked vendor call) ──────────────────────────────────────


class TestFetchBenchmarkRoc:
    def _csv(self, closes: list[float]) -> str:
        header = f"# Stock data for SPY from 2026-06-01 to 2026-07-19\n# Total records: {len(closes)}\n\n"
        lines = ["Date,Open,High,Low,Close,Volume"]
        base = datetime(2026, 6, 1)
        for i, c in enumerate(closes):
            d = (base + timedelta(days=i)).strftime("%Y-%m-%d")
            lines.append(f"{d},{c},{c},{c},{c},1000000")
        return header + "\n".join(lines)

    def test_computes_20d_roc_from_csv(self):
        closes = [100.0] * 20 + [110.0]  # 21 rows: 20-day-ago close 100 -> now 110
        raw = self._csv(closes)
        with patch("tradingagents.dataflows.interface.route_to_vendor", return_value=raw):
            roc = fetch_benchmark_roc("SPY", "2026-07-19", lookback_days=20)
        assert roc == pytest.approx(10.0)

    def test_no_data_available_returns_none(self):
        with patch(
            "tradingagents.dataflows.interface.route_to_vendor",
            return_value="NO_DATA_AVAILABLE: no usable data",
        ):
            assert fetch_benchmark_roc("SPY", "2026-07-19") is None

    def test_raised_exception_returns_none(self):
        with patch(
            "tradingagents.dataflows.interface.route_to_vendor",
            side_effect=ConnectionError("network unreachable"),
        ):
            assert fetch_benchmark_roc("SPY", "2026-07-19") is None

    def test_insufficient_history_returns_none(self):
        raw = self._csv([100.0, 101.0, 102.0])
        with patch("tradingagents.dataflows.interface.route_to_vendor", return_value=raw):
            assert fetch_benchmark_roc("SPY", "2026-07-19", lookback_days=20) is None

    def test_invalid_trade_date_returns_none(self):
        assert fetch_benchmark_roc("SPY", "not-a-date") is None


class TestResolveBenchmarkTicker:
    def test_explicit_override_wins(self):
        config = {"benchmark_ticker": "QQQ", "benchmark_map": {"": "SPY"}}
        assert resolve_benchmark_ticker("AAPL", config) == "QQQ"

    def test_suffix_match(self):
        config = {"benchmark_ticker": None, "benchmark_map": {".T": "^N225", "": "SPY"}}
        assert resolve_benchmark_ticker("7203.T", config) == "^N225"

    def test_default_us_fallback(self):
        config = {"benchmark_ticker": None, "benchmark_map": {".T": "^N225", "": "SPY"}}
        assert resolve_benchmark_ticker("AAPL", config) == "SPY"

    def test_missing_benchmark_map_falls_back_to_spy(self):
        assert resolve_benchmark_ticker("AAPL", {}) == "SPY"


# ── assemble_swing_precompute integration (mocked vendor calls) ────────────


class TestAssembleSwingPrecompute:
    def test_full_precompute_wires_everything_together(self):
        envelope = _envelope_from_records(_make_ohlcv(100, trend="up"))

        with (
            patch(
                "tradingagents.agents.trader.swing_trader_computation.fetch_benchmark_roc",
                return_value=2.0,
            ) as mock_bench,
            patch(
                "tradingagents.agents.trader.swing_trader_computation.fetch_earnings_calendar",
                return_value={"status": "ok", "earnings_date": "2026-08-15", "days_to_earnings": 30},
            ) as mock_earn,
        ):
            result = assemble_swing_precompute(
                market_report=envelope,
                ticker="TEST",
                trade_date="2026-07-19",
                holding_period_days=5,
                benchmark="SPY",
            )

        mock_bench.assert_called_once_with("SPY", "2026-07-19")
        mock_earn.assert_called_once_with("TEST", "2026-07-19")

        assert "regime_gate" in result
        assert "trade_setup" in result
        assert "relative_strength" in result
        assert "earnings_check" in result
        assert result["regime_gate"]["regime"] == "trending_up"
        assert result["relative_strength"]["benchmark_roc"] == 2.0
        # 30 trading-ish days out is well outside a 5-day holding window.
        assert result["earnings_check"]["has_earnings_in_window"] is False

    def test_vendor_failures_degrade_the_whole_precompute_gracefully(self):
        envelope = _envelope_from_records(_make_ohlcv(100, trend="up"))
        with (
            patch(
                "tradingagents.agents.trader.swing_trader_computation.fetch_benchmark_roc",
                return_value=None,
            ),
            patch(
                "tradingagents.agents.trader.swing_trader_computation.fetch_earnings_calendar",
                return_value={"status": "unknown", "earnings_date": None, "days_to_earnings": None},
            ),
        ):
            result = assemble_swing_precompute(
                market_report=envelope,
                ticker="TEST",
                trade_date="2026-07-19",
            )
        assert result["relative_strength"]["relative_strength"] is None
        assert result["earnings_check"]["status"] == "unknown"


# ── Renderer Tests (unchanged; verified correct by design review) ──────────


class TestSwingDecisionRenderer:
    def test_render_buy_decision(self):
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


# ── Node-level tests (mocked LLM, mocked pre-compute assembly) ─────────────


def _make_swing_state(**overrides) -> dict:
    envelope = _envelope_from_records(_make_ohlcv(100, trend="up"))
    state = {
        "company_of_interest": "AAPL",
        "asset_type": "stock",
        "trade_date": "2026-07-19",
        "market_report": json.dumps(envelope),
        "sentiment_report": None,
        "news_report": None,
        "fundamentals_report": None,
        "investment_plan": "**Recommendation**: Buy\n**Rationale**: Strong technicals.",
        "swing_past_context": "",
    }
    state.update(overrides)
    return state


def _mock_precompute():
    return {
        "regime_gate": {
            "regime": "trending_up",
            "allow_pullback_long": True,
            "allow_pullback_short": False,
            "allow_catalyst": True,
            "hold_bias": False,
            "confidence": 1.0,
        },
        "trade_setup": {
            "bias": "BUY",
            "entry_trigger": "Pullback to VWMA (150.0)",
            "stop_loss": 145.0,
            "stop_loss_source": "trade_setup",
            "take_profit": 165.0,
            "risk_reward": 2.0,
            "close": 150.0,
        },
        "relative_strength": {"roc_20d": 8.0, "benchmark_roc": 2.0, "relative_strength": 6.0},
        "earnings_check": {
            "status": "ok",
            "has_earnings_in_window": False,
            "earnings_date": "2026-09-01",
            "should_avoid_non_catalyst": False,
        },
    }


@pytest.fixture(autouse=True)
def _patch_precompute():
    """Node-level tests exercise the node's own logic (clamp, fallback,
    prompt assembly) — the pre-compute internals are covered separately
    above, so replace the (network-touching) assembler with a fixed,
    already-tested-shape result."""
    with patch(
        "tradingagents.agents.trader.swing_trader.assemble_swing_precompute",
        return_value=_mock_precompute(),
    ) as mocked:
        yield mocked


class TestSwingTraderNode:
    """Node-level tests. The LLM is mocked directly (a MagicMock passed to
    create_swing_trader), the same pattern test_structured_agents.py uses for
    the Trader/Research Manager nodes — those also take an ``llm`` object
    directly rather than going through the ``create_llm_client`` factory that
    ``mock_llm_client`` patches, so that fixture isn't the applicable
    mocking point here."""

    def test_holding_period_clamped_at_call_site(self):
        proposal = SwingDecision(
            action=SwingAction.BUY,
            conviction=0.8,
            holding_period_days=30,  # exceeds default max of 15
            entry_price=150.0,
            stop_loss=145.0,
            take_profit=165.0,
            exit_conditions="Time stop",
            setup_type="pullback",
            key_drivers=["RSI oversold [market]"],
            thesis="Pullback in uptrend.",
        )
        structured = MagicMock()
        structured.invoke.return_value = proposal
        llm = MagicMock()
        llm.with_structured_output.return_value = structured

        node = create_swing_trader(llm)
        result = node(_make_swing_state())

        assert result["swing_structured_data"]["holding_period_days"] == 15
        assert "**Holding Period**: 15 trading days" in result["swing_trade_decision"]

    def test_holding_period_within_cap_is_unchanged(self):
        proposal = SwingDecision(
            action=SwingAction.BUY,
            conviction=0.8,
            holding_period_days=7,
            entry_price=150.0,
            stop_loss=145.0,
            take_profit=165.0,
            exit_conditions="Time stop",
            setup_type="pullback",
            key_drivers=["RSI oversold [market]"],
            thesis="Pullback in uptrend.",
        )
        structured = MagicMock()
        structured.invoke.return_value = proposal
        llm = MagicMock()
        llm.with_structured_output.return_value = structured

        node = create_swing_trader(llm)
        result = node(_make_swing_state())
        assert result["swing_structured_data"]["holding_period_days"] == 7

    def test_structured_path_writes_state_fields(self):
        proposal = SwingDecision(
            action=SwingAction.SELL,
            conviction=0.65,
            holding_period_days=4,
            entry_price=150.0,
            stop_loss=155.0,
            take_profit=138.0,
            exit_conditions="Time stop",
            setup_type="catalyst",
            key_drivers=["Earnings miss expected [news]"],
            thesis="Short the drift.",
        )
        structured = MagicMock()
        structured.invoke.return_value = proposal
        llm = MagicMock()
        llm.with_structured_output.return_value = structured

        node = create_swing_trader(llm)
        result = node(_make_swing_state())

        assert result["sender"] == "Swing Trader"
        assert "**Action**: Sell" in result["swing_trade_decision"]
        assert result["swing_structured_data"]["setup_type"] == "catalyst"
        assert result["swing_trade_decision"] in result["messages"][0].content

    def test_falls_back_to_freetext_when_structured_invocation_fails(self):
        plain_response = "**Action**: Hold\n\nNo qualifying setup today."
        llm = MagicMock()
        structured = MagicMock()
        structured.invoke.side_effect = RuntimeError("malformed structured response")
        llm.with_structured_output.return_value = structured
        llm.invoke.return_value = MagicMock(content=plain_response)

        node = create_swing_trader(llm)
        result = node(_make_swing_state())

        assert result["swing_trade_decision"] == plain_response
        assert "swing_structured_data" not in result

    def test_falls_back_to_freetext_when_structured_output_unsupported(self):
        plain_response = "**Action**: Buy\n\nPullback confirmed."
        llm = MagicMock()
        llm.with_structured_output.side_effect = NotImplementedError("provider unsupported")
        llm.invoke.return_value = MagicMock(content=plain_response)

        node = create_swing_trader(llm)
        result = node(_make_swing_state())

        assert result["swing_trade_decision"] == plain_response

    def test_prompt_contains_computed_values_and_contract(self):
        captured = {}
        structured = MagicMock()

        def _invoke(prompt):
            captured["prompt"] = prompt
            return SwingDecision(
                action=SwingAction.BUY,
                conviction=0.8,
                holding_period_days=5,
                entry_price=150.0,
                stop_loss=145.0,
                take_profit=165.0,
                exit_conditions="Time stop",
                setup_type="pullback",
                key_drivers=["RSI oversold [market]"],
                thesis="Pullback in uptrend.",
            )

        structured.invoke.side_effect = _invoke
        llm = MagicMock()
        llm.with_structured_output.return_value = structured

        node = create_swing_trader(llm)
        node(_make_swing_state())

        user_content = captured["prompt"][1]["content"]
        # Pre-computed values from the (mocked) precompute must be present.
        assert "trending_up" in user_content
        assert "pullback LONG allowed=True" in user_content
        assert "pullback SHORT allowed=False" in user_content
        assert "6.00%" in user_content  # relative strength
        assert '"stop_loss": 145.0' in user_content  # candidate trade setup json
        # Aggressiveness contract elements (design section 5).
        assert "HOLD only if" in user_content
        assert "must act" in user_content
        assert "Flip BUY" in user_content
        assert "Holding period: 1" in user_content

    def test_prompt_reflects_earnings_in_window_warning(self):
        with patch(
            "tradingagents.agents.trader.swing_trader.assemble_swing_precompute",
            return_value={
                **_mock_precompute(),
                "earnings_check": {
                    "status": "ok",
                    "has_earnings_in_window": True,
                    "earnings_date": "2026-07-22",
                    "should_avoid_non_catalyst": True,
                },
            },
        ):
            captured = {}
            structured = MagicMock()

            def _invoke(prompt):
                captured["prompt"] = prompt
                return SwingDecision(
                    action=SwingAction.HOLD,
                    conviction=0.4,
                    holding_period_days=1,
                    exit_conditions="Earnings in window; avoid non-catalyst entry.",
                    setup_type="none",
                    key_drivers=["Earnings in window [market]"],
                    thesis="Standing aside ahead of earnings.",
                )

            structured.invoke.side_effect = _invoke
            llm = MagicMock()
            llm.with_structured_output.return_value = structured

            node = create_swing_trader(llm)
            node(_make_swing_state())

        user_content = captured["prompt"][1]["content"]
        assert "avoid non-catalyst setups" in user_content


# ── Knowledge base (wiki tools) integration tests ───────────────────────────


class TestSwingTraderWithWikiTools:
    """Tests for knowledge_base_enabled gating and run_structured_with_tools integration."""

    def test_knowledge_base_enabled_invokes_run_structured_with_tools(self):
        """When knowledge_base_enabled=True, the swing trader uses run_structured_with_tools
        with [search_strategy_wiki] and max_rounds from config."""
        from tradingagents.dataflows.config import set_config

        proposal = SwingDecision(
            action=SwingAction.BUY,
            conviction=0.75,
            holding_period_days=5,
            entry_price=150.0,
            stop_loss=145.0,
            take_profit=165.0,
            exit_conditions="Time stop after 5 days",
            setup_type="pullback",
            key_drivers=["RSI oversold [market]"],
            thesis="Pullback entry in uptrend.",
        )

        # Mock run_structured_with_tools to capture its invocation
        with patch(
            "tradingagents.agents.trader.swing_trader.run_structured_with_tools"
        ) as mock_run_tools:
            mock_run_tools.return_value = (proposal, None, [])

            set_config({"knowledge_base_enabled": True, "knowledge_base_tool_max_rounds": 3})

            llm = MagicMock()
            llm.with_structured_output.return_value = MagicMock()
            node = create_swing_trader(llm)
            result = node(_make_swing_state())

        # Assert run_structured_with_tools was called with correct parameters
        mock_run_tools.assert_called_once()
        call_args = mock_run_tools.call_args
        assert call_args[0][0] is llm  # llm passed
        assert call_args[0][2][0].name == "search_strategy_wiki"  # tools list
        assert call_args[0][3] is SwingDecision  # response_model
        assert call_args[1]["max_rounds"] == 3  # max_rounds from config
        assert call_args[1]["agent_name"] == "Swing Trader"

        # Assert the result has structured data and correct rendering
        assert result["swing_structured_data"]["holding_period_days"] == 5
        assert "**Action**: Buy" in result["swing_trade_decision"]

    def test_knowledge_base_disabled_still_routes_through_shared_helper(self):
        """When knowledge_base_enabled=False, the swing trader still calls
        run_structured_with_tools -- with no tools and max_rounds=0 (issue #153).

        Before #153 this configuration took a hand-rolled
        ``structured_llm.invoke``/except path that bypassed the shared helper
        entirely, so the free-text fallback (#152) and the schema-repair retry
        (#153) never applied to it. The single structured call is unchanged; it
        just happens inside the shared helper now.
        """
        from tradingagents.dataflows.config import set_config

        proposal = SwingDecision(
            action=SwingAction.BUY,
            conviction=0.75,
            holding_period_days=5,
            entry_price=150.0,
            stop_loss=145.0,
            take_profit=165.0,
            exit_conditions="Time stop after 5 days",
            setup_type="pullback",
            key_drivers=["RSI oversold [market]"],
            thesis="Pullback entry in uptrend.",
        )

        set_config({"knowledge_base_enabled": False})

        structured = MagicMock()
        structured.invoke.return_value = proposal
        llm = MagicMock()
        llm.with_structured_output.return_value = structured
        llm.bind_tools = MagicMock(
            side_effect=AssertionError("bind_tools must not be called with no tools")
        )

        node = create_swing_trader(llm)
        result = node(_make_swing_state())

        # Exactly one structured call, made through the shared helper: no tools
        # bound (asserted by the bind_tools side effect above) and no tool loop.
        structured.invoke.assert_called_once()

        # Result should still have structured data and correct rendering
        assert result["swing_structured_data"]["holding_period_days"] == 5
        assert "**Action**: Buy" in result["swing_trade_decision"]

    def test_knowledge_base_disabled_passes_no_tools_and_zero_rounds(self):
        """The knowledge-base-off configuration must reach the shared helper with
        tools=[] and max_rounds=0, so no tool-loop behavior is introduced."""
        from tradingagents.dataflows.config import set_config

        proposal = SwingDecision(
            action=SwingAction.HOLD,
            conviction=0.4,
            holding_period_days=3,
            exit_conditions="No setup",
            setup_type="none",
            key_drivers=["Choppy tape [market]"],
            thesis="Stand aside.",
        )

        with patch(
            "tradingagents.agents.trader.swing_trader.run_structured_with_tools"
        ) as mock_run_tools:
            mock_run_tools.return_value = (proposal, None, [])

            set_config({"knowledge_base_enabled": False, "knowledge_base_tool_max_rounds": 3})

            llm = MagicMock()
            node = create_swing_trader(llm)
            node(_make_swing_state())

        mock_run_tools.assert_called_once()
        call_args = mock_run_tools.call_args
        assert call_args[0][2] == []  # no tools
        assert call_args[0][3] is SwingDecision
        assert call_args[1]["max_rounds"] == 0

    def test_retries_with_repair_instruction_when_kb_disabled(self):
        """Issue #153: the schema-repair retry fires with knowledge_base_enabled=False.

        That configuration used to take a hand-rolled ``structured_llm.invoke``
        path that bypassed ``run_structured_with_tools``, so the retry never ran
        there. The swing trader now always routes through the shared helper
        (no tools, max_rounds=0), so the retry applies in both configurations.
        """
        from tradingagents.dataflows.config import set_config

        proposal = SwingDecision(
            action=SwingAction.SELL,
            conviction=0.7,
            holding_period_days=4,
            entry_price=150.0,
            stop_loss=155.0,
            take_profit=138.0,
            exit_conditions="Time stop after 4 days",
            setup_type="catalyst",
            key_drivers=["Guidance cut [news]"],
            thesis="Fade the pop.",
        )

        structured_calls = []

        def _structured_invoke(messages):
            structured_calls.append(messages)
            if len(structured_calls) == 1:
                raise ValueError("Malformed JSON from a weak model")
            return proposal

        structured = MagicMock()
        structured.invoke = _structured_invoke

        llm = MagicMock()
        llm.with_structured_output.return_value = structured
        llm.bind_tools = MagicMock(
            side_effect=AssertionError("bind_tools must not be called with no tools")
        )
        llm.invoke = MagicMock(
            side_effect=AssertionError("free-text fallback must not be reached")
        )

        set_config({"knowledge_base_enabled": False})

        node = create_swing_trader(llm)
        result = node(_make_swing_state())

        # Exactly two structured attempts: the original and the repair retry.
        assert len(structured_calls) == 2
        # The retry appended a repair instruction naming the legal actions.
        repair_instruction = structured_calls[1][-1].content
        assert "Reply with ONLY valid JSON" in repair_instruction
        assert '"Sell"' in repair_instruction
        # The retry's result is what the node returned -- no free-text fallback.
        assert "**Action**: Sell" in result["swing_trade_decision"]
        assert result["swing_structured_data"]["setup_type"] == "catalyst"

    def test_knowledge_base_enabled_with_tool_call_then_structured_result(self):
        """When knowledge base is enabled and run_structured_with_tools executes a tool
        call and then returns a structured decision, the swing trader renders the decision
        correctly with all expected fields."""
        from tradingagents.dataflows.config import set_config

        proposal = SwingDecision(
            action=SwingAction.SELL,
            conviction=0.65,
            holding_period_days=4,
            entry_price=150.0,
            stop_loss=155.0,
            take_profit=138.0,
            exit_conditions="Time stop",
            setup_type="catalyst",
            key_drivers=["Earnings miss expected [news]"],
            thesis="Short the drift.",
        )

        # Simulate a message trace that run_structured_with_tools would return
        # (we don't need to construct actual tool_call messages; we just verify
        # the swing trader handles the returned proposal correctly)
        mock_trace = []

        with patch(
            "tradingagents.agents.trader.swing_trader.run_structured_with_tools"
        ) as mock_run_tools:
            mock_run_tools.return_value = (proposal, None, mock_trace)

            set_config({"knowledge_base_enabled": True})

            llm = MagicMock()
            llm.with_structured_output.return_value = MagicMock()
            node = create_swing_trader(llm)
            result = node(_make_swing_state())

        # Verify result structure
        assert "swing_structured_data" in result
        assert result["swing_structured_data"]["setup_type"] == "catalyst"
        assert "**Action**: Sell" in result["swing_trade_decision"]
        assert result["sender"] == "Swing Trader"
        # Verify that rendering worked correctly
        assert "**Conviction**: 0.65" in result["swing_trade_decision"]
        assert "**Holding Period**: 4 trading days" in result["swing_trade_decision"]

    def test_knowledge_base_enabled_with_fallback_text(self):
        """When run_structured_with_tools returns fallback_text (structured path failed),
        the swing trader uses the fallback and does not include swing_structured_data."""
        from tradingagents.dataflows.config import set_config

        fallback_decision = "**Action**: Hold\n\nNo qualifying setup today."
        mock_trace = [{"role": "system"}, {"role": "user"}]

        with patch(
            "tradingagents.agents.trader.swing_trader.run_structured_with_tools"
        ) as mock_run_tools:
            mock_run_tools.return_value = (None, fallback_decision, mock_trace)

            set_config({"knowledge_base_enabled": True})

            llm = MagicMock()
            llm.with_structured_output.return_value = MagicMock()
            node = create_swing_trader(llm)
            result = node(_make_swing_state())

        # Fallback path should be used
        assert result["swing_trade_decision"] == fallback_decision
        assert "swing_structured_data" not in result
        assert result["sender"] == "Swing Trader"

    def test_holding_period_clamped_with_knowledge_base_enabled(self):
        """Even when knowledge base is enabled and run_structured_with_tools is used,
        holding_period_days must still be clamped at the call site."""
        from tradingagents.dataflows.config import set_config

        proposal = SwingDecision(
            action=SwingAction.BUY,
            conviction=0.8,
            holding_period_days=30,  # exceeds default max of 15
            entry_price=150.0,
            stop_loss=145.0,
            take_profit=165.0,
            exit_conditions="Time stop",
            setup_type="pullback",
            key_drivers=["RSI oversold [market]"],
            thesis="Pullback in uptrend.",
        )

        with patch(
            "tradingagents.agents.trader.swing_trader.run_structured_with_tools"
        ) as mock_run_tools:
            mock_run_tools.return_value = (proposal, None, [])

            set_config({"knowledge_base_enabled": True, "swing_trader_max_holding_days": 15})

            llm = MagicMock()
            llm.with_structured_output.return_value = MagicMock()
            node = create_swing_trader(llm)
            result = node(_make_swing_state())

        # Verify clamping occurred
        assert result["swing_structured_data"]["holding_period_days"] == 15
        assert "**Holding Period**: 15 trading days" in result["swing_trade_decision"]

    def test_regime_and_hold_bias_injected_with_knowledge_base(self):
        """When run_structured_with_tools returns a structured result, the regime
        and hold_bias from precompute are still injected into swing_structured_data."""
        from tradingagents.dataflows.config import set_config

        proposal = SwingDecision(
            action=SwingAction.HOLD,
            conviction=0.50,
            holding_period_days=1,
            exit_conditions="Regime blocks entry",
            setup_type="none",
            key_drivers=["Volatile regime [market]"],
            thesis="No setup today.",
        )

        with patch(
            "tradingagents.agents.trader.swing_trader.run_structured_with_tools"
        ) as mock_run_tools:
            mock_run_tools.return_value = (proposal, None, [])

            with patch(
                "tradingagents.agents.trader.swing_trader.assemble_swing_precompute",
                return_value={
                    **_mock_precompute(),
                    "regime_gate": {
                        "regime": "volatile",
                        "hold_bias": True,
                        "allow_pullback_long": False,
                        "allow_pullback_short": False,
                        "allow_catalyst": False,
                    },
                },
            ):
                set_config({"knowledge_base_enabled": True})

                llm = MagicMock()
                llm.with_structured_output.return_value = MagicMock()
                node = create_swing_trader(llm)
                result = node(_make_swing_state())

        # Verify regime and hold_bias were injected
        assert result["swing_structured_data"]["regime"] == "volatile"
        assert result["swing_structured_data"]["hold_bias"] is True
