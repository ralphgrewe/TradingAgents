"""
Deterministic pre-computation module for the Swing Trader node.

Pure-Python step (no LLM) that evaluates the regime gate, assembles candidate
setup levels, computes relative strength, and flags earnings within the holding
window. All numeric derivation happens here; the LLM sees computed values in
the prompt and judges/weighs them (LEARNINGS.md rule).

This module mirrors the pattern of market_indicators_computation.py but scoped
to swing-trader-specific setup assembly and regime gating.
"""

import json
from typing import Any


def assess_regime_gate(market_report: str | dict) -> dict[str, Any]:
    """
    Evaluate the regime gate from the market analyst report.

    Gate logic (from design section 2):
    - trending_up → pullback longs OK
    - trending_down → stand aside or pullback shorts
    - ranging → catalyst setups only
    - volatile → HOLD bias / reduced conviction

    Args:
        market_report: JSON string or parsed dict from market analyst

    Returns:
        dict with keys:
        - regime: str, one of "trending_up", "trending_down", "ranging", "volatile", or "unknown"
        - regime_allows_pullback: bool, True if regime permits pullback entry
        - regime_allows_catalyst: bool, True if regime permits catalyst entry
        - confidence: float 0.0-1.0, regime confidence from the report
    """
    try:
        envelope = json.loads(market_report) if isinstance(market_report, str) else market_report

        if not isinstance(envelope, dict):
            return {
                "regime": "unknown",
                "regime_allows_pullback": False,
                "regime_allows_catalyst": False,
                "confidence": 0.0,
            }

        details = envelope.get("details", {})
        regime = details.get("market_regime", "unknown")
        confidence = float(envelope.get("confidence") or 0.0)

        # Regime gate rules
        regime_allows_pullback = regime in ("trending_up",)
        regime_allows_catalyst = regime in ("trending_up", "trending_down", "ranging")

        return {
            "regime": regime,
            "regime_allows_pullback": regime_allows_pullback,
            "regime_allows_catalyst": regime_allows_catalyst,
            "confidence": confidence,
        }

    except (json.JSONDecodeError, ValueError, TypeError, AttributeError):
        return {
            "regime": "unknown",
            "regime_allows_pullback": False,
            "regime_allows_catalyst": False,
            "confidence": 0.0,
        }


def extract_trade_setup(market_report: str | dict) -> dict[str, Any] | None:
    """
    Extract the quant trade_setup from market analyst envelope.

    The Market Analyst computes entry_trigger, stop_loss, take_profit, and
    risk_reward deterministically; the swing trader uses these as computed
    anchors (LEARNINGS.md rule) rather than re-deriving.

    Args:
        market_report: JSON string or parsed dict from market analyst

    Returns:
        dict with keys: bias, entry_trigger, stop_loss, take_profit, risk_reward, etc.
        Or None if not available.
    """
    try:
        envelope = json.loads(market_report) if isinstance(market_report, str) else market_report

        if not isinstance(envelope, dict):
            return None

        return envelope.get("details", {}).get("trade_setup")

    except (json.JSONDecodeError, ValueError, TypeError, AttributeError):
        return None


def compute_relative_strength(
    market_report: str | dict,
    benchmark_roc: float | None = None,
) -> dict[str, Any]:
    """
    Compute 20-day relative strength vs. benchmark.

    Relative strength (RS) = stock ROC - benchmark ROC over 20 days.
    Positive RS means the stock outperformed; negative means underperformed.
    Used as a momentum confirmation for long setups.

    Args:
        market_report: JSON envelope from market analyst
        benchmark_roc: 20-day ROC of the benchmark (e.g., SPY), or None if unavailable

    Returns:
        dict with keys:
        - roc_20d: float or None, stock's 20-day ROC
        - benchmark_roc: float or None, benchmark's 20-day ROC
        - relative_strength: float or None, roc_20d - benchmark_roc
    """
    try:
        envelope = json.loads(market_report) if isinstance(market_report, str) else market_report

        if not isinstance(envelope, dict):
            return {
                "roc_20d": None,
                "benchmark_roc": benchmark_roc,
                "relative_strength": None,
            }

        details = envelope.get("details", {})
        roc = details.get("roc", {})
        stock_roc_20d = roc.get("20d")

        rs = None
        if stock_roc_20d is not None and benchmark_roc is not None:
            rs = stock_roc_20d - benchmark_roc

        return {
            "roc_20d": stock_roc_20d,
            "benchmark_roc": benchmark_roc,
            "relative_strength": rs,
        }

    except (json.JSONDecodeError, ValueError, TypeError, AttributeError):
        return {
            "roc_20d": None,
            "benchmark_roc": benchmark_roc,
            "relative_strength": None,
        }


def check_earnings_in_window(
    earnings_calendar: dict | None,
    holding_period_days: int,
    setup_type: str = "none",
) -> dict[str, Any]:
    """
    Check if earnings date falls within the intended holding window.

    Standard swing-risk practice (design section 1.1): don't hold through
    unplanned earnings unless the setup IS the earnings play (catalyst).

    Args:
        earnings_calendar: dict from get_earnings_calendar with keys
            - "earnings_date": str (YYYY-MM-DD) or None
            - "days_to_earnings": int or None
        holding_period_days: intended holding period in trading days
        setup_type: "pullback", "catalyst", or "none"

    Returns:
        dict with keys:
        - has_earnings_in_window: bool
        - earnings_date: str or None
        - days_to_earnings: int or None
        - should_avoid_non_catalyst: bool, True if earnings block non-catalyst setups
    """
    if earnings_calendar is None:
        return {
            "has_earnings_in_window": False,
            "earnings_date": None,
            "days_to_earnings": None,
            "should_avoid_non_catalyst": False,
        }

    try:
        earnings_date = earnings_calendar.get("earnings_date")
        days_to_earnings = earnings_calendar.get("days_to_earnings")

        if days_to_earnings is None or earnings_date is None:
            return {
                "has_earnings_in_window": False,
                "earnings_date": None,
                "days_to_earnings": None,
                "should_avoid_non_catalyst": False,
            }

        # Earnings in window if 0 < days_to_earnings < holding_period
        # (0 = today, already past; >= holding_period = after expected exit)
        in_window = 0 < days_to_earnings < holding_period_days

        # Block if earnings in window AND setup is not catalyst
        should_avoid = in_window and setup_type != "catalyst"

        return {
            "has_earnings_in_window": in_window,
            "earnings_date": earnings_date,
            "days_to_earnings": days_to_earnings,
            "should_avoid_non_catalyst": should_avoid,
        }

    except (TypeError, AttributeError, KeyError):
        return {
            "has_earnings_in_window": False,
            "earnings_date": None,
            "days_to_earnings": None,
            "should_avoid_non_catalyst": False,
        }


def assemble_swing_precompute(
    market_report: str | dict | None,
    earnings_calendar: dict | None,
    holding_period_days: int = 5,
    benchmark_roc: float | None = None,
) -> dict[str, Any]:
    """
    Assemble all pre-compute inputs into one dict for the LLM prompt.

    This is the main entry point; it pulls all the pieces together.

    Args:
        market_report: JSON envelope from market analyst (or None)
        earnings_calendar: earnings calendar result (or None)
        holding_period_days: candidate holding window (default 5, typical for swing)
        benchmark_roc: 20-day ROC of benchmark for relative strength (or None)

    Returns:
        dict with all pre-compute results, suitable for embedding in prompt
    """
    regime_gate = assess_regime_gate(market_report)
    trade_setup = extract_trade_setup(market_report)
    relative_strength = compute_relative_strength(market_report, benchmark_roc)

    # For earnings check, assume "pullback" as default setup_type until LLM decides
    earnings_check = check_earnings_in_window(
        earnings_calendar,
        holding_period_days,
        setup_type="pullback",
    )

    return {
        "regime_gate": regime_gate,
        "trade_setup": trade_setup,
        "relative_strength": relative_strength,
        "earnings_check": earnings_check,
    }
