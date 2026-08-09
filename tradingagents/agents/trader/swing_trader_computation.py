"""
Deterministic pre-computation module for the Swing Trader node.

Pure-Python step (no LLM, except where noted) that evaluates the regime gate,
assembles candidate setup levels, computes relative strength vs. a benchmark,
and flags earnings within the holding window. All numeric derivation happens
here; the LLM sees computed values in the prompt and judges/weighs them
(LEARNINGS.md rule).

This module mirrors the pattern of market_indicators_computation.py but scoped
to swing-trader-specific setup assembly and regime gating. It also owns the
two live vendor calls the design (issue #26 design comment, sections 2/4)
assigns to the swing trader's own pre-compute step rather than an analyst:
the earnings calendar (get_earnings_calendar, #90) and the benchmark ROC
(get_stock_data), both routed through route_to_vendor. Both degrade to
None/"unknown" on any failure — never raise into the node.
"""

from __future__ import annotations

import io
import json
import logging
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from tradingagents.agents.analysts.market_indicators_computation import confidence_to_score

logger = logging.getLogger(__name__)

# ATR multiplier used as the fallback stop-loss sizing, matching the Market
# Analyst's own trade_setup derivation (market_indicators_computation.py).
ATR_STOP_MULTIPLIER = 1.5


def _parse_envelope(market_report: str | dict | None) -> dict | None:
    """Parse a market analyst JSON envelope (string or already-parsed dict)."""
    if market_report is None:
        return None
    try:
        envelope = json.loads(market_report) if isinstance(market_report, str) else market_report
    except (json.JSONDecodeError, TypeError):
        return None
    return envelope if isinstance(envelope, dict) else None


def _find_indicator_value(details: dict, name: str) -> float | None:
    """Look up a single indicator's value from details.indicators (a list of
    {"indicator": ..., "value": ..., ...} dicts, per market_indicators_computation.py)."""
    for entry in details.get("indicators") or []:
        if isinstance(entry, dict) and entry.get("indicator") == name:
            return entry.get("value")
    return None


def assess_regime_gate(market_report: str | dict | None) -> dict[str, Any]:
    """
    Evaluate the regime gate from the market analyst report.

    Four distinct branches (design section 2 / section 1 recommendation):
    - trending_up    -> pullback longs allowed; catalyst overlay also allowed.
    - trending_down  -> pullback SHORTS allowed (not longs); catalyst allowed.
      Distinct from ranging: a trending market still supports a directional
      pullback trade, just in the opposite direction.
    - ranging        -> no pullback either direction; catalyst setups only.
    - volatile       -> HOLD bias; neither pullback direction nor catalyst is
      cleared (the survey's documented failure mode for every strategy in
      choppy/volatile conditions).
    - unknown        -> nothing cleared, no HOLD bias (we simply don't know).

    Args:
        market_report: JSON string or parsed dict from market analyst

    Returns:
        dict with keys:
        - regime: str, one of "trending_up", "trending_down", "ranging",
          "volatile", or "unknown"
        - allow_pullback_long: bool
        - allow_pullback_short: bool
        - allow_catalyst: bool
        - hold_bias: bool, True if the regime itself argues for HOLD
        - confidence: float 0.0-1.0, regime confidence from the report
          (market analyst confidence is HIGH/MEDIUM/LOW, mapped via the
          same confidence_to_score used for memory-log scoring)
    """
    envelope = _parse_envelope(market_report)
    if envelope is None:
        return {
            "regime": "unknown",
            "allow_pullback_long": False,
            "allow_pullback_short": False,
            "allow_catalyst": False,
            "hold_bias": False,
            "confidence": 0.0,
        }

    details = envelope.get("details") or {}
    regime = details.get("market_regime") or "unknown"
    confidence = confidence_to_score(envelope.get("confidence"))

    branches = {
        "trending_up": {
            "allow_pullback_long": True,
            "allow_pullback_short": False,
            "allow_catalyst": True,
            "hold_bias": False,
        },
        "trending_down": {
            "allow_pullback_long": False,
            "allow_pullback_short": True,
            "allow_catalyst": True,
            "hold_bias": False,
        },
        "ranging": {
            "allow_pullback_long": False,
            "allow_pullback_short": False,
            "allow_catalyst": True,
            "hold_bias": False,
        },
        "volatile": {
            "allow_pullback_long": False,
            "allow_pullback_short": False,
            "allow_catalyst": False,
            "hold_bias": True,
        },
    }
    gate = branches.get(
        regime,
        {
            "allow_pullback_long": False,
            "allow_pullback_short": False,
            "allow_catalyst": False,
            "hold_bias": False,
        },
    )

    return {"regime": regime, "confidence": confidence, **gate}


def extract_trade_setup(market_report: str | dict | None) -> dict[str, Any] | None:
    """
    Assemble the candidate swing setup from the market analyst envelope.

    Stop-loss fallback chain (design section 2: "assembles candidate setup
    levels (entry anchor, stop from trade_setup/ATR/swing low, target,
    reward:risk)"), in order:
    1. The analyst's own ``trade_setup.stop_loss`` (already ATR-anchored).
    2. Direct ATR fallback: close -/+ ``ATR_STOP_MULTIPLIER`` * ATR
       (long/short) when trade_setup didn't produce a stop (e.g. HOLD bias
       at analyst compute time) but ATR and close are available now.
    3. Structural fallback: the most recent confirmed swing low (long) /
       swing high (short) from ``details.swing_indicators``.

    Reward:risk is computed here (numeric), never left for the LLM to derive
    (LEARNINGS.md rule).

    Args:
        market_report: JSON string or parsed dict from market analyst

    Returns:
        dict with keys: bias, entry_trigger, stop_loss, stop_loss_source,
        take_profit, risk_reward, close. None only when the envelope itself
        can't be parsed.
    """
    envelope = _parse_envelope(market_report)
    if envelope is None:
        return None

    details = envelope.get("details") or {}
    trade_setup = details.get("trade_setup") or {}
    close = details.get("close")
    bias = trade_setup.get("bias", "HOLD")
    swing_indicators = details.get("swing_indicators") or {}

    stop_loss = trade_setup.get("stop_loss")
    stop_loss_source = "trade_setup" if stop_loss is not None else None

    if stop_loss is None and bias in ("BUY", "SELL"):
        atr_val = _find_indicator_value(details, "atr")
        if close is not None and atr_val is not None:
            if bias == "BUY":
                stop_loss = round(close - ATR_STOP_MULTIPLIER * atr_val, 2)
            else:
                stop_loss = round(close + ATR_STOP_MULTIPLIER * atr_val, 2)
            stop_loss_source = "atr_fallback"

    if stop_loss is None and bias in ("BUY", "SELL"):
        swing_low = swing_indicators.get("swing_low")
        swing_high = swing_indicators.get("swing_high")
        if bias == "BUY" and swing_low is not None:
            stop_loss = swing_low.get("value")
            stop_loss_source = "swing_level_fallback"
        elif bias == "SELL" and swing_high is not None:
            stop_loss = swing_high.get("value")
            stop_loss_source = "swing_level_fallback"

    take_profit = trade_setup.get("take_profit")

    risk_reward = None
    if close is not None and stop_loss is not None and take_profit is not None:
        risk = abs(close - stop_loss)
        reward = abs(take_profit - close)
        if risk > 0:
            risk_reward = round(reward / risk, 2)

    return {
        "bias": bias,
        "entry_trigger": trade_setup.get("entry_trigger"),
        "stop_loss": stop_loss,
        "stop_loss_source": stop_loss_source,
        "take_profit": take_profit,
        "risk_reward": risk_reward,
        "close": close,
    }


def compute_relative_strength(
    market_report: str | dict | None,
    benchmark_roc: float | None = None,
) -> dict[str, Any]:
    """
    Compute 20-day relative strength vs. benchmark.

    Relative strength (RS) = stock ROC - benchmark ROC over 20 days.
    Positive RS means the stock outperformed; negative means underperformed.
    Used as a momentum confirmation for long setups.

    Reads the real #89 envelope path: ``details.swing_indicators.
    rate_of_change.roc_20d`` (not ``details.roc.20d``, which never existed).

    Args:
        market_report: JSON envelope from market analyst
        benchmark_roc: 20-day ROC of the benchmark (e.g., SPY), or None if unavailable

    Returns:
        dict with keys:
        - roc_20d: float or None, stock's 20-day ROC
        - benchmark_roc: float or None, benchmark's 20-day ROC
        - relative_strength: float or None, roc_20d - benchmark_roc
    """
    envelope = _parse_envelope(market_report)
    if envelope is None:
        return {
            "roc_20d": None,
            "benchmark_roc": benchmark_roc,
            "relative_strength": None,
        }

    details = envelope.get("details") or {}
    swing_indicators = details.get("swing_indicators") or {}
    roc_section = swing_indicators.get("rate_of_change") or {}
    stock_roc_20d = roc_section.get("roc_20d")

    relative_strength = None
    if stock_roc_20d is not None and benchmark_roc is not None:
        relative_strength = round(stock_roc_20d - benchmark_roc, 2)

    return {
        "roc_20d": stock_roc_20d,
        "benchmark_roc": benchmark_roc,
        "relative_strength": relative_strength,
    }


def resolve_benchmark_ticker(ticker: str, config: dict) -> str:
    """
    Pick the benchmark ticker for relative-strength comparison.

    Mirrors ``TradingAgentsGraph._resolve_benchmark`` (trading_graph.py
    ~line 194): an explicit ``config["benchmark_ticker"]`` wins; otherwise
    the ticker's suffix is matched against ``config["benchmark_map"]``,
    falling back to the empty-suffix entry (SPY by default). Kept as a
    standalone function here (rather than reusing the graph instance
    method) because the swing trader node only has the module-level
    ``get_config()`` dict available, not a ``TradingAgentsGraph`` instance.
    """
    explicit = config.get("benchmark_ticker")
    if explicit:
        return explicit
    benchmark_map = config.get("benchmark_map") or {}
    ticker_upper = ticker.upper()
    for suffix, benchmark in benchmark_map.items():
        if suffix and ticker_upper.endswith(suffix.upper()):
            return benchmark
    return benchmark_map.get("", "SPY")


def fetch_benchmark_roc(
    benchmark: str,
    trade_date: str,
    lookback_days: int = 20,
) -> float | None:
    """
    Compute the benchmark's trailing N-trading-day rate of change via
    ``get_stock_data`` (routed through ``route_to_vendor``), for the
    relative-strength comparison in ``compute_relative_strength``.

    Any failure — bad date, vendor/network error, insufficient history —
    degrades to None. The swing prompt then tells the LLM relative strength
    is unavailable rather than fabricating a number (never crashes the node).
    """
    try:
        end_dt = datetime.strptime(trade_date, "%Y-%m-%d")
    except (TypeError, ValueError):
        return None

    # Request a generous calendar window so ~lookback_days trading days are
    # covered even across weekends/holidays (no holiday calendar available,
    # consistent with the rest of this codebase, e.g. memory/resolve.py).
    start_dt = end_dt - timedelta(days=int(lookback_days * 1.6) + 5)

    try:
        from tradingagents.dataflows.interface import route_to_vendor

        raw = route_to_vendor(
            "get_stock_data", benchmark, start_dt.strftime("%Y-%m-%d"), trade_date
        )
    except Exception as exc:
        logger.warning("Benchmark ROC unavailable for %s: %s", benchmark, exc)
        return None

    if not isinstance(raw, str) or raw.startswith(("NO_DATA_AVAILABLE", "Error", "DATA_UNAVAILABLE")):
        return None

    try:
        df = pd.read_csv(io.StringIO(raw), comment="#")
    except Exception as exc:
        logger.warning("Could not parse benchmark stock data for %s: %s", benchmark, exc)
        return None

    if "Close" not in df.columns or len(df) < lookback_days + 1:
        return None

    closes = df["Close"].tolist()
    current = closes[-1]
    past = closes[-(lookback_days + 1)]
    if not past:
        return None

    return round(((current - past) / abs(past)) * 100, 2)


def fetch_earnings_calendar(ticker: str, trade_date: str) -> dict[str, Any]:
    """
    Fetch and parse the earnings calendar for ``ticker`` via the routed
    vendor function (``get_earnings_calendar``, #90), for the
    earnings-in-window check.

    ``route_to_vendor`` returns formatted text, not a dict: a
    "# Earnings Calendar ..." block on success, or one of several sentinel
    strings (``NO_DATA_AVAILABLE``, ``NO_EARNINGS_CALENDAR_AVAILABLE`` for
    non-equity symbols, ``UNKNOWN`` for a bad date, or an "Error retrieving
    ..." string for a genuine vendor/network failure). All of those, plus
    any raised exception, degrade to ``{"status": "unknown", ...}`` here —
    never crash the node.

    Returns:
        dict with keys: status ("ok" or "unknown"), earnings_date
        (YYYY-MM-DD string or None), days_to_earnings (int, calendar days,
        or None).
    """
    try:
        from tradingagents.dataflows.interface import route_to_vendor

        raw = route_to_vendor("get_earnings_calendar", ticker, trade_date)
    except Exception as exc:
        logger.warning("Earnings calendar unavailable for %s: %s", ticker, exc)
        return {"status": "unknown", "earnings_date": None, "days_to_earnings": None}

    return _parse_earnings_calendar_text(raw)


def _parse_earnings_calendar_text(raw: Any) -> dict[str, Any]:
    """Parse get_earnings_calendar's formatted text (see y_finance.py) into a dict."""
    unknown = {"status": "unknown", "earnings_date": None, "days_to_earnings": None}

    if not raw or not isinstance(raw, str):
        return unknown

    sentinel_prefixes = (
        "NO_DATA_AVAILABLE",
        "NO_EARNINGS_CALENDAR_AVAILABLE",
        "UNKNOWN",
        "Error retrieving",
        "DATA_UNAVAILABLE",
    )
    if raw.startswith(sentinel_prefixes):
        return unknown

    earnings_date = None
    days_to_earnings = None
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("Next Earnings Date:"):
            value = line.split(":", 1)[1].strip()
            if value and "Not scheduled" not in value:
                earnings_date = value
        elif line.startswith("Days Until Next Earnings:"):
            value = line.split(":", 1)[1].strip()
            try:
                days_to_earnings = int(value)
            except ValueError:
                days_to_earnings = None

    if earnings_date is None or days_to_earnings is None:
        return unknown

    return {"status": "ok", "earnings_date": earnings_date, "days_to_earnings": days_to_earnings}


def check_earnings_in_window(
    earnings_info: dict | None,
    trade_date: str,
    holding_period_days: int,
    setup_type: str = "none",
) -> dict[str, Any]:
    """
    Check if the next earnings date falls within the intended holding window.

    Standard swing-risk practice (design section 1.1 / section 5): don't hold
    through unplanned earnings unless the setup IS the earnings play
    (catalyst). The window end is computed with a trading-day offset
    (``numpy.busday_offset``, weekends-only, no holiday calendar — the same
    convention ``memory/resolve.py`` uses) so ``holding_period_days`` (a
    trading-day count) is compared against a real calendar date rather than
    naively against ``days_to_earnings`` (which the vendor reports in
    calendar days).

    Args:
        earnings_info: dict from fetch_earnings_calendar/_parse_earnings_calendar_text
            with keys "status" ("ok"/"unknown"), "earnings_date", "days_to_earnings"
        trade_date: current date (YYYY-MM-DD), the start of the holding window
        holding_period_days: intended holding period in trading days
        setup_type: "pullback", "catalyst", or "none"

    Returns:
        dict with keys:
        - status: "ok" or "unknown"
        - has_earnings_in_window: bool
        - earnings_date: str or None
        - should_avoid_non_catalyst: bool, True if earnings blocks non-catalyst setups
    """
    unknown_result = {
        "status": "unknown",
        "has_earnings_in_window": False,
        "earnings_date": None,
        "should_avoid_non_catalyst": False,
    }

    if not earnings_info or earnings_info.get("status") != "ok":
        return unknown_result

    earnings_date_str = earnings_info.get("earnings_date")

    try:
        trade_date_np = np.datetime64(trade_date, "D")
        earnings_date_np = np.datetime64(earnings_date_str, "D")
        window_end_np = np.busday_offset(
            trade_date_np, holding_period_days, roll="forward"
        )
        in_window = bool(trade_date_np < earnings_date_np <= window_end_np)
    except (ValueError, TypeError):
        return unknown_result

    should_avoid = in_window and setup_type != "catalyst"

    return {
        "status": "ok",
        "has_earnings_in_window": in_window,
        "earnings_date": earnings_date_str,
        "should_avoid_non_catalyst": should_avoid,
    }


def assemble_swing_precompute(
    market_report: str | dict | None,
    ticker: str,
    trade_date: str,
    holding_period_days: int = 5,
    benchmark: str = "SPY",
) -> dict[str, Any]:
    """
    Assemble all pre-compute inputs into one dict for the LLM prompt.

    This is the main entry point the Swing Trader node calls; it pulls
    together the regime gate, candidate setup, relative strength (fetching
    the benchmark ROC live), and the earnings-in-window flag (fetching the
    earnings calendar live). Both vendor calls degrade gracefully — see
    ``fetch_benchmark_roc`` / ``fetch_earnings_calendar`` — so a network or
    vendor failure never aborts the node; it just tells the LLM the input is
    unavailable.

    Args:
        market_report: JSON envelope from market analyst (or None)
        ticker: the instrument under analysis (for the earnings calendar call)
        trade_date: current trading date, YYYY-MM-DD
        holding_period_days: candidate holding window (trading days; default
            5, typical for a pullback swing trade — the LLM may declare a
            different final holding_period_days, capped at the call site)
        benchmark: benchmark ticker for relative strength (resolve via
            ``resolve_benchmark_ticker`` before calling this)

    Returns:
        dict with all pre-compute results, suitable for embedding in prompt
    """
    regime_gate = assess_regime_gate(market_report)
    trade_setup = extract_trade_setup(market_report)

    benchmark_roc = fetch_benchmark_roc(benchmark, trade_date)
    relative_strength = compute_relative_strength(market_report, benchmark_roc)

    earnings_info = fetch_earnings_calendar(ticker, trade_date)
    setup_type_hint = (
        "pullback" if trade_setup and trade_setup.get("bias") in ("BUY", "SELL") else "none"
    )
    earnings_check = check_earnings_in_window(
        earnings_info, trade_date, holding_period_days, setup_type=setup_type_hint
    )

    return {
        "regime_gate": regime_gate,
        "trade_setup": trade_setup,
        "relative_strength": relative_strength,
        "earnings_check": earnings_check,
    }
