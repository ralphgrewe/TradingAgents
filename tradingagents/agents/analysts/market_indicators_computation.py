"""
Deterministic indicator computation module for market analyst.

Adapted from skills/quant/compute_indicators.py, but integrated into the
pipeline to be called synchronously. This module computes all technical
indicators, signal/confidence from OHLCV data, and builds the JSON envelope
payload — the LLM contributes only a one-line summary.

Key design principle: pure computation, no LLM, deterministic signal/confidence.
"""

import json
import math
from typing import Any

import pandas as pd

# ── Indicator functions ────────────────────────────────────────────────────────

def sma(series: pd.Series, length: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=length).mean()


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=length).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=length).mean()
    rs = gain / loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """MACD Histogram (MACD - Signal Line)."""
    ema_fast = series.ewm(span=fast).mean()
    ema_slow = series.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    return macd_line - signal_line


def bbands(series: pd.Series, length: int = 20, std: float = 2) -> tuple[pd.Series, pd.Series]:
    """Bollinger Bands (upper, lower)."""
    sma_val = series.rolling(window=length).mean()
    stdev = series.rolling(window=length).std()
    return sma_val + (stdev * std), sma_val - (stdev * std)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    """Average True Range."""
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=length).mean()


def vwma(close: pd.Series, volume: pd.Series, length: int = 20) -> pd.Series:
    """Volume-Weighted Moving Average."""
    return (close * volume).rolling(window=length).sum() / volume.rolling(window=length).sum()


# ── Swing trading indicators (issue #89) ──────────────────────────────────────

def rolling_n_day_high_low(high: pd.Series, low: pd.Series, n: int = 20) -> tuple[pd.Series, pd.Series]:
    """
    Compute rolling N-day high and low.

    Args:
        high: Series of high prices
        low: Series of low prices
        n: number of trading days (default 20)

    Returns:
        (rolling_high, rolling_low) tuple of Series
    """
    rolling_high = high.rolling(window=n).max()
    rolling_low = low.rolling(window=n).min()
    return rolling_high, rolling_low


def find_swing_high_low(high: pd.Series, low: pd.Series, lookback: int = 2) -> tuple[dict | None, dict | None]:
    """
    Find the most recent confirmed swing high and swing low.

    A swing high is a bar whose high is the extreme (highest) of the lookback bars on each side.
    A swing low is a bar whose low is the extreme (lowest) of the lookback bars on each side.
    We use "confirmed" pivots: ones with at least lookback completed bars after them.

    Vectorized via a centered rolling max/min over a window of `2 * lookback + 1`
    bars: a bar is a pivot iff its own high/low equals the window extreme.
    Pandas' centered rolling only yields a value once `lookback` bars exist on
    both sides, which is exactly the "confirmed" boundary condition, so no
    separate index-bounds bookkeeping is needed.

    Args:
        high: Series of high prices
        low: Series of low prices
        lookback: number of bars on each side to check (default 2)

    Returns:
        (swing_high_dict, swing_low_dict) where each dict has keys:
        - value: the swing price
        - bar_index: the (positional) index of the swing bar
        - bars_since: how many bars ago the swing was
        Or None if not enough confirmed pivots exist.
    """
    n = len(high)
    window = lookback * 2 + 1
    if n < window:
        return None, None

    high = high.reset_index(drop=True)
    low = low.reset_index(drop=True)

    rolling_max = high.rolling(window=window, center=True).max()
    rolling_min = low.rolling(window=window, center=True).min()

    is_swing_high = high == rolling_max
    is_swing_low = low == rolling_min

    swing_high_positions = is_swing_high[is_swing_high].index
    swing_low_positions = is_swing_low[is_swing_low].index

    swing_high_idx = int(swing_high_positions[-1]) if len(swing_high_positions) else None
    swing_low_idx = int(swing_low_positions[-1]) if len(swing_low_positions) else None

    swing_high = None
    if swing_high_idx is not None:
        swing_high = {
            "value": v(high.iloc[swing_high_idx]),
            "bar_index": swing_high_idx,
            "bars_since": n - 1 - swing_high_idx,
        }

    swing_low = None
    if swing_low_idx is not None:
        swing_low = {
            "value": v(low.iloc[swing_low_idx]),
            "bar_index": swing_low_idx,
            "bars_since": n - 1 - swing_low_idx,
        }

    return swing_high, swing_low


def rate_of_change(close: pd.Series, periods: list[int]) -> dict[int, float | None]:
    """
    Compute rate of change (%) for multiple periods.

    ROC = ((close_today - close_n_days_ago) / close_n_days_ago) * 100

    Args:
        close: Series of close prices
        periods: list of periods to compute (e.g. [5, 20, 63])

    Returns:
        dict mapping period -> ROC % (or None if insufficient history)
    """
    result = {}
    for period in periods:
        if len(close) >= period + 1:
            current = close.iloc[-1]
            past = close.iloc[-(period + 1)]
            if past != 0:
                roc = ((current - past) / abs(past)) * 100
                result[period] = v(roc)
            else:
                result[period] = None
        else:
            result[period] = None
    return result


def distance_from_52week_high(high: pd.Series, close: pd.Series) -> float | None:
    """
    Compute distance from 52-week high as % below the max.

    Distance = ((max_high - current_close) / max_high) * 100

    "Current price" is the latest bar's close (the module's established
    convention elsewhere), compared against the max high of the past 252
    trading days.

    Returns None if insufficient history (< 252 bars).
    """
    if len(high) < 252:
        return None

    # Look back 252 trading days
    lookback_high = high.iloc[-252:].max()
    current_close = close.iloc[-1]

    if lookback_high > 0:
        distance = ((lookback_high - current_close) / lookback_high) * 100
        return v(distance)

    return None


def volume_surge_ratio(volume: pd.Series) -> float | None:
    """
    Compute volume surge ratio: latest day's volume / prior 20-day average volume.

    The baseline average excludes the latest bar itself, so a surge on the
    latest day is not diluted by its own volume.

    Returns None if insufficient history (< 21 bars: 20 baseline days + latest).
    """
    if len(volume) < 21:
        return None

    latest_vol = volume.iloc[-1]
    avg_vol = volume.iloc[-21:-1].mean()

    if avg_vol > 0:
        ratio = latest_vol / avg_vol
        return v(ratio)

    return None


# ── Helpers ────────────────────────────────────────────────────────────────────

def v(x: Any) -> float | None:
    """Safely convert to float, returning None for NaN/inf."""
    if x is None or (hasattr(x, '__float__') and pd.isna(x)):
        return None
    try:
        f = float(x)
        return None if math.isnan(f) or math.isinf(f) else round(f, 4)
    except (TypeError, ValueError):
        return None


def trend(cur: float | None, prv: float | None) -> str:
    """Determine trend: Rising/Falling/Flat."""
    if cur is None or prv is None:
        return "Flat"
    if cur > prv * 1.001:
        return "Rising"
    if cur < prv * 0.999:
        return "Falling"
    return "Flat"


ROLES = {
    "sma_50": "Medium-term trend, dynamic support/resistance",
    "macdh": "Direction + strength + divergence",
    "rsi": "Overbought/Oversold momentum (70/30 thresholds)",
    "boll_ub": "Upper ~2σ band — breakout / resistance",
    "boll_lb": "Lower ~2σ band — breakdown / support",
    "atr": "Required for stop-loss sizing; not a directional signal",
    "vwma": "Volume-weighted trend confirmation",
}


def interpret_signal(indicator: str, val: float | None, prv: float | None, close: float | None) -> str:
    """Return Bullish/Bearish/Neutral for an indicator."""
    tr = trend(val, prv)
    if indicator == "sma_50":
        if val and close is not None:
            if close > val:
                return "Bullish"
            if close < val:
                return "Bearish"
    elif indicator == "macdh":
        if val is not None and val > 0 and tr == "Rising":
            return "Bullish"
        if val is not None and val < 0 and tr == "Falling":
            return "Bearish"
    elif indicator == "rsi":
        if val is not None:
            if val < 30:
                return "Bullish"
            if val > 70:
                return "Bearish"
    elif indicator == "boll_ub":
        if val is not None and close is not None and close >= val:
            return "Bearish"
    elif indicator == "boll_lb":
        if val is not None and close is not None:
            if close <= val:
                return "Bearish"
            if close > val * 1.02:
                return "Bullish"
    elif indicator == "atr":
        return "Neutral"  # sizing only
    elif indicator == "vwma":
        if val is not None and close is not None and close > val:
            return "Bullish"
        elif val is not None and close is not None and close < val:
            return "Bearish"
    return "Neutral"


# ── Memory wiring helpers (issue #8 compat) ────────────────────────────────────

CONFIDENCE_SCORE = {"HIGH": 1.0, "MEDIUM": 0.6, "LOW": 0.3}


def confidence_to_score(confidence: str) -> float:
    """Map HIGH/MEDIUM/LOW confidence to numeric score."""
    return CONFIDENCE_SCORE.get(str(confidence).upper(), 0.3)


def build_key_drivers(details: dict) -> dict:
    """Build key_drivers payload for memory_store_decision."""
    convergence = details.get("convergence") or {}
    return {
        "market_regime": details.get("market_regime"),
        "confirms": convergence.get("confirms", []),
        "conflicts": convergence.get("conflicts", []),
        "trade_setup": details.get("trade_setup"),
    }


# ── Main computation ──────────────────────────────────────────────────────────

def compute_indicators(records: list[dict], ticker: str) -> dict[str, Any]:
    """
    Compute the quant envelope (signal, confidence, summary, details) from OHLCV.

    Args:
        records: list of dicts from yfinance (Date, Open, High, Low, Close, Volume)
        ticker: stock ticker symbol

    Returns:
        dict with keys: signal, confidence, summary, details
        Deterministic: same input → same output, every time.
    """
    if not records:
        return {
            "signal": None,
            "confidence": None,
            "summary": "Insufficient data",
            "details": {
                "as_of": None,
                "close": None,
                "market_regime": None,
                "indicators": [],
                "convergence": {"confirms": [], "conflicts": [], "missing": []},
                "trade_setup": None,
            },
        }

    df = pd.DataFrame(records)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    if len(df) < 2:
        return {
            "signal": None,
            "confidence": None,
            "summary": "Insufficient historical data",
            "details": {
                "as_of": None,
                "close": None,
                "market_regime": None,
                "indicators": [],
                "convergence": {"confirms": [], "conflicts": [], "missing": []},
                "trade_setup": None,
            },
        }

    # ── Compute indicators ────────────────────────────────────────────────────
    df["sma_50"] = sma(df["Close"], 50)
    df["macdh"] = macd(df["Close"])
    df["rsi"] = rsi(df["Close"], 14)
    df["boll_ub"], df["boll_lb"] = bbands(df["Close"], 20, 2)
    df["atr"] = atr(df["High"], df["Low"], df["Close"], 14)
    df["vwma"] = vwma(df["Close"], df["Volume"], 20)

    # ── Swing trading indicators (issue #89) ──────────────────────────────────
    df["n20_high"], df["n20_low"] = rolling_n_day_high_low(df["High"], df["Low"], n=20)
    swing_high, swing_low = find_swing_high_low(df["High"], df["Low"], lookback=2)
    roc_vals = rate_of_change(df["Close"], [5, 20, 63])
    distance_52w = distance_from_52week_high(df["High"], df["Close"])
    vol_surge = volume_surge_ratio(df["Volume"])

    last, prev = df.iloc[-1], df.iloc[-2]
    close = v(last["Close"])

    ind_keys = ["sma_50", "macdh", "rsi", "boll_ub", "boll_lb", "atr", "vwma"]
    raw = {k: {"value": v(last[k]), "prev": v(prev[k])} for k in ind_keys}

    # ── Build indicators list ─────────────────────────────────────────────────
    indicators = []
    for k in ind_keys:
        val, prv = raw[k]["value"], raw[k]["prev"]
        tr = trend(val, prv)
        sig = interpret_signal(k, val, prv, close)
        indicators.append({
            "indicator": k,
            "value": val,
            "prev": prv,
            "trend": tr,
            "signal": sig,
            "role": ROLES[k],
        })

    # ── Convergence ───────────────────────────────────────────────────────────
    bull = [i["indicator"] for i in indicators if i["signal"] == "Bullish"]
    bear = [i["indicator"] for i in indicators if i["signal"] == "Bearish"]
    miss = [i["indicator"] for i in indicators if i["value"] is None]
    n_bull, n_bear, n_total = len(bull), len(bear), len(ind_keys)

    if n_bull > n_bear:
        confirms = bull
        conflicts = [f"{k} bearish" for k in bear]
    elif n_bear > n_bull:
        confirms = bear
        conflicts = [f"{k} bullish" for k in bull]
    else:
        confirms = []
        conflicts = [f"{k} bullish" for k in bull] + [f"{k} bearish" for k in bear]

    convergence = {"confirms": confirms, "conflicts": conflicts, "missing": miss}

    # ── Trade setup ───────────────────────────────────────────────────────────
    atr_val = raw["atr"]["value"]
    boll_ub_v = raw["boll_ub"]["value"]
    boll_lb_v = raw["boll_lb"]["value"]
    vwma_v = raw["vwma"]["value"]
    sma_v = raw["sma_50"]["value"]
    K = 1.5

    if n_bull > n_bear:
        bias = "BUY"
        entry = f"Pullback to VWMA ({vwma_v}) or confirmed close above SMA-50 ({sma_v})"
        stop = round(close - K * atr_val, 2) if (close and atr_val) else None
        take = round(boll_ub_v, 2) if boll_ub_v else None
        sf = f"close - {K} × ATR = {close} - {K} × {atr_val} = {stop}" if stop else None
    elif n_bear > n_bull:
        bias = "SELL"
        entry = f"Confirmed close below SMA-50 ({sma_v}) with MACDH negative"
        stop = round(close + K * atr_val, 2) if (close and atr_val) else None
        take = round(boll_lb_v, 2) if boll_lb_v else None
        sf = f"close + {K} × ATR = {close} + {K} × {atr_val} = {stop}" if stop else None
    else:
        bias = "HOLD"
        entry = "Wait for directional confirmation"
        stop = None
        take = None
        sf = None

    risk = abs(close - stop) if (stop and close) else None
    reward = abs(take - close) if (take and close) else None
    rr = f"1:{round(reward/risk, 1)}" if (risk and reward and risk > 0) else "N/A"

    trade_setup = {
        "bias": bias,
        "entry_trigger": entry,
        "stop_loss": stop,
        "stop_loss_formula": sf,
        "take_profit": take,
        "risk_reward": rr,
    }

    # ── Market regime ─────────────────────────────────────────────────────────
    boll_width_pct = ((boll_ub_v - boll_lb_v) / close) if (boll_ub_v and boll_lb_v and close) else None
    sma_tr = trend(raw["sma_50"]["value"], raw["sma_50"]["prev"])

    if sma_tr == "Rising" and n_bull > n_bear:
        market_regime = "trending_up"
    elif sma_tr == "Falling" and n_bear > n_bull:
        market_regime = "trending_down"
    elif boll_width_pct and boll_width_pct > 0.08:
        market_regime = "volatile"
    else:
        market_regime = "ranging"

    # ── Confidence ────────────────────────────────────────────────────────────
    conv_ratio = abs(n_bull - n_bear) / n_total if n_total else 0
    if conv_ratio >= 0.6:
        confidence = "HIGH"
    elif conv_ratio >= 0.3:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"
    if miss:
        lvl = ["LOW", "MEDIUM", "HIGH"]
        confidence = lvl[min(lvl.index(confidence), lvl.index("MEDIUM"))]

    summary = (
        f"{bias} ({market_regime}) — "
        f"{n_bull} bullish, {n_bear} bearish, {len(miss)} missing"
    )

    # ── Build swing trading indicators section ──────────────────────────────────
    swing_indicators = {}

    # N-day high/low
    n20_high_val = v(df["n20_high"].iloc[-1])
    n20_low_val = v(df["n20_low"].iloc[-1])
    if n20_high_val is not None or n20_low_val is not None:
        swing_indicators["n20_high"] = n20_high_val
        swing_indicators["n20_low"] = n20_low_val

    # Swing high/low — replace the internal `bar_index` (meaningless once the
    # envelope is serialized) with the pivot's calendar date, matching the
    # envelope's other calendar-date fields (details.as_of).
    if swing_high is not None:
        swing_indicators["swing_high"] = {
            "value": swing_high["value"],
            "date": str(df["Date"].iloc[swing_high["bar_index"]].date()),
            "bars_since": swing_high["bars_since"],
        }
    if swing_low is not None:
        swing_indicators["swing_low"] = {
            "value": swing_low["value"],
            "date": str(df["Date"].iloc[swing_low["bar_index"]].date()),
            "bars_since": swing_low["bars_since"],
        }

    # Rate of change
    roc_section = {}
    for period, roc_val in roc_vals.items():
        if roc_val is not None:
            roc_section[f"roc_{period}d"] = roc_val
    if roc_section:
        swing_indicators["rate_of_change"] = roc_section

    # Distance from 52-week high
    if distance_52w is not None:
        swing_indicators["distance_from_52week_high_pct"] = distance_52w

    # Volume surge ratio
    if vol_surge is not None:
        swing_indicators["volume_surge_ratio"] = vol_surge

    result = {
        "signal": bias,
        "confidence": confidence,
        "summary": summary,
        "details": {
            "as_of": str(last["Date"].date()),
            "close": close,
            "market_regime": market_regime,
            "indicators": indicators,
            "convergence": convergence,
            "trade_setup": trade_setup,
            "swing_indicators": swing_indicators if swing_indicators else None,
        },
    }

    return result


def build_json_envelope(
    signal: str | None,
    confidence: str | None,
    summary: str,
    details: dict,
    ticker: str,
    date: str,
) -> str:
    """
    Build the JSON envelope per skills/SCHEMA.md.

    Returns JSON string serialized for storage in market_report.
    """
    envelope = {
        "skill": "market-analyst",
        "ticker": ticker,
        "date": date,
        "signal": signal,
        "confidence": confidence,
        "summary": summary,
        "details": details,
    }
    return json.dumps(envelope, indent=2)
