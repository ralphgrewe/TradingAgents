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
from typing import Any, Optional

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


# ── Helpers ────────────────────────────────────────────────────────────────────

def v(x: Any) -> Optional[float]:
    """Safely convert to float, returning None for NaN/inf."""
    if x is None or (hasattr(x, '__float__') and pd.isna(x)):
        return None
    try:
        f = float(x)
        return None if math.isnan(f) or math.isinf(f) else round(f, 4)
    except (TypeError, ValueError):
        return None


def trend(cur: Optional[float], prv: Optional[float]) -> str:
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


def interpret_signal(indicator: str, val: Optional[float], prv: Optional[float], close: Optional[float]) -> str:
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
        if val is not None and close is not None:
            if close >= val:
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
        if val is not None and close is not None:
            if close > val:
                return "Bullish"
            if close < val:
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
        },
    }

    return result


def build_json_envelope(
    signal: Optional[str],
    confidence: Optional[str],
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
