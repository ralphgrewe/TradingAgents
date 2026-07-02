#!/usr/bin/env python3
"""
compute_indicators.py — Extended quant indicator script (A3)

Usage:
  python compute_indicators.py <price_history.json> <TICKER>

<price_history.json> is the raw output of yfinance_get_price_history saved to a file.

Computes SMA-50, MACDH, RSI, Bollinger bands, ATR, VWMA in pure pandas/numpy
(no pandas-ta dependency).

Prints JSON with keys:
  signal, confidence, summary, details
  where details matches the quant-indicator-analyst details payload exactly.

Exit 0 on success, 1 on error.
"""

import json, sys, math
from pathlib import Path

try:
    import pandas as pd
    import numpy as np
except ImportError:
    import subprocess
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "pandas", "numpy",
         "--break-system-packages"],
        stdout=subprocess.DEVNULL,
    )
    import pandas as pd
    import numpy as np


# ── indicator functions ────────────────────────────────────────────────────────

def sma(series, length):
    return series.rolling(window=length).mean()


def rsi(series, length=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=length).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=length).mean()
    rs = gain / loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))


def macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast).mean()
    ema_slow = series.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    return macd_line - signal_line


def bbands(series, length=20, std=2):
    sma_val = series.rolling(window=length).mean()
    stdev = series.rolling(window=length).std()
    return sma_val + (stdev * std), sma_val - (stdev * std)


def atr(high, low, close, length=14):
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=length).mean()


def vwma(close, volume, length=20):
    return (close * volume).rolling(window=length).sum() / volume.rolling(window=length).sum()


# ── helpers ────────────────────────────────────────────────────────────────────

def v(x):
    if x is None or (hasattr(x, '__float__') and pd.isna(x)):
        return None
    try:
        f = float(x)
        return None if math.isnan(f) or math.isinf(f) else round(f, 4)
    except (TypeError, ValueError):
        return None


def trend(cur, prv):
    if cur is None or prv is None:
        return "Flat"
    if cur > prv * 1.001:
        return "Rising"
    if cur < prv * 0.999:
        return "Falling"
    return "Flat"


ROLES = {
    "sma_50":  "Medium-term trend, dynamic support/resistance",
    "macdh":   "Direction + strength + divergence",
    "rsi":     "Overbought/Oversold momentum (70/30 thresholds)",
    "boll_ub": "Upper ~2σ band — breakout / resistance",
    "boll_lb": "Lower ~2σ band — breakdown / support",
    "atr":     "Required for stop-loss sizing; not a directional signal",
    "vwma":    "Volume-weighted trend confirmation",
}


def interpret_signal(indicator, val, prv, close):
    """Return Bullish / Bearish / Neutral for a given indicator."""
    tr = trend(val, prv)
    if indicator == "sma_50":
        if val and close > val:  return "Bullish"
        if val and close < val:  return "Bearish"
    elif indicator == "macdh":
        if val and val > 0 and tr == "Rising":  return "Bullish"
        if val and val < 0 and tr == "Falling": return "Bearish"
    elif indicator == "rsi":
        if val and val < 30: return "Bullish"
        if val and val > 70: return "Bearish"
    elif indicator == "boll_ub":
        if val and close >= val: return "Bearish"
    elif indicator == "boll_lb":
        if val and close <= val: return "Bearish"
        if val and close > val * 1.02: return "Bullish"
    elif indicator == "atr":
        return "Neutral"   # sizing only
    elif indicator == "vwma":
        if val and close > val: return "Bullish"
        if val and close < val: return "Bearish"
    return "Neutral"


# ── memory wiring (issue #8) ─────────────────────────────────────────────────
#
# The quant skill is the memory-system pilot (issue #8): SKILL.md calls the
# memory_* MCP tools directly, but the payloads it sends to
# memory_store_decision must be built from this script's deterministic
# output, not re-derived by the model. Keeping that construction here (pure
# functions of the already-computed `result`/`details` dict, no I/O) means
# it is exercised by the same unit tests as the rest of the indicator math
# and can never itself introduce non-determinism into `signal`/`confidence`.

# Same HIGH/MEDIUM/LOW -> numeric convention as skills/trader/score_trader.py's
# `conf_weight` (also documented in tradingagents/memory/stats.py's confidence
# bucket derivation) — reused here rather than inventing a second mapping.
CONFIDENCE_SCORE = {"HIGH": 1.0, "MEDIUM": 0.6, "LOW": 0.3}


def confidence_to_score(confidence):
    """Map the envelope's HIGH/MEDIUM/LOW confidence to the numeric score
    `memory_store_decision`'s `confidence` argument expects. Unknown/missing
    values fall back to 0.3 (LOW), matching `conf_weight`'s default."""
    return CONFIDENCE_SCORE.get(str(confidence).upper(), 0.3)


def build_key_drivers(details):
    """Build the `key_drivers` payload for `memory_store_decision` from this
    script's `details` output: `market_regime`, `convergence.confirms` +
    `.conflicts`, and `trade_setup`, verbatim (issue #8). No new derivation —
    every value here already exists in `details`."""
    convergence = details.get("convergence") or {}
    return {
        "market_regime": details.get("market_regime"),
        "confirms": convergence.get("confirms", []),
        "conflicts": convergence.get("conflicts", []),
        "trade_setup": details.get("trade_setup"),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def compute(records, ticker):
    """Compute the quant envelope (`signal`, `confidence`, `summary`,
    `details`) for `ticker` from raw OHLCV `records`
    (`yfinance_get_price_history`-shaped list of dicts).

    Pure function of `records`/`ticker` only — deliberately takes no memory
    / past-context input, so the deterministic signal/confidence computed
    here can never be influenced by injected lessons (issue #8's "Verify"
    requirement). `main()` is a thin CLI wrapper around this.
    """
    df = pd.DataFrame(records)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # ── Compute indicators ────────────────────────────────────────────────────
    df["sma_50"]  = sma(df["Close"], 50)
    df["macdh"]   = macd(df["Close"])
    df["rsi"]     = rsi(df["Close"], 14)
    df["boll_ub"], df["boll_lb"] = bbands(df["Close"], 20, 2)
    df["atr"]     = atr(df["High"], df["Low"], df["Close"], 14)
    df["vwma"]    = vwma(df["Close"], df["Volume"], 20)

    last, prev = df.iloc[-1], df.iloc[-2]
    close = v(last["Close"])

    ind_keys = ["sma_50", "macdh", "rsi", "boll_ub", "boll_lb", "atr", "vwma"]
    raw = {k: {"value": v(last[k]), "prev": v(prev[k])} for k in ind_keys}

    # ── Build indicators list ─────────────────────────────────────────────────
    indicators = []
    for k in ind_keys:
        val, prv = raw[k]["value"], raw[k]["prev"]
        tr  = trend(val, prv)
        sig = interpret_signal(k, val, prv, close)
        indicators.append({
            "indicator": k,
            "value":  val,
            "prev":   prv,
            "trend":  tr,
            "signal": sig,
            "role":   ROLES[k],
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
        confirms  = []
        conflicts = [f"{k} bullish" for k in bull] + [f"{k} bearish" for k in bear]

    convergence = {"confirms": confirms, "conflicts": conflicts, "missing": miss}

    # ── Trade setup ───────────────────────────────────────────────────────────
    atr_val    = raw["atr"]["value"]
    boll_ub_v  = raw["boll_ub"]["value"]
    boll_lb_v  = raw["boll_lb"]["value"]
    vwma_v     = raw["vwma"]["value"]
    sma_v      = raw["sma_50"]["value"]
    K          = 1.5

    if n_bull > n_bear:
        bias = "BUY"
        entry  = f"Pullback to VWMA ({vwma_v}) or confirmed close above SMA-50 ({sma_v})"
        stop   = round(close - K * atr_val, 2) if (close and atr_val) else None
        take   = round(boll_ub_v, 2) if boll_ub_v else None
        sf     = f"close - {K} × ATR = {close} - {K} × {atr_val} = {stop}" if stop else None
    elif n_bear > n_bull:
        bias = "SELL"
        entry  = f"Confirmed close below SMA-50 ({sma_v}) with MACDH negative"
        stop   = round(close + K * atr_val, 2) if (close and atr_val) else None
        take   = round(boll_lb_v, 2) if boll_lb_v else None
        sf     = f"close + {K} × ATR = {close} + {K} × {atr_val} = {stop}" if stop else None
    else:
        bias = "HOLD"
        entry  = "Wait for directional confirmation"
        stop   = None
        take   = None
        sf     = None

    risk   = abs(close - stop) if (stop and close) else None
    reward = abs(take  - close) if (take  and close) else None
    rr     = f"1:{round(reward/risk, 1)}" if (risk and reward and risk > 0) else "N/A"

    trade_setup = {
        "bias":              bias,
        "entry_trigger":     entry,
        "stop_loss":         stop,
        "stop_loss_formula": sf,
        "take_profit":       take,
        "risk_reward":       rr,
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
        "signal":     bias,
        "confidence": confidence,
        "summary":    summary,
        "details": {
            "as_of":         str(last["Date"].date()),
            "close":         close,
            "market_regime": market_regime,
            "indicators":    indicators,
            "convergence":   convergence,
            "trade_setup":   trade_setup,
        },
    }

    return result


def main():
    if len(sys.argv) < 3:
        print("Usage: compute_indicators.py <price_history.json> <TICKER>",
              file=sys.stderr)
        sys.exit(1)

    try:
        raw_json = Path(sys.argv[1]).read_text()
        ticker   = sys.argv[2].upper()
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    records = json.loads(raw_json)
    result = compute(records, ticker)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
