#!/usr/bin/env python3
"""
rebalance.py — Portfolio allocation script (A4)

Usage:
  python rebalance.py <params.json>

params.json schema:
  {
    "style":         "aggressive" | "conservative",
    "depot_id":      str,
    "universe":      [ticker, ...],
    "mode":          "fresh" | "cached",
    "date":          "YYYY-MM-DD",
    "signals":       {TICKER: {"signal": "BUY|HOLD|SELL", "confidence": "HIGH|MEDIUM|LOW"}},
    "prices":        {TICKER: float},
    "pre_snapshot":  {"equity": float, "cash": float,
                      "positions": {symbol: {"shares": int, "price": float, "market_value": float}}},
    "trades_executed": [{"symbol":str,"side":"buy|sell","quantity":int,
                         "status":"executed|pending|rejected","message":str,"fill_price":float}],
    "post_snapshot": {"equity": float, "cash": float, "positions": {...}}
  }

Prints the full portfolio-manager envelope JSON.

Exit 0 on success, 1 on error.
"""

import json
import math
import sys
from pathlib import Path

# ── Style tables ──────────────────────────────────────────────────────────────

STYLE_PARAMS = {
    "aggressive": {
        "max_positions": 5,
        "cash_reserve":  0.05,
        "slot":          0.19,
        "position_cap":  0.35,
        "min_trade_size": 1,
    },
    "conservative": {
        "max_positions": 10,
        "cash_reserve":  0.15,
        "slot":          0.085,
        "position_cap":  0.15,
        "min_trade_size": 2,
    },
}

# Aggressive: raw_weight = multiplier × slot
AGG_MULT = {
    ("BUY",  "HIGH"):   2.0,
    ("BUY",  "MEDIUM"): 1.0,
    ("BUY",  "LOW"):    0.5,
    ("HOLD", "HIGH"):   0.0,
    ("HOLD", "MEDIUM"): 0.0,
    ("HOLD", "LOW"):    0.0,
    ("SELL", "HIGH"):   0.0,
    ("SELL", "MEDIUM"): 0.0,
    ("SELL", "LOW"):    0.0,
}

# Conservative: raw_weight = multiplier × slot  (HOLD/SELL handled separately)
CON_MULT = {
    ("BUY",  "HIGH"):   1.0,
    ("BUY",  "MEDIUM"): 0.75,
    ("BUY",  "LOW"):    0.4,
}


# ── helpers ───────────────────────────────────────────────────────────────────

def raw_weight(style, sig, conf, current_weight, slot):
    key = (sig.upper(), conf.upper())
    if style == "aggressive":
        return AGG_MULT.get(key, 0.0) * slot

    # conservative
    if sig.upper() == "HOLD":
        return current_weight          # keep current allocation up to slot (clamped later)
    if sig.upper() == "SELL":
        return current_weight * 0.5 if conf.upper() != "HIGH" else 0.0
    return CON_MULT.get(key, 0.0) * slot


def normalise(weights: dict, cash_reserve: float, position_cap: float) -> dict:
    """Scale to (1 - cash_reserve), clamp to position_cap, re-normalise once."""
    total = sum(weights.values())
    if total == 0:
        return dict.fromkeys(weights, 0.0)

    target = 1.0 - cash_reserve
    scaled = {k: v / total * target for k, v in weights.items()}

    # Clamp
    clamped = {k: min(v, position_cap) for k, v in scaled.items()}

    # Re-normalise if any cap fired
    if any(scaled[k] > position_cap for k in scaled):
        total2 = sum(clamped.values())
        if total2 > 0:
            clamped = {k: v / total2 * target for k, v in clamped.items()}
            clamped = {k: min(v, position_cap) for k, v in clamped.items()}

    return clamped


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: rebalance.py <params.json>", file=sys.stderr)
        sys.exit(1)

    try:
        params = json.loads(Path(sys.argv[1]).read_text())
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    style    = params["style"]
    depot_id = params["depot_id"]
    universe = params["universe"]
    mode     = params.get("mode", "fresh")
    date     = params["date"]
    signals  = params["signals"]
    prices   = params["prices"]
    pre      = params["pre_snapshot"]
    trades   = params.get("trades_executed", [])
    post     = params["post_snapshot"]

    sp           = STYLE_PARAMS[style]
    slot         = sp["slot"]
    position_cap = sp["position_cap"]
    cash_reserve = sp["cash_reserve"]
    min_trade    = sp["min_trade_size"]

    pre_equity    = pre["equity"]
    pre_positions = pre.get("positions", {})

    # ── Step 4: Compute allocation ────────────────────────────────────────────

    # 4a+4b: raw weights (HOLD handling embedded in raw_weight())
    raw_w = {}
    for ticker in universe:
        sig_data = signals.get(ticker, {"signal": "HOLD", "confidence": "MEDIUM"})
        sig  = sig_data.get("signal", "HOLD")
        conf = sig_data.get("confidence", "MEDIUM")
        curr_mv  = pre_positions.get(ticker, {}).get("market_value", 0)
        curr_w   = curr_mv / pre_equity if pre_equity > 0 else 0.0
        raw_w[ticker] = raw_weight(style, sig, conf, curr_w, slot)

    # 4c: normalise + clamp
    target_w = normalise(raw_w, cash_reserve, position_cap)

    # 4e+4f: target shares and deltas
    allocation = {}
    for ticker in universe:
        price = prices.get(ticker)
        if not price:
            continue
        target_dollars = pre_equity * target_w.get(ticker, 0.0)
        target_shs     = math.floor(target_dollars / price)
        current_shs    = int(pre_positions.get(ticker, {}).get("shares", 0))
        delta          = target_shs - current_shs

        # min trade size filter
        if abs(delta) < min_trade:
            delta = 0

        allocation[ticker] = {
            "raw_weight":     round(raw_w.get(ticker, 0.0), 4),
            "target_weight":  round(target_w.get(ticker, 0.0), 4),
            "target_shares":  target_shs,
            "current_shares": current_shs,
            "delta":          delta,
            "price":          price,
        }

    # ── Step 7: Envelope assembly ─────────────────────────────────────────────
    post_equity    = post["equity"]
    n_buys  = sum(1 for t in trades if t.get("side") == "buy"  and t.get("status") != "rejected")
    n_sells = sum(1 for t in trades if t.get("side") == "sell" and t.get("status") != "rejected")
    rejected = [t for t in trades if t.get("status") == "rejected"]

    details = {
        "style":      style,
        "depot_id":   depot_id,
        "universe":   universe,
        "mode":       mode,
        "style_params": sp,
        "signals":    signals,
        "allocation": allocation,
        "trades_executed":  trades,
        "rejected_orders":  rejected,
        "pre_snapshot":  pre,
        "post_snapshot": post,
        "equity_change": round(post_equity - pre_equity, 2),
    }

    envelope = {
        "skill":      "portfolio-manager",
        "ticker":     None,
        "date":       date,
        "signal":     None,
        "confidence": None,
        "summary": (
            f"{style} rebalance of {depot_id}: "
            f"{n_buys} buys, {n_sells} sells — equity ${post_equity:,.0f}"
        ),
        "details": details,
    }

    print(json.dumps(envelope, indent=2))


if __name__ == "__main__":
    main()
