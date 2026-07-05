"""Deterministic portfolio allocation engine.

Ported from ``skills/portfolio-manager/rebalance.py`` (the Claude Cowork
skill's reference implementation) so the same style tables and allocation
math can run inside ``run_trading_agents.py``'s portfolio mode. The porting
goal is fidelity: the style parameters, weight tables, normalisation, and
share/delta math below are intentionally identical to the skill version —
only the I/O shell (params.json / envelope printing) is dropped in favour
of plain function calls that ``tradingagents.portfolio.runner`` composes.

Two style tables are supported:

- **aggressive** — 5 slots, 5% cash reserve, 35% position cap, 1-share min
  trade. HOLD = exit: any position not actively rated BUY is sold.
- **conservative** — 10 slots, 15% cash reserve, 15% position cap, 2-share
  min trade. HOLD = keep (no trade); SELL trims 50% of the current weight
  unless confidence is HIGH, in which case it's a full exit.

``RATING_TO_SIGNAL`` maps the pipeline's 5-tier ``PortfolioRating`` (Buy /
Overweight / Hold / Underweight / Sell) onto the style tables' 3-tier
signal + conviction vocabulary (BUY/HOLD/SELL x HIGH/MEDIUM/LOW). It is a
plain data table so the mapping is easy to retune without touching the
allocation math.
"""

from __future__ import annotations

import math
from typing import Any

# ── Style tables ──────────────────────────────────────────────────────────────

STYLE_PARAMS: dict[str, dict[str, float | int]] = {
    "aggressive": {
        "max_positions": 5,
        "cash_reserve": 0.05,
        "slot": 0.19,
        "position_cap": 0.35,
        "min_trade_size": 1,
    },
    "conservative": {
        "max_positions": 10,
        "cash_reserve": 0.15,
        "slot": 0.085,
        "position_cap": 0.15,
        "min_trade_size": 2,
    },
}

# Aggressive: raw_weight = multiplier x slot
AGG_MULT: dict[tuple[str, str], float] = {
    ("BUY", "HIGH"): 2.0,
    ("BUY", "MEDIUM"): 1.0,
    ("BUY", "LOW"): 0.5,
    ("HOLD", "HIGH"): 0.0,
    ("HOLD", "MEDIUM"): 0.0,
    ("HOLD", "LOW"): 0.0,
    ("SELL", "HIGH"): 0.0,
    ("SELL", "MEDIUM"): 0.0,
    ("SELL", "LOW"): 0.0,
}

# Conservative: raw_weight = multiplier x slot (HOLD/SELL handled separately)
CON_MULT: dict[tuple[str, str], float] = {
    ("BUY", "HIGH"): 1.0,
    ("BUY", "MEDIUM"): 0.75,
    ("BUY", "LOW"): 0.4,
}

# Default mapping from the pipeline's final 5-tier PortfolioRating to the
# style tables' (signal, confidence) vocabulary. A plain data table so it's
# easy to retune independently of the allocation math above.
RATING_TO_SIGNAL: dict[str, tuple[str, str]] = {
    "Buy": ("BUY", "HIGH"),
    "Overweight": ("BUY", "MEDIUM"),
    "Hold": ("HOLD", "MEDIUM"),
    "Underweight": ("SELL", "MEDIUM"),
    "Sell": ("SELL", "HIGH"),
}


def rating_to_signal(rating: str) -> dict[str, str]:
    """Map a 5-tier ``PortfolioRating`` string to a ``{signal, confidence}`` dict.

    Unknown ratings fall back to HOLD/MEDIUM (no-op for aggressive, keep
    current weight for conservative) rather than raising, so a single bad
    rating doesn't abort the whole portfolio run.
    """
    signal, confidence = RATING_TO_SIGNAL.get(rating, ("HOLD", "MEDIUM"))
    return {"signal": signal, "confidence": confidence}


# ── helpers ───────────────────────────────────────────────────────────────────


def raw_weight(style: str, sig: str, conf: str, current_weight: float, slot: float) -> float:
    """Compute the raw (pre-normalisation) target weight for one position."""
    key = (sig.upper(), conf.upper())
    if style == "aggressive":
        return AGG_MULT.get(key, 0.0) * slot

    # conservative
    if sig.upper() == "HOLD":
        return current_weight  # keep current allocation up to slot (clamped later)
    if sig.upper() == "SELL":
        return current_weight * 0.5 if conf.upper() != "HIGH" else 0.0
    return CON_MULT.get(key, 0.0) * slot


def normalise(weights: dict[str, float], cash_reserve: float, position_cap: float) -> dict[str, float]:
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


def compute_allocation(
    style: str,
    universe: list[str],
    signals: dict[str, dict[str, str]],
    prices: dict[str, float],
    pre_equity: float,
    pre_positions: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Compute the target allocation for every ticker in ``universe``.

    Args:
        style: "aggressive" or "conservative".
        universe: tickers to allocate over (tickers held but not in the
            universe are simply not touched by the caller).
        signals: ``{ticker: {"signal": "BUY|HOLD|SELL", "confidence": "HIGH|MEDIUM|LOW"}}``.
        prices: ``{ticker: price}``. Tickers with no price are skipped (no
            allocation entry emitted for them).
        pre_equity: total portfolio equity before rebalancing.
        pre_positions: ``{ticker: {"shares": int, "price": float, "market_value": float}}``.

    Returns:
        ``{ticker: {raw_weight, target_weight, target_shares, current_shares,
        delta, price}}`` for each ticker with a known price.
    """
    sp = STYLE_PARAMS[style]
    slot = sp["slot"]
    position_cap = sp["position_cap"]
    cash_reserve = sp["cash_reserve"]
    min_trade = sp["min_trade_size"]

    # raw weights (HOLD/SELL handling embedded in raw_weight())
    raw_w: dict[str, float] = {}
    for ticker in universe:
        sig_data = signals.get(ticker, {"signal": "HOLD", "confidence": "MEDIUM"})
        sig = sig_data.get("signal", "HOLD")
        conf = sig_data.get("confidence", "MEDIUM")
        curr_mv = pre_positions.get(ticker, {}).get("market_value", 0)
        curr_w = curr_mv / pre_equity if pre_equity > 0 else 0.0
        raw_w[ticker] = raw_weight(style, sig, conf, curr_w, slot)

    # normalise + clamp
    target_w = normalise(raw_w, cash_reserve, position_cap)

    # target shares and deltas
    allocation: dict[str, dict[str, Any]] = {}
    for ticker in universe:
        price = prices.get(ticker)
        if not price:
            continue
        target_dollars = pre_equity * target_w.get(ticker, 0.0)
        target_shs = math.floor(target_dollars / price)
        current_shs = int(pre_positions.get(ticker, {}).get("shares", 0))
        delta = target_shs - current_shs

        # min trade size filter
        if abs(delta) < min_trade:
            delta = 0

        allocation[ticker] = {
            "raw_weight": round(raw_w.get(ticker, 0.0), 4),
            "target_weight": round(target_w.get(ticker, 0.0), 4),
            "target_shares": target_shs,
            "current_shares": current_shs,
            "delta": delta,
            "price": price,
        }

    return allocation
