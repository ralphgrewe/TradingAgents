#!/usr/bin/env python3
"""
score_trader.py — Trader scoring script (A1)

Usage: python score_trader.py <run_dir>
  run_dir: path like runs/YYYYMMDD/TICKER

Reads:
  <run_dir>/fundamental-analyst.json
  <run_dir>/financial-news-analyst.json
  <run_dir>/quant-indicator-analyst.json

Prints JSON with keys:
  signals, composite_score, thresholds, analyst_aggregates,
  conflicts, signal, confidence

Exit 0 on success, 1 on error.

The scoring math lives in `compute(fund, news, quant)`, a pure function of
the three loaded analyst envelopes with no memory/context input — `main()`
is a thin CLI wrapper around it. `build_key_drivers` is a separate pure
helper (issue #11) used by `SKILL.md`'s memory wiring to build the
`key_drivers` payload for `memory_store_decision` from this script's
`result` plus the `rationale`/`risk_note` prose written afterward.
"""

import json
import sys
from pathlib import Path


def load(path):
    with open(path) as f:
        return json.load(f)


def conf_weight(c):
    return {"HIGH": 1.0, "MEDIUM": 0.6, "LOW": 0.3}.get(str(c).upper(), 0.3)


def sig_score(s):
    return {"BUY": 1, "HOLD": 0, "SELL": -1}.get(str(s).upper(), 0)


def score_to_signal(s):
    if s > 0.15:
        return "BUY"
    if s < -0.15:
        return "SELL"
    return "HOLD"


def aggregate(signals_subset):
    total = sum(s["weighted_score"] for s in signals_subset)
    return score_to_signal(total)


# ── memory wiring (issue #11) ────────────────────────────────────────────────
#
# The trader skill is wired into the shared memory core in issue #11, one
# tier above the three analyst skills (quant #8, fundamental #9, news #10):
# it synthesizes their signals rather than producing one from raw data
# itself. `conf_weight` above already is this module's HIGH/MEDIUM/LOW ->
# numeric convention (the same one skills/quant/compute_indicators.py's and
# skills/fundamental/compute_ratios.py's `confidence_to_score` reuse) — no
# second mapping is added here. `build_key_drivers` mirrors those two
# modules' helper of the same name: a pure reshaping function, with no
# memory/context parameter, so it is structurally incapable of feeding
# injected lessons back into the composite score.


def build_key_drivers(result, rationale=None, risk_note=None):
    """Build the `key_drivers` payload for `memory_store_decision` (issue #11):
    `composite_score`, `analyst_aggregates`, and `conflicts` taken verbatim
    from `compute()`'s `result`, plus the `rationale`/`risk_note` prose
    written by the Haiku subagent in SKILL.md Steps 3-6 (not part of
    `result` — passed in separately since they are not computed by this
    script). No new derivation — every value here already exists elsewhere
    in the trader envelope."""
    return {
        "composite_score": result.get("composite_score"),
        "analyst_aggregates": result.get("analyst_aggregates"),
        "conflicts": result.get("conflicts", []),
        "rationale": rationale,
        "risk_note": risk_note,
    }


def compute(fund, news, quant):
    """Compute the trader's composite score/signal/confidence (Steps 2-5 of
    SKILL.md) from the three already-loaded analyst envelopes.

    Pure function of its arguments only — deliberately takes no memory /
    past-context input, so the deterministic composite-score logic here can
    never be influenced by injected lessons (issue #11's "Verify"
    requirement: `signal`/`confidence` unchanged by the memory addition,
    mirroring issue #8/#9's `compute()` functions). `main()` is a thin CLI
    wrapper around this.
    """
    fd = fund["details"]
    nd = news["details"]
    qd = quant["details"]

    # ── Step 2: Extract signals ──────────────────────────────────────────────

    # Fundamental
    f1_raw = fd["value"]["signal"]
    f1_sw  = conf_weight(fd["value"]["confidence"])
    f2_raw = fd["growth"]["signal"]
    f2_sw  = conf_weight(fd["growth"]["confidence"])
    f3_map = {"BULLISH": "BUY", "NEUTRAL": "HOLD", "BEARISH": "SELL"}
    f3_raw = f3_map.get(str(fd["insider_sentiment"]).upper(), "HOLD")
    f3_sw  = 0.5

    # News
    n1_raw = nd["conservative"]["rating"]
    n1_sw  = float(nd["conservative"]["confidence"])
    n2_raw = nd["risky"]["rating"]
    n2_sw  = float(nd["risky"]["confidence"])

    fc     = nd["fundamentals_counts"]
    fc_pos = fc["positive"]["count"]
    fc_neg = fc["negative"]["count"]
    n3_raw = "BUY" if fc_pos > fc_neg else ("SELL" if fc_neg > fc_pos else "HOLD")
    n3_sw  = max(
        fc["positive"].get("avg_confidence", 0),
        fc.get("neutral", {}).get("avg_confidence", 0),
        fc["negative"].get("avg_confidence", 0),
    )

    sc     = nd["sentiment_counts"]
    sc_pos = sc["positive"]["count"]
    sc_neg = sc["negative"]["count"]
    n4_raw = "BUY" if sc_pos > sc_neg else ("SELL" if sc_neg > sc_pos else "HOLD")
    n4_sw  = max(
        sc["positive"].get("avg_confidence", 0),
        sc.get("neutral", {}).get("avg_confidence", 0),
        sc["negative"].get("avg_confidence", 0),
    )

    # Quant
    q1_raw = qd["trade_setup"]["bias"]
    q1_sw  = 1.0

    indicators = qd["indicators"]
    n_bull = sum(1 for i in indicators if i.get("signal") == "Bullish")
    n_bear = sum(1 for i in indicators if i.get("signal") == "Bearish")
    n_total = len(indicators)
    q2_raw = "BUY" if n_bull > n_bear else ("SELL" if n_bear > n_bull else "HOLD")
    q2_sw  = (abs(n_bull - n_bear) / n_total) if n_total > 0 else 0.0

    # ── Step 3: Score ────────────────────────────────────────────────────────
    AW_F = 0.35
    AW_N = 0.25
    AW_Q = 0.40

    def ws(raw, sw, aw):
        return round(sig_score(raw) * sw * aw, 4)

    signals = [
        {"id": "F1", "source": "fundamental", "raw": f1_raw, "signal_weight": f1_sw,        "analyst_weight": AW_F, "weighted_score": ws(f1_raw, f1_sw, AW_F)},
        {"id": "F2", "source": "fundamental", "raw": f2_raw, "signal_weight": f2_sw,        "analyst_weight": AW_F, "weighted_score": ws(f2_raw, f2_sw, AW_F)},
        {"id": "F3", "source": "fundamental", "raw": f3_raw, "signal_weight": f3_sw,        "analyst_weight": AW_F, "weighted_score": ws(f3_raw, f3_sw, AW_F)},
        {"id": "N1", "source": "news",        "raw": n1_raw, "signal_weight": round(n1_sw, 4), "analyst_weight": AW_N, "weighted_score": ws(n1_raw, n1_sw, AW_N)},
        {"id": "N2", "source": "news",        "raw": n2_raw, "signal_weight": round(n2_sw, 4), "analyst_weight": AW_N, "weighted_score": ws(n2_raw, n2_sw, AW_N)},
        {"id": "N3", "source": "news",        "raw": n3_raw, "signal_weight": round(n3_sw, 4), "analyst_weight": AW_N, "weighted_score": ws(n3_raw, n3_sw, AW_N)},
        {"id": "N4", "source": "news",        "raw": n4_raw, "signal_weight": round(n4_sw, 4), "analyst_weight": AW_N, "weighted_score": ws(n4_raw, n4_sw, AW_N)},
        {"id": "Q1", "source": "quant",       "raw": q1_raw, "signal_weight": q1_sw,        "analyst_weight": AW_Q, "weighted_score": ws(q1_raw, q1_sw, AW_Q)},
        {"id": "Q2", "source": "quant",       "raw": q2_raw, "signal_weight": round(q2_sw, 4), "analyst_weight": AW_Q, "weighted_score": ws(q2_raw, q2_sw, AW_Q)},
    ]

    composite = round(sum(s["weighted_score"] for s in signals), 4)
    decision  = score_to_signal(composite)

    # ── Step 4: Conflict checks ──────────────────────────────────────────────
    f_sigs = [s for s in signals if s["source"] == "fundamental"]
    n_sigs = [s for s in signals if s["source"] == "news"]
    q_sigs = [s for s in signals if s["source"] == "quant"]

    f_agg = aggregate(f_sigs)
    n_agg = aggregate(n_sigs)
    q_agg = aggregate(q_sigs)

    conflicts   = []
    reinforced  = False

    # Fund vs quant conflict: both F1 and F2 are HIGH-confidence SELL but Q1 is BUY
    if (f1_raw == "SELL" and str(fd["value"]["confidence"]).upper()  == "HIGH"
            and f2_raw == "SELL" and str(fd["growth"]["confidence"]).upper() == "HIGH"
            and q1_raw == "BUY"):
        conflicts.append("fundamentals-vs-quant")
        if decision == "BUY":
            decision = "HOLD"

    # Reinforce SELL: N4 SELL with confidence > 0.8 and fundamentals agree
    if n4_raw == "SELL" and n4_sw > 0.8 and f_agg == "SELL":
        reinforced = True

    # ── Step 5: Confidence ───────────────────────────────────────────────────
    agg_set = {f_agg, n_agg, q_agg}
    if len(agg_set) == 1 or reinforced:
        confidence = "HIGH"
    elif len(agg_set) == 2:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    if conflicts:
        confidence = "LOW"

    result = {
        "signals": signals,
        "composite_score": composite,
        "thresholds": {"buy": 0.15, "sell": -0.15},
        "analyst_aggregates": {
            "fundamental": f_agg,
            "news":        n_agg,
            "quant":       q_agg,
        },
        "conflicts":  conflicts,
        "signal":     decision,
        "confidence": confidence,
    }

    return result


def main():
    if len(sys.argv) < 2:
        print("Usage: score_trader.py <run_dir>", file=sys.stderr)
        sys.exit(1)

    run_dir = Path(sys.argv[1])

    try:
        fund  = load(run_dir / "fundamental-analyst.json")
        news  = load(run_dir / "financial-news-analyst.json")
        quant = load(run_dir / "quant-indicator-analyst.json")
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    result = compute(fund, news, quant)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
