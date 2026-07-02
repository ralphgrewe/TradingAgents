"""SQLite-backed shared memory core — statistics path (issue #23).

Computes "hard statistics" (hit-rates, average forward returns, confidence
calibration) over resolved ``decisions`` rows (written by ``store_decision``,
issue #5, and resolved by ``resolve_pending``, issue #6). This is the
factual backbone for the periodic-review step (issue #8): *that* a given
agent was wrong 7/9 times on a ticker is computed here; *why* is worked out
there. Also exposes a small CLI (``python -m tradingagents.memory.stats``)
so stats can be checked by hand between runs.

Design choices (mirrors the shape/conventions of ``query.py``, issue #7,
since both modules need a "plain dict for scripts/tests" + "prompt-ready
markdown" dual output):

- **Resolved rows only**: every query filters on ``resolved_at IS NOT NULL``,
  same precedent as ``get_past_context`` — an unresolved decision has no
  forward return yet, so there is nothing to score it against.
- **Ticker matching is exact-string, not normalized**: the ``ticker=``
  filter is matched against the stored ``decisions.ticker`` column as-is,
  with no call to ``dataflows.symbol_utils.normalize_symbol``. This mirrors
  both ``resolve_pending``'s and ``get_past_context``'s own ``ticker=``
  filters (issues #6/#7), which are exact-match for the same reason
  documented in their module docstrings — keeping matching semantics
  consistent across the read paths rather than introducing a third
  normalization behavior. Callers that need alias-aware matching should
  normalize their own ``ticker`` argument before calling.
- **``since`` filter**: an inclusive lower bound on ``decision_date``
  (``decision_date >= since``), compared as a plain ISO ``YYYY-MM-DD``
  string — same direct-string-comparison approach ``resolve.py`` already
  uses for date ordering (``as_of <= decision_date``), which works
  correctly for ISO 8601 dates without needing to parse them.
- **Signal normalization for scoring**: ``decisions.signal`` is free-form
  text (see ``store.py`` docstring) because two different signal
  vocabularies are in use elsewhere in this repo — the 3-tier
  ``BUY/HOLD/SELL`` used by the ``skills/`` envelope (``skills/SCHEMA.md``)
  and the 5-tier ``Buy/Overweight/Hold/Underweight/Sell`` scale used by the
  legacy graph (``tradingagents/agents/utils/rating.py``). This module
  normalizes both onto the 3-tier scale for scoring purposes — Buy/
  Overweight -> ``BUY``, Sell/Underweight -> ``SELL``, Hold -> ``HOLD``
  (case-insensitive) — since "was this bullish/neutral/bearish call right"
  is the same question regardless of which vocabulary produced it, and the
  issue explicitly asks for "average forward return per signal
  (BUY/HOLD/SELL)". Rows whose ``signal`` does not match any of the six
  known spellings are **excluded** from every statistic (not counted in any
  sample size) rather than guessed at — a good statistics module should not
  silently misclassify an unrecognized label.
- **Hit-rate / correctness definition** (the one genuinely open design
  choice the issue leaves to this module, called out explicitly in the
  issue body — "define HOLD handling explicitly"):

  - ``BUY``-normalized: correct iff ``forward_return > 0``.
  - ``SELL``-normalized: correct iff ``forward_return < 0``.
  - ``HOLD``-normalized: correct iff ``abs(forward_return) <= HOLD_CORRECT_THRESHOLD``
    (a "small move" band around zero — a HOLD call is a bet that the price
    won't move much either way, not a bet on the exact sign of a near-zero
    return).
  - ``HOLD_CORRECT_THRESHOLD = 0.02`` (2%) is a module-level constant, kept
    as a single documented knob (matching the ``DEFAULT_HORIZON_DAYS``
    precedent in ``resolve.py``) rather than an unstated magic number.
    There is no existing "small return" constant anywhere else in this repo
    to inherit; 2% over the ``DEFAULT_HORIZON_DAYS`` (10 trading day, ~2
    calendar week) resolution window is a conventional "noise band" size for
    a single-name equity move and errs toward a stricter bar for HOLD than
    for BUY/SELL (exact zero also counts as *not* correct for BUY/SELL,
    which is consistent — a flat return validates neither a bullish nor a
    bearish call).
- **Confidence bucket boundaries**: ``decisions.confidence`` is a numeric
  ``REAL`` column (0.0-1.0, see ``store.py`` docstring), while the
  ``skills/`` envelope's categorical confidence is ``HIGH/MEDIUM/LOW``
  (``skills/SCHEMA.md``). This module buckets the numeric column onto that
  same three-level scale rather than inventing a fourth vocabulary. Boundary
  values are derived from the one existing numeric<->categorical mapping in
  this repo, ``skills/trader/score_trader.py``'s ``conf_weight``
  (``HIGH -> 1.0``, ``MEDIUM -> 0.6``, ``LOW -> 0.3``): bucket boundaries
  are placed at the midpoints between those anchors, i.e.
  ``LOW: confidence < 0.45``, ``MEDIUM: 0.45 <= confidence < 0.8``,
  ``HIGH: confidence >= 0.8``. Rows with ``confidence IS NULL`` are excluded
  from the calibration table (there is nothing to bucket).
- **Grouping**: per the issue body, hit-rate/average-return statistics are
  reported per ``(agent, ticker)`` and per ``agent`` overall (across all its
  tickers); calibration is reported per ``agent`` per confidence bucket
  (calibration is a property of how an agent uses its own confidence scale,
  so it is not meaningful to mix agents into one bucket).
- **Determinism**: like ``query.py``, this module makes no LLM calls and no
  network I/O — a SQL read plus arithmetic plus deterministic string
  formatting.
"""

from __future__ import annotations

import argparse
import json as json_module
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from tradingagents.memory.store import get_connection

# "Small move" band for HOLD correctness — see module docstring.
HOLD_CORRECT_THRESHOLD = 0.02

# Confidence-bucket boundaries — see module docstring for how these were
# derived from skills/trader/score_trader.py's conf_weight anchors.
_CONFIDENCE_LOW_MAX = 0.45  # confidence < this -> LOW
_CONFIDENCE_MEDIUM_MAX = 0.8  # this <= confidence < ... handled below -> MEDIUM; else HIGH

# Ordered 3-tier scale used for scoring/reporting (independent of which of
# the two input vocabularies a given row's raw ``signal`` was written in).
SIGNALS = ("BUY", "HOLD", "SELL")

# Confidence buckets, ordered high -> low for markdown rendering.
CONFIDENCE_BUCKETS = ("HIGH", "MEDIUM", "LOW")

_BULLISH_SIGNALS = {"buy", "overweight"}
_BEARISH_SIGNALS = {"sell", "underweight"}
_NEUTRAL_SIGNALS = {"hold"}


def _normalize_signal(signal: Optional[str]) -> Optional[str]:
    """Map a raw ``decisions.signal`` value onto ``BUY``/``HOLD``/``SELL``.

    Case-insensitive; accepts both the 3-tier skills vocabulary
    (``BUY``/``HOLD``/``SELL``) and the legacy 5-tier scale
    (``Buy``/``Overweight``/``Hold``/``Underweight``/``Sell``). Returns
    ``None`` for any other value (row is excluded from statistics — see
    module docstring).
    """
    if signal is None:
        return None
    s = signal.strip().lower()
    if s in _BULLISH_SIGNALS:
        return "BUY"
    if s in _BEARISH_SIGNALS:
        return "SELL"
    if s in _NEUTRAL_SIGNALS:
        return "HOLD"
    return None


def _is_correct(norm_signal: str, forward_return: float) -> bool:
    """Whether a normalized signal was "correct" given a realized forward return.

    See module docstring for the exact BUY/HOLD/SELL correctness rules.
    """
    if norm_signal == "BUY":
        return forward_return > 0
    if norm_signal == "SELL":
        return forward_return < 0
    # HOLD
    return abs(forward_return) <= HOLD_CORRECT_THRESHOLD


def _confidence_bucket(confidence: Optional[float]) -> Optional[str]:
    """Bucket a numeric ``confidence`` value onto HIGH/MEDIUM/LOW, or ``None``."""
    if confidence is None:
        return None
    if confidence < _CONFIDENCE_LOW_MAX:
        return "LOW"
    if confidence < _CONFIDENCE_MEDIUM_MAX:
        return "MEDIUM"
    return "HIGH"


def _empty_signal_breakdown() -> Dict[str, Dict[str, Any]]:
    return {
        sig: {"n": 0, "hit_rate": None, "avg_forward_return": None} for sig in SIGNALS
    }


def _summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate a list of scored rows (each with ``norm_signal``/``correct``/
    ``forward_return``) into ``{n, hit_rate, by_signal}``.
    """
    by_signal_rows: Dict[str, List[Dict[str, Any]]] = {sig: [] for sig in SIGNALS}
    for row in rows:
        by_signal_rows[row["norm_signal"]].append(row)

    by_signal = _empty_signal_breakdown()
    for sig, sig_rows in by_signal_rows.items():
        if not sig_rows:
            continue
        n = len(sig_rows)
        hits = sum(1 for r in sig_rows if r["correct"])
        avg_return = sum(r["forward_return"] for r in sig_rows) / n
        by_signal[sig] = {
            "n": n,
            "hit_rate": hits / n,
            "avg_forward_return": avg_return,
        }

    n_total = len(rows)
    hit_rate = (sum(1 for r in rows if r["correct"]) / n_total) if n_total else None
    return {"n": n_total, "hit_rate": hit_rate, "by_signal": by_signal}


def get_statistics(
    agent: Optional[str] = None,
    ticker: Optional[str] = None,
    since: Optional[str] = None,
    db_path: Union[str, Path, None] = None,
) -> Dict[str, Any]:
    """Compute hit-rate / average-return / calibration statistics from resolved decisions.

    Only resolved rows (``resolved_at IS NOT NULL``) are considered. Rows
    whose ``signal`` does not normalize to one of BUY/HOLD/SELL (see
    ``_normalize_signal``) are excluded from every statistic.

    Args:
        agent: If given, restrict to this exact agent identifier.
        ticker: If given, restrict to this exact (un-normalized) ticker
            string — matched verbatim, consistent with ``resolve_pending``
            and ``get_past_context`` (see module docstring).
        since: If given, an inclusive lower bound on ``decision_date``
            (``YYYY-MM-DD``, compared as a string).
        db_path: Optional override for the DB path (see
            ``tradingagents.memory.store.resolve_db_path``).

    Returns:
        A plain, JSON-serializable dict:

        ``{"filters": {"agent", "ticker", "since"},
           "per_agent_ticker": [{"agent", "ticker", "n", "hit_rate",
                                  "by_signal": {"BUY"/"HOLD"/"SELL":
                                      {"n", "hit_rate", "avg_forward_return"}}}, ...],
           "per_agent": [{"agent", "n", "hit_rate", "by_signal": {...}}, ...],
           "calibration": [{"agent", "bucket", "n", "hit_rate"}, ...]}``

        Lists are sorted for deterministic output: ``per_agent_ticker`` by
        ``(agent, ticker)``; ``per_agent`` by ``agent``; ``calibration`` by
        ``(agent, bucket)`` with bucket ordered HIGH, MEDIUM, LOW.
    """
    conn = get_connection(db_path)
    try:
        query = (
            "SELECT agent, ticker, signal, confidence, forward_return, decision_date "
            "FROM decisions WHERE resolved_at IS NOT NULL"
        )
        params: list[Any] = []
        if agent is not None:
            query += " AND agent = ?"
            params.append(agent)
        if ticker is not None:
            query += " AND ticker = ?"
            params.append(ticker)
        if since is not None:
            query += " AND decision_date >= ?"
            params.append(since)
        raw_rows = conn.execute(query, params).fetchall()
    finally:
        conn.close()

    scored_rows: List[Dict[str, Any]] = []
    for row in raw_rows:
        norm_signal = _normalize_signal(row["signal"])
        if norm_signal is None:
            continue  # unrecognized signal vocabulary — excluded, see docstring
        forward_return = row["forward_return"]
        if forward_return is None:
            continue  # defensive: resolved rows should always have this set
        scored_rows.append(
            {
                "agent": row["agent"],
                "ticker": row["ticker"],
                "norm_signal": norm_signal,
                "forward_return": forward_return,
                "correct": _is_correct(norm_signal, forward_return),
                "confidence_bucket": _confidence_bucket(row["confidence"]),
            }
        )

    # --- per (agent, ticker) ---
    by_agent_ticker_rows: Dict[tuple, List[Dict[str, Any]]] = {}
    for row in scored_rows:
        key = (row["agent"], row["ticker"])
        by_agent_ticker_rows.setdefault(key, []).append(row)

    per_agent_ticker = [
        {"agent": a, "ticker": t, **_summarize(rows)}
        for (a, t), rows in sorted(by_agent_ticker_rows.items())
    ]

    # --- per agent overall ---
    by_agent_rows: Dict[str, List[Dict[str, Any]]] = {}
    for row in scored_rows:
        by_agent_rows.setdefault(row["agent"], []).append(row)

    per_agent = [
        {"agent": a, **_summarize(rows)} for a, rows in sorted(by_agent_rows.items())
    ]

    # --- calibration: per agent, per confidence bucket ---
    by_agent_bucket_rows: Dict[tuple, List[Dict[str, Any]]] = {}
    for row in scored_rows:
        bucket = row["confidence_bucket"]
        if bucket is None:
            continue  # no confidence recorded — excluded from calibration
        key = (row["agent"], bucket)
        by_agent_bucket_rows.setdefault(key, []).append(row)

    bucket_order = {b: i for i, b in enumerate(CONFIDENCE_BUCKETS)}
    calibration = [
        {
            "agent": a,
            "bucket": bucket,
            "n": len(rows),
            "hit_rate": sum(1 for r in rows if r["correct"]) / len(rows),
        }
        for (a, bucket), rows in sorted(
            by_agent_bucket_rows.items(), key=lambda kv: (kv[0][0], bucket_order[kv[0][1]])
        )
    ]

    return {
        "filters": {"agent": agent, "ticker": ticker, "since": since},
        "per_agent_ticker": per_agent_ticker,
        "per_agent": per_agent,
        "calibration": calibration,
    }


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

_NO_DATA_MESSAGE = "*No resolved decisions match these filters yet.*"


def _fmt_pct(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _fmt_return(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value * 100:+.2f}%"


def _hit_rate_table_rows(records: List[Dict[str, Any]], key_cols: List[str]) -> List[str]:
    lines = []
    for rec in records:
        key_values = [str(rec[k]) for k in key_cols]
        by_signal = rec["by_signal"]
        signal_cells = [_fmt_return(by_signal[sig]["avg_forward_return"]) for sig in SIGNALS]
        lines.append(
            "| "
            + " | ".join(key_values)
            + f" | {rec['n']} | {_fmt_pct(rec['hit_rate'])} | "
            + " | ".join(signal_cells)
            + " |"
        )
    return lines


def format_statistics_markdown(stats: Dict[str, Any]) -> str:
    """Render the ``get_statistics`` dict as a prompt-ready markdown block.

    Always returns a well-formed, non-empty markdown string — safe to
    inject into a prompt unconditionally, mirroring ``get_past_context``'s
    "no matching rows" handling (issue #7).
    """
    filters = stats["filters"]
    filter_desc = ", ".join(
        f"{k}={v if v is not None else 'any'}" for k, v in filters.items()
    )
    parts: List[str] = ["## Memory statistics", f"**Filters:** {filter_desc}"]

    if not stats["per_agent_ticker"] and not stats["per_agent"]:
        parts.append(_NO_DATA_MESSAGE)
        return "\n\n".join(parts)

    signal_header = " | ".join(f"Avg return ({sig})" for sig in SIGNALS)

    if stats["per_agent_ticker"]:
        parts.append("### Hit-rate by agent x ticker")
        header = f"| Agent | Ticker | N | Hit-rate | {signal_header} |"
        sep = "|---|---|---|---|" + "---|" * len(SIGNALS)
        rows = _hit_rate_table_rows(stats["per_agent_ticker"], ["agent", "ticker"])
        parts.append("\n".join([header, sep, *rows]))

    if stats["per_agent"]:
        parts.append("### Hit-rate by agent (overall)")
        header = f"| Agent | N | Hit-rate | {signal_header} |"
        sep = "|---|---|---|" + "---|" * len(SIGNALS)
        rows = _hit_rate_table_rows(stats["per_agent"], ["agent"])
        parts.append("\n".join([header, sep, *rows]))

    if stats["calibration"]:
        parts.append("### Calibration by confidence bucket")
        header = "| Agent | Bucket | N | Hit-rate |"
        sep = "|---|---|---|---|"
        rows = [
            f"| {rec['agent']} | {rec['bucket']} | {rec['n']} | {_fmt_pct(rec['hit_rate'])} |"
            for rec in stats["calibration"]
        ]
        parts.append("\n".join([header, sep, *rows]))

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# CLI entry point: python -m tradingagents.memory.stats
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tradingagents.memory.stats",
        description=(
            "Compute hit-rate / average-return / calibration statistics from "
            "resolved memory-core decisions."
        ),
    )
    parser.add_argument("--agent", default=None, help="Restrict to this exact agent id.")
    parser.add_argument("--ticker", default=None, help="Restrict to this exact ticker (un-normalized).")
    parser.add_argument(
        "--since", default=None, help="Inclusive lower bound on decision_date (YYYY-MM-DD)."
    )
    parser.add_argument("--db-path", default=None, help="Override the memory DB path.")
    parser.add_argument(
        "--json", action="store_true", help="Print the raw dict as JSON instead of markdown."
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    stats = get_statistics(
        agent=args.agent, ticker=args.ticker, since=args.since, db_path=args.db_path
    )
    if args.json:
        print(json_module.dumps(stats, indent=2))
    else:
        print(format_statistics_markdown(stats))


if __name__ == "__main__":
    main()
