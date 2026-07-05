#!/usr/bin/env python3
"""find_patterns.py — deterministic pattern-detection for the memory-review skill (issue #14).

`skills/memory-review/SKILL.md` is markdown-driven, like the other skills under `skills/` (see
`skills/quant/compute_indicators.py` / `skills/trader/score_trader.py` for the established split:
a plain Python script does the mechanical computation, the skill's live LLM step does the
reasoning and writes files with the `Read`/`Write` tools). This module is that mechanical half for
memory-review:

1. ``find_patterns`` — the "hard numbers" half of the issue's step 1/2 split ("pulls
   memory_get_statistics ... reasons about *why* the patterns exist"). Flags every
   ``(agent, ticker)`` cell in ``tradingagents.memory.stats.get_statistics``'s ``per_agent_ticker``
   output whose hit-rate diverges sharply from that same agent's hit-rate on *its other* tickers —
   e.g. "fundamental was wrong 7/9 times on TSLA [...] but right elsewhere" from the issue body is
   exactly this shape: one cell far below (or above) the agent's own baseline. This is a
   statistical *signal*, not a claim about causation — the "why" is left entirely to the skill's
   LLM step, fed by:
2. ``gather_context_rows`` — pulls the raw resolved ``decisions`` rows (``key_drivers``, ``thesis``,
   ``lesson``) for a flagged ``(agent, ticker)`` pair, the material the issue calls out
   explicitly ("this is where the stored key_drivers/thesis context pays off").
3. ``format_findings_markdown`` — renders findings + gathered context as a single prompt-ready
   markdown block, mirroring the shape of ``stats.format_statistics_markdown`` /
   ``query.get_past_context`` (always well-formed, safe to inject unconditionally).
4. ``default_heuristic_line`` / ``draft_heuristics_markdown`` / ``write_heuristics_md`` — a
   mechanical fallback "distillation" plus the file-write helper. The issue's ExpeL-style ask is
   for a *curated* heuristic grounded in the "why" reasoning — that reasoning only happens live, in
   the skill's LLM step, once it has read ``format_findings_markdown``'s context rows. The
   mechanical `default_heuristic_line` exists so (a) the pattern-detection pipeline is fully
   testable end-to-end without an LLM in the loop (this repo's tests never call live LLM APIs —
   see ``tests/conftest.py`` / CLAUDE.md), and (b) a heuristics.md always has *some* well-formed
   content even before a human/LLM curation pass improves the wording — the skill is expected to
   overwrite these lines with real root-cause reasoning, not just accept the template verbatim.

Design choices:

- **Divergence definition**: for a given ``(agent, ticker)`` cell with ``n >= min_n`` resolved
  decisions, compare its hit-rate against the same agent's weighted hit-rate on every *other*
  ticker it has covered ("elsewhere"), provided that baseline also has ``n >= min_n`` decisions
  backing it (otherwise there's nothing to compare against). A cell is flagged if the absolute gap
  is at least ``divergence_threshold``. This directly matches the issue's framing ("wrong 7/9 times
  on TSLA ... right elsewhere") — the comparison is always "this ticker vs. this same agent's other
  tickers", never across agents (an agent's own confidence/skill baseline is the only fair
  reference point for "is this ticker special for this agent").
- **``min_n`` / ``divergence_threshold`` defaults** (3 resolved decisions, 30 percentage points):
  kept as documented module-level constants (same precedent as ``stats.HOLD_CORRECT_THRESHOLD``)
  rather than unstated magic numbers. 3 is the smallest sample size where a hit-rate is more than a
  single coin flip; 30 points is a deliberately high bar so routine noise doesn't get reported as a
  "pattern" worth an agent's finite heuristics budget (the issue asks for "short, curated"
  heuristics.md files, not an exhaustive stats dump).
- **No ticker normalization**: ``gather_context_rows`` matches ``ticker`` verbatim against the
  stored column, no call to ``symbol_utils.normalize_symbol`` — same precedent and same rationale
  as ``stats.py``/``query.py`` (see their module docstrings): normalization is a vendor-lookup /
  cross-alias concern that doesn't apply to matching already-stored rows.
- **Heuristics file location**: ``runs/memory/heuristics/<agent>.md``, sibling to the memory DB's
  own default (``runs/memory/memory.db``, see ``store.DEFAULT_DB_PATH``), overridable via the
  ``TRADINGAGENTS_HEURISTICS_DIR`` env var — same precedence pattern (explicit arg > env var >
  built-in default) as ``store.resolve_db_path``. One file per agent (not per ticker/date) because
  a heuristic is a lesson about how to read *that agent's* signals in general, meant to be injected
  into every future run of that agent, not scoped to a single decision.
- **Overwrite, not append, in ``write_heuristics_md``**: the issue asks for a "short, curated"
  file, i.e. a maintained document that gets rewritten as understanding improves — not an
  ever-growing decision log (that's what the ``decisions`` table already is). Callers own producing
  the full desired body (including preserving any existing curated lines they want to keep) before
  calling this.
- **Determinism**: like ``stats.py``/``query.py``, this module makes no LLM calls and no network
  I/O — a SQL read plus arithmetic plus deterministic string formatting/file write.
"""

from __future__ import annotations

import argparse
import json as json_module
import os
import sys
from pathlib import Path
from typing import Any

# Unlike the other skills/*.py scripts (compute_indicators.py, score_trader.py, ...), this module
# intentionally depends on tradingagents.memory — it needs the exact same schema/correctness-rule
# knowledge as the statistics module rather than duplicating it (see module docstring). Those
# scripts run standalone with no such dependency; this one runs against this repo's own memory
# core, so make sure the repo root (two levels up from skills/memory-review/) is importable even
# when this file is executed directly (``python skills/memory-review/find_patterns.py``), since
# Python only auto-adds the script's own directory to sys.path, not the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tradingagents.memory.stats import _is_correct, _normalize_signal, get_statistics  # noqa: E402
from tradingagents.memory.store import get_connection  # noqa: E402

# --- Pattern-detection knobs (see module docstring) -------------------------
DEFAULT_MIN_SAMPLE = 3
DEFAULT_DIVERGENCE_THRESHOLD = 0.3  # 30 percentage points

# --- Context-gathering default ----------------------------------------------
DEFAULT_CONTEXT_LIMIT = 10

# --- Heuristics file location (see module docstring) -------------------------
DEFAULT_HEURISTICS_DIR = Path("runs") / "memory" / "heuristics"
_HEURISTICS_ENV_VAR = "TRADINGAGENTS_HEURISTICS_DIR"


def resolve_heuristics_dir(heuristics_dir: str | Path | None = None) -> Path:
    """Resolve the effective heuristics directory.

    Precedence: explicit ``heuristics_dir`` argument > ``TRADINGAGENTS_HEURISTICS_DIR`` env var >
    built-in default (``runs/memory/heuristics``, relative to cwd) — mirrors
    ``tradingagents.memory.store.resolve_db_path``.
    """
    if heuristics_dir is not None:
        return Path(heuristics_dir).expanduser()
    env_value = os.environ.get(_HEURISTICS_ENV_VAR)
    if env_value:
        return Path(env_value).expanduser()
    return DEFAULT_HEURISTICS_DIR


def heuristics_path(agent: str, heuristics_dir: str | Path | None = None) -> Path:
    """Path to one agent's curated heuristics file: ``<heuristics_dir>/<agent>.md``."""
    return resolve_heuristics_dir(heuristics_dir) / f"{agent}.md"


# ---------------------------------------------------------------------------
# Pattern detection
# ---------------------------------------------------------------------------


def find_patterns(
    agent: str | None = None,
    ticker: str | None = None,
    since: str | None = None,
    min_n: int = DEFAULT_MIN_SAMPLE,
    divergence_threshold: float = DEFAULT_DIVERGENCE_THRESHOLD,
    db_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Flag ``(agent, ticker)`` cells whose hit-rate diverges from that agent's other tickers.

    Built on top of ``tradingagents.memory.stats.get_statistics`` — this function adds no new SQL,
    it re-derives "this ticker's baseline" from the same ``per_agent_ticker`` breakdown
    ``get_statistics`` already computes.

    Args:
        agent/ticker/since/db_path: forwarded to ``get_statistics`` — same filter semantics
            (exact-match ticker, inclusive ``since`` lower bound; see ``stats.py``).
        min_n: minimum resolved-decision count for a cell to be eligible, and separately for its
            "elsewhere" baseline to be eligible as a comparison point.
        divergence_threshold: minimum absolute hit-rate gap (0.0-1.0) between a cell and its
            agent's elsewhere-baseline to be flagged.

    Returns:
        A list of finding dicts, sorted by descending ``abs(delta)`` (most divergent first), each:
        ``{"agent", "ticker", "n", "hit_rate", "baseline_hit_rate", "baseline_n", "delta",
           "direction"}`` where ``delta = baseline_hit_rate - hit_rate`` (positive ->
        ``direction="underperforms"``: this ticker is worse than the agent's baseline; negative ->
        ``direction="overperforms"``).
    """
    stats = get_statistics(agent=agent, ticker=ticker, since=since, db_path=db_path)

    by_agent: dict[str, list[dict[str, Any]]] = {}
    for cell in stats["per_agent_ticker"]:
        by_agent.setdefault(cell["agent"], []).append(cell)

    findings: list[dict[str, Any]] = []
    for agent_id, cells in by_agent.items():
        for cell in cells:
            if cell["n"] < min_n or cell["hit_rate"] is None:
                continue
            others = [c for c in cells if c["ticker"] != cell["ticker"]]
            n_elsewhere = sum(c["n"] for c in others)
            if n_elsewhere < min_n:
                continue  # not enough of a baseline to compare against
            hits_elsewhere = sum(round(c["hit_rate"] * c["n"]) for c in others)
            baseline_hit_rate = hits_elsewhere / n_elsewhere
            delta = baseline_hit_rate - cell["hit_rate"]
            if abs(delta) < divergence_threshold:
                continue
            findings.append(
                {
                    "agent": agent_id,
                    "ticker": cell["ticker"],
                    "n": cell["n"],
                    "hit_rate": cell["hit_rate"],
                    "baseline_hit_rate": baseline_hit_rate,
                    "baseline_n": n_elsewhere,
                    "delta": delta,
                    "direction": "underperforms" if delta > 0 else "overperforms",
                }
            )

    findings.sort(key=lambda f: (-abs(f["delta"]), f["agent"], f["ticker"]))
    return findings


# ---------------------------------------------------------------------------
# Context gathering — the "why" material (issue step 2)
# ---------------------------------------------------------------------------


def gather_context_rows(
    agent: str,
    ticker: str,
    db_path: str | Path | None = None,
    limit: int = DEFAULT_CONTEXT_LIMIT,
    misses_only: bool = False,
) -> list[dict[str, Any]]:
    """Fetch resolved decision rows for ``(agent, ticker)`` with their key_drivers/thesis/lesson.

    This is the raw material the memory-review skill's LLM step reasons over to explain *why* a
    pattern flagged by ``find_patterns`` exists. Reuses ``stats._normalize_signal`` /
    ``stats._is_correct`` (rather than re-implementing the HOLD-threshold correctness rule) so
    "correct"/"miss" here is defined identically to how ``find_patterns``'s underlying statistics
    were computed.

    Args:
        agent/ticker: exact-match filters (no ticker normalization — see module docstring).
        limit: maximum rows returned, most recent (``decision_date`` DESC, ``id`` DESC) first —
            same ordering convention as ``query.get_past_context``.
        misses_only: if True, only rows scored "incorrect" are returned (useful once a finding's
            direction is "underperforms" and the reasoning step only cares about the misses).

    Returns:
        A list of dicts: ``{"decision_date", "signal", "confidence", "key_drivers" (parsed from
        JSON, or None), "thesis", "lesson", "forward_return", "correct"}``. ``correct`` is ``None``
        if the row's signal doesn't normalize to BUY/HOLD/SELL (see ``stats._normalize_signal``).
    """
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            """
            SELECT decision_date, signal, confidence, key_drivers, thesis, lesson, forward_return
            FROM decisions
            WHERE resolved_at IS NOT NULL AND agent = ? AND ticker = ?
            ORDER BY decision_date DESC, id DESC
            LIMIT ?
            """,
            (agent, ticker, limit),
        ).fetchall()
    finally:
        conn.close()

    out: list[dict[str, Any]] = []
    for row in rows:
        norm_signal = _normalize_signal(row["signal"])
        correct: bool | None = None
        if norm_signal is not None and row["forward_return"] is not None:
            correct = _is_correct(norm_signal, row["forward_return"])
        if misses_only and correct is not False:
            continue
        out.append(
            {
                "decision_date": row["decision_date"],
                "signal": row["signal"],
                "confidence": row["confidence"],
                "key_drivers": json_module.loads(row["key_drivers"]) if row["key_drivers"] else None,
                "thesis": row["thesis"],
                "lesson": row["lesson"],
                "forward_return": row["forward_return"],
                "correct": correct,
            }
        )
    return out


def _context_key(finding: dict[str, Any]) -> tuple[str, str]:
    return (finding["agent"], finding["ticker"])


def gather_all_context(
    findings: list[dict[str, Any]],
    db_path: str | Path | None = None,
    limit: int = DEFAULT_CONTEXT_LIMIT,
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Convenience: ``gather_context_rows`` for every ``(agent, ticker)`` in ``findings``."""
    return {
        _context_key(f): gather_context_rows(f["agent"], f["ticker"], db_path=db_path, limit=limit)
        for f in findings
    }


# ---------------------------------------------------------------------------
# Markdown rendering — findings block (prompt-ready, for the skill's reasoning step)
# ---------------------------------------------------------------------------

_NO_FINDINGS_MESSAGE = "*No divergent per-(agent, ticker) patterns found.*"


def format_findings_markdown(
    findings: list[dict[str, Any]],
    context_by_key: dict[tuple[str, str], list[dict[str, Any]]] | None = None,
) -> str:
    """Render ``find_patterns`` output (+ optional gathered context) as prompt-ready markdown.

    Always returns a well-formed, non-empty markdown string — safe to inject into a prompt
    unconditionally, same convention as ``stats.format_statistics_markdown`` /
    ``query.get_past_context``.
    """
    if not findings:
        return "## Memory review findings\n\n" + _NO_FINDINGS_MESSAGE

    parts: list[str] = ["## Memory review findings"]
    for finding in findings:
        hits = round(finding["hit_rate"] * finding["n"])
        pct = f"{finding['hit_rate'] * 100:.0f}%"
        baseline_pct = f"{finding['baseline_hit_rate'] * 100:.0f}%"
        parts.append(
            f"### {finding['agent']} × {finding['ticker']}: "
            f"{finding['direction']} baseline "
            f"({hits}/{finding['n']} correct, {pct} vs {baseline_pct} elsewhere on "
            f"{finding['baseline_n']} other decisions)"
        )
        rows = (context_by_key or {}).get(_context_key(finding), [])
        if rows:
            lines = []
            for row in rows:
                mark = "MISS" if row["correct"] is False else ("HIT" if row["correct"] else "?")
                detail = row["thesis"] or row["lesson"] or ""
                lines.append(
                    f"- [{mark}] {row['decision_date']} | {row['signal']} "
                    f"(confidence {row['confidence']}): {detail}"
                )
            parts.append("\n".join(lines))

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Heuristics distillation (mechanical fallback) + persistence
# ---------------------------------------------------------------------------


def default_heuristic_line(finding: dict[str, Any]) -> str:
    """Mechanically-derived, one-line draft heuristic for a finding — see module docstring.

    This is a template, not a substitute for the skill's live reasoning about *why* the pattern
    exists (which needs ``gather_context_rows``' key_drivers/thesis). It exists so the pipeline is
    testable without an LLM and so a heuristics.md always has well-formed content even before a
    curation pass improves the wording.
    """
    return (
        f"- **{finding['ticker']}**: {finding['agent']} {finding['direction']} its own baseline "
        f"on this ticker ({finding['hit_rate'] * 100:.0f}% vs "
        f"{finding['baseline_hit_rate'] * 100:.0f}% elsewhere, n={finding['n']}) — investigate "
        f"root cause before trusting this agent's signal on {finding['ticker']} at face value."
    )


def draft_heuristics_markdown(agent: str, findings: list[dict[str, Any]]) -> str:
    """Render a full draft heuristics.md body for one agent from ``find_patterns`` output.

    Mechanical distillation only (see ``default_heuristic_line``) — the memory-review skill is
    expected to start from this, reason about *why* each pattern exists using
    ``gather_context_rows``, and rewrite these lines into genuinely curated, ExpeL-style
    heuristics before persisting the final version via ``write_heuristics_md``.
    """
    agent_findings = [f for f in findings if f["agent"] == agent]
    lines = [
        f"# Heuristics: {agent}",
        "",
        "*Distilled from tradingagents/memory statistics "
        "— see skills/memory-review/SKILL.md.*",
        "",
    ]
    if not agent_findings:
        lines.append("*No divergent patterns found yet.*")
    else:
        lines.extend(default_heuristic_line(f) for f in agent_findings)
    return "\n".join(lines) + "\n"


def write_heuristics_md(
    agent: str,
    body_markdown: str,
    heuristics_dir: str | Path | None = None,
) -> Path:
    """Persist one agent's curated heuristics file (overwrite, not append — see module docstring).

    Thin persistence helper — creates the heuristics directory if needed and writes
    ``body_markdown`` verbatim to ``heuristics_path(agent, heuristics_dir)``. No formatting logic
    of its own: callers (the memory-review skill, live, or ``draft_heuristics_markdown`` for the
    mechanical fallback) own producing well-formed markdown first.
    """
    path = heuristics_path(agent, heuristics_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body_markdown, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# CLI entry point: python skills/memory-review/find_patterns.py
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="find_patterns.py",
        description=(
            "Detect divergent per-(agent, ticker) hit-rate patterns in the shared memory core "
            "and render them as a prompt-ready findings block."
        ),
    )
    parser.add_argument("--agent", default=None, help="Restrict to this exact agent id.")
    parser.add_argument("--ticker", default=None, help="Restrict to this exact ticker (un-normalized).")
    parser.add_argument("--since", default=None, help="Inclusive lower bound on decision_date (YYYY-MM-DD).")
    parser.add_argument("--db-path", default=None, help="Override the memory DB path.")
    parser.add_argument("--min-n", type=int, default=DEFAULT_MIN_SAMPLE, help="Minimum sample size per cell.")
    parser.add_argument(
        "--divergence", type=float, default=DEFAULT_DIVERGENCE_THRESHOLD,
        help="Minimum absolute hit-rate gap (0.0-1.0) to flag a cell.",
    )
    parser.add_argument(
        "--context-limit", type=int, default=DEFAULT_CONTEXT_LIMIT,
        help="Max context rows gathered per flagged (agent, ticker).",
    )
    parser.add_argument(
        "--write-draft", action="store_true",
        help="Also write a mechanical draft heuristics.md per flagged agent (for the skill's "
        "LLM step to then curate).",
    )
    parser.add_argument("--heuristics-dir", default=None, help="Override the heuristics directory.")
    parser.add_argument("--json", action="store_true", help="Print raw findings/context as JSON instead of markdown.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    findings = find_patterns(
        agent=args.agent,
        ticker=args.ticker,
        since=args.since,
        min_n=args.min_n,
        divergence_threshold=args.divergence,
        db_path=args.db_path,
    )
    context_by_key = gather_all_context(findings, db_path=args.db_path, limit=args.context_limit)

    if args.write_draft:
        agents = sorted({f["agent"] for f in findings})
        for agent_id in agents:
            write_heuristics_md(
                agent_id,
                draft_heuristics_markdown(agent_id, findings),
                heuristics_dir=args.heuristics_dir,
            )

    if args.json:
        printable = {
            "findings": findings,
            "context": {f"{a}::{t}": rows for (a, t), rows in context_by_key.items()},
        }
        print(json_module.dumps(printable, indent=2))
    else:
        print(format_findings_markdown(findings, context_by_key))


if __name__ == "__main__":
    main()
