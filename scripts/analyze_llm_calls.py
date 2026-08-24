#!/usr/bin/env python3
"""Analyze LLM call logs and prompt dumps across runs (issue #143).

Aggregates every ``reports/*/llm_calls.jsonl`` (written by
``tradingagents/llm_call_log.py``, issue #138) and, when present, the
matching ``reports/*/prompts/*.json`` dumps (issue #139) into per-agent
statistics, a top-N largest-prompts listing, truncation-candidate detection,
and per-prompt segment composition. Read-only: this script never mutates the
``reports/`` tree.

Usage::

    ./venv/bin/python scripts/analyze_llm_calls.py
    ./venv/bin/python scripts/analyze_llm_calls.py --reports-dir reports --format json
    ./venv/bin/python scripts/analyze_llm_calls.py --format markdown --sort-by estimated_prompt_tokens_max --top 10
    ./venv/bin/python scripts/analyze_llm_calls.py --sort-by duration_total
    ./venv/bin/python scripts/analyze_llm_calls.py --segment --format json --top 5

Design notes (second pass, post design-review — see issue #143 comments):

- The chars/4 token estimator is imported from ``tradingagents.llm_call_log``
  (not reimplemented) so the two can never drift apart.
- Every ``CallRecord`` remembers the run directory it was loaded from
  (``run_dir``), so a prompt-dump path pooled from a multi-run corpus can
  still be resolved to an absolute, directly-openable path
  (``resolved_dump_path()``) without guessing which run a ticker belongs to.
- Rendering is a two-stage pipeline: ``build_report_data``/``build_segment_data``
  compute a plain-dict "report model" (sorting, capping, path resolution all
  happen exactly once, here), and the three ``render_*`` functions format that
  same dict as text, markdown, or JSON — so ``--format`` and ``--segment``
  compose freely instead of some combinations silently reusing another
  format's renderer.
- ``--sort-by`` is restricted to the actual keys ``PerAgentStats.compute_stats()``
  produces (``AGENT_SORT_KEYS``), so an unrecognized value is an argparse error
  at startup, not a silent no-op.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tradingagents.llm_call_log import _CHARS_PER_TOKEN_ESTIMATE, _TOKEN_COUNT_METHOD_HEURISTIC

# ─── Segment anchors (define in one place for easy extension) ──────────────


SEGMENT_ANCHORS = [
    "Past analyses of ",
    "Recent cross-ticker lessons:",
    "Analyst Reports:",
    "**Risk Analysts Debate History:**",
    "**Available Tools:**",
    # Add more anchors here as they are discovered in the corpus.
]


def _split_prompt_into_segments(prompt_text: str) -> list[tuple[str, str, int]]:
    """Split a prompt by anchor lines into (segment_name, content, char_count) tuples.

    Puts everything before the first anchor into a "header/instructions" segment.
    An empty prompt produces a single "empty_prompt" segment; an anchor-less
    prompt produces a single "full_prompt" segment covering the whole text.

    Returns:
        List of (segment_name, content, char_count) tuples, in prompt order.
    """
    if not prompt_text:
        return [("empty_prompt", "", 0)]

    anchors_found: list[tuple[int, str]] = []
    for anchor in SEGMENT_ANCHORS:
        pos = prompt_text.find(anchor)
        if pos != -1:
            anchors_found.append((pos, anchor))

    if not anchors_found:
        return [("full_prompt", prompt_text, len(prompt_text))]

    anchors_found.sort()
    segments: list[tuple[str, str, int]] = []

    first_anchor_pos = anchors_found[0][0]
    if first_anchor_pos > 0:
        header_text = prompt_text[:first_anchor_pos]
        segments.append(("header/instructions", header_text, len(header_text)))

    for i, (pos, anchor) in enumerate(anchors_found):
        segment_name = anchor.strip().rstrip(":").rstrip()
        if len(segment_name) > 40:
            segment_name = segment_name[:37] + "..."
        segment_name = segment_name.lower().replace(" ", "_")

        next_pos = anchors_found[i + 1][0] if i + 1 < len(anchors_found) else len(prompt_text)
        content = prompt_text[pos:next_pos]
        segments.append((segment_name, content, len(content)))

    return segments


# ─── Call record aggregation and statistics ────────────────────────────────


@dataclass
class CallRecord:
    """A single LLM call from an ``llm_calls.jsonl`` record.

    ``run_dir`` is the directory the record was loaded from (not part of the
    JSONL schema itself) — it is what lets ``resolved_dump_path()`` turn the
    JSONL's run-relative ``prompt_dump_path`` into a path that can be opened
    directly regardless of which run directory it came from.
    """

    timestamp: str
    run_id: str
    ticker: str
    date: str
    agent: str
    model: str
    message_count: int
    prompt_chars: int
    prompt_tokens_estimated: int
    token_count_method: str
    input_tokens: int | None
    output_tokens: int | None
    duration_seconds: float
    error: str | None
    prompt_dump_path: str | None
    run_dir: Path = field(default_factory=lambda: Path("."))

    @classmethod
    def from_jsonl_dict(cls, d: dict[str, Any], run_dir: Path = Path(".")) -> CallRecord:
        return cls(
            timestamp=d.get("timestamp", ""),
            run_id=d.get("run_id", ""),
            ticker=d.get("ticker", ""),
            date=d.get("date", ""),
            agent=d.get("agent", "unknown"),
            model=d.get("model", "unknown"),
            message_count=d.get("message_count", 0),
            prompt_chars=d.get("prompt_chars", 0),
            prompt_tokens_estimated=d.get("prompt_tokens_estimated", 0),
            # Pre-#147 records predate this field entirely; those numbers were
            # always the chars/4 heuristic, so a missing field defaults to it
            # (see tradingagents/llm_call_log.py's module docstring).
            token_count_method=d.get("token_count_method") or _TOKEN_COUNT_METHOD_HEURISTIC,
            input_tokens=d.get("input_tokens"),
            output_tokens=d.get("output_tokens"),
            duration_seconds=d.get("duration_seconds", 0.0),
            error=d.get("error"),
            prompt_dump_path=d.get("prompt_dump_path"),
            run_dir=run_dir,
        )

    def resolved_dump_path(self) -> Path | None:
        """Return an absolute, directly-openable path to this call's prompt dump.

        ``prompt_dump_path`` in the JSONL record is relative to its own run
        directory (e.g. ``"prompts/<run_id>.json"``); resolving it against
        ``self.run_dir`` (captured at load time, not re-derived later) is what
        makes Top-N/truncation listings openable once records from many runs
        are pooled, even when the same ticker has multiple run directories.
        Does not check the file actually exists — callers that need to open
        it should handle ``OSError``.
        """
        if not self.prompt_dump_path:
            return None
        p = Path(self.prompt_dump_path)
        if p.is_absolute():
            return p
        return (self.run_dir / p).resolve()


# Keys PerAgentStats.compute_stats() actually produces (besides "agent"),
# and therefore the only valid --sort-by values. Restricting to this list
# turns a typo'd/renamed key into an argparse error instead of a silent no-op.
AGENT_SORT_KEYS = (
    "call_count",
    "error_count",
    "estimated_prompt_tokens_mean",
    "estimated_prompt_tokens_median",
    "estimated_prompt_tokens_p95",
    "estimated_prompt_tokens_max",
    "reported_input_tokens_max",
    "output_tokens_total",
    "output_tokens_max",
    "duration_total",
    "duration_max",
)


@dataclass
class PerAgentStats:
    """Aggregated statistics for one agent across all calls."""

    agent: str
    call_count: int = 0
    error_count: int = 0
    estimated_prompt_tokens_list: list[int] = field(default_factory=list)
    reported_input_tokens_list: list[int] = field(default_factory=list)
    output_tokens_list: list[int] = field(default_factory=list)
    duration_list: list[float] = field(default_factory=list)
    total_wall_time: float = 0.0

    def add_call(self, record: CallRecord) -> None:
        self.call_count += 1
        self.estimated_prompt_tokens_list.append(record.prompt_tokens_estimated)
        if record.input_tokens is not None:
            self.reported_input_tokens_list.append(record.input_tokens)
        if record.output_tokens is not None:
            self.output_tokens_list.append(record.output_tokens)
        self.duration_list.append(record.duration_seconds)
        self.total_wall_time += record.duration_seconds
        if record.error is not None:
            self.error_count += 1

    def compute_stats(self) -> dict[str, Any]:
        """Compute derived statistics. Keys match ``AGENT_SORT_KEYS`` plus ``agent``/``call_count``/``error_count``."""
        stats: dict[str, Any] = {
            "agent": self.agent,
            "call_count": self.call_count,
            "error_count": self.error_count,
        }

        if self.estimated_prompt_tokens_list:
            sorted_tokens = sorted(self.estimated_prompt_tokens_list)
            stats["estimated_prompt_tokens_mean"] = statistics.mean(
                self.estimated_prompt_tokens_list
            )
            stats["estimated_prompt_tokens_median"] = statistics.median(sorted_tokens)
            stats["estimated_prompt_tokens_p95"] = (
                sorted_tokens[int(len(sorted_tokens) * 0.95)]
                if len(sorted_tokens) > 1
                else sorted_tokens[0]
            )
            stats["estimated_prompt_tokens_max"] = max(self.estimated_prompt_tokens_list)
        else:
            stats["estimated_prompt_tokens_mean"] = None
            stats["estimated_prompt_tokens_median"] = None
            stats["estimated_prompt_tokens_p95"] = None
            stats["estimated_prompt_tokens_max"] = 0

        if self.reported_input_tokens_list:
            stats["reported_input_tokens_max"] = max(self.reported_input_tokens_list)
        else:
            stats["reported_input_tokens_max"] = None

        if self.output_tokens_list:
            stats["output_tokens_total"] = sum(self.output_tokens_list)
            stats["output_tokens_max"] = max(self.output_tokens_list)
        else:
            stats["output_tokens_total"] = None
            stats["output_tokens_max"] = None

        if self.duration_list:
            stats["duration_total"] = sum(self.duration_list)
            stats["duration_max"] = max(self.duration_list)
        else:
            stats["duration_total"] = None
            stats["duration_max"] = None

        return stats


@dataclass
class CorpusAggregation:
    """Top-level aggregation across all run directories."""

    reports_dir: Path
    run_count: int = 0
    call_count: int = 0
    total_wall_time: float = 0.0
    model_counts: Counter[str] = field(default_factory=Counter)
    per_agent_stats: dict[str, PerAgentStats] = field(default_factory=dict)
    all_records: list[CallRecord] = field(default_factory=list)
    skipped_dirs: list[str] = field(default_factory=list)

    @property
    def models_seen(self) -> set[str]:
        return set(self.model_counts)


def load_llm_calls_from_directory(reports_dir: Path) -> CorpusAggregation:
    """Load all llm_calls.jsonl files from reports_dir and aggregate stats.

    Handles, without raising:
    - A nonexistent or empty ``reports_dir``.
    - A run directory with no ``llm_calls.jsonl`` (e.g. only ``prompts/``,
      the opt-in dump directory, present) — skipped and counted.
    - Malformed JSON lines within an otherwise valid ``llm_calls.jsonl`` —
      skipped and counted; other lines in the same file still load.
    """
    agg = CorpusAggregation(reports_dir=reports_dir)

    if not reports_dir.exists():
        return agg

    for run_dir in sorted(reports_dir.iterdir()):
        if not run_dir.is_dir() or run_dir.name.startswith("."):
            continue

        llm_calls_file = run_dir / "llm_calls.jsonl"
        if not llm_calls_file.exists():
            agg.skipped_dirs.append(run_dir.name)
            continue

        agg.run_count += 1

        try:
            with open(llm_calls_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record_dict = json.loads(line)
                    except json.JSONDecodeError as e:
                        agg.skipped_dirs.append(
                            f"{run_dir.name} (malformed JSON in llm_calls.jsonl: {e})"
                        )
                        continue
                    record = CallRecord.from_jsonl_dict(record_dict, run_dir=run_dir)
                    agg.all_records.append(record)
                    agg.call_count += 1
                    agg.total_wall_time += record.duration_seconds
                    agg.model_counts[record.model] += 1

                    if record.agent not in agg.per_agent_stats:
                        agg.per_agent_stats[record.agent] = PerAgentStats(agent=record.agent)
                    agg.per_agent_stats[record.agent].add_call(record)
        except OSError as e:
            agg.skipped_dirs.append(f"{run_dir.name} (read error: {e})")

    return agg


def _fmt_num(value: float | None, digits: int = 2) -> str:
    """Format a number for display, or 'n/a' if None."""
    return "n/a" if value is None else f"{value:.{digits}f}"


def _fmt_int(value: int | None) -> str:
    """Format an integer for display, or 'n/a' if None."""
    return "n/a" if value is None else str(value)


# ─── Truncation detection ────────────────────────────────────────────────────


@dataclass
class TruncationCandidate:
    """A call that may have had its prompt truncated."""

    agent: str
    run_dir: str
    run_id: str
    estimated_tokens: int
    reported_tokens: int | None
    ratio: float | None  # reported / estimated
    dump_path: str | None


def detect_truncation_candidates(
    records: list[CallRecord], threshold: float = 0.8
) -> list[TruncationCandidate]:
    """Flag calls where reported input_tokens << estimated prompt_tokens.

    A ratio < threshold suggests truncation. Returns candidates sorted by
    ratio ascending (most-truncated first).
    """
    candidates: list[TruncationCandidate] = []

    for record in records:
        if record.input_tokens is None or record.prompt_tokens_estimated == 0:
            continue

        ratio = record.input_tokens / record.prompt_tokens_estimated
        if ratio < threshold:
            dump_path = record.resolved_dump_path()
            candidates.append(
                TruncationCandidate(
                    agent=record.agent,
                    run_dir=record.run_dir.name,
                    run_id=record.run_id,
                    estimated_tokens=record.prompt_tokens_estimated,
                    reported_tokens=record.input_tokens,
                    ratio=ratio,
                    dump_path=str(dump_path) if dump_path else None,
                )
            )

    candidates.sort(key=lambda c: c.ratio if c.ratio is not None else 1.0)
    return candidates


# ─── Report data model (sorting/capping/path-resolution happens once, here) ─


def build_report_data(
    agg: CorpusAggregation,
    *,
    sort_by: str = "call_count",
    top_n: int = 10,
    truncation_threshold: float = 0.8,
) -> dict[str, Any]:
    """Build the plain-dict report model shared by all three ``render_*`` functions.

    Sorting the per-agent table happens against the actual dict
    ``PerAgentStats.compute_stats()`` returns (not the ``PerAgentStats``
    instance, which lacks several of these keys) so ``sort_by`` in
    ``AGENT_SORT_KEYS`` always has an effect.
    """
    per_agent = [s.compute_stats() for s in agg.per_agent_stats.values()]
    per_agent.sort(
        key=lambda s: s.get(sort_by) if s.get(sort_by) is not None else -1,
        reverse=True,
    )

    sorted_by_size = sorted(agg.all_records, key=lambda r: r.prompt_tokens_estimated, reverse=True)
    top_prompts = []
    for r in sorted_by_size[:top_n]:
        ratio = (
            r.input_tokens / r.prompt_tokens_estimated
            if (r.input_tokens is not None and r.prompt_tokens_estimated)
            else None
        )
        dump_path = r.resolved_dump_path()
        top_prompts.append(
            {
                "agent": r.agent,
                "ticker": r.ticker,
                "run_dir": r.run_dir.name,
                "estimated_tokens": r.prompt_tokens_estimated,
                "reported_tokens": r.input_tokens,
                "ratio": ratio,
                "dump_path": str(dump_path) if dump_path else None,
            }
        )

    candidates = detect_truncation_candidates(agg.all_records, threshold=truncation_threshold)
    truncation_candidates = [
        {
            "agent": c.agent,
            "run_dir": c.run_dir,
            "run_id": c.run_id,
            "estimated_tokens": c.estimated_tokens,
            "reported_tokens": c.reported_tokens,
            "ratio": c.ratio,
            "dump_path": c.dump_path,
        }
        for c in candidates[:top_n]
    ]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reports_dir": str(agg.reports_dir),
        "sort_by": sort_by,
        "top_n": top_n,
        "truncation_threshold": truncation_threshold,
        "corpus": {
            "run_count": agg.run_count,
            "call_count": agg.call_count,
            "total_wall_time": agg.total_wall_time,
            "model_counts": dict(agg.model_counts),
            "skipped_dirs_count": len(agg.skipped_dirs),
            "skipped_dirs": list(agg.skipped_dirs),
        },
        "per_agent": per_agent,
        "top_prompts": top_prompts,
        "truncation_candidates": truncation_candidates,
    }


def render_json_report(data: dict[str, Any]) -> str:
    """Render the report model as machine-readable JSON, suitable for diffing two runs."""
    return json.dumps(data, indent=2, default=str)


def render_markdown_report(data: dict[str, Any]) -> str:
    """Render the report model as markdown (headers + pipe tables)."""
    lines: list[str] = []
    lines.append("# LLM Calls Analysis Report")
    lines.append("")
    lines.append(f"Generated: {data['generated_at']}")
    lines.append(f"Reports directory: {data['reports_dir']}")
    lines.append("")

    c = data["corpus"]
    lines.append("## Corpus Summary")
    lines.append(f"- Run directories found: {c['run_count']}")
    lines.append(f"- Total LLM calls: {c['call_count']}")
    lines.append(f"- Total wall time (all calls): {_fmt_num(c['total_wall_time'])} seconds")
    if c["model_counts"]:
        mix = ", ".join(f"{m} ({n})" for m, n in c["model_counts"].items())
        lines.append(f"- Model mix: {mix}")
    else:
        lines.append("- Model mix: (none)")
    if c["skipped_dirs_count"]:
        lines.append(
            f"- Skipped directories: {c['skipped_dirs_count']} (missing or malformed llm_calls.jsonl)"
        )
    lines.append("")

    lines.append(f"## Per-Agent Statistics (sorted by `{data['sort_by']}`)")
    lines.append(
        "| Agent | Calls | Est Prompt (mean/median/p95/max) | Input (max) | Output (total/max) | Duration (total/max) | Errors |"
    )
    lines.append("|---|---:|---|---:|---|---|---:|")
    for s in data["per_agent"]:
        prompt_str = (
            f"{_fmt_num(s['estimated_prompt_tokens_mean'], 0)}/"
            f"{_fmt_num(s['estimated_prompt_tokens_median'], 0)}/"
            f"{_fmt_num(s['estimated_prompt_tokens_p95'], 0)}/"
            f"{_fmt_int(s['estimated_prompt_tokens_max'])}"
        )
        output_str = f"{_fmt_int(s['output_tokens_total'])}/{_fmt_int(s['output_tokens_max'])}"
        duration_str = f"{_fmt_num(s['duration_total'], 1)}/{_fmt_num(s['duration_max'], 2)}"
        lines.append(
            f"| {s['agent']} | {s['call_count']} | {prompt_str} | "
            f"{_fmt_int(s['reported_input_tokens_max'])} | {output_str} | {duration_str} | {s['error_count']} |"
        )
    lines.append("")

    lines.append(f"## Top {data['top_n']} Largest Prompts")
    lines.append("| # | Agent | Run directory | Est tokens | Reported | Ratio | Dump path |")
    lines.append("|---:|---|---|---:|---:|---|---|")
    for i, p in enumerate(data["top_prompts"], 1):
        ratio_str = _fmt_num(p["ratio"], 3) if p["ratio"] is not None else "n/a"
        lines.append(
            f"| {i} | {p['agent']} | {p['run_dir']} | {p['estimated_tokens']} | "
            f"{_fmt_int(p['reported_tokens'])} | {ratio_str} | {p['dump_path'] or 'n/a'} |"
        )
    lines.append("")

    lines.append(
        f"## Truncation Candidates (reported input tokens far below estimate; "
        f"threshold={data['truncation_threshold']})"
    )
    if data["truncation_candidates"]:
        lines.append("| Agent | Run directory | Est tokens | Reported | Ratio | Dump path |")
        lines.append("|---|---|---:|---:|---|---|")
        for cand in data["truncation_candidates"]:
            lines.append(
                f"| {cand['agent']} | {cand['run_dir']} | {cand['estimated_tokens']} | "
                f"{_fmt_int(cand['reported_tokens'])} | {_fmt_num(cand['ratio'], 3)} | "
                f"{cand['dump_path'] or 'n/a'} |"
            )
    else:
        lines.append("No truncation candidates detected.")
    lines.append("")

    return "\n".join(lines) + "\n"


def render_text_report(data: dict[str, Any]) -> str:
    """Render the report model as plain text (fixed-width columns, no markdown markup).

    Deliberately distinct from ``render_markdown_report``: no ``#``/``**``/``|``
    markup, so ``--format text`` and ``--format markdown`` produce genuinely
    different output rather than the same markdown tables twice.
    """
    lines: list[str] = []
    lines.append("LLM CALLS ANALYSIS REPORT")
    lines.append("=" * 60)
    lines.append(f"Generated: {data['generated_at']}")
    lines.append(f"Reports directory: {data['reports_dir']}")
    lines.append("")

    c = data["corpus"]
    lines.append("Corpus Summary")
    lines.append("-" * 60)
    lines.append(f"Run directories found : {c['run_count']}")
    lines.append(f"Total LLM calls       : {c['call_count']}")
    lines.append(f"Total wall time (s)   : {_fmt_num(c['total_wall_time'])}")
    if c["model_counts"]:
        mix = ", ".join(f"{m}={n}" for m, n in c["model_counts"].items())
        lines.append(f"Model mix             : {mix}")
    else:
        lines.append("Model mix             : (none)")
    if c["skipped_dirs_count"]:
        lines.append(
            f"Skipped directories   : {c['skipped_dirs_count']} (missing or malformed llm_calls.jsonl)"
        )
    lines.append("")

    lines.append(f"Per-Agent Statistics (sorted by {data['sort_by']})")
    lines.append("-" * 60)
    lines.append(
        f"{'Agent':<24}{'Calls':>7}{'EstPromptMean':>15}{'EstPromptMax':>14}"
        f"{'InputMax':>10}{'OutTotal':>10}{'DurTotal':>10}{'Errors':>8}"
    )
    for s in data["per_agent"]:
        lines.append(
            f"{s['agent'][:23]:<24}"
            f"{s['call_count']:>7}"
            f"{_fmt_num(s['estimated_prompt_tokens_mean'], 0):>15}"
            f"{_fmt_int(s['estimated_prompt_tokens_max']):>14}"
            f"{_fmt_int(s['reported_input_tokens_max']):>10}"
            f"{_fmt_int(s['output_tokens_total']):>10}"
            f"{_fmt_num(s['duration_total'], 1):>10}"
            f"{s['error_count']:>8}"
        )
    lines.append("")

    lines.append(f"Top {data['top_n']} Largest Prompts")
    lines.append("-" * 60)
    for i, p in enumerate(data["top_prompts"], 1):
        ratio_str = _fmt_num(p["ratio"], 3) if p["ratio"] is not None else "n/a"
        lines.append(f"{i}. {p['agent']} [{p['run_dir']}]")
        lines.append(
            f"   estimated={p['estimated_tokens']} reported={_fmt_int(p['reported_tokens'])} "
            f"ratio={ratio_str}"
        )
        lines.append(f"   dump: {p['dump_path'] or 'n/a'}")
    lines.append("")

    lines.append(f"Truncation Candidates (threshold={data['truncation_threshold']})")
    lines.append("-" * 60)
    if data["truncation_candidates"]:
        for cand in data["truncation_candidates"]:
            lines.append(
                f"{cand['agent']} [{cand['run_dir']}] estimated={cand['estimated_tokens']} "
                f"reported={_fmt_int(cand['reported_tokens'])} ratio={_fmt_num(cand['ratio'], 3)} "
                f"dump={cand['dump_path'] or 'n/a'}"
            )
    else:
        lines.append("No truncation candidates detected.")
    lines.append("")

    return "\n".join(lines) + "\n"


# ─── Segment mode ────────────────────────────────────────────────────────────


def build_segment_data(agg: CorpusAggregation, top_n: int = 5) -> dict[str, Any]:
    """Build the segment-mode report model for the ``top_n`` largest prompts.

    Uses each record's own ``run_dir`` (captured at load time) to resolve its
    prompt dump, rather than fuzzy-matching the ticker against an unsorted
    directory listing — the latter can pick the wrong run when the same
    ticker has multiple run directories. A dump that is missing, unreadable,
    or not valid JSON is skipped (counted in ``missing_dump_count``) rather
    than crashing the run.
    """
    sorted_records = sorted(agg.all_records, key=lambda r: r.prompt_tokens_estimated, reverse=True)
    prompts: list[dict[str, Any]] = []
    missing_dump_count = 0

    for record in sorted_records[:top_n]:
        entry: dict[str, Any] = {
            "agent": record.agent,
            "ticker": record.ticker,
            "run_id": record.run_id,
            "run_dir": record.run_dir.name,
            "total_estimated_tokens": record.prompt_tokens_estimated,
            "dump_path": None,
            "segments": None,
            "error": None,
        }

        dump_path = record.resolved_dump_path()
        if dump_path is None:
            entry["error"] = "no prompt dump recorded for this call"
            missing_dump_count += 1
            prompts.append(entry)
            continue

        entry["dump_path"] = str(dump_path)
        try:
            dump_data = json.loads(dump_path.read_text(encoding="utf-8"))
        except OSError:
            entry["error"] = "prompt dump file not found"
            missing_dump_count += 1
            prompts.append(entry)
            continue
        except json.JSONDecodeError as e:
            entry["error"] = f"malformed prompt dump JSON: {e}"
            missing_dump_count += 1
            prompts.append(entry)
            continue

        prompt_text = ""
        if isinstance(dump_data, dict):
            for msg in dump_data.get("messages", []):
                if isinstance(msg, dict):
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        prompt_text += content + "\n"

        segments = _split_prompt_into_segments(prompt_text)
        total_chars = sum(chars for _, _, chars in segments) or 1
        entry["segments"] = [
            {
                "name": name,
                "chars": chars,
                "estimated_tokens": chars // _CHARS_PER_TOKEN_ESTIMATE,
                "pct_of_prompt": round(chars / total_chars * 100, 1),
            }
            for name, _, chars in segments
        ]
        prompts.append(entry)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reports_dir": str(agg.reports_dir),
        "top_n": top_n,
        "prompts": prompts,
        "missing_dump_count": missing_dump_count,
    }


def render_segment_report(data: dict[str, Any], report_format: str = "text") -> str:
    """Render the segment-mode report model as text, markdown, or JSON.

    Respects ``report_format`` the same way the main report does (fixes
    ``--segment --format json`` previously always emitting the same
    free-text output regardless of ``--format``).
    """
    if report_format == "json":
        return json.dumps(data, indent=2, default=str)

    is_md = report_format == "markdown"
    lines: list[str] = []
    if is_md:
        lines.append("# Prompt Segment Analysis")
    else:
        lines.append("PROMPT SEGMENT ANALYSIS")
        lines.append("=" * 60)
    lines.append("")
    lines.append(f"Generated: {data['generated_at']}")
    lines.append(f"Reports directory: {data['reports_dir']}")
    if data["missing_dump_count"]:
        lines.append(f"Prompt dumps skipped (missing/unreadable/malformed): {data['missing_dump_count']}")
    lines.append("")

    for i, p in enumerate(data["prompts"], 1):
        header = f"#{i}: {p['agent']} / {p['ticker']} [{p['run_dir']}] ({p['run_id']})"
        if is_md:
            lines.append(f"## {header}")
        else:
            lines.append(header)
            lines.append("-" * len(header))
        lines.append(f"Total estimated tokens: {p['total_estimated_tokens']}")
        lines.append("")

        if p["error"]:
            lines.append(f"(skipped: {p['error']})")
            lines.append("")
            continue

        if is_md:
            lines.append("| Segment | Chars | Est tokens | % of prompt |")
            lines.append("|---|---:|---:|---:|")
            for seg in p["segments"]:
                lines.append(
                    f"| {seg['name']} | {seg['chars']} | {seg['estimated_tokens']} | {seg['pct_of_prompt']}% |"
                )
        else:
            for seg in p["segments"]:
                lines.append(
                    f"  {seg['name']:<30}{seg['chars']:>8} chars  "
                    f"~{seg['estimated_tokens']:>6} tok  {seg['pct_of_prompt']:>5.1f}%"
                )
        lines.append("")

    return "\n".join(lines) + "\n"


# ─── CLI ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze LLM call logs and prompt dumps from reports/ (issue #143)"
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("reports"),
        help="Reports directory containing run subdirectories (default: reports)",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json", "markdown"],
        default="text",
        help="Output format (default: text). Applies to --segment mode too.",
    )
    parser.add_argument(
        "--sort-by",
        choices=AGENT_SORT_KEYS,
        default="call_count",
        help="Sort the per-agent table by this column, descending (default: call_count). "
        "Ignored in --segment mode.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of rows in the Top-N largest prompts and truncation-candidates "
        "listings; in --segment mode, the number of prompts to analyze (default: 10)",
    )
    parser.add_argument(
        "--truncation-threshold",
        type=float,
        default=0.8,
        help="Flag a call as a truncation candidate when reported input_tokens / "
        "estimated prompt tokens falls below this ratio (default: 0.8)",
    )
    parser.add_argument(
        "--segment",
        action="store_true",
        help="Segment mode: split the --top largest prompts by anchor lines and report "
        "per-segment chars/estimated-token composition instead of the main report.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output file path (default: stdout)",
    )

    args = parser.parse_args()

    agg = load_llm_calls_from_directory(args.reports_dir)

    if args.segment:
        segment_data = build_segment_data(agg, top_n=args.top)
        report = render_segment_report(segment_data, report_format=args.format)
    else:
        data = build_report_data(
            agg,
            sort_by=args.sort_by,
            top_n=args.top,
            truncation_threshold=args.truncation_threshold,
        )
        if args.format == "json":
            report = render_json_report(data)
        elif args.format == "markdown":
            report = render_markdown_report(data)
        else:
            report = render_text_report(data)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        print(f"Report written to: {args.out}")
    else:
        print(report, end="")


if __name__ == "__main__":
    main()
