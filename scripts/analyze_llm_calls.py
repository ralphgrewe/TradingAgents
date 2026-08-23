#!/usr/bin/env python3
"""Analyze LLM call logs and prompt dumps across runs (issue #143).

Aggregates llm_calls.jsonl and prompts/*.json from reports/ to provide per-agent
statistics, per-prompt segment composition, and truncation detection.

Usage::

    ./venv/bin/python scripts/analyze_llm_calls.py
    ./venv/bin/python scripts/analyze_llm_calls.py --reports-dir reports --format json
    ./venv/bin/python scripts/analyze_llm_calls.py --format text --sort-by duration --top 10
    ./venv/bin/python scripts/analyze_llm_calls.py --segment --format markdown

"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Token estimator: reuse chars/4 heuristic from tradingagents/llm_call_log.py
_CHARS_PER_TOKEN_ESTIMATE = 4


# ─── Segment anchors (define in one place for easy extension) ──────────────


SEGMENT_ANCHORS = [
    "Past analyses of ",
    "Recent cross-ticker lessons:",
    "Analyst Reports:",
    "**Risk Analysts Debate History:**",
    "**Available Tools:**",
    # Add more anchors here as they are discovered in the corpus
]


def _split_prompt_into_segments(prompt_text: str) -> list[tuple[str, str, int]]:
    """Split a prompt by anchor lines into (segment_name, content, char_count) tuples.

    Puts everything before the first anchor into a "header/instructions" segment.
    Unmatched or anchor-less prompts produce a single "full_prompt" segment.
    Segment names are derived from the anchor text or "header/instructions"/"full_prompt".

    Returns:
        List of (segment_name, content, char_count) tuples.
    """
    if not prompt_text:
        return [("empty_prompt", "", 0)]

    # Find all anchor positions
    anchors_found: list[tuple[int, str]] = []
    for anchor in SEGMENT_ANCHORS:
        pos = prompt_text.find(anchor)
        if pos != -1:
            anchors_found.append((pos, anchor))

    if not anchors_found:
        # No anchors: treat entire prompt as one segment
        return [("full_prompt", prompt_text, len(prompt_text))]

    # Sort anchors by position
    anchors_found.sort()
    segments: list[tuple[str, str, int]] = []

    # Add header/instructions (before first anchor)
    first_anchor_pos = anchors_found[0][0]
    if first_anchor_pos > 0:
        header_text = prompt_text[:first_anchor_pos]
        segments.append(("header/instructions", header_text, len(header_text)))

    # Add segments for each anchor
    for i, (pos, anchor) in enumerate(anchors_found):
        # Segment name: use anchor text (truncate for readability)
        segment_name = anchor.strip().rstrip(":").rstrip()
        if len(segment_name) > 40:
            segment_name = segment_name[:37] + "..."
        segment_name = segment_name.lower().replace(" ", "_")

        # Content: from this anchor to the next anchor (or end of text)
        next_pos = anchors_found[i + 1][0] if i + 1 < len(anchors_found) else len(prompt_text)
        content = prompt_text[pos:next_pos]
        segments.append((segment_name, content, len(content)))

    return segments


# ─── Call record aggregation and statistics ────────────────────────────────


@dataclass
class CallRecord:
    """A single LLM call from an llm_calls.jsonl record."""

    timestamp: str
    run_id: str
    ticker: str
    date: str
    agent: str
    model: str
    message_count: int
    prompt_chars: int
    prompt_tokens_estimated: int
    input_tokens: int | None
    output_tokens: int | None
    duration_seconds: float
    error: str | None
    prompt_dump_path: str | None

    @classmethod
    def from_jsonl_dict(cls, d: dict[str, Any]) -> CallRecord:
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
            input_tokens=d.get("input_tokens"),
            output_tokens=d.get("output_tokens"),
            duration_seconds=d.get("duration_seconds", 0.0),
            error=d.get("error"),
            prompt_dump_path=d.get("prompt_dump_path"),
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
        """Compute derived statistics."""
        stats: dict[str, Any] = {
            "agent": self.agent,
            "call_count": self.call_count,
            "error_count": self.error_count,
        }

        # Estimated prompt tokens
        if self.estimated_prompt_tokens_list:
            sorted_tokens = sorted(self.estimated_prompt_tokens_list)
            stats["estimated_prompt_tokens_mean"] = statistics.mean(
                self.estimated_prompt_tokens_list
            )
            stats["estimated_prompt_tokens_median"] = statistics.median(sorted_tokens)
            stats["estimated_prompt_tokens_p95"] = sorted_tokens[
                int(len(sorted_tokens) * 0.95)
            ] if len(sorted_tokens) > 1 else sorted_tokens[0]
            stats["estimated_prompt_tokens_max"] = max(self.estimated_prompt_tokens_list)
        else:
            stats["estimated_prompt_tokens_mean"] = None
            stats["estimated_prompt_tokens_median"] = None
            stats["estimated_prompt_tokens_p95"] = None
            stats["estimated_prompt_tokens_max"] = 0

        # Reported input tokens
        if self.reported_input_tokens_list:
            stats["reported_input_tokens_max"] = max(self.reported_input_tokens_list)
        else:
            stats["reported_input_tokens_max"] = None

        # Output tokens
        if self.output_tokens_list:
            stats["output_tokens_total"] = sum(self.output_tokens_list)
            stats["output_tokens_max"] = max(self.output_tokens_list)
        else:
            stats["output_tokens_total"] = None
            stats["output_tokens_max"] = None

        # Duration
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
    models_seen: set[str] = field(default_factory=set)
    per_agent_stats: dict[str, PerAgentStats] = field(default_factory=dict)
    all_records: list[CallRecord] = field(default_factory=list)
    skipped_dirs: list[str] = field(default_factory=list)
    missing_dumps: int = 0


def load_llm_calls_from_directory(reports_dir: Path) -> CorpusAggregation:
    """Load all llm_calls.jsonl files from reports_dir and aggregate stats.

    Handles:
    - Missing llm_calls.jsonl (report dir exists but no calls logged)
    - Missing prompts/ or individual prompt dump files
    - Malformed JSON
    """
    agg = CorpusAggregation(reports_dir=reports_dir)

    if not reports_dir.exists():
        return agg

    # Iterate over all report directories (ticker_date_timestamp format)
    for run_dir in sorted(reports_dir.iterdir()):
        if not run_dir.is_dir() or run_dir.name.startswith("."):
            continue

        llm_calls_file = run_dir / "llm_calls.jsonl"
        if not llm_calls_file.exists():
            agg.skipped_dirs.append(run_dir.name)
            continue

        agg.run_count += 1

        # Load JSONL records
        try:
            with open(llm_calls_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record_dict = json.loads(line)
                        record = CallRecord.from_jsonl_dict(record_dict)
                        agg.all_records.append(record)
                        agg.call_count += 1
                        agg.total_wall_time += record.duration_seconds
                        agg.models_seen.add(record.model)

                        # Aggregate per-agent stats
                        if record.agent not in agg.per_agent_stats:
                            agg.per_agent_stats[record.agent] = PerAgentStats(
                                agent=record.agent
                            )
                        agg.per_agent_stats[record.agent].add_call(record)
                    except json.JSONDecodeError as e:
                        agg.skipped_dirs.append(
                            f"{run_dir.name} (malformed JSON in llm_calls.jsonl: {e})"
                        )
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
    prompt_dump_path: str | None


def detect_truncation_candidates(
    records: list[CallRecord], threshold: float = 0.8
) -> list[TruncationCandidate]:
    """Flag calls where reported input_tokens << estimated prompt_tokens.

    A ratio < threshold suggests truncation. Returns sorted list.
    """
    candidates: list[TruncationCandidate] = []

    for record in records:
        if (
            record.input_tokens is None
            or record.prompt_tokens_estimated == 0
        ):
            continue

        ratio = record.input_tokens / record.prompt_tokens_estimated
        if ratio < threshold:
            candidates.append(
                TruncationCandidate(
                    agent=record.agent,
                    run_dir=record.ticker,
                    run_id=record.run_id,
                    estimated_tokens=record.prompt_tokens_estimated,
                    reported_tokens=record.input_tokens,
                    ratio=ratio,
                    prompt_dump_path=record.prompt_dump_path,
                )
            )

    # Sort by ratio (lowest first = most truncated)
    candidates.sort(key=lambda c: c.ratio if c.ratio is not None else 1.0)
    return candidates


# ─── Output formatting ──────────────────────────────────────────────────────


def render_text_report(
    agg: CorpusAggregation,
    sort_by: str = "call_count",
    top_n: int = 10,
    include_truncation: bool = True,
    include_segment_mode: bool = False,
    reports_dir_override: Path | None = None,
) -> str:
    """Render corpus aggregation as human-readable text."""
    lines: list[str] = []
    lines.append("# LLM Calls Analysis Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"Reports directory: {agg.reports_dir}")
    lines.append("")

    # Corpus summary
    lines.append("## Corpus Summary")
    lines.append(f"- Run directories found: {agg.run_count}")
    lines.append(f"- Total LLM calls: {agg.call_count}")
    lines.append(f"- Total wall time (all calls): {_fmt_num(agg.total_wall_time)} seconds")
    lines.append(f"- Models used: {len(agg.models_seen)}")
    lines.append(f"  - {', '.join(sorted(agg.models_seen))}")
    if agg.skipped_dirs:
        lines.append(
            f"- Skipped directories: {len(agg.skipped_dirs)} (missing or malformed llm_calls.jsonl)"
        )
    lines.append("")

    # Per-agent table
    lines.append("## Per-Agent Statistics")
    lines.append(
        "| Agent | Calls | Est Prompt (mean/p95/max) | Input (max) | Output (total/max) | Duration (total/max) | Errors |"
    )
    lines.append("|---|---:|---|---|---|---|---:|")

    sorted_agents = sorted(
        agg.per_agent_stats.values(),
        key=lambda s: (
            getattr(s, sort_by.replace("-", "_"), 0)
            if sort_by not in ["call_count", "duration_total"]
            else (s.call_count if sort_by == "call_count" else s.total_wall_time)
        ),
        reverse=True,
    )

    for agent_stats in sorted_agents:
        stats = agent_stats.compute_stats()
        prompt_tokens_str = (
            f"{_fmt_num(stats['estimated_prompt_tokens_mean'], 0)}/{_fmt_num(stats['estimated_prompt_tokens_p95'], 0)}/{_fmt_num(stats['estimated_prompt_tokens_max'], 0)}"
        )
        output_tokens_str = (
            f"{_fmt_int(stats['output_tokens_total'])}/{_fmt_int(stats['output_tokens_max'])}"
        )
        duration_str = (
            f"{_fmt_num(stats['duration_total'], 1)}/{_fmt_num(stats['duration_max'], 2)}"
        )
        lines.append(
            f"| {stats['agent']} | {stats['call_count']} | {prompt_tokens_str} | {_fmt_int(stats['reported_input_tokens_max'])} | {output_tokens_str} | {duration_str} | {stats['error_count']} |"
        )

    lines.append("")

    # Top-N largest prompts
    lines.append(f"## Top {top_n} Largest Prompts")
    sorted_by_prompt_size = sorted(
        agg.all_records, key=lambda r: r.prompt_tokens_estimated, reverse=True
    )
    for i, record in enumerate(sorted_by_prompt_size[:top_n], 1):
        lines.append(f"{i}. {record.agent} / {record.ticker}")
        lines.append(
            f"   - Estimated tokens: {record.prompt_tokens_estimated}, Reported input: {record.input_tokens or 'n/a'}"
        )
        if record.prompt_dump_path:
            lines.append(f"   - Dump: {record.prompt_dump_path}")
        lines.append("")

    # Truncation candidates
    if include_truncation:
        candidates = detect_truncation_candidates(agg.all_records)
        lines.append("## Potential Truncation Candidates (reported input << estimated prompt)")
        if candidates:
            lines.append(
                "| Agent | Run | Est. tokens | Reported | Ratio | Dump path |"
            )
            lines.append("|---|---|---:|---:|---|---|")
            for cand in candidates[:20]:  # Show top 20
                lines.append(
                    f"| {cand.agent} | {cand.run_dir} | {cand.estimated_tokens} | {cand.reported_tokens} | {_fmt_num(cand.ratio, 3)} | {cand.prompt_dump_path or 'n/a'} |"
                )
        else:
            lines.append("No truncation candidates detected.")
        lines.append("")

    return "\n".join(lines) + "\n"


def render_json_report(
    agg: CorpusAggregation,
    sort_by: str = "call_count",
    top_n: int = 10,
    include_truncation: bool = True,
) -> str:
    """Render corpus aggregation as machine-readable JSON."""
    output: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "corpus": {
            "reports_dir": str(agg.reports_dir),
            "run_count": agg.run_count,
            "call_count": agg.call_count,
            "total_wall_time": agg.total_wall_time,
            "models": sorted(agg.models_seen),
            "skipped_dirs_count": len(agg.skipped_dirs),
        },
        "per_agent": {},
    }

    for agent_name in sorted(agg.per_agent_stats.keys()):
        stats = agg.per_agent_stats[agent_name].compute_stats()
        output["per_agent"][agent_name] = stats

    # Top-N largest prompts
    sorted_by_size = sorted(
        agg.all_records, key=lambda r: r.prompt_tokens_estimated, reverse=True
    )
    output["top_largest_prompts"] = [
        {
            "agent": r.agent,
            "ticker": r.ticker,
            "estimated_tokens": r.prompt_tokens_estimated,
            "reported_input_tokens": r.input_tokens,
            "dump_path": r.prompt_dump_path,
        }
        for r in sorted_by_size[:top_n]
    ]

    # Truncation candidates
    if include_truncation:
        candidates = detect_truncation_candidates(agg.all_records)
        output["truncation_candidates"] = [
            {
                "agent": c.agent,
                "run_dir": c.run_dir,
                "estimated_tokens": c.estimated_tokens,
                "reported_tokens": c.reported_tokens,
                "ratio": c.ratio,
                "dump_path": c.prompt_dump_path,
            }
            for c in candidates[:20]
        ]

    return json.dumps(output, indent=2)


def render_markdown_report(
    agg: CorpusAggregation,
    sort_by: str = "call_count",
    top_n: int = 10,
    include_truncation: bool = True,
) -> str:
    """Render corpus aggregation as markdown (similar to text but markdown-flavored)."""
    return render_text_report(
        agg,
        sort_by=sort_by,
        top_n=top_n,
        include_truncation=include_truncation,
    )


def render_segment_analysis(
    agg: CorpusAggregation, reports_dir: Path
) -> str:
    """Analyze segment composition of the largest prompts."""
    lines: list[str] = []
    lines.append("# Prompt Segment Analysis")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append("")

    sorted_by_size = sorted(
        agg.all_records, key=lambda r: r.prompt_tokens_estimated, reverse=True
    )

    # Analyze top 5 largest prompts
    for i, record in enumerate(sorted_by_size[:5], 1):
        lines.append(f"## #{i}: {record.agent} / {record.ticker} ({record.run_id})")
        lines.append(f"Total estimated tokens: {record.prompt_tokens_estimated}")
        lines.append("")

        # Try to load and segment the prompt
        if record.prompt_dump_path:
            dump_path = None
            # Try to find the actual dump file
            for run_dir in reports_dir.iterdir():
                if run_dir.is_dir() and record.ticker in run_dir.name:
                    potential_path = run_dir / record.prompt_dump_path
                    if potential_path.exists():
                        dump_path = potential_path
                        break

            if dump_path and dump_path.exists():
                try:
                    with open(dump_path, encoding="utf-8") as f:
                        dump_data = json.load(f)
                        # Extract full prompt text from messages
                        prompt_text = ""
                        if isinstance(dump_data, dict):
                            messages = dump_data.get("messages", [])
                            for msg in messages:
                                if isinstance(msg, dict):
                                    content = msg.get("content", "")
                                    if isinstance(content, str):
                                        prompt_text += content + "\n"

                        if prompt_text:
                            segments = _split_prompt_into_segments(prompt_text)
                            total_chars = sum(c for _, _, c in segments)
                            lines.append("| Segment | Chars | Token Share (est.) |")
                            lines.append("|---|---:|---|")
                            for seg_name, _, char_count in segments:
                                tokens = char_count // _CHARS_PER_TOKEN_ESTIMATE
                                pct = (char_count / total_chars * 100) if total_chars > 0 else 0
                                lines.append(f"| {seg_name} | {char_count} | {tokens} (~{pct:.1f}%) |")
                            lines.append("")
                except (json.JSONDecodeError, OSError) as e:
                    lines.append(f"Error reading dump: {e}")
                    lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze LLM call logs from reports/ (issue #143)"
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
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--sort-by",
        default="call_count",
        help="Sort per-agent table by column (e.g. call_count, estimated_prompt_tokens_max, duration_total)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of top largest prompts to list (default: 10)",
    )
    parser.add_argument(
        "--segment",
        action="store_true",
        help="Enable segment mode: split prompts by anchor lines and report composition",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output file path (default: stdout)",
    )

    args = parser.parse_args()

    # Load and aggregate
    agg = load_llm_calls_from_directory(args.reports_dir)

    # Render
    if args.segment:
        report = render_segment_analysis(agg, args.reports_dir)
    elif args.format == "json":
        report = render_json_report(
            agg, sort_by=args.sort_by, top_n=args.top, include_truncation=True
        )
    elif args.format == "markdown":
        report = render_markdown_report(
            agg, sort_by=args.sort_by, top_n=args.top, include_truncation=True
        )
    else:  # text
        report = render_text_report(
            agg,
            sort_by=args.sort_by,
            top_n=args.top,
            include_truncation=True,
            reports_dir_override=args.reports_dir,
        )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        print(f"Report written to: {args.out}")
    else:
        print(report, end="")


if __name__ == "__main__":
    main()
