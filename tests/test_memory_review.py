"""Tests for skills/memory-review/find_patterns.py — the memory-review skill's deterministic
pattern-detection + heuristics-persistence helpers (issue #14).

`skills/memory-review/SKILL.md` is markdown-driven, like the other skills under `skills/` — it has
no test harness of its own (same precedent as `test_quant_memory.py` for `skills/quant/SKILL.md`).
What *is* testable, and is exercised here, is the deterministic half of the pipeline the skill
documents: statistics -> pattern detection -> context gathering -> findings markdown -> mechanical
heuristics draft -> persistence. This directly mirrors the issue's own "Verify" section: "seed a
DB with a planted pattern (e.g. one agent systematically wrong on one ticker, right elsewhere),
run the review, confirm the pattern is surfaced in findings and lands in that agent's
heuristics.md" — everything up to the live LLM curation step (which cannot be exercised without a
real provider, and this repo's tests never call live LLM APIs — see CLAUDE.md / conftest.py).

No live LLM or network access is used.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from tradingagents.memory.store import get_connection, store_decision

# ---------------------------------------------------------------------------
# Load skills/memory-review/find_patterns.py as a module (it lives outside any
# Python package under skills/, so it's not importable via a dotted path) —
# same loading pattern as test_quant_memory.py.
# ---------------------------------------------------------------------------

_MEMORY_REVIEW_DIR = Path(__file__).resolve().parents[1] / "skills" / "memory-review"
_SPEC = importlib.util.spec_from_file_location(
    "memory_review_find_patterns", _MEMORY_REVIEW_DIR / "find_patterns.py"
)
find_patterns_mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(find_patterns_mod)


def _db_path(tmp_path):
    return tmp_path / "memory.db"


def _seed_resolved(
    db_path,
    agent,
    ticker,
    date,
    signal,
    confidence,
    forward_return,
    key_drivers=None,
    thesis="Some thesis.",
    lesson="Some lesson.",
):
    """Write a fully resolved decision row directly — bypasses resolve_pending's LLM/yfinance
    mechanics, same precedent as test_memory_stats.py's `_seed_resolved` helper."""
    store_decision(
        agent=agent,
        ticker=ticker,
        date=date,
        signal=signal,
        confidence=confidence,
        key_drivers=key_drivers,
        thesis=thesis,
        db_path=db_path,
    )
    conn = get_connection(db_path)
    try:
        conn.execute(
            """
            UPDATE decisions
            SET horizon_days = ?, forward_return = ?, lesson = ?,
                resolved_at = ?
            WHERE agent = ? AND ticker = ? AND decision_date = ?
            """,
            (10, forward_return, lesson, "2026-02-01T00:00:00+00:00", agent, ticker, date),
        )
        conn.commit()
    finally:
        conn.close()


def _seed_planted_pattern(db_path):
    """Seed the exact scenario the issue's Verify section describes: "fundamental was wrong 7/9
    times on TSLA — the misses cluster around earnings surprises the ratios can't see" — right
    elsewhere (AAPL).

    fundamental/TSLA (9 rows): 2 correct, 7 incorrect (misses tagged with an
    "earnings_surprise" key_driver — the "why" material for the reasoning step).
    fundamental/AAPL (6 rows): all correct (elsewhere baseline).
    quant/TSLA (4 rows), quant/AAPL (4 rows): both ~50% — a *non*-pattern control that should
    NOT be flagged (divergence well under the default 30-point threshold).
    """
    # TSLA misses: BUY calls that went down, tagged as an earnings-surprise miss.
    for i in range(7):
        _seed_resolved(
            db_path,
            "fundamental",
            "TSLA",
            f"2026-01-{i + 1:02d}",
            "Buy",
            0.75,
            -0.04,
            key_drivers={"ratio_signal": "undervalued", "earnings_surprise": True},
            thesis="Ratios look cheap heading into earnings; undervalued on P/E and FCF yield.",
            lesson="Missed an earnings-driven guidance cut the ratios couldn't see.",
        )
    # TSLA hits: 2 correct calls.
    for i in range(7, 9):
        _seed_resolved(
            db_path,
            "fundamental",
            "TSLA",
            f"2026-01-{i + 1:02d}",
            "Buy",
            0.75,
            0.03,
            key_drivers={"ratio_signal": "undervalued", "earnings_surprise": False},
            thesis="Ratios look cheap; no major earnings catalyst pending.",
            lesson="Steady fundamentals-driven gain, no surprise.",
        )
    # AAPL: 6/6 correct — the "right elsewhere" baseline.
    for i in range(6):
        _seed_resolved(
            db_path,
            "fundamental",
            "AAPL",
            f"2026-02-{i + 1:02d}",
            "Buy",
            0.8,
            0.05,
            key_drivers={"ratio_signal": "undervalued", "earnings_surprise": False},
            thesis="Steady fundamentals, no earnings surprise risk flagged.",
            lesson="Correct bullish call.",
        )
    # quant/TSLA + quant/AAPL: ~50% each, no meaningful divergence -- control.
    for i, ret in enumerate([0.02, -0.02, 0.02, -0.02]):
        _seed_resolved(db_path, "quant", "TSLA", f"2026-03-{i + 1:02d}", "Buy", 0.6, ret)
    for i, ret in enumerate([0.02, -0.02, 0.02, -0.02]):
        _seed_resolved(db_path, "quant", "AAPL", f"2026-03-{i + 1:02d}", "Buy", 0.6, ret)


# ---------------------------------------------------------------------------
# find_patterns: the planted pattern is surfaced, the control is not.
# ---------------------------------------------------------------------------


def test_find_patterns_surfaces_the_planted_pattern(tmp_path):
    db_path = _db_path(tmp_path)
    _seed_planted_pattern(db_path)

    findings = find_patterns_mod.find_patterns(db_path=db_path)

    by_key = {(f["agent"], f["ticker"]): f for f in findings}
    assert ("fundamental", "TSLA") in by_key

    finding = by_key[("fundamental", "TSLA")]
    assert finding["n"] == 9
    assert finding["hit_rate"] == pytest.approx(2 / 9)
    assert finding["baseline_hit_rate"] == pytest.approx(1.0)
    assert finding["baseline_n"] == 6
    assert finding["direction"] == "underperforms"
    assert finding["delta"] == pytest.approx(1.0 - 2 / 9)

    # AAPL is fundamental's only other ticker, so it's the mirror image of the same divergence
    # (this agent only covers two tickers here) -- flagged too, but in the opposite direction.
    aapl_finding = by_key[("fundamental", "AAPL")]
    assert aapl_finding["direction"] == "overperforms"
    assert aapl_finding["hit_rate"] == pytest.approx(1.0)
    assert aapl_finding["baseline_hit_rate"] == pytest.approx(2 / 9)

    # The quant control (both tickers ~50%, no real divergence) is not flagged.
    assert ("quant", "TSLA") not in by_key
    assert ("quant", "AAPL") not in by_key


def test_find_patterns_respects_min_n_and_divergence_threshold(tmp_path):
    db_path = _db_path(tmp_path)
    _seed_planted_pattern(db_path)

    # Absurdly high min_n -> nothing qualifies.
    assert find_patterns_mod.find_patterns(db_path=db_path, min_n=100) == []

    # Absurdly high divergence threshold -> nothing qualifies.
    assert find_patterns_mod.find_patterns(db_path=db_path, divergence_threshold=0.99) == []

    # Loosening the threshold enough should pick up the quant control too.
    loose = find_patterns_mod.find_patterns(db_path=db_path, divergence_threshold=0.01)
    loose_keys = {(f["agent"], f["ticker"]) for f in loose}
    assert ("fundamental", "TSLA") in loose_keys


def test_find_patterns_filters_by_agent(tmp_path):
    db_path = _db_path(tmp_path)
    _seed_planted_pattern(db_path)

    findings = find_patterns_mod.find_patterns(agent="fundamental", db_path=db_path)
    assert {f["agent"] for f in findings} == {"fundamental"}


# ---------------------------------------------------------------------------
# gather_context_rows: the "why" material -- key_drivers/thesis on the misses.
# ---------------------------------------------------------------------------


def test_gather_context_rows_returns_key_drivers_and_marks_misses(tmp_path):
    db_path = _db_path(tmp_path)
    _seed_planted_pattern(db_path)

    rows = find_patterns_mod.gather_context_rows(
        "fundamental", "TSLA", db_path=db_path, limit=20
    )
    assert len(rows) == 9

    misses = [r for r in rows if r["correct"] is False]
    hits = [r for r in rows if r["correct"] is True]
    assert len(misses) == 7
    assert len(hits) == 2

    # The stored key_drivers survive the JSON round trip and carry the earnings-surprise flag
    # that explains *why* -- this is the material the skill's LLM step reasons over.
    assert all(m["key_drivers"]["earnings_surprise"] is True for m in misses)
    assert all(h["key_drivers"]["earnings_surprise"] is False for h in hits)
    assert all("earnings" in m["lesson"].lower() for m in misses)


def test_gather_context_rows_misses_only(tmp_path):
    db_path = _db_path(tmp_path)
    _seed_planted_pattern(db_path)

    rows = find_patterns_mod.gather_context_rows(
        "fundamental", "TSLA", db_path=db_path, limit=20, misses_only=True
    )
    assert len(rows) == 7
    assert all(r["correct"] is False for r in rows)


def test_gather_context_rows_respects_limit(tmp_path):
    db_path = _db_path(tmp_path)
    _seed_planted_pattern(db_path)

    rows = find_patterns_mod.gather_context_rows("fundamental", "TSLA", db_path=db_path, limit=3)
    assert len(rows) == 3


# ---------------------------------------------------------------------------
# format_findings_markdown: findings + context render as one prompt-ready block.
# ---------------------------------------------------------------------------


def test_format_findings_markdown_surfaces_pattern_and_context(tmp_path):
    db_path = _db_path(tmp_path)
    _seed_planted_pattern(db_path)

    findings = find_patterns_mod.find_patterns(db_path=db_path)
    context = find_patterns_mod.gather_all_context(findings, db_path=db_path)
    md = find_patterns_mod.format_findings_markdown(findings, context)

    assert "## Memory review findings" in md
    assert "fundamental × TSLA" in md
    assert "underperforms" in md
    assert "2/9 correct" in md
    assert "22%" in md  # 2/9 rounded
    assert "100%" in md  # baseline
    assert "[MISS]" in md
    assert "earnings" in md.lower()


def test_format_findings_markdown_empty_case():
    md = find_patterns_mod.format_findings_markdown([])
    assert "## Memory review findings" in md
    assert "No divergent" in md


# ---------------------------------------------------------------------------
# Heuristics distillation + persistence -- "lands in that agent's heuristics.md".
# ---------------------------------------------------------------------------


def test_default_heuristic_line_names_ticker_and_stats():
    finding = {
        "agent": "fundamental",
        "ticker": "TSLA",
        "n": 9,
        "hit_rate": 2 / 9,
        "baseline_hit_rate": 1.0,
        "baseline_n": 6,
        "delta": 1.0 - 2 / 9,
        "direction": "underperforms",
    }
    line = find_patterns_mod.default_heuristic_line(finding)
    assert "TSLA" in line
    assert "fundamental" in line
    assert "22%" in line
    assert "100%" in line


def test_draft_heuristics_markdown_scopes_to_one_agent(tmp_path):
    db_path = _db_path(tmp_path)
    _seed_planted_pattern(db_path)
    findings = find_patterns_mod.find_patterns(db_path=db_path)

    draft = find_patterns_mod.draft_heuristics_markdown("fundamental", findings)
    assert "# Heuristics: fundamental" in draft
    assert "TSLA" in draft

    # A different agent with no findings gets the "nothing yet" placeholder.
    empty_draft = find_patterns_mod.draft_heuristics_markdown("news", findings)
    assert "# Heuristics: news" in empty_draft
    assert "No divergent patterns found yet." in empty_draft


def test_write_heuristics_md_lands_pattern_in_agents_file(tmp_path):
    db_path = _db_path(tmp_path)
    heuristics_dir = tmp_path / "heuristics"
    _seed_planted_pattern(db_path)

    findings = find_patterns_mod.find_patterns(db_path=db_path)
    draft = find_patterns_mod.draft_heuristics_markdown("fundamental", findings)
    written_path = find_patterns_mod.write_heuristics_md(
        "fundamental", draft, heuristics_dir=heuristics_dir
    )

    expected_path = heuristics_dir / "fundamental.md"
    assert written_path == expected_path
    assert expected_path.exists()

    content = expected_path.read_text(encoding="utf-8")
    assert "TSLA" in content
    assert "fundamental" in content

    # quant had no flagged findings -- its file should simply not exist (nothing was written for
    # it), demonstrating the pattern "lands" specifically in the agent it belongs to.
    assert not (heuristics_dir / "quant.md").exists()


def test_write_heuristics_md_overwrites_not_appends(tmp_path):
    heuristics_dir = tmp_path / "heuristics"
    path = find_patterns_mod.write_heuristics_md("fundamental", "first version\n", heuristics_dir=heuristics_dir)
    find_patterns_mod.write_heuristics_md("fundamental", "second version\n", heuristics_dir=heuristics_dir)

    content = path.read_text(encoding="utf-8")
    assert content == "second version\n"


# ---------------------------------------------------------------------------
# Heuristics directory resolution -- explicit arg > env var > default.
# ---------------------------------------------------------------------------


def test_resolve_heuristics_dir_precedence(tmp_path, monkeypatch):
    monkeypatch.delenv("TRADINGAGENTS_HEURISTICS_DIR", raising=False)
    assert find_patterns_mod.resolve_heuristics_dir() == find_patterns_mod.DEFAULT_HEURISTICS_DIR

    env_dir = tmp_path / "env_heuristics"
    monkeypatch.setenv("TRADINGAGENTS_HEURISTICS_DIR", str(env_dir))
    assert find_patterns_mod.resolve_heuristics_dir() == env_dir

    explicit_dir = tmp_path / "explicit_heuristics"
    assert find_patterns_mod.resolve_heuristics_dir(explicit_dir) == explicit_dir


def test_heuristics_path_is_one_file_per_agent(tmp_path):
    p = find_patterns_mod.heuristics_path("fundamental", heuristics_dir=tmp_path)
    assert p == tmp_path / "fundamental.md"


# ---------------------------------------------------------------------------
# End-to-end: full review pipeline, exactly as the issue's Verify section describes.
# ---------------------------------------------------------------------------


def test_end_to_end_review_pipeline_surfaces_and_persists_the_pattern(tmp_path):
    db_path = _db_path(tmp_path)
    heuristics_dir = tmp_path / "heuristics"
    _seed_planted_pattern(db_path)

    # 1. Pull statistics-derived findings.
    findings = find_patterns_mod.find_patterns(db_path=db_path)
    assert any(f["agent"] == "fundamental" and f["ticker"] == "TSLA" for f in findings)

    # 2. Gather the key_drivers/thesis context behind the flagged cell.
    context = find_patterns_mod.gather_all_context(findings, db_path=db_path)
    assert ("fundamental", "TSLA") in context

    # 3. Render findings for the reasoning step.
    md = find_patterns_mod.format_findings_markdown(findings, context)
    assert "TSLA" in md and "fundamental" in md

    # 4. Distill + persist per agent.
    for agent_id in sorted({f["agent"] for f in findings}):
        draft = find_patterns_mod.draft_heuristics_markdown(agent_id, findings)
        find_patterns_mod.write_heuristics_md(agent_id, draft, heuristics_dir=heuristics_dir)

    fundamental_md = (heuristics_dir / "fundamental.md").read_text(encoding="utf-8")
    assert "TSLA" in fundamental_md
    assert "investigate root cause" in fundamental_md


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_write_draft_and_json(tmp_path, capsys):
    db_path = _db_path(tmp_path)
    heuristics_dir = tmp_path / "heuristics"
    _seed_planted_pattern(db_path)

    find_patterns_mod.main(
        [
            "--db-path", str(db_path),
            "--heuristics-dir", str(heuristics_dir),
            "--write-draft",
            "--json",
        ]
    )

    out = capsys.readouterr().out
    payload = json.loads(out)
    assert any(f["agent"] == "fundamental" and f["ticker"] == "TSLA" for f in payload["findings"])
    assert "fundamental::TSLA" in payload["context"]

    assert (heuristics_dir / "fundamental.md").exists()
    assert "TSLA" in (heuristics_dir / "fundamental.md").read_text(encoding="utf-8")
