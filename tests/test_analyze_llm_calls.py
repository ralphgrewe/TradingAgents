"""Tests for scripts/analyze_llm_calls.py (issue #143).

Covers aggregation math, the segment splitter, truncation detection, the
rendering/CLI layer (text/json/markdown for both the main report and segment
mode), and robustness to missing/partial/malformed directories — all against
synthetic fixture directories created in ``tmp_path``, never the real
``reports/`` tree (which is not checked in and differs per machine).

This is a rewrite after a failed haiku attempt (design review comment on
issue #143) whose bugs lived entirely in the rendering/CLI layer that had
zero test coverage; this file specifically targets that layer in addition to
the aggregation layer the original tests already covered.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import tradingagents.llm_call_log as llm_call_log
from scripts.analyze_llm_calls import (
    _CHARS_PER_TOKEN_ESTIMATE,
    AGENT_SORT_KEYS,
    CallRecord,
    PerAgentStats,
    _split_prompt_into_segments,
    build_report_data,
    build_segment_data,
    detect_truncation_candidates,
    load_llm_calls_from_directory,
    render_json_report,
    render_markdown_report,
    render_segment_report,
    render_text_report,
)

pytestmark = pytest.mark.unit


# ─── Fixture helpers ────────────────────────────────────────────────────────


def _make_llm_calls_jsonl(report_dir: Path, records: list[dict]) -> None:
    """Write synthetic llm_calls.jsonl records."""
    report_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = report_dir / "llm_calls.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def _make_prompt_dump(report_dir: Path, run_id: str, messages: list[dict]) -> Path:
    """Write a synthetic prompt dump JSON file, mirroring the schema in llm_call_log.py."""
    prompts_dir = report_dir / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    dump_path = prompts_dir / f"{run_id}.json"
    dump_data = {"format": "chat_messages", "messages": messages}
    dump_path.write_text(json.dumps(dump_data), encoding="utf-8")
    return dump_path


def _call_record(**overrides) -> dict:
    """A minimal, complete llm_calls.jsonl record dict, with overrides applied."""
    base = {
        "timestamp": "2026-08-23T12:00:00Z",
        "run_id": "run-1",
        "ticker": "AAPL",
        "date": "2026-08-23",
        "agent": "Market Analyst",
        "model": "gpt-4",
        "message_count": 2,
        "prompt_chars": 500,
        "prompt_tokens_estimated": 125,
        "input_tokens": 110,
        "output_tokens": 50,
        "duration_seconds": 1.5,
        "error": None,
        "prompt_dump_path": None,
    }
    base.update(overrides)
    return base


def _record(**overrides) -> CallRecord:
    """A CallRecord built directly (not via JSONL), for unit-level tests."""
    d = _call_record(**overrides)
    run_dir = overrides.pop("run_dir", Path("."))
    return CallRecord.from_jsonl_dict(d, run_dir=run_dir)


# ─── Token estimator reuse (design review point #1) ────────────────────────


def test_token_estimator_is_imported_not_reimplemented():
    """analyze_llm_calls must import the estimator from llm_call_log, not hardcode a copy."""
    assert _CHARS_PER_TOKEN_ESTIMATE == llm_call_log._CHARS_PER_TOKEN_ESTIMATE


# ─── Segment splitting ──────────────────────────────────────────────────────


class TestSegmentSplitting:
    def test_empty_prompt_returns_single_empty_segment(self):
        segments = _split_prompt_into_segments("")
        assert segments == [("empty_prompt", "", 0)]

    def test_prompt_without_anchors_is_single_segment(self):
        text = "This is a simple prompt with no anchors."
        segments = _split_prompt_into_segments(text)
        assert len(segments) == 1
        assert segments[0] == ("full_prompt", text, len(text))

    def test_prompt_with_one_anchor(self):
        text = "Header text\n**Available Tools:**\nTool A\nTool B"
        segments = _split_prompt_into_segments(text)
        assert len(segments) == 2
        assert segments[0][0] == "header/instructions"
        assert "**Available Tools:**" in segments[1][1]

    def test_prompt_with_multiple_anchors_in_order(self):
        text = "Header\nPast analyses of X\nAnalyst Reports:\nReports here"
        segments = _split_prompt_into_segments(text)
        assert len(segments) >= 2
        assert any("header" in s[0].lower() for s in segments)

    def test_segment_char_count_is_correct(self):
        text = "A" * 100 + "Analyst Reports:" + "B" * 50
        segments = _split_prompt_into_segments(text)
        total_chars = sum(c for _, _, c in segments)
        assert total_chars == len(text)

    def test_segment_names_are_sanitized(self):
        text = "Start\n**Risk Analysts Debate History:**\nDebate text"
        segments = _split_prompt_into_segments(text)
        for seg_name, _, _ in segments:
            assert seg_name.islower() or seg_name in ("full_prompt", "empty_prompt")

    def test_no_anchors_detected_when_all_missing(self):
        text = "Some random text without any of the standard anchors."
        segments = _split_prompt_into_segments(text)
        assert len(segments) == 1
        assert segments[0][0] == "full_prompt"


# ─── Call record aggregation ────────────────────────────────────────────────


class TestCallRecord:
    def test_from_jsonl_dict_complete(self):
        record = CallRecord.from_jsonl_dict(_call_record(agent="Market Analyst"))
        assert record.agent == "Market Analyst"
        assert record.prompt_tokens_estimated == 125
        assert record.error is None

    def test_from_jsonl_dict_missing_optional_fields(self):
        minimal_dict = {"agent": "Unknown", "timestamp": "2026-08-23T12:00:00Z"}
        record = CallRecord.from_jsonl_dict(minimal_dict)
        assert record.agent == "Unknown"
        assert record.input_tokens is None
        assert record.error is None

    def test_run_dir_captured_at_load_time(self, tmp_path):
        run_dir = tmp_path / "AAPL_2026-08-23_run1"
        record = CallRecord.from_jsonl_dict(_call_record(), run_dir=run_dir)
        assert record.run_dir == run_dir

    def test_resolved_dump_path_none_when_no_dump(self):
        record = CallRecord.from_jsonl_dict(_call_record(prompt_dump_path=None))
        assert record.resolved_dump_path() is None

    def test_resolved_dump_path_resolves_against_run_dir(self, tmp_path):
        run_dir = tmp_path / "AAPL_2026-08-23_run1"
        _make_prompt_dump(run_dir, "run-1", [{"role": "HumanMessage", "content": "hi"}])
        record = CallRecord.from_jsonl_dict(
            _call_record(prompt_dump_path="prompts/run-1.json"), run_dir=run_dir
        )
        resolved = record.resolved_dump_path()
        assert resolved is not None
        assert resolved.is_absolute()
        assert resolved.exists()
        assert resolved.read_text(encoding="utf-8")

    def test_resolved_dump_path_distinguishes_same_ticker_different_runs(self, tmp_path):
        """Two runs of the same ticker must not resolve to each other's dump file."""
        run1 = tmp_path / "AAPL_2026-08-23_run1"
        run2 = tmp_path / "AAPL_2026-08-23_run2"
        _make_prompt_dump(run1, "run-1", [{"role": "HumanMessage", "content": "from run1"}])
        _make_prompt_dump(run2, "run-1", [{"role": "HumanMessage", "content": "from run2"}])

        record1 = CallRecord.from_jsonl_dict(
            _call_record(prompt_dump_path="prompts/run-1.json"), run_dir=run1
        )
        record2 = CallRecord.from_jsonl_dict(
            _call_record(prompt_dump_path="prompts/run-1.json"), run_dir=run2
        )

        content1 = json.loads(record1.resolved_dump_path().read_text(encoding="utf-8"))
        content2 = json.loads(record2.resolved_dump_path().read_text(encoding="utf-8"))
        assert content1["messages"][0]["content"] == "from run1"
        assert content2["messages"][0]["content"] == "from run2"


class TestPerAgentStats:
    def test_add_call_increments_count(self):
        stats = PerAgentStats(agent="Test Agent")
        stats.add_call(_record(agent="Test Agent", duration_seconds=1.0))
        assert stats.call_count == 1
        assert stats.total_wall_time == 1.0

    def test_compute_stats_with_single_call(self):
        stats = PerAgentStats(agent="Test")
        stats.add_call(
            _record(
                agent="Test",
                prompt_tokens_estimated=100,
                input_tokens=95,
                output_tokens=20,
                duration_seconds=1.0,
            )
        )
        computed = stats.compute_stats()
        assert computed["call_count"] == 1
        assert computed["error_count"] == 0
        assert computed["estimated_prompt_tokens_max"] == 100
        assert computed["output_tokens_total"] == 20

    def test_compute_stats_with_multiple_calls(self):
        stats = PerAgentStats(agent="Test")
        for i in range(3):
            stats.add_call(
                _record(
                    agent="Test",
                    run_id=f"run-{i}",
                    prompt_tokens_estimated=100 + i * 25,
                    input_tokens=95 + i * 20,
                    output_tokens=20 + i * 5,
                    duration_seconds=1.0 + i * 0.5,
                )
            )
        computed = stats.compute_stats()
        assert computed["call_count"] == 3
        assert computed["estimated_prompt_tokens_mean"] is not None
        assert computed["estimated_prompt_tokens_max"] == 150
        assert computed["output_tokens_total"] == 75  # 20 + 25 + 30

    def test_error_count_incremented_on_error_record(self):
        stats = PerAgentStats(agent="Test")
        stats.add_call(
            _record(
                agent="Test",
                input_tokens=None,
                output_tokens=None,
                error="APIError: timeout",
            )
        )
        computed = stats.compute_stats()
        assert computed["error_count"] == 1

    def test_compute_stats_keys_match_agent_sort_keys(self):
        """Every AGENT_SORT_KEYS entry must be an actual compute_stats() key (design review #2)."""
        stats = PerAgentStats(agent="Test")
        stats.add_call(_record(agent="Test"))
        computed = stats.compute_stats()
        for key in AGENT_SORT_KEYS:
            assert key in computed


# ─── Truncation detection ───────────────────────────────────────────────────


class TestTruncationDetection:
    def test_no_truncation_when_ratio_high(self):
        records = [_record(prompt_tokens_estimated=100, input_tokens=95)]
        candidates = detect_truncation_candidates(records, threshold=0.8)
        assert len(candidates) == 0

    def test_truncation_when_ratio_low(self, tmp_path):
        run_dir = tmp_path / "AAPL_2026-08-23_run1"
        records = [
            _record(
                prompt_tokens_estimated=100,
                input_tokens=60,
                prompt_dump_path="prompts/run-1.json",
                run_dir=run_dir,
            )
        ]
        candidates = detect_truncation_candidates(records, threshold=0.8)
        assert len(candidates) == 1
        assert candidates[0].ratio == pytest.approx(0.6)
        # run_dir must be the actual run directory name, not the ticker (design review #4).
        assert candidates[0].run_dir == run_dir.name
        assert candidates[0].run_dir != "AAPL"

    def test_truncation_candidates_sorted_by_ratio(self):
        records = [
            _record(run_id=f"run-{i}", prompt_tokens_estimated=100, input_tokens=int(pct))
            for i, pct in enumerate([70, 50, 90, 40])
        ]
        candidates = detect_truncation_candidates(records, threshold=0.8)
        assert candidates[0].ratio < candidates[-1].ratio

    def test_skip_records_with_none_input_tokens(self):
        records = [_record(prompt_tokens_estimated=100, input_tokens=None)]
        candidates = detect_truncation_candidates(records)
        assert len(candidates) == 0

    def test_threshold_is_configurable(self):
        """The threshold must actually be usable as a tunable knob (design review #7)."""
        records = [_record(prompt_tokens_estimated=100, input_tokens=85)]
        assert detect_truncation_candidates(records, threshold=0.8) == []
        assert len(detect_truncation_candidates(records, threshold=0.9)) == 1


# ─── Directory loading ───────────────────────────────────────────────────────


class TestDirectoryLoading:
    def test_nonexistent_reports_dir_returns_empty_agg(self, tmp_path):
        agg = load_llm_calls_from_directory(tmp_path / "nonexistent")
        assert agg.run_count == 0
        assert agg.call_count == 0

    def test_empty_reports_dir_returns_empty_agg(self, tmp_path):
        agg = load_llm_calls_from_directory(tmp_path)
        assert agg.run_count == 0
        assert agg.call_count == 0

    def test_skip_dir_without_llm_calls_jsonl(self, tmp_path):
        run_dir = tmp_path / "AAPL_2026-08-23_20260823_091531"
        run_dir.mkdir()
        (run_dir / "prompts").mkdir()  # Has prompts dir but no jsonl
        agg = load_llm_calls_from_directory(tmp_path)
        assert agg.run_count == 0
        assert len(agg.skipped_dirs) == 1

    def test_load_single_call_from_jsonl(self, tmp_path):
        run_dir = tmp_path / "AAPL_2026-08-23_test"
        _make_llm_calls_jsonl(run_dir, [_call_record()])
        agg = load_llm_calls_from_directory(tmp_path)
        assert agg.run_count == 1
        assert agg.call_count == 1
        assert "Market Analyst" in agg.per_agent_stats
        assert agg.per_agent_stats["Market Analyst"].call_count == 1

    def test_records_carry_their_source_run_dir(self, tmp_path):
        run_dir = tmp_path / "AAPL_2026-08-23_test"
        _make_llm_calls_jsonl(run_dir, [_call_record()])
        agg = load_llm_calls_from_directory(tmp_path)
        assert agg.all_records[0].run_dir == run_dir

    def test_load_multiple_agents_from_single_run(self, tmp_path):
        run_dir = tmp_path / "AAPL_2026-08-23_test"
        _make_llm_calls_jsonl(
            run_dir,
            [
                _call_record(run_id="run-1", agent="Market Analyst"),
                _call_record(run_id="run-2", agent="Researcher"),
            ],
        )
        agg = load_llm_calls_from_directory(tmp_path)
        assert agg.call_count == 2
        assert len(agg.per_agent_stats) == 2
        assert agg.per_agent_stats["Market Analyst"].call_count == 1
        assert agg.per_agent_stats["Researcher"].call_count == 1

    def test_load_multiple_runs(self, tmp_path):
        for ticker in ["AAPL", "MSFT"]:
            run_dir = tmp_path / f"{ticker}_2026-08-23_test"
            _make_llm_calls_jsonl(
                run_dir, [_call_record(run_id=f"run-{ticker}", ticker=ticker, agent="Analyst")]
            )
        agg = load_llm_calls_from_directory(tmp_path)
        assert agg.run_count == 2
        assert agg.call_count == 2

    def test_skip_malformed_json_in_jsonl(self, tmp_path):
        run_dir = tmp_path / "AAPL_2026-08-23_test"
        run_dir.mkdir(parents=True)
        jsonl_path = run_dir / "llm_calls.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            f.write('{"valid": "json"}\n')
            f.write("not valid json at all\n")
        agg = load_llm_calls_from_directory(tmp_path)
        assert len(agg.skipped_dirs) > 0

    def test_aggregate_wall_time_from_multiple_calls(self, tmp_path):
        run_dir = tmp_path / "AAPL_2026-08-23_test"
        _make_llm_calls_jsonl(
            run_dir,
            [
                _call_record(run_id="run-1", agent="A1", duration_seconds=1.5),
                _call_record(run_id="run-2", agent="A1", duration_seconds=2.5),
            ],
        )
        agg = load_llm_calls_from_directory(tmp_path)
        assert agg.total_wall_time == pytest.approx(4.0)

    def test_models_seen_and_model_counts(self, tmp_path):
        run_dir = tmp_path / "AAPL_2026-08-23_test"
        _make_llm_calls_jsonl(
            run_dir,
            [
                _call_record(run_id="run-1", agent="A1", model="gpt-4"),
                _call_record(run_id="run-2", agent="A2", model="gpt-4-turbo"),
                _call_record(run_id="run-3", agent="A1", model="gpt-4"),
            ],
        )
        agg = load_llm_calls_from_directory(tmp_path)
        assert agg.models_seen == {"gpt-4", "gpt-4-turbo"}
        # "model mix" means counts, not just distinct names.
        assert agg.model_counts["gpt-4"] == 2
        assert agg.model_counts["gpt-4-turbo"] == 1


class TestCorpusAggregationEndToEnd:
    def test_realistic_multi_ticker_multi_agent_corpus(self, tmp_path):
        tickers = ["AAPL", "MSFT", "NVDA"]
        agents = ["Market Analyst", "Sentiment Analyst", "Researcher"]
        models = ["gpt-4", "gpt-4-turbo"]

        run_id_counter = 0
        for ticker in tickers:
            run_dir = tmp_path / f"{ticker}_2026-08-23_test_{len(tickers)}"
            records = []
            for agent in agents:
                for model in models:
                    run_id_counter += 1
                    records.append(
                        _call_record(
                            run_id=f"run-{run_id_counter}",
                            ticker=ticker,
                            agent=agent,
                            model=model,
                            prompt_chars=400 + run_id_counter * 50,
                            prompt_tokens_estimated=100 + run_id_counter * 12,
                            input_tokens=95 + run_id_counter * 10,
                            output_tokens=20 + run_id_counter * 2,
                            duration_seconds=1.0 + run_id_counter * 0.1,
                        )
                    )
            _make_llm_calls_jsonl(run_dir, records)

        agg = load_llm_calls_from_directory(tmp_path)

        assert agg.run_count == 3
        assert agg.call_count == 3 * len(agents) * len(models)
        assert len(agg.per_agent_stats) == len(agents)
        for agent in agents:
            assert agent in agg.per_agent_stats
            assert agg.per_agent_stats[agent].call_count == 3 * len(models)
        assert len(agg.models_seen) == len(models)

    def test_corpus_with_mix_of_valid_and_invalid_dirs(self, tmp_path):
        run1 = tmp_path / "AAPL_2026-08-23_001"
        _make_llm_calls_jsonl(run1, [_call_record()])

        run2 = tmp_path / "MSFT_2026-08-23_002"
        run2.mkdir()  # no jsonl: skipped

        run3 = tmp_path / "NVDA_2026-08-23_003"
        run3.mkdir()
        (run3 / "llm_calls.jsonl").write_text("not json")  # malformed: skipped

        agg = load_llm_calls_from_directory(tmp_path)

        assert agg.run_count == 2  # run1 and run3 both have llm_calls.jsonl
        assert agg.call_count == 1  # only run1 parsed successfully
        assert len(agg.skipped_dirs) >= 1

    def test_missing_prompt_dump_file_does_not_crash_loading(self, tmp_path):
        """A prompt_dump_path may point at a file that was never written (opt-in dumps)."""
        run_dir = tmp_path / "AAPL_2026-08-23_001"
        _make_llm_calls_jsonl(
            run_dir, [_call_record(prompt_dump_path="prompts/does-not-exist.json")]
        )
        agg = load_llm_calls_from_directory(tmp_path)
        assert agg.call_count == 1
        record = agg.all_records[0]
        # Path resolution itself must not require the file to exist.
        assert record.resolved_dump_path() is not None
        assert not record.resolved_dump_path().exists()


# ─── Report data model (sort, cap, path resolution) ─────────────────────────


class TestBuildReportData:
    def _corpus(self, tmp_path):
        run_dir = tmp_path / "AAPL_2026-08-23_test"
        _make_llm_calls_jsonl(
            run_dir,
            [
                _call_record(
                    run_id="run-1",
                    agent="Fast Agent",
                    prompt_tokens_estimated=50,
                    duration_seconds=0.5,
                ),
                _call_record(
                    run_id="run-2",
                    agent="Slow Big Agent",
                    prompt_tokens_estimated=500,
                    duration_seconds=10.0,
                ),
            ],
        )
        return load_llm_calls_from_directory(tmp_path)

    def test_sort_by_estimated_prompt_tokens_max_changes_order(self, tmp_path):
        agg = self._corpus(tmp_path)
        data = build_report_data(agg, sort_by="estimated_prompt_tokens_max")
        agents_in_order = [row["agent"] for row in data["per_agent"]]
        assert agents_in_order[0] == "Slow Big Agent"

    def test_sort_by_duration_total_changes_order(self, tmp_path):
        agg = self._corpus(tmp_path)
        data = build_report_data(agg, sort_by="duration_total")
        agents_in_order = [row["agent"] for row in data["per_agent"]]
        assert agents_in_order[0] == "Slow Big Agent"

    def test_different_sort_keys_produce_different_orders(self, tmp_path):
        """Regression test for design review #2: --sort-by must actually change row order."""
        run_dir = tmp_path / "AAPL_2026-08-23_test"
        _make_llm_calls_jsonl(
            run_dir,
            [
                _call_record(
                    run_id="run-1",
                    agent="ManyCallsAgent",
                    prompt_tokens_estimated=10,
                    duration_seconds=0.1,
                ),
                _call_record(
                    run_id="run-2",
                    agent="ManyCallsAgent",
                    prompt_tokens_estimated=10,
                    duration_seconds=0.1,
                ),
                _call_record(
                    run_id="run-3",
                    agent="BigPromptAgent",
                    prompt_tokens_estimated=9999,
                    duration_seconds=0.1,
                ),
            ],
        )
        agg = load_llm_calls_from_directory(tmp_path)

        by_calls = [r["agent"] for r in build_report_data(agg, sort_by="call_count")["per_agent"]]
        by_tokens = [
            r["agent"]
            for r in build_report_data(agg, sort_by="estimated_prompt_tokens_max")["per_agent"]
        ]
        assert by_calls[0] == "ManyCallsAgent"
        assert by_tokens[0] == "BigPromptAgent"
        assert by_calls != by_tokens

    def test_top_n_caps_top_prompts_and_truncation_candidates(self, tmp_path):
        run_dir = tmp_path / "AAPL_2026-08-23_test"
        records = [
            _call_record(run_id=f"run-{i}", prompt_tokens_estimated=100 + i, input_tokens=10)
            for i in range(30)
        ]
        _make_llm_calls_jsonl(run_dir, records)
        agg = load_llm_calls_from_directory(tmp_path)

        data = build_report_data(agg, top_n=5)
        assert len(data["top_prompts"]) == 5
        assert len(data["truncation_candidates"]) == 5

        data_more = build_report_data(agg, top_n=15)
        assert len(data_more["top_prompts"]) == 15
        assert len(data_more["truncation_candidates"]) == 15

    def test_truncation_threshold_is_wired_through(self, tmp_path):
        run_dir = tmp_path / "AAPL_2026-08-23_test"
        _make_llm_calls_jsonl(run_dir, [_call_record(prompt_tokens_estimated=100, input_tokens=85)])
        agg = load_llm_calls_from_directory(tmp_path)

        assert build_report_data(agg, truncation_threshold=0.8)["truncation_candidates"] == []
        assert len(build_report_data(agg, truncation_threshold=0.9)["truncation_candidates"]) == 1

    def test_top_prompts_include_run_dir_and_openable_dump_path(self, tmp_path):
        run_dir = tmp_path / "AAPL_2026-08-23_test"
        _make_prompt_dump(run_dir, "run-1", [{"role": "HumanMessage", "content": "hi"}])
        _make_llm_calls_jsonl(
            run_dir, [_call_record(run_id="run-1", prompt_dump_path="prompts/run-1.json")]
        )
        agg = load_llm_calls_from_directory(tmp_path)
        data = build_report_data(agg)
        top = data["top_prompts"][0]
        assert top["run_dir"] == run_dir.name
        assert Path(top["dump_path"]).exists()

    def test_model_mix_reported_in_corpus_summary(self, tmp_path):
        run_dir = tmp_path / "AAPL_2026-08-23_test"
        _make_llm_calls_jsonl(
            run_dir,
            [
                _call_record(run_id="run-1", model="gpt-4"),
                _call_record(run_id="run-2", model="gpt-4"),
                _call_record(run_id="run-3", model="gpt-4-turbo"),
            ],
        )
        agg = load_llm_calls_from_directory(tmp_path)
        data = build_report_data(agg)
        assert data["corpus"]["model_counts"] == {"gpt-4": 2, "gpt-4-turbo": 1}


# ─── Rendering: text / markdown / json must genuinely differ ───────────────


class TestRenderers:
    def _data(self, tmp_path):
        run_dir = tmp_path / "AAPL_2026-08-23_test"
        _make_llm_calls_jsonl(
            run_dir,
            [_call_record(run_id="run-1", prompt_tokens_estimated=100, input_tokens=50)],
        )
        agg = load_llm_calls_from_directory(tmp_path)
        return build_report_data(agg)

    def test_json_report_round_trips_and_has_expected_shape(self, tmp_path):
        data = self._data(tmp_path)
        parsed = json.loads(render_json_report(data))
        assert parsed["corpus"]["call_count"] == 1
        assert "per_agent" in parsed
        assert "top_prompts" in parsed
        assert "truncation_candidates" in parsed

    def test_markdown_report_uses_markdown_tables(self, tmp_path):
        data = self._data(tmp_path)
        md = render_markdown_report(data)
        assert "| Agent |" in md
        assert md.startswith("# LLM Calls Analysis Report")

    def test_text_report_is_not_markdown(self, tmp_path):
        """Design review #6: text and markdown must be genuinely distinct renderers."""
        data = self._data(tmp_path)
        text = render_text_report(data)
        assert "|" not in text
        assert "# " not in text
        assert "**" not in text

    def test_text_and_markdown_reports_differ(self, tmp_path):
        data = self._data(tmp_path)
        assert render_text_report(data) != render_markdown_report(data)

    def test_truncation_column_present_in_all_three_formats(self, tmp_path):
        run_dir = tmp_path / "AAPL_2026-08-23_test"
        _make_llm_calls_jsonl(
            run_dir,
            [_call_record(run_id="run-1", prompt_tokens_estimated=100, input_tokens=50)],
        )
        agg = load_llm_calls_from_directory(tmp_path)
        data = build_report_data(agg, truncation_threshold=0.8)
        assert data["truncation_candidates"], "fixture must actually produce a candidate"

        assert "Truncation" in render_text_report(data)
        assert "Truncation" in render_markdown_report(data)
        parsed = json.loads(render_json_report(data))
        assert len(parsed["truncation_candidates"]) == 1


# ─── Segment mode rendering ──────────────────────────────────────────────────


class TestSegmentMode:
    def _agg_with_dump(self, tmp_path, content: str):
        run_dir = tmp_path / "AAPL_2026-08-23_test"
        _make_prompt_dump(run_dir, "run-1", [{"role": "HumanMessage", "content": content}])
        _make_llm_calls_jsonl(
            run_dir,
            [
                _call_record(
                    run_id="run-1",
                    prompt_tokens_estimated=len(content) // 4,
                    prompt_dump_path="prompts/run-1.json",
                )
            ],
        )
        return load_llm_calls_from_directory(tmp_path)

    def test_segment_json_format_is_valid_json_and_distinct_from_text(self, tmp_path):
        """Design review #5: --segment --format json must respect --format, not always emit text."""
        agg = self._agg_with_dump(tmp_path, "Header\nAnalyst Reports:\nSome report body")
        data = build_segment_data(agg, top_n=5)

        json_report = render_segment_report(data, report_format="json")
        parsed = json.loads(json_report)  # must not raise
        assert parsed["prompts"][0]["segments"] is not None

        text_report = render_segment_report(data, report_format="text")
        assert text_report != json_report
        assert "|" not in text_report

        md_report = render_segment_report(data, report_format="markdown")
        assert "| Segment |" in md_report
        assert md_report != text_report

    def test_segment_mode_reports_run_dir_not_ticker(self, tmp_path):
        agg = self._agg_with_dump(tmp_path, "Header\nAnalyst Reports:\nbody")
        data = build_segment_data(agg, top_n=5)
        assert data["prompts"][0]["run_dir"] == "AAPL_2026-08-23_test"

    def test_segment_mode_handles_missing_dump_file(self, tmp_path):
        run_dir = tmp_path / "AAPL_2026-08-23_test"
        _make_llm_calls_jsonl(
            run_dir,
            [_call_record(run_id="run-1", prompt_dump_path="prompts/missing.json")],
        )
        agg = load_llm_calls_from_directory(tmp_path)
        data = build_segment_data(agg, top_n=5)
        assert data["missing_dump_count"] == 1
        assert data["prompts"][0]["error"] is not None
        # Must not raise when rendered either.
        render_segment_report(data, report_format="text")
        render_segment_report(data, report_format="json")

    def test_segment_mode_handles_malformed_dump_json(self, tmp_path):
        run_dir = tmp_path / "AAPL_2026-08-23_test"
        prompts_dir = run_dir / "prompts"
        prompts_dir.mkdir(parents=True)
        (prompts_dir / "run-1.json").write_text("not valid json")
        _make_llm_calls_jsonl(
            run_dir,
            [_call_record(run_id="run-1", prompt_dump_path="prompts/run-1.json")],
        )
        agg = load_llm_calls_from_directory(tmp_path)
        data = build_segment_data(agg, top_n=5)
        assert data["missing_dump_count"] == 1
        assert "malformed" in data["prompts"][0]["error"]

    def test_segment_mode_resolves_correct_run_when_same_ticker_has_multiple_runs(self, tmp_path):
        """Regression for design review #3/#4: fuzzy ticker-substring matching over an
        unsorted directory listing could previously pick the wrong run."""
        run1 = tmp_path / "AAPL_2026-08-23_run1"
        run2 = tmp_path / "AAPL_2026-08-23_run2"
        _make_prompt_dump(run1, "run-a", [{"role": "HumanMessage", "content": "Analyst Reports:\nfrom run1"}])
        _make_prompt_dump(run2, "run-b", [{"role": "HumanMessage", "content": "Analyst Reports:\nfrom run2 much much bigger content here padding padding"}])
        _make_llm_calls_jsonl(
            run1,
            [
                _call_record(
                    run_id="run-a",
                    ticker="AAPL",
                    prompt_tokens_estimated=10,
                    prompt_dump_path="prompts/run-a.json",
                )
            ],
        )
        _make_llm_calls_jsonl(
            run2,
            [
                _call_record(
                    run_id="run-b",
                    ticker="AAPL",
                    prompt_tokens_estimated=9999,
                    prompt_dump_path="prompts/run-b.json",
                )
            ],
        )
        agg = load_llm_calls_from_directory(tmp_path)
        data = build_segment_data(agg, top_n=1)
        # The largest-estimated-tokens record (run-b, in run2) must resolve its own dump.
        assert data["prompts"][0]["run_id"] == "run-b"
        assert data["prompts"][0]["run_dir"] == "AAPL_2026-08-23_run2"
        assert data["prompts"][0]["error"] is None


# ─── CLI (main) ──────────────────────────────────────────────────────────────


class TestCLI:
    def _make_corpus(self, base: Path):
        run_dir = base / "reports" / "AAPL_2026-08-23_test"
        _make_llm_calls_jsonl(
            run_dir,
            [_call_record(run_id="run-1", prompt_tokens_estimated=100, input_tokens=50)],
        )

    def _run(self, tmp_path, *extra_args):
        self._make_corpus(tmp_path)
        cmd = [
            sys.executable,
            "scripts/analyze_llm_calls.py",
            "--reports-dir",
            str(tmp_path / "reports"),
            *extra_args,
        ]
        return subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).resolve().parent.parent)

    def test_default_invocation_succeeds(self, tmp_path):
        result = self._run(tmp_path)
        assert result.returncode == 0, result.stderr
        assert "LLM CALLS ANALYSIS REPORT" in result.stdout

    def test_format_json_produces_valid_json(self, tmp_path):
        result = self._run(tmp_path, "--format", "json")
        assert result.returncode == 0, result.stderr
        json.loads(result.stdout)  # must not raise

    def test_format_markdown_differs_from_text(self, tmp_path):
        text_result = self._run(tmp_path, "--format", "text")
        md_result = self._run(tmp_path, "--format", "markdown")
        assert text_result.stdout != md_result.stdout
        assert "|" in md_result.stdout

    def test_invalid_sort_by_is_an_argparse_error_not_a_silent_noop(self, tmp_path):
        result = self._run(tmp_path, "--sort-by", "not_a_real_key")
        assert result.returncode != 0
        assert "invalid choice" in result.stderr

    def test_sort_by_estimated_prompt_tokens_max_is_accepted(self, tmp_path):
        """The exact example given in --help must actually work (design review #2)."""
        result = self._run(tmp_path, "--sort-by", "estimated_prompt_tokens_max")
        assert result.returncode == 0, result.stderr

    def test_segment_format_json_is_valid_json(self, tmp_path):
        result = self._run(tmp_path, "--segment", "--format", "json")
        assert result.returncode == 0, result.stderr
        json.loads(result.stdout)  # must not raise; previously ignored --format entirely

    def test_out_flag_writes_file(self, tmp_path):
        self._make_corpus(tmp_path)
        out_path = tmp_path / "report.md"
        cmd = [
            sys.executable,
            "scripts/analyze_llm_calls.py",
            "--reports-dir",
            str(tmp_path / "reports"),
            "--format",
            "markdown",
            "--out",
            str(out_path),
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=Path(__file__).resolve().parent.parent
        )
        assert result.returncode == 0, result.stderr
        assert out_path.exists()
        assert "# LLM Calls Analysis Report" in out_path.read_text(encoding="utf-8")
