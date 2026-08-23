"""Tests for scripts/analyze_llm_calls.py (issue #143).

Covers aggregation math, segment splitting, truncation detection, and robustness
to missing/partial directories using synthetic fixture data.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.analyze_llm_calls import (
    CallRecord,
    PerAgentStats,
    _split_prompt_into_segments,
    detect_truncation_candidates,
    load_llm_calls_from_directory,
)

pytestmark = pytest.mark.unit


# ─── Fixture helpers ────────────────────────────────────────────────────────


def _make_llm_calls_jsonl(
    report_dir: Path, records: list[dict]
) -> None:
    """Write synthetic llm_calls.jsonl records."""
    report_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = report_dir / "llm_calls.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def _make_prompt_dump(
    report_dir: Path, run_id: str, messages: list[dict]
) -> Path:
    """Write a synthetic prompt dump JSON file."""
    prompts_dir = report_dir / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    dump_path = prompts_dir / f"{run_id}.json"
    dump_data = {"format": "chat_messages", "messages": messages}
    dump_path.write_text(json.dumps(dump_data), encoding="utf-8")
    return dump_path


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
        # Should have header + tools segment
        assert len(segments) == 2
        assert segments[0][0] == "header/instructions"
        assert "**Available Tools:**" in segments[1][1]

    def test_prompt_with_multiple_anchors_in_order(self):
        text = "Header\nPast analyses of X\nAnalyst Reports:\nReports here"
        segments = _split_prompt_into_segments(text)
        # Header, then segments for each anchor
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
        # All segment names should be lowercase with underscores
        for seg_name, _, _ in segments:
            assert seg_name.islower() or seg_name == "full_prompt" or seg_name == "empty_prompt"

    def test_no_anchors_detected_when_all_missing(self):
        text = "Some random text without any of the standard anchors."
        segments = _split_prompt_into_segments(text)
        assert len(segments) == 1
        assert segments[0][0] == "full_prompt"


# ─── Call record aggregation ────────────────────────────────────────────────


class TestCallRecord:
    def test_from_jsonl_dict_complete(self):
        record_dict = {
            "timestamp": "2026-08-23T12:00:00Z",
            "run_id": "run-123",
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
            "prompt_dump_path": "prompts/run-123.json",
        }
        record = CallRecord.from_jsonl_dict(record_dict)
        assert record.agent == "Market Analyst"
        assert record.prompt_tokens_estimated == 125
        assert record.error is None

    def test_from_jsonl_dict_missing_optional_fields(self):
        minimal_dict = {
            "agent": "Unknown",
            "timestamp": "2026-08-23T12:00:00Z",
        }
        record = CallRecord.from_jsonl_dict(minimal_dict)
        assert record.agent == "Unknown"
        assert record.input_tokens is None
        assert record.error is None


class TestPerAgentStats:
    def test_add_call_increments_count(self):
        stats = PerAgentStats(agent="Test Agent")
        record = CallRecord(
            timestamp="2026-08-23T12:00:00Z",
            run_id="run-1",
            ticker="AAPL",
            date="2026-08-23",
            agent="Test Agent",
            model="gpt-4",
            message_count=2,
            prompt_chars=400,
            prompt_tokens_estimated=100,
            input_tokens=95,
            output_tokens=20,
            duration_seconds=1.0,
            error=None,
            prompt_dump_path=None,
        )
        stats.add_call(record)
        assert stats.call_count == 1
        assert stats.total_wall_time == 1.0

    def test_compute_stats_with_single_call(self):
        stats = PerAgentStats(agent="Test")
        record = CallRecord(
            timestamp="2026-08-23T12:00:00Z",
            run_id="run-1",
            ticker="AAPL",
            date="2026-08-23",
            agent="Test",
            model="gpt-4",
            message_count=1,
            prompt_chars=400,
            prompt_tokens_estimated=100,
            input_tokens=95,
            output_tokens=20,
            duration_seconds=1.0,
            error=None,
            prompt_dump_path=None,
        )
        stats.add_call(record)
        computed = stats.compute_stats()
        assert computed["call_count"] == 1
        assert computed["error_count"] == 0
        assert computed["estimated_prompt_tokens_max"] == 100
        assert computed["output_tokens_total"] == 20

    def test_compute_stats_with_multiple_calls(self):
        stats = PerAgentStats(agent="Test")
        for i in range(3):
            record = CallRecord(
                timestamp=f"2026-08-23T12:00:{i}0Z",
                run_id=f"run-{i}",
                ticker="AAPL",
                date="2026-08-23",
                agent="Test",
                model="gpt-4",
                message_count=1,
                prompt_chars=400 + i * 100,
                prompt_tokens_estimated=100 + i * 25,
                input_tokens=95 + i * 20,
                output_tokens=20 + i * 5,
                duration_seconds=1.0 + i * 0.5,
                error=None,
                prompt_dump_path=None,
            )
            stats.add_call(record)

        computed = stats.compute_stats()
        assert computed["call_count"] == 3
        assert computed["estimated_prompt_tokens_mean"] is not None
        assert computed["estimated_prompt_tokens_max"] == 150
        assert computed["output_tokens_total"] == 75  # 20 + 25 + 30

    def test_error_count_incremented_on_error_record(self):
        stats = PerAgentStats(agent="Test")
        error_record = CallRecord(
            timestamp="2026-08-23T12:00:00Z",
            run_id="run-1",
            ticker="AAPL",
            date="2026-08-23",
            agent="Test",
            model="gpt-4",
            message_count=1,
            prompt_chars=400,
            prompt_tokens_estimated=100,
            input_tokens=None,
            output_tokens=None,
            duration_seconds=0.5,
            error="APIError: timeout",
            prompt_dump_path=None,
        )
        stats.add_call(error_record)
        computed = stats.compute_stats()
        assert computed["error_count"] == 1


# ─── Truncation detection ───────────────────────────────────────────────────


class TestTruncationDetection:
    def test_no_truncation_when_ratio_high(self):
        records = [
            CallRecord(
                timestamp="2026-08-23T12:00:00Z",
                run_id="run-1",
                ticker="AAPL",
                date="2026-08-23",
                agent="Analyst",
                model="gpt-4",
                message_count=1,
                prompt_chars=400,
                prompt_tokens_estimated=100,
                input_tokens=95,  # 95% of 100
                output_tokens=20,
                duration_seconds=1.0,
                error=None,
                prompt_dump_path=None,
            )
        ]
        candidates = detect_truncation_candidates(records, threshold=0.8)
        assert len(candidates) == 0

    def test_truncation_when_ratio_low(self):
        records = [
            CallRecord(
                timestamp="2026-08-23T12:00:00Z",
                run_id="run-1",
                ticker="AAPL",
                date="2026-08-23",
                agent="Analyst",
                model="gpt-4",
                message_count=1,
                prompt_chars=400,
                prompt_tokens_estimated=100,
                input_tokens=60,  # 60% of 100
                output_tokens=20,
                duration_seconds=1.0,
                error=None,
                prompt_dump_path="prompts/run-1.json",
            )
        ]
        candidates = detect_truncation_candidates(records, threshold=0.8)
        assert len(candidates) == 1
        assert candidates[0].ratio == pytest.approx(0.6)

    def test_truncation_candidates_sorted_by_ratio(self):
        records = []
        for i, ratio_pct in enumerate([70, 50, 90, 40]):
            est_tokens = 100
            rep_tokens = int(est_tokens * ratio_pct / 100)
            records.append(
                CallRecord(
                    timestamp=f"2026-08-23T12:00:{i}0Z",
                    run_id=f"run-{i}",
                    ticker="AAPL",
                    date="2026-08-23",
                    agent="Analyst",
                    model="gpt-4",
                    message_count=1,
                    prompt_chars=400,
                    prompt_tokens_estimated=est_tokens,
                    input_tokens=rep_tokens,
                    output_tokens=20,
                    duration_seconds=1.0,
                    error=None,
                    prompt_dump_path=None,
                )
            )
        candidates = detect_truncation_candidates(records, threshold=0.8)
        # Should be sorted by ratio (lowest first)
        assert candidates[0].ratio < candidates[-1].ratio

    def test_skip_records_with_none_input_tokens(self):
        records = [
            CallRecord(
                timestamp="2026-08-23T12:00:00Z",
                run_id="run-1",
                ticker="AAPL",
                date="2026-08-23",
                agent="Analyst",
                model="gpt-4",
                message_count=1,
                prompt_chars=400,
                prompt_tokens_estimated=100,
                input_tokens=None,  # Not reported
                output_tokens=20,
                duration_seconds=1.0,
                error=None,
                prompt_dump_path=None,
            )
        ]
        candidates = detect_truncation_candidates(records)
        assert len(candidates) == 0


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
        _make_llm_calls_jsonl(
            run_dir,
            [
                {
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
            ],
        )
        agg = load_llm_calls_from_directory(tmp_path)
        assert agg.run_count == 1
        assert agg.call_count == 1
        assert "Market Analyst" in agg.per_agent_stats
        assert agg.per_agent_stats["Market Analyst"].call_count == 1

    def test_load_multiple_agents_from_single_run(self, tmp_path):
        run_dir = tmp_path / "AAPL_2026-08-23_test"
        _make_llm_calls_jsonl(
            run_dir,
            [
                {
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
                },
                {
                    "timestamp": "2026-08-23T12:01:00Z",
                    "run_id": "run-2",
                    "ticker": "AAPL",
                    "date": "2026-08-23",
                    "agent": "Researcher",
                    "model": "gpt-4",
                    "message_count": 3,
                    "prompt_chars": 1000,
                    "prompt_tokens_estimated": 250,
                    "input_tokens": 230,
                    "output_tokens": 100,
                    "duration_seconds": 3.0,
                    "error": None,
                    "prompt_dump_path": None,
                },
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
                run_dir,
                [
                    {
                        "timestamp": "2026-08-23T12:00:00Z",
                        "run_id": f"run-{ticker}",
                        "ticker": ticker,
                        "date": "2026-08-23",
                        "agent": "Analyst",
                        "model": "gpt-4",
                        "message_count": 1,
                        "prompt_chars": 400,
                        "prompt_tokens_estimated": 100,
                        "input_tokens": 90,
                        "output_tokens": 20,
                        "duration_seconds": 1.0,
                        "error": None,
                        "prompt_dump_path": None,
                    }
                ],
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
        # Should skip due to malformed JSON
        assert len(agg.skipped_dirs) > 0

    def test_aggregate_wall_time_from_multiple_calls(self, tmp_path):
        run_dir = tmp_path / "AAPL_2026-08-23_test"
        _make_llm_calls_jsonl(
            run_dir,
            [
                {
                    "timestamp": "2026-08-23T12:00:00Z",
                    "run_id": "run-1",
                    "ticker": "AAPL",
                    "date": "2026-08-23",
                    "agent": "A1",
                    "model": "gpt-4",
                    "message_count": 1,
                    "prompt_chars": 400,
                    "prompt_tokens_estimated": 100,
                    "input_tokens": 90,
                    "output_tokens": 20,
                    "duration_seconds": 1.5,
                    "error": None,
                    "prompt_dump_path": None,
                },
                {
                    "timestamp": "2026-08-23T12:01:00Z",
                    "run_id": "run-2",
                    "ticker": "AAPL",
                    "date": "2026-08-23",
                    "agent": "A1",
                    "model": "gpt-4",
                    "message_count": 1,
                    "prompt_chars": 400,
                    "prompt_tokens_estimated": 100,
                    "input_tokens": 90,
                    "output_tokens": 20,
                    "duration_seconds": 2.5,
                    "error": None,
                    "prompt_dump_path": None,
                },
            ],
        )
        agg = load_llm_calls_from_directory(tmp_path)
        assert agg.total_wall_time == pytest.approx(4.0)

    def test_models_seen_aggregated(self, tmp_path):
        run_dir = tmp_path / "AAPL_2026-08-23_test"
        _make_llm_calls_jsonl(
            run_dir,
            [
                {
                    "timestamp": "2026-08-23T12:00:00Z",
                    "run_id": "run-1",
                    "ticker": "AAPL",
                    "date": "2026-08-23",
                    "agent": "A1",
                    "model": "gpt-4",
                    "message_count": 1,
                    "prompt_chars": 400,
                    "prompt_tokens_estimated": 100,
                    "input_tokens": 90,
                    "output_tokens": 20,
                    "duration_seconds": 1.0,
                    "error": None,
                    "prompt_dump_path": None,
                },
                {
                    "timestamp": "2026-08-23T12:01:00Z",
                    "run_id": "run-2",
                    "ticker": "AAPL",
                    "date": "2026-08-23",
                    "agent": "A2",
                    "model": "gpt-4-turbo",
                    "message_count": 1,
                    "prompt_chars": 400,
                    "prompt_tokens_estimated": 100,
                    "input_tokens": 90,
                    "output_tokens": 20,
                    "duration_seconds": 1.0,
                    "error": None,
                    "prompt_dump_path": None,
                },
            ],
        )
        agg = load_llm_calls_from_directory(tmp_path)
        assert "gpt-4" in agg.models_seen
        assert "gpt-4-turbo" in agg.models_seen
        assert len(agg.models_seen) == 2


# ─── End-to-end corpus loading ───────────────────────────────────────────────


class TestCorpusAggregationEndToEnd:
    def test_realistic_multi_ticker_multi_agent_corpus(self, tmp_path):
        """Simulate a realistic corpus with multiple tickers and agents."""
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
                        {
                            "timestamp": f"2026-08-23T12:{run_id_counter:02d}:00Z",
                            "run_id": f"run-{run_id_counter}",
                            "ticker": ticker,
                            "date": "2026-08-23",
                            "agent": agent,
                            "model": model,
                            "message_count": 2,
                            "prompt_chars": 400 + run_id_counter * 50,
                            "prompt_tokens_estimated": 100 + run_id_counter * 12,
                            "input_tokens": 95 + run_id_counter * 10,
                            "output_tokens": 20 + run_id_counter * 2,
                            "duration_seconds": 1.0 + run_id_counter * 0.1,
                            "error": None,
                            "prompt_dump_path": None,
                        }
                    )
            _make_llm_calls_jsonl(run_dir, records)

        agg = load_llm_calls_from_directory(tmp_path)

        # Verify aggregation
        assert agg.run_count == 3  # One per ticker
        assert agg.call_count == 3 * len(agents) * len(models)
        assert len(agg.per_agent_stats) == len(agents)
        for agent in agents:
            assert agent in agg.per_agent_stats
            assert agg.per_agent_stats[agent].call_count == 3 * len(models)
        assert len(agg.models_seen) == len(models)

    def test_corpus_with_mix_of_valid_and_invalid_dirs(self, tmp_path):
        """Test robustness: mix of valid runs, missing jsonl, and malformed data."""
        # Valid run
        run1 = tmp_path / "AAPL_2026-08-23_001"
        _make_llm_calls_jsonl(run1, [{"timestamp": "2026-08-23T12:00:00Z", "run_id": "run-1", "ticker": "AAPL", "date": "2026-08-23", "agent": "A", "model": "m", "message_count": 1, "prompt_chars": 400, "prompt_tokens_estimated": 100, "input_tokens": 90, "output_tokens": 20, "duration_seconds": 1.0, "error": None, "prompt_dump_path": None}])

        # Dir without jsonl (should be skipped)
        run2 = tmp_path / "MSFT_2026-08-23_002"
        run2.mkdir()

        # Dir with malformed jsonl
        run3 = tmp_path / "NVDA_2026-08-23_003"
        run3.mkdir()
        (run3 / "llm_calls.jsonl").write_text("not json")

        agg = load_llm_calls_from_directory(tmp_path)

        # Should load the valid one, count both directories with jsonl files
        assert agg.run_count == 2  # Both run1 and run3 have llm_calls.jsonl
        assert agg.call_count == 1  # But only run1 was successfully parsed
        assert len(agg.skipped_dirs) >= 1  # run2 has no jsonl, run3 has malformed jsonl
