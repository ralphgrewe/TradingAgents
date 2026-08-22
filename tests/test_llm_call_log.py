"""Tests for the per-call LLM call log handler (issue #138, part of #137).

``LLMCallLogHandler`` (``tradingagents/llm_call_log.py``) appends one JSONL
record per LLM call, keyed by the LangChain callback ``run_id`` so
concurrently running analysts (``analyst_concurrency_limit``) pair up
start/end events correctly instead of racing on "the last call". These
tests feed synthetic callback events (no real LLM calls, per
``tests/conftest.py`` conventions) and assert on the resulting JSONL
records, run_id pairing/duration, thread-safety under interleaving, and the
end-of-run summary aggregation.
"""

from __future__ import annotations

import json
import threading
import uuid

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from tradingagents.llm_call_log import LLMCallLogHandler, summarize_records


def _chat_result(content: str, input_tokens=None, output_tokens=None) -> LLMResult:
    usage_metadata = None
    if input_tokens is not None or output_tokens is not None:
        usage_metadata = {
            "input_tokens": input_tokens or 0,
            "output_tokens": output_tokens or 0,
            "total_tokens": (input_tokens or 0) + (output_tokens or 0),
        }
    message = AIMessage(content=content, usage_metadata=usage_metadata)
    return LLMResult(generations=[[ChatGeneration(message=message)]])


def test_basic_call_produces_one_jsonl_record(tmp_path):
    log_path = tmp_path / "llm_calls.jsonl"
    handler = LLMCallLogHandler(log_path)

    run_id = uuid.uuid4()
    messages = [[HumanMessage(content="hello"), HumanMessage(content="world!")]]
    handler.on_chat_model_start(
        {"kwargs": {"model_name": "gpt-4o-mini"}},
        messages,
        run_id=run_id,
        metadata={"langgraph_node": "Market Analyst"},
    )
    handler.on_llm_end(
        _chat_result("response text", input_tokens=42, output_tokens=7),
        run_id=run_id,
    )

    assert log_path.exists()
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])

    assert record["agent"] == "Market Analyst"
    assert record["model"] == "gpt-4o-mini"
    assert record["message_count"] == 2
    assert record["prompt_chars"] == len("hello") + len("world!")
    assert record["prompt_tokens_estimated"] == record["prompt_chars"] // 4
    assert record["input_tokens"] == 42
    assert record["output_tokens"] == 7
    assert record["run_id"] == str(run_id)
    assert record["duration_seconds"] >= 0
    assert "timestamp" in record


def test_missing_langgraph_node_metadata_falls_back_to_unknown(tmp_path):
    handler = LLMCallLogHandler(tmp_path / "llm_calls.jsonl")
    run_id = uuid.uuid4()

    handler.on_chat_model_start(
        {"kwargs": {"model_name": "gpt-4o-mini"}},
        [[HumanMessage(content="hi")]],
        run_id=run_id,
        metadata=None,
    )
    handler.on_llm_end(_chat_result("ok"), run_id=run_id)

    record = handler.get_records()[0]
    assert record["agent"] == "unknown"


def test_missing_usage_metadata_reports_none(tmp_path):
    handler = LLMCallLogHandler(tmp_path / "llm_calls.jsonl")
    run_id = uuid.uuid4()

    handler.on_chat_model_start(
        {"kwargs": {"model_name": "local-model"}},
        [[HumanMessage(content="hi")]],
        run_id=run_id,
        metadata={"langgraph_node": "Trader"},
    )
    # No usage_metadata on the response (provider doesn't report it).
    handler.on_llm_end(_chat_result("ok"), run_id=run_id)

    record = handler.get_records()[0]
    assert record["input_tokens"] is None
    assert record["output_tokens"] is None


def test_end_without_matching_start_is_skipped_not_crashed(tmp_path):
    log_path = tmp_path / "llm_calls.jsonl"
    handler = LLMCallLogHandler(log_path)

    # No corresponding on_chat_model_start/on_llm_start was ever recorded.
    handler.on_llm_end(_chat_result("orphaned"), run_id=uuid.uuid4())

    assert handler.get_records() == []
    assert not log_path.exists()


def test_llm_start_legacy_prompts_path(tmp_path):
    handler = LLMCallLogHandler(tmp_path / "llm_calls.jsonl")
    run_id = uuid.uuid4()

    handler.on_llm_start(
        {"kwargs": {"model": "text-davinci-003"}},
        ["a short prompt", "another one"],
        run_id=run_id,
        metadata={"langgraph_node": "Legacy Node"},
    )
    handler.on_llm_end(_chat_result("done"), run_id=run_id)

    record = handler.get_records()[0]
    assert record["model"] == "text-davinci-003"
    assert record["message_count"] == 2
    assert record["prompt_chars"] == len("a short prompt") + len("another one")


def test_run_id_pairing_survives_interleaving(tmp_path):
    """Two overlapping calls (start A, start B, end B, end A) must not cross-pair."""
    handler = LLMCallLogHandler(tmp_path / "llm_calls.jsonl")
    run_a, run_b = uuid.uuid4(), uuid.uuid4()

    handler.on_chat_model_start(
        {"kwargs": {"model_name": "model-a"}},
        [[HumanMessage(content="A" * 40)]],
        run_id=run_a,
        metadata={"langgraph_node": "Node A"},
    )
    handler.on_chat_model_start(
        {"kwargs": {"model_name": "model-b"}},
        [[HumanMessage(content="B" * 8)]],
        run_id=run_b,
        metadata={"langgraph_node": "Node B"},
    )
    # B finishes first even though it started second.
    handler.on_llm_end(_chat_result("b done", input_tokens=1, output_tokens=1), run_id=run_b)
    handler.on_llm_end(_chat_result("a done", input_tokens=2, output_tokens=2), run_id=run_a)

    records = {r["agent"]: r for r in handler.get_records()}
    assert records["Node A"]["model"] == "model-a"
    assert records["Node A"]["prompt_chars"] == 40
    assert records["Node B"]["model"] == "model-b"
    assert records["Node B"]["prompt_chars"] == 8


def test_concurrent_calls_from_multiple_threads_all_recorded(tmp_path):
    """Simulates concurrent analysts (analyst_concurrency_limit > 1) hammering the handler."""
    handler = LLMCallLogHandler(tmp_path / "llm_calls.jsonl")
    n_threads = 8

    def worker(i: int) -> None:
        run_id = uuid.uuid4()
        handler.on_chat_model_start(
            {"kwargs": {"model_name": "m"}},
            [[HumanMessage(content="x" * (i + 1))]],
            run_id=run_id,
            metadata={"langgraph_node": f"Analyst {i}"},
        )
        handler.on_llm_end(_chat_result("y", input_tokens=i, output_tokens=i), run_id=run_id)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    records = handler.get_records()
    assert len(records) == n_threads
    agents = {r["agent"] for r in records}
    assert agents == {f"Analyst {i}" for i in range(n_threads)}

    # File on disk must also contain exactly one line per call (no torn writes).
    lines = (tmp_path / "llm_calls.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == n_threads
    for line in lines:
        json.loads(line)  # each line parses as standalone JSON


def test_disabled_handler_produces_no_file(tmp_path):
    log_path = tmp_path / "llm_calls.jsonl"
    handler = LLMCallLogHandler(log_path, enabled=False)
    run_id = uuid.uuid4()

    handler.on_chat_model_start(
        {"kwargs": {"model_name": "m"}},
        [[HumanMessage(content="hi")]],
        run_id=run_id,
        metadata={"langgraph_node": "Node"},
    )
    handler.on_llm_end(_chat_result("ok"), run_id=run_id)

    assert not log_path.exists()
    assert handler.get_records() == []
    assert handler.get_summary() == {}

    summary_path = tmp_path / "llm_calls_summary.json"
    handler.write_summary(summary_path)
    assert not summary_path.exists()


def test_none_log_path_is_equivalent_to_disabled():
    handler = LLMCallLogHandler(None)
    assert handler.enabled is False

    run_id = uuid.uuid4()
    handler.on_chat_model_start(
        {"kwargs": {"model_name": "m"}},
        [[HumanMessage(content="hi")]],
        run_id=run_id,
        metadata={"langgraph_node": "Node"},
    )
    handler.on_llm_end(_chat_result("ok"), run_id=run_id)
    assert handler.get_records() == []


def test_summary_aggregates_per_agent(tmp_path):
    handler = LLMCallLogHandler(tmp_path / "llm_calls.jsonl")

    def call(agent, content_len, input_tokens, output_tokens):
        run_id = uuid.uuid4()
        handler.on_chat_model_start(
            {"kwargs": {"model_name": "m"}},
            [[HumanMessage(content="x" * content_len)]],
            run_id=run_id,
            metadata={"langgraph_node": agent},
        )
        handler.on_llm_end(
            _chat_result("resp", input_tokens=input_tokens, output_tokens=output_tokens),
            run_id=run_id,
        )

    call("Market Analyst", 400, 100, 10)   # 100 estimated tokens
    call("Market Analyst", 40, 10, 2)      # 10 estimated tokens
    call("News Analyst", 8000, 2000, 50)   # 2000 estimated tokens

    summary = handler.get_summary()

    assert summary["Market Analyst"]["call_count"] == 2
    assert summary["Market Analyst"]["total_prompt_tokens_estimated"] == 110
    assert summary["Market Analyst"]["max_prompt_tokens_estimated"] == 100
    assert summary["Market Analyst"]["total_output_tokens"] == 12

    assert summary["News Analyst"]["call_count"] == 1
    assert summary["News Analyst"]["total_prompt_tokens_estimated"] == 2000
    assert summary["News Analyst"]["max_prompt_tokens_estimated"] == 2000
    assert summary["News Analyst"]["total_output_tokens"] == 50


def test_write_summary_writes_json_file(tmp_path):
    handler = LLMCallLogHandler(tmp_path / "llm_calls.jsonl")
    run_id = uuid.uuid4()
    handler.on_chat_model_start(
        {"kwargs": {"model_name": "m"}},
        [[HumanMessage(content="hello")]],
        run_id=run_id,
        metadata={"langgraph_node": "Trader"},
    )
    handler.on_llm_end(_chat_result("ok", input_tokens=5, output_tokens=1), run_id=run_id)

    summary_path = tmp_path / "nested" / "llm_calls_summary.json"
    handler.write_summary(summary_path)

    assert summary_path.exists()
    written = json.loads(summary_path.read_text(encoding="utf-8"))
    assert written["Trader"]["call_count"] == 1


def test_summarize_records_standalone_function():
    records = [
        {"agent": "A", "prompt_tokens_estimated": 10, "output_tokens": 1},
        {"agent": "A", "prompt_tokens_estimated": 30, "output_tokens": None},
        {"agent": "B", "prompt_tokens_estimated": 5, "output_tokens": 2},
    ]
    summary = summarize_records(records)
    assert summary["A"] == {
        "call_count": 2,
        "total_prompt_tokens_estimated": 40,
        "max_prompt_tokens_estimated": 30,
        "total_output_tokens": 1,
    }
    assert summary["B"]["call_count"] == 1


def test_default_config_llm_call_log_enabled_is_true():
    from tradingagents.default_config import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["llm_call_log_enabled"] is True


def test_env_override_disables_llm_call_log(monkeypatch):
    import importlib

    import tradingagents.default_config as default_config_module

    monkeypatch.setenv("TRADINGAGENTS_LLM_CALL_LOG_ENABLED", "false")
    try:
        reloaded = importlib.reload(default_config_module)
        assert reloaded.DEFAULT_CONFIG["llm_call_log_enabled"] is False
    finally:
        monkeypatch.delenv("TRADINGAGENTS_LLM_CALL_LOG_ENABLED", raising=False)
        importlib.reload(default_config_module)
