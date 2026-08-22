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
from pathlib import Path

import pytest
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
    assert record["error"] is None
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
        "error_count": 0,
    }
    assert summary["B"]["call_count"] == 1


def test_llm_error_writes_a_record_and_clears_the_pending_entry(tmp_path):
    """A failed call must still be logged (and must not leak its _pending entry)."""
    log_path = tmp_path / "llm_calls.jsonl"
    handler = LLMCallLogHandler(log_path)
    run_id = uuid.uuid4()

    handler.on_chat_model_start(
        {"kwargs": {"model_name": "qwen3:14b"}},
        [[HumanMessage(content="x" * 400)]],
        run_id=run_id,
        metadata={"langgraph_node": "Researcher"},
    )
    handler.on_llm_error(TimeoutError("request timed out after 600s"), run_id=run_id)

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])

    # Same shape as a successful record...
    assert record["agent"] == "Researcher"
    assert record["model"] == "qwen3:14b"
    assert record["prompt_chars"] == 400
    assert record["prompt_tokens_estimated"] == 100
    assert record["run_id"] == str(run_id)
    assert record["duration_seconds"] >= 0
    # ...with no usage to report and the failure recorded.
    assert record["input_tokens"] is None
    assert record["output_tokens"] is None
    assert "TimeoutError" in record["error"]
    assert "request timed out after 600s" in record["error"]

    # The pending entry was popped, so it does not leak for the rest of the run.
    assert handler._pending == {}


def test_llm_error_with_empty_message_still_names_the_exception_type(tmp_path):
    handler = LLMCallLogHandler(tmp_path / "llm_calls.jsonl")
    run_id = uuid.uuid4()

    handler.on_chat_model_start(
        {"kwargs": {"model_name": "m"}},
        [[HumanMessage(content="hi")]],
        run_id=run_id,
        metadata={"langgraph_node": "Trader"},
    )
    handler.on_llm_error(ConnectionError(), run_id=run_id)

    assert handler.get_records()[0]["error"] == "ConnectionError"


def test_llm_error_without_matching_start_is_skipped_not_crashed(tmp_path):
    log_path = tmp_path / "llm_calls.jsonl"
    handler = LLMCallLogHandler(log_path)

    handler.on_llm_error(RuntimeError("orphaned"), run_id=uuid.uuid4())

    assert handler.get_records() == []
    assert not log_path.exists()


def test_llm_error_does_not_disturb_a_concurrent_in_flight_call(tmp_path):
    """One call failing must not consume another run_id's pending bookkeeping."""
    handler = LLMCallLogHandler(tmp_path / "llm_calls.jsonl")
    run_ok, run_bad = uuid.uuid4(), uuid.uuid4()

    handler.on_chat_model_start(
        {"kwargs": {"model_name": "model-ok"}},
        [[HumanMessage(content="O" * 12)]],
        run_id=run_ok,
        metadata={"langgraph_node": "Node OK"},
    )
    handler.on_chat_model_start(
        {"kwargs": {"model_name": "model-bad"}},
        [[HumanMessage(content="B" * 20)]],
        run_id=run_bad,
        metadata={"langgraph_node": "Node BAD"},
    )
    handler.on_llm_error(ValueError("boom"), run_id=run_bad)
    handler.on_llm_end(_chat_result("fine", input_tokens=3, output_tokens=4), run_id=run_ok)

    records = {r["agent"]: r for r in handler.get_records()}
    assert records["Node BAD"]["error"].startswith("ValueError")
    assert records["Node BAD"]["output_tokens"] is None
    assert records["Node OK"]["error"] is None
    assert records["Node OK"]["output_tokens"] == 4
    assert handler._pending == {}


def test_summary_counts_errored_calls_separately(tmp_path):
    handler = LLMCallLogHandler(tmp_path / "llm_calls.jsonl")

    for outcome in ("ok", "error"):
        run_id = uuid.uuid4()
        handler.on_chat_model_start(
            {"kwargs": {"model_name": "m"}},
            [[HumanMessage(content="x" * 40)]],
            run_id=run_id,
            metadata={"langgraph_node": "Researcher"},
        )
        if outcome == "ok":
            handler.on_llm_end(_chat_result("r", input_tokens=1, output_tokens=9), run_id=run_id)
        else:
            handler.on_llm_error(TimeoutError("nope"), run_id=run_id)

    stats = handler.get_summary()["Researcher"]
    assert stats["call_count"] == 2
    assert stats["error_count"] == 1
    # The failed call still sent its prompt, so it counts towards prompt tokens.
    assert stats["total_prompt_tokens_estimated"] == 20
    assert stats["total_output_tokens"] == 9


# -- run context: per-ticker attribution / output routing ---------------------


def _emit_call(handler, agent="Market Analyst", content="hello", output_tokens=1):
    run_id = uuid.uuid4()
    handler.on_chat_model_start(
        {"kwargs": {"model_name": "m"}},
        [[HumanMessage(content=content)]],
        run_id=run_id,
        metadata={"langgraph_node": agent},
    )
    handler.on_llm_end(_chat_result("ok", input_tokens=1, output_tokens=output_tokens), run_id=run_id)


def test_start_run_tags_records_with_ticker_and_date(tmp_path):
    handler = LLMCallLogHandler(tmp_path / "llm_calls.jsonl")

    handler.start_run(ticker="AAPL", date="2024-01-15")
    _emit_call(handler)
    handler.start_run(ticker="MSFT", date="2024-01-16")
    _emit_call(handler)

    records = handler.get_records()
    assert [(r["ticker"], r["date"]) for r in records] == [
        ("AAPL", "2024-01-15"),
        ("MSFT", "2024-01-16"),
    ]

    # And the snapshot/summary can be narrowed back down to one ticker run.
    assert len(handler.get_records(ticker="AAPL")) == 1
    assert handler.get_summary(ticker="MSFT", date="2024-01-16")["Market Analyst"]["call_count"] == 1
    assert handler.get_summary(ticker="TSLA") == {}


def test_records_without_a_run_context_have_null_ticker_and_date(tmp_path):
    """cli/main.py never calls start_run() (one ticker per run) — that must stay valid."""
    handler = LLMCallLogHandler(tmp_path / "llm_calls.jsonl")
    _emit_call(handler)

    record = handler.get_records()[0]
    assert record["ticker"] is None
    assert record["date"] is None


def test_start_run_routes_records_to_the_given_per_ticker_file(tmp_path):
    fallback = tmp_path / "llm_calls.jsonl"
    handler = LLMCallLogHandler(fallback)

    aapl_log = tmp_path / "AAPL_2024-01-15_120000" / "llm_calls.jsonl"
    msft_log = tmp_path / "MSFT_2024-01-15_120500" / "llm_calls.jsonl"

    handler.start_run(ticker="AAPL", date="2024-01-15", log_path=aapl_log)
    _emit_call(handler)
    handler.start_run(ticker="MSFT", date="2024-01-15", log_path=msft_log)
    _emit_call(handler)
    _emit_call(handler)

    # Nothing was pooled into a shared batch-level file.
    assert not fallback.exists()
    assert len(aapl_log.read_text(encoding="utf-8").strip().splitlines()) == 1
    assert len(msft_log.read_text(encoding="utf-8").strip().splitlines()) == 2
    assert json.loads(aapl_log.read_text(encoding="utf-8").splitlines()[0])["ticker"] == "AAPL"


def test_start_run_on_a_disabled_handler_writes_nothing(tmp_path):
    handler = LLMCallLogHandler(tmp_path / "llm_calls.jsonl", enabled=False)
    per_ticker = tmp_path / "AAPL_2024-01-15_120000" / "llm_calls.jsonl"

    handler.start_run(ticker="AAPL", date="2024-01-15", log_path=per_ticker)
    _emit_call(handler)

    assert not per_ticker.exists()
    assert not per_ticker.parent.exists()
    assert handler.get_records() == []


def test_write_summary_can_be_scoped_to_one_ticker_run(tmp_path):
    handler = LLMCallLogHandler(tmp_path / "llm_calls.jsonl")

    handler.start_run(ticker="AAPL", date="2024-01-15")
    _emit_call(handler, agent="Trader", output_tokens=3)
    handler.start_run(ticker="MSFT", date="2024-01-15")
    _emit_call(handler, agent="Trader", output_tokens=5)

    scoped = tmp_path / "AAPL" / "llm_calls_summary.json"
    handler.write_summary(scoped, ticker="AAPL", date="2024-01-15")
    assert json.loads(scoped.read_text(encoding="utf-8"))["Trader"]["total_output_tokens"] == 3

    batch = tmp_path / "llm_calls_summary.json"
    handler.write_summary(batch)
    assert json.loads(batch.read_text(encoding="utf-8"))["Trader"]["total_output_tokens"] == 8


def test_batch_run_writes_one_log_per_ticker_report_dir(tmp_path, monkeypatch):
    """run_trading_agents.py multi-ticker batch: every record is attributable to its ticker.

    Drives ``run_trading_agents.main()`` with a mocked ``TradingAgentsGraph``
    whose ``propagate`` replays synthetic callback events into the real
    handler the script constructed, then asserts each ticker got its own
    ``llm_calls.jsonl`` inside its own ``{ticker}_{date}_{timestamp}`` report
    directory (the layout ``save_report_to_disk`` already uses).
    """
    from unittest.mock import MagicMock, patch

    import run_trading_agents

    stocks_file = tmp_path / "stocks.json"
    stocks_file.write_text(
        json.dumps(
            [
                {"ticker": "AAPL", "date": "2024-01-15"},
                {"ticker": "MSFT", "date": "2024-01-16"},
            ]
        ),
        encoding="utf-8",
    )
    report_dir = tmp_path / "reports"

    captured: dict[str, LLMCallLogHandler] = {}

    def _build_graph(*args, **kwargs):
        captured["handler"] = kwargs["callbacks"][0]
        instance = MagicMock()

        def _propagate(ticker, date):
            handler = captured["handler"]
            _emit_call(handler, agent="Market Analyst", content="m" * 40)
            _emit_call(handler, agent="Trader", content="t" * 8)
            return {"final_trade_decision": "HOLD"}, "HOLD"

        instance.propagate.side_effect = _propagate
        return instance

    with (
        patch("run_trading_agents.TradingAgentsGraph", side_effect=_build_graph),
        patch(
            "run_trading_agents.save_report_to_disk",
            side_effect=lambda state, ticker, path: (Path(path) / "report.md", {}),
        ),
        patch(
            "sys.argv",
            [
                "run_trading_agents.py",
                str(stocks_file),
                "--use-dates-from-json",
                "--report-dir",
                str(report_dir),
            ],
        ),
    ):
        run_trading_agents.main()

    logs = sorted(report_dir.glob("*/llm_calls.jsonl"))
    assert len(logs) == 2, f"expected one log per ticker, got {logs}"
    # No pooled batch-level JSONL alongside the per-ticker ones.
    assert not (report_dir / "llm_calls.jsonl").exists()

    by_ticker = {}
    for log in logs:
        records = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
        tickers = {r["ticker"] for r in records}
        assert len(tickers) == 1, "a ticker's log must contain only that ticker's calls"
        ticker = tickers.pop()
        by_ticker[ticker] = (log, records)
        assert log.parent.name.startswith(f"{ticker}_{records[0]['date']}_")
        # Each ticker's per-agent summary sits next to its own JSONL.
        summary = json.loads((log.parent / "llm_calls_summary.json").read_text(encoding="utf-8"))
        assert summary["Market Analyst"]["call_count"] == 1
        assert summary["Trader"]["call_count"] == 1

    assert set(by_ticker) == {"AAPL", "MSFT"}
    assert by_ticker["AAPL"][1][0]["date"] == "2024-01-15"
    assert by_ticker["MSFT"][1][0]["date"] == "2024-01-16"

    # The batch-level roll-up still aggregates every ticker.
    batch = json.loads((report_dir / "llm_calls_summary.json").read_text(encoding="utf-8"))
    assert batch["Market Analyst"]["call_count"] == 2
    assert batch["Trader"]["call_count"] == 2


def test_batch_run_flushes_the_call_log_summary_when_a_ticker_fails(tmp_path):
    """A failed ticker run must still leave its per-call log + summary on disk."""
    from unittest.mock import MagicMock, patch

    import run_trading_agents

    stocks_file = tmp_path / "stocks.json"
    stocks_file.write_text(
        json.dumps([{"ticker": "AAPL", "date": "2024-01-15"}]), encoding="utf-8"
    )
    report_dir = tmp_path / "reports"

    def _build_graph(*args, **kwargs):
        handler = kwargs["callbacks"][0]
        instance = MagicMock()

        def _propagate(ticker, date):
            _emit_call(handler, agent="Market Analyst")
            raise ConnectionError("Failed to connect to Ollama")

        instance.propagate.side_effect = _propagate
        return instance

    with (
        patch("run_trading_agents.TradingAgentsGraph", side_effect=_build_graph),
        patch(
            "sys.argv",
            [
                "run_trading_agents.py",
                str(stocks_file),
                "--use-dates-from-json",
                "--report-dir",
                str(report_dir),
            ],
        ),
        pytest.raises(SystemExit) as exc_info,
    ):
        run_trading_agents.main()

    assert exc_info.value.code == 1
    logs = list(report_dir.glob("AAPL_2024-01-15_*/llm_calls.jsonl"))
    assert len(logs) == 1
    summary = json.loads(
        (logs[0].parent / "llm_calls_summary.json").read_text(encoding="utf-8")
    )
    assert summary["Market Analyst"]["call_count"] == 1


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
