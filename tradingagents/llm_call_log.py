"""Per-call LLM call log: a JSONL callback handler + end-of-run summary (issue #138).

Part of the #137 context-size instrumentation effort. ``cli/stats_handler.py``
(``StatsCallbackHandler``) only accumulates aggregate call counts and total
tokens in/out across a whole run, which is not enough to diagnose *which*
agent/node blew up the context window on a given call. ``LLMCallLogHandler``
records one JSON object per LLM call instead, so a run can be inspected call
by call after the fact.

This module lives under ``tradingagents/`` (not ``cli/``) so it is importable
by both ``run_trading_agents.py`` and the interactive CLI (``cli/main.py``).

Design notes:

- LangGraph passes the executing node name to LangChain callbacks via the
  callback ``metadata`` dict under the key ``"langgraph_node"``. We read that
  key (falling back to ``"unknown"`` when absent, e.g. for calls made outside
  a compiled graph) rather than tracking "current node" ourselves, since
  analysts can run concurrently (``analyst_concurrency_limit`` in
  ``tradingagents/graph/analyst_execution.py``) and there is no single
  "current" node while that's happening.
- Every ``on_llm_start``/``on_chat_model_start`` call carries a ``run_id``
  (a ``uuid.UUID``) that its matching ``on_llm_end`` call also carries.
  In-flight start/end pairs are keyed by that ``run_id`` (never by "the last
  call") so interleaved calls from concurrent analysts pair up correctly.
- Prompt size is reported both as a raw character count and as an estimated
  token count using the chars/4 heuristic already used elsewhere in this
  codebase (see ``tradingagents/dataflows/tavily_search.py``), explicitly
  labeled as an estimate. ``input_tokens``/``output_tokens`` are the
  provider-reported figures from ``usage_metadata`` on the response when
  available (e.g. Ollama via the OpenAI-compatible endpoint supplies these),
  else ``None``.
- Records are appended as JSONL (one JSON object per line) to a file the
  caller supplies. This module does not decide *where* that file lives —
  callers follow the results-directory layout already used for a run's other
  outputs (see ``tradingagents/reporting.py`` and the callers in
  ``run_trading_agents.py`` / ``cli/main.py``).
- A single handler instance can span several ticker runs, because callbacks
  are bound into the LLM clients when ``TradingAgentsGraph`` is constructed
  (``trading_graph.py`` puts them in ``llm_kwargs``) and
  ``run_trading_agents.py`` builds that graph once for a whole ``stocks.json``
  batch. ``start_run()`` therefore lets the caller retarget the handler at the
  ticker/date it is about to analyze: every record is tagged with that
  ``ticker``/``date``, and the JSONL destination can be switched to that
  ticker's own report directory so a batch run does not pool every ticker's
  calls into one undifferentiated file.
- Failed calls are logged too: ``on_llm_error`` pops the same ``run_id``
  bookkeeping ``on_llm_end`` would have popped (otherwise a failed call leaks
  its pending entry for the rest of the run) and writes a record with the same
  shape, with ``input_tokens``/``output_tokens`` as ``None`` and the ``error``
  field populated. Successful records carry ``"error": None`` so both kinds of
  record share one schema.
"""

from __future__ import annotations

import json
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import LLMResult

# chars/4 heuristic for estimating token counts from character counts,
# consistent with tradingagents/dataflows/tavily_search.py's evidence-pack
# budgeting. This is explicitly an estimate, not a real tokenizer count.
_CHARS_PER_TOKEN_ESTIMATE = 4


def _message_text_len(message: BaseMessage) -> int:
    """Return the character length of a (possibly multimodal) message's content."""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        total = 0
        for block in content:
            if isinstance(block, str):
                total += len(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    total += len(text)
        return total
    return len(str(content))


def _extract_model_name(serialized: dict[str, Any] | None, kwargs: dict[str, Any]) -> str:
    """Best-effort model name extraction from ``serialized``/callback ``kwargs``.

    ``invocation_params`` (built by ``BaseChatModel._get_invocation_params``)
    reliably carries a ``"model"`` key across providers (verified for
    langchain-openai and langchain-anthropic); ``serialized["kwargs"]`` is a
    fallback for handlers that don't receive ``invocation_params`` (e.g. a
    plain ``on_llm_start`` call from a non-chat LLM wrapper).
    """
    invocation_params = kwargs.get("invocation_params")
    if isinstance(invocation_params, dict):
        model = invocation_params.get("model") or invocation_params.get("model_name")
        if model:
            return str(model)

    if isinstance(serialized, dict):
        serialized_kwargs = serialized.get("kwargs")
        if isinstance(serialized_kwargs, dict):
            model = serialized_kwargs.get("model") or serialized_kwargs.get("model_name")
            if model:
                return str(model)
        name = serialized.get("name")
        if name:
            return str(name)

    return "unknown"


def _extract_agent_name(metadata: dict[str, Any] | None) -> str:
    """Return the executing LangGraph node name, or "unknown" when absent."""
    if isinstance(metadata, dict):
        node = metadata.get("langgraph_node")
        if node:
            return str(node)
    return "unknown"


def _extract_usage(response: LLMResult) -> tuple[int | None, int | None]:
    """Return ``(input_tokens, output_tokens)`` reported by the provider, else ``(None, None)``."""
    try:
        generation = response.generations[0][0]
    except (IndexError, TypeError, AttributeError):
        return None, None

    message = getattr(generation, "message", None)
    if isinstance(message, AIMessage):
        usage_metadata = getattr(message, "usage_metadata", None)
        if usage_metadata:
            return usage_metadata.get("input_tokens"), usage_metadata.get("output_tokens")
    return None, None


def _format_error(error: BaseException | Any) -> str:
    """Render a callback error as ``"TypeName: message"`` (bare type name when the message is empty)."""
    message = str(error).strip()
    type_name = type(error).__name__
    return f"{type_name}: {message}" if message else type_name


class LLMCallLogHandler(BaseCallbackHandler):
    """Callback handler that appends one JSONL record per LLM call.

    Args:
        log_path: File to append JSONL records to. Parent directories are
            created on first write. May be ``None`` (equivalent to
            ``enabled=False``): no file is ever written. ``start_run()`` can
            point subsequent records at a different file.
        enabled: When ``False``, the handler is a no-op — no file is created
            and no records are kept in memory. Callers wire this to the
            ``llm_call_log_enabled`` config key (default ``True``).
        ticker / date: Optional run context stamped onto every record. Usually
            set via ``start_run()`` rather than here, since one handler
            instance can outlive a single ticker run (see module docstring).
    """

    def __init__(
        self,
        log_path: str | Path | None,
        enabled: bool = True,
        ticker: str | None = None,
        date: str | None = None,
    ) -> None:
        super().__init__()
        self.log_path = Path(log_path) if log_path is not None else None
        self.enabled = enabled and self.log_path is not None

        self._lock = threading.Lock()
        # run_id (str(UUID)) -> start-time bookkeeping, so interleaved calls
        # from concurrently running analysts pair up correctly instead of
        # racing on "the last call started".
        self._pending: dict[str, dict[str, Any]] = {}
        self._records: list[dict[str, Any]] = []
        self._ticker = ticker
        self._date = date

        if self.enabled:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    # -- run context ----------------------------------------------------------

    def start_run(
        self,
        ticker: str | None = None,
        date: str | None = None,
        log_path: str | Path | None = None,
    ) -> None:
        """Retarget the handler at the ticker/date run that is about to start.

        Every record written from now on is tagged with ``ticker``/``date``,
        and — when ``log_path`` is given — appended to that file instead of the
        one the handler was constructed with. Callbacks are bound into the LLM
        clients at ``TradingAgentsGraph`` construction time, so a multi-ticker
        batch shares one handler instance; this is how each ticker's calls end
        up in its own report directory (and stay attributable even if the JSONL
        files are later concatenated).

        Records already written are left alone: ``get_records()`` /
        ``get_summary()`` keep accumulating across runs and can be filtered
        back down per ticker/date.
        """
        with self._lock:
            self._ticker = ticker
            self._date = date
            if log_path is not None and self.enabled:
                self.log_path = Path(log_path)
                self.log_path.parent.mkdir(parents=True, exist_ok=True)

    # -- call-start handlers -------------------------------------------------

    def _record_start(
        self,
        run_id: Any,
        agent: str,
        model: str,
        message_count: int,
        prompt_chars: int,
    ) -> None:
        with self._lock:
            self._pending[str(run_id)] = {
                "start_monotonic": time.monotonic(),
                "agent": agent,
                "model": model,
                "message_count": message_count,
                "prompt_chars": prompt_chars,
            }

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: Any,
        parent_run_id: Any = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if not self.enabled:
            return
        flat_messages = [m for batch in messages for m in batch]
        prompt_chars = sum(_message_text_len(m) for m in flat_messages)
        self._record_start(
            run_id,
            agent=_extract_agent_name(metadata),
            model=_extract_model_name(serialized, kwargs),
            message_count=len(flat_messages),
            prompt_chars=prompt_chars,
        )

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: Any,
        parent_run_id: Any = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if not self.enabled:
            return
        prompt_chars = sum(len(p) for p in prompts)
        self._record_start(
            run_id,
            agent=_extract_agent_name(metadata),
            model=_extract_model_name(serialized, kwargs),
            message_count=len(prompts),
            prompt_chars=prompt_chars,
        )

    # -- call-end handlers ----------------------------------------------------

    def _finish(
        self,
        run_id: Any,
        input_tokens: int | None,
        output_tokens: int | None,
        error: str | None,
    ) -> None:
        """Pop the pending start for ``run_id`` and append its completed record.

        Shared by ``on_llm_end`` and ``on_llm_error`` so a failed call is
        bookkept (and logged) exactly like a successful one.
        """
        with self._lock:
            start = self._pending.pop(str(run_id), None)
            ticker, date = self._ticker, self._date
            log_path = self.log_path
        if start is None:
            # No matching start record (e.g. handler was constructed mid-run,
            # or this is a call type we don't track the start of). Skip
            # rather than emit a record with a bogus/zero duration.
            return

        duration_seconds = time.monotonic() - start["start_monotonic"]
        prompt_chars = start["prompt_chars"]

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": str(run_id),
            "ticker": ticker,
            "date": date,
            "agent": start["agent"],
            "model": start["model"],
            "message_count": start["message_count"],
            "prompt_chars": prompt_chars,
            "prompt_tokens_estimated": prompt_chars // _CHARS_PER_TOKEN_ESTIMATE,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "duration_seconds": round(duration_seconds, 3),
            "error": error,
        }

        with self._lock:
            self._records.append(record)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        if not self.enabled:
            return
        input_tokens, output_tokens = _extract_usage(response)
        self._finish(run_id, input_tokens, output_tokens, error=None)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Log a failed LLM call (timeout, transport error, provider error, ...).

        Without this, the call's ``_pending`` entry would never be popped and
        the failure — the most interesting event for context-size/timeout
        diagnostics — would be missing from the log entirely. Token counts are
        ``None`` because there is no response to read ``usage_metadata`` from.
        """
        if not self.enabled:
            return
        self._finish(run_id, input_tokens=None, output_tokens=None, error=_format_error(error))

    # -- summary ---------------------------------------------------------------

    def get_records(
        self, ticker: str | None = None, date: str | None = None
    ) -> list[dict[str, Any]]:
        """Return a snapshot of the records written so far.

        ``ticker``/``date`` narrow the snapshot to one ticker run of a
        multi-ticker batch; omitting both returns everything.
        """
        with self._lock:
            records = list(self._records)
        if ticker is not None:
            records = [r for r in records if r.get("ticker") == ticker]
        if date is not None:
            records = [r for r in records if r.get("date") == date]
        return records

    def get_summary(
        self, ticker: str | None = None, date: str | None = None
    ) -> dict[str, dict[str, Any]]:
        """Return the per-agent aggregate (call count, prompt/output tokens) computed so far."""
        return summarize_records(self.get_records(ticker=ticker, date=date))

    def write_summary(
        self,
        summary_path: str | Path,
        ticker: str | None = None,
        date: str | None = None,
    ) -> None:
        """Write ``get_summary(ticker, date)`` to ``summary_path`` as JSON.

        No-op when the handler is disabled, matching the "no file when
        llm_call_log_enabled=False" behavior of the JSONL log itself.
        """
        if not self.enabled:
            return
        summary_path = Path(summary_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary = self.get_summary(ticker=ticker, date=date)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def summarize_records(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Compute the per-agent aggregate (call count, prompt tokens, output tokens) from JSONL records.

    Returns a dict keyed by agent/node name with:
      - ``call_count``: number of LLM calls attributed to this agent.
      - ``total_prompt_tokens_estimated`` / ``max_prompt_tokens_estimated``:
        sum/max of each call's chars/4 estimate (always available).
      - ``total_output_tokens``: sum of provider-reported ``output_tokens``
        (calls where the provider didn't report usage contribute 0).
      - ``error_count``: how many of those calls failed (records with a
        non-null ``error``). Failed calls still count towards ``call_count``
        and the prompt-token figures — the prompt was sent — but contribute no
        output tokens, so this keeps the other numbers readable.
    """
    aggregates: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "call_count": 0,
            "total_prompt_tokens_estimated": 0,
            "max_prompt_tokens_estimated": 0,
            "total_output_tokens": 0,
            "error_count": 0,
        }
    )
    for record in records:
        agent = record.get("agent", "unknown")
        bucket = aggregates[agent]
        bucket["call_count"] += 1
        estimated = record.get("prompt_tokens_estimated", 0) or 0
        bucket["total_prompt_tokens_estimated"] += estimated
        bucket["max_prompt_tokens_estimated"] = max(bucket["max_prompt_tokens_estimated"], estimated)
        bucket["total_output_tokens"] += record.get("output_tokens") or 0
        if record.get("error"):
            bucket["error_count"] += 1

    return dict(aggregates)
