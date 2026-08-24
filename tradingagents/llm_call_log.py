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
- Prompt size is reported both as a raw character count (``prompt_chars``)
  and as a token count (``prompt_tokens_estimated``) — the field name and
  meaning (a token count for the prompt) are unchanged from before issue
  #147, but the *value* now comes from a real tokenizer where one is
  available, falling back to the chars/4 heuristic already used elsewhere in
  this codebase (see ``tradingagents/dataflows/tavily_search.py``) otherwise.
  A new field, ``token_count_method``, names which of the two produced the
  number for that record (see "Tokenizer strategy" below) so a consumer can
  tell a real count from an approximation instead of having to guess.
  ``input_tokens``/``output_tokens`` are the provider-reported figures from
  ``usage_metadata`` on the response when available (e.g. Ollama via the
  OpenAI-compatible endpoint supplies these), else ``None``.

Tokenizer strategy (issue #147)
--------------------------------
``_CHARS_PER_TOKEN_ESTIMATE`` (chars/4) was the only token-count mechanism
through issue #138/#143. Measured against the ~33-run corpus under
``reports/*/llm_calls.jsonl``, its error against provider-reported
``input_tokens`` was large and asymmetric enough (issue #146/#147) that the
number could not be used to reason about context-window headroom. This
module now picks a counting mechanism per call, based on which LangChain
chat-model class made it (read from ``invocation_params["_type"]``, the
``_llm_type`` LangChain attaches to every chat model — see
``_extract_llm_type``), and always labels the result:

- **``"tiktoken"``** — used when ``_type == "openai-chat"``, i.e. the call
  went through ``langchain_openai.ChatOpenAI`` or a subclass of it. Every
  OpenAI-compatible provider this codebase's ``OpenAIClient`` routes to
  (openai, xai, deepseek, qwen/qwen-cn, glm/glm-cn, minimax/minimax-cn,
  ollama, openrouter, mistral, kimi, groq, nvidia, openai_compatible — see
  ``tradingagents/llm_clients/factory.py`` and ``openai_client.py``) shares
  this LangChain class regardless of what model the base_url actually points
  at. ``tiktoken.encoding_for_model`` is tried first (exact for genuine
  OpenAI model names); an unrecognized name (the common case here, since
  most of these providers serve their own model names, e.g. Ollama's
  ``ministral-3:3b``) falls back to a fixed modern encoding
  (``_TIKTOKEN_FALLBACK_ENCODING = "o200k_base"``) as a family-level
  approximation, not a claim of an exact match to that model's own
  tokenizer. Anthropic (``_type == "anthropic-chat"``) is deliberately
  *not* in this set: the ``anthropic`` SDK's only token-counting entry
  point, ``client.messages.count_tokens``, is a network call to Anthropic's
  Token Count API (verified against ``anthropic`` 0.120.0's source), not a
  local computation — making a network round trip on every LLM call's
  logging path is the wrong trade for a hot path, so Anthropic calls use the
  heuristic below instead. Google, Azure, Perplexity, Bedrock and Intel XPU
  calls do the same: no local/offline tokenizer for those was evaluated as
  clearly better than an explicitly-labeled heuristic, so they are left on
  it rather than reaching for an approximation with an undocumented error
  bar.
- **``"heuristic_chars_per_token"``** — the chars/4 estimate, used whenever
  the tiktoken path above doesn't apply, tiktoken raises for any reason
  (unavailable, unexpected input, ...), or the calling code predates this
  field entirely (see "Backward compatibility" below). This is graceful
  degradation by construction: nothing on this path raises out to the
  caller (see "must never break a run" below).

**Calibration.** Run against the same ~33-run / 521-call corpus using the
prompt dumps under ``reports/*/prompts/`` (``llm_call_log_prompts=True`` was
set for these runs), re-tokenizing each prompt's actual text with tiktoken
and comparing to that call's provider-reported ``input_tokens`` (520 calls
had both a dump and a reported ``input_tokens``; all via Ollama, models
``ministral-3:3b``/``ministral-3:8b``, family ``"openai-chat"``):

  - tiktoken ratio to reported ``input_tokens``: mean 1.96x, median 1.59x,
    36.7% of calls within ±20%.
  - chars/4 ratio to reported ``input_tokens`` (same calls): mean 1.90x,
    median 1.50x, 26.9% of calls within ±20%.
  - tiktoken and chars/4 track each other closely (tiktoken averages ~5%
    higher token counts than chars/4 on this corpus) and neither is
    consistently closer to reported ``input_tokens`` than the other
    (tiktoken closer on 242/520 calls, chars/4 closer on 272/520, 6 ties).

Two things follow from this. First, tiktoken is a real, modest improvement
in the "close enough" bucket (+10 points within ±20%) even against a model
family (Ministral via Ollama) it was never designed for, which is the
expected outcome of a family-level approximation rather than an exact
match — for genuine OpenAI models it is exact. Second, and more
importantly: neither counter tracks reported ``input_tokens`` well on this
corpus, and they disagree with it in the *same direction and rough
magnitude* as each other. That is consistent with the anomaly living in
``input_tokens``/the provider-reported path rather than in the
counting-instrument change made here — which is exactly the question issue
#146's next sub-issue (diagnosing the truncation/anomaly itself) is scoped
to answer, not this one. This module's job was only to make the measuring
instrument itself sound and its error bar documented, which the above does.

**No import-time cost, no crash risk.** ``tiktoken`` is imported lazily
inside ``_get_tiktoken_encoding``, matching the lazy-import-per-provider
convention in ``tradingagents/llm_clients/factory.py`` — importing this
module never pulls in ``tiktoken`` for a run that never hits the
OpenAI-compatible path. Encodings are cached per model name
(``_TIKTOKEN_ENCODING_CACHE``) so the (comparatively expensive) BPE rank
load happens once per model, not once per call, across the lifetime of the
process. ``_count_prompt_tokens`` never raises: a missing tokenizer, an
unrecognized model, or any exception mid-encode all fall back to the
heuristic rather than failing the call that triggered the log record —
instrumentation failing a trading run is a worse outcome than an imprecise
number.

**Backward compatibility.** Pre-#147 records in the existing
``reports/*/llm_calls.jsonl`` corpus have no ``token_count_method`` field at
all (it did not exist yet) — ``scripts/analyze_llm_calls.py`` and any other
reader should treat a missing/absent field as ``"heuristic_chars_per_token"``,
which is what those records' numbers always were.

Design notes (continued):

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
# budgeting. This is explicitly an estimate, not a real tokenizer count;
# used as the fallback whenever a real tokenizer isn't available (see
# "Tokenizer strategy" in the module docstring).
_CHARS_PER_TOKEN_ESTIMATE = 4

# Names written to the new `token_count_method` field, distinguishing a real
# tokenizer count from the chars/4 heuristic fallback (issue #147).
_TOKEN_COUNT_METHOD_TIKTOKEN = "tiktoken"
_TOKEN_COUNT_METHOD_HEURISTIC = "heuristic_chars_per_token"

# LangChain invocation_params["_type"] values that indicate the call went
# through langchain_openai.ChatOpenAI or a subclass of it — every provider
# tradingagents/llm_clients/openai_client.py's OpenAIClient routes to (see
# "Tokenizer strategy" in the module docstring for the full provider list
# and the reasoning). tiktoken only ships true encodings for OpenAI's own
# model names, so applying it to this whole family is a deliberate
# family-level approximation for the non-OpenAI members, not a claim of
# exactness — see the calibration write-up above.
_TIKTOKEN_LLM_TYPES = frozenset({"openai-chat"})

# Fallback tiktoken encoding for model names tiktoken.encoding_for_model
# doesn't recognize (the common case for this family — see above). o200k_base
# is the encoding OpenAI's current-generation models use; picked as a modern,
# general-purpose BPE vocabulary rather than a claim of exactness for
# non-OpenAI models.
_TIKTOKEN_FALLBACK_ENCODING = "o200k_base"

# Cache of model name -> tiktoken encoding, so the (comparatively expensive)
# BPE rank load happens once per model name, not once per LLM call. Guarded
# by _TIKTOKEN_LOCK since concurrent analysts (analyst_concurrency_limit) can
# race to populate it from different threads.
_TIKTOKEN_ENCODING_CACHE: dict[str, Any] = {}
_TIKTOKEN_LOCK = threading.Lock()


def _get_tiktoken_encoding(model: str) -> Any:
    """Return a cached tiktoken encoding for ``model``, loading it on first use.

    ``tiktoken`` is imported lazily here (not at module import time) so a run
    that never hits the OpenAI-compatible path never pays its import cost,
    matching the lazy-import-per-provider convention in
    ``tradingagents/llm_clients/factory.py``. Raises whatever ``tiktoken``
    raises on a genuine failure (e.g. the package isn't installed) — callers
    are expected to catch broadly, since this is not the fallback path itself
    (see ``_count_prompt_tokens``).
    """
    with _TIKTOKEN_LOCK:
        cached = _TIKTOKEN_ENCODING_CACHE.get(model)
        if cached is not None:
            return cached

        import tiktoken

        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding(_TIKTOKEN_FALLBACK_ENCODING)
        _TIKTOKEN_ENCODING_CACHE[model] = encoding
        return encoding


def _count_prompt_tokens(text: str, model: str, llm_type: str | None) -> tuple[int, str]:
    """Return ``(token_count, method)`` for a prompt's full text.

    Uses tiktoken when ``llm_type`` names a ChatOpenAI-family provider (see
    ``_TIKTOKEN_LLM_TYPES``); otherwise, and on any failure loading or
    running the tokenizer (missing dependency, unexpected input, ...), falls
    back to the chars/4 heuristic. Never raises: a broken tokenizer must
    degrade the number, not fail the LLM call it's logging (see the module
    docstring's "must never break a run" note).
    """
    if llm_type in _TIKTOKEN_LLM_TYPES:
        try:
            encoding = _get_tiktoken_encoding(model)
            # disallowed_special=() so a prompt that happens to contain a
            # substring that looks like a tiktoken special token (e.g. text
            # copied from a chat transcript) is counted as ordinary text
            # instead of raising ValueError.
            token_count = len(encoding.encode(text, disallowed_special=()))
            return token_count, _TOKEN_COUNT_METHOD_TIKTOKEN
        except Exception:
            pass  # fall through to the heuristic below
    return len(text) // _CHARS_PER_TOKEN_ESTIMATE, _TOKEN_COUNT_METHOD_HEURISTIC


def _message_text(message: BaseMessage) -> str:
    """Return the text content of a (possibly multimodal) message, concatenated."""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(content)


def _message_text_len(message: BaseMessage) -> int:
    """Return the character length of a (possibly multimodal) message's content."""
    return len(_message_text(message))


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


def _extract_llm_type(kwargs: dict[str, Any]) -> str | None:
    """Return the LangChain ``_type`` invocation param, or ``None`` when absent.

    ``BaseChatModel._get_invocation_params`` includes ``"_type"`` — derived
    from each chat model class's ``_llm_type`` property — alongside
    ``"model"`` for every provider this codebase wires up (verified for
    ``ChatOpenAI``/its ``OpenAIClient`` subclasses: ``"openai-chat"``;
    ``ChatAnthropic``: ``"anthropic-chat"``; ``ChatGoogleGenerativeAI``:
    ``"chat-google-generative-ai"``; ``AzureChatOpenAI``:
    ``"azure-openai-chat"``). It is the tokenizer-family signal used by
    ``_count_prompt_tokens`` (see the module docstring's "Tokenizer
    strategy") because, unlike the model name, it doesn't depend on
    providers using recognizable name prefixes — several OpenAI-compatible
    providers in this codebase serve custom or locally-hosted model names
    (e.g. Ollama's ``ministral-3:3b``) that a name-based heuristic would
    misclassify.
    """
    invocation_params = kwargs.get("invocation_params")
    if isinstance(invocation_params, dict):
        llm_type = invocation_params.get("_type")
        if isinstance(llm_type, str):
            return llm_type
    return None


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


def _format_messages_for_dump(messages: list[BaseMessage] | list[str] | None) -> dict[str, Any]:
    """Format messages for a prompt dump file.

    Returns a dict with 'format' (indicating the message type) and 'messages'
    (a list of message dicts with 'role' and 'content').
    """
    if messages is None:
        return {"format": "none", "messages": []}

    # Check if we have BaseMessage objects (from on_chat_model_start)
    if messages and isinstance(messages[0], BaseMessage):
        formatted_messages = []
        for msg in messages:
            role = type(msg).__name__  # e.g., "HumanMessage", "AIMessage"
            content = getattr(msg, "content", "")
            if isinstance(content, list):
                # Multimodal content: serialize as list
                content = content
            elif isinstance(content, str):
                pass  # Already a string
            else:
                content = str(content)
            formatted_messages.append({"role": role, "content": content})
        return {"format": "chat_messages", "messages": formatted_messages}
    else:
        # String prompts from on_llm_start
        formatted_messages = [{"role": "prompt", "content": msg} for msg in (messages or [])]
        return {"format": "prompts", "messages": formatted_messages}


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
        dump_prompts: When ``True`` and ``enabled=True``, full prompts are
            written to disk under a ``prompts/`` subdirectory. Callers wire
            this to the ``llm_call_log_prompts`` config key (default ``False``).
        ticker / date: Optional run context stamped onto every record. Usually
            set via ``start_run()`` rather than here, since one handler
            instance can outlive a single ticker run (see module docstring).
    """

    def __init__(
        self,
        log_path: str | Path | None,
        enabled: bool = True,
        dump_prompts: bool = False,
        ticker: str | None = None,
        date: str | None = None,
    ) -> None:
        super().__init__()
        self.log_path = Path(log_path) if log_path is not None else None
        self.enabled = enabled and self.log_path is not None
        self.dump_prompts = dump_prompts and self.enabled
        self.prompts_dir = None
        if self.enabled:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            if self.dump_prompts:
                self.prompts_dir = self.log_path.parent / "prompts"
                self.prompts_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        # run_id (str(UUID)) -> start-time bookkeeping, so interleaved calls
        # from concurrently running analysts pair up correctly instead of
        # racing on "the last call started".
        self._pending: dict[str, dict[str, Any]] = {}
        self._records: list[dict[str, Any]] = []
        self._ticker = ticker
        self._date = date

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
                if self.dump_prompts:
                    self.prompts_dir = self.log_path.parent / "prompts"
                    self.prompts_dir.mkdir(parents=True, exist_ok=True)

    # -- call-start handlers -------------------------------------------------

    def _record_start(
        self,
        run_id: Any,
        agent: str,
        model: str,
        message_count: int,
        prompt_chars: int,
        token_count: int,
        token_count_method: str,
        messages: list[BaseMessage] | None = None,
    ) -> None:
        with self._lock:
            self._pending[str(run_id)] = {
                "start_monotonic": time.monotonic(),
                "agent": agent,
                "model": model,
                "message_count": message_count,
                "prompt_chars": prompt_chars,
                "token_count": token_count,
                "token_count_method": token_count_method,
                "messages": messages if self.dump_prompts else None,
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
        prompt_text = "".join(_message_text(m) for m in flat_messages)
        model = _extract_model_name(serialized, kwargs)
        llm_type = _extract_llm_type(kwargs)
        token_count, token_count_method = _count_prompt_tokens(prompt_text, model, llm_type)
        self._record_start(
            run_id,
            agent=_extract_agent_name(metadata),
            model=model,
            message_count=len(flat_messages),
            prompt_chars=len(prompt_text),
            token_count=token_count,
            token_count_method=token_count_method,
            messages=flat_messages if self.dump_prompts else None,
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
        prompt_text = "".join(prompts)
        model = _extract_model_name(serialized, kwargs)
        llm_type = _extract_llm_type(kwargs)
        token_count, token_count_method = _count_prompt_tokens(prompt_text, model, llm_type)
        # For legacy on_llm_start, we don't have message objects, just strings.
        # Store them as a simple list if dumping prompts is enabled.
        messages = prompts if self.dump_prompts else None
        self._record_start(
            run_id,
            agent=_extract_agent_name(metadata),
            model=model,
            message_count=len(prompts),
            prompt_chars=len(prompt_text),
            token_count=token_count,
            token_count_method=token_count_method,
            messages=messages,
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
            prompts_dir = self.prompts_dir
        if start is None:
            # No matching start record (e.g. handler was constructed mid-run,
            # or this is a call type we don't track the start of). Skip
            # rather than emit a record with a bogus/zero duration.
            return

        duration_seconds = time.monotonic() - start["start_monotonic"]
        prompt_chars = start["prompt_chars"]

        # Write prompt dump if enabled
        prompt_dump_path = None
        if self.dump_prompts and prompts_dir is not None:
            dump_filename = f"{run_id}.json"
            dump_file_path = prompts_dir / dump_filename
            prompt_dump_data = _format_messages_for_dump(start.get("messages"))
            dump_file_path.write_text(
                json.dumps(prompt_dump_data, indent=2, default=str),
                encoding="utf-8",
            )
            # Store relative path for the JSONL record
            prompt_dump_path = f"prompts/{dump_filename}"

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": str(run_id),
            "ticker": ticker,
            "date": date,
            "agent": start["agent"],
            "model": start["model"],
            "message_count": start["message_count"],
            "prompt_chars": prompt_chars,
            "prompt_tokens_estimated": start["token_count"],
            "token_count_method": start["token_count_method"],
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "duration_seconds": round(duration_seconds, 3),
            "error": error,
            "prompt_dump_path": prompt_dump_path,
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
        sum/max of each call's ``prompt_tokens_estimated`` (a real tokenizer
        count or the chars/4 heuristic fallback — see ``token_count_method``
        on the individual record; always available either way).
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
