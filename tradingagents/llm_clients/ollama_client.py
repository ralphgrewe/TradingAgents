"""Client for Ollama, routed through its native ``/api/chat`` endpoint (issue #169).

Ollama's OpenAI-compatible endpoint (``/v1/chat/completions``, previously served by
``OpenAIClient``/``OllamaChatOpenAI`` in ``openai_client.py``) silently drops both
``options.num_ctx`` and ``think`` -- verified live against Ollama 0.32.3 (see issue
#169's problem statement): the same request against the native ``/api/chat`` endpoint
honors both. Everything issues #149/#154 (num_ctx) and #155 (think) built was therefore
a no-op on the wire; every Ollama run executed at the daemon's VRAM-tiered auto-fit
context window (observed landing as low as 4096) regardless of what this codebase
computed and attached to the request.

This module migrates the ``ollama`` provider off the OpenAI-compatible family entirely
and onto ``langchain_ollama.ChatOllama``, a dedicated (non-OpenAI-compatible) client in
the same shape as ``anthropic_client.py``/``google_client.py`` -- see
``factory.create_llm_client``, which now routes ``provider == "ollama"`` here *before*
the ``is_openai_compatible`` check.

Three things carry over from the old wiring, re-pointed rather than reimplemented:

- **num_ctx derivation (issue #154).** ``OllamaNumCtxDerivation`` (``llm_call_log.py``)
  stays the single arithmetic source shared by ``ContextWindowGuardHandler`` (aborts
  before dispatch) and ``LLMCallLogHandler`` (audits the value sent). ``ChatOllama`` has
  no ``_get_request_payload`` hook (that's a ``langchain_openai.ChatOpenAI`` construct);
  ``_chat_params`` is the equivalent per-request seam on the native client, overridden
  below on ``NormalizedChatOllama``.
- **think mode (issue #155).** Now a genuine three-state knob at the config level
  (``ollama_think``: ``True``/``False``/``None``) mapped onto ``ChatOllama``'s
  ``reasoning`` field, which distinguishes "explicitly off" from "unset" the same way --
  ``ChatOllama._chat_params`` sends ``think: self.reasoning`` verbatim, and the
  underlying ``ollama`` SDK client drops ``None``-valued fields from the request body
  entirely (``ChatRequest(...).model_dump(exclude_none=True)``), so ``reasoning=None``
  means "no think field sent at all", distinct from ``reasoning=False`` ("think: false").
- **Structured output method (issue #161).** ``"auto"`` still resolves to
  ``json_schema`` for Ollama (grammar-constrained at decode time); an explicit
  ``structured_output_method`` config value or a ``method=`` argument still wins.
"""

import os
from typing import Any

from langchain_core.messages import BaseMessage
from langchain_core.runnables.config import ensure_config
from langchain_ollama import ChatOllama

from tradingagents.dataflows.config import get_config
from tradingagents.llm_call_log import (
    OllamaNumCtxDerivation,
    _count_prompt_tokens,
    _extract_agent_name,
    _message_text,
)

from .base_client import BaseLLMClient, normalize_content
from .capabilities import get_capabilities
from .validators import validate_model

# Ollama's own default when no base URL is configured at all. No "/v1" suffix --
# unlike the OpenAI-compatible endpoint this client replaces, the native client talks
# to the bare host and appends its own "/api/..." paths.
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"

# Tiktoken-family tokenizer estimation applies to Ollama regardless of which endpoint
# serves it (see llm_call_log.py's "Tokenizer strategy" -- the #147/#148 calibration
# corpus was entirely Ollama-served models). ChatOllama reports its own _llm_type
# ("chat-ollama"), so that string -- not "openai-chat" -- is what
# TradingAgentsGraph._get_provider_kwargs / this module's _chat_params override must
# pass through to `_count_prompt_tokens` for the estimator to keep firing.
OLLAMA_LLM_TYPE = "chat-ollama"


def _resolve_base_url(explicit: str | None) -> str:
    """Resolve the Ollama base URL: explicit > ``OLLAMA_BASE_URL`` env > default.

    A resolved value ending in ``/v1`` or ``/v1/`` -- the documented shape for the old
    OpenAI-compatible endpoint (issues #149/#154/#155) -- has that suffix stripped, so
    existing setups with ``.../v1`` already in their config or ``OLLAMA_BASE_URL`` keep
    working against the native endpoint instead of 404ing (issue #169 edge case).
    """
    base_url = explicit or os.environ.get("OLLAMA_BASE_URL") or DEFAULT_OLLAMA_BASE_URL
    stripped = base_url.rstrip("/")
    if stripped.endswith("/v1"):
        stripped = stripped[: -len("/v1")]
    return stripped or DEFAULT_OLLAMA_BASE_URL


class NormalizedChatOllama(ChatOllama):
    """ChatOllama with normalized content output, per-request num_ctx, and method dispatch.

    ``invoke`` normalizes content the same way every other client in this codebase does
    (see ``base_client.normalize_content``) for consistent downstream handling.

    ``_chat_params`` is ``ChatOllama``'s per-request assembly hook -- every
    ``_generate``/``_agenerate`` call (sync or async, streamed or not) routes its
    outgoing ``options``/``think``/``format`` dict through here, which is what makes it
    the right override point for the #154 per-request ``num_ctx`` derivation, mirroring
    ``OllamaChatOpenAI._get_request_payload`` in ``openai_client.py`` (the equivalent
    hook on the now-removed OpenAI-compatible path). Only ever attached
    (``self._ollama_num_ctx_derivation``) when no explicit ``num_ctx`` was configured --
    see ``OllamaClient.get_llm`` -- so an explicit value and the derivation are never
    both live at once, exactly as before #169.

    ``_chat_params`` has no ``run_manager``/callback ``metadata`` in its own signature
    (unlike ``ContextWindowGuardHandler``/``LLMCallLogHandler``, which receive it as an
    explicit callback argument) -- and ``self.quick_thinking_llm``/``self.deep_thinking_llm``
    are each a single ``NormalizedChatOllama`` instance shared across every agent that
    uses that thinking tier (see ``TradingAgentsGraph.__init__``), so the agent name can't
    be bound once at construction time either. Issue #170's per-agent headroom override
    therefore reads the LangGraph node name the same way ``ensure_config()`` recovers it
    for an un-configured ``BaseChatModel.invoke()`` call: LangGraph runs each node's
    callable via ``context.run(...)`` inside ``langgraph._internal._runnable.set_config_context``,
    which sets ``langchain_core.runnables.config.var_child_runnable_config`` -- an ambient
    ``contextvars.ContextVar`` that survives through every nested, unconfigured function
    call made during that node's synchronous execution (including the one several frames
    below that ends up here). Calling ``ensure_config()`` again from within ``_chat_params``
    reads that same ambient value, which is exactly how the callback handlers' ``metadata``
    argument already gets ``{"langgraph_node": ...}`` populated without any agent code
    threading ``config`` through by hand. This also means concurrent analysts
    (``analyst_concurrency_limit``) don't race each other here: LangGraph copies the
    context per node invocation, so each node's ``ensure_config()`` reads see only that
    node's own metadata.
    """

    _ollama_num_ctx_derivation: OllamaNumCtxDerivation | None = None

    def invoke(self, input, config=None, **kwargs):
        return normalize_content(super().invoke(input, config, **kwargs))

    def _chat_params(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        params = super()._chat_params(messages, stop=stop, **kwargs)
        derivation = self._ollama_num_ctx_derivation
        if derivation is None or not derivation.applies_to(self.model):
            return params

        prompt_text = "".join(_message_text(m) for m in messages)
        token_count, _method = _count_prompt_tokens(prompt_text, self.model, OLLAMA_LLM_TYPE)
        # issue #170: recover the executing LangGraph node name from the ambient
        # RunnableConfig (see the class docstring above for why this, rather than an
        # instance attribute or an explicit kwarg, is the right mechanism here) so a
        # configured per-agent headroom override actually reaches the outgoing
        # request -- not just the audit log and the abort check.
        agent = _extract_agent_name(ensure_config().get("metadata"))
        needed = derivation.needed_tokens(token_count, agent=agent)
        num_ctx = min(needed, derivation.num_ctx_max)

        # super()._chat_params() already built an "options" dict from whichever of
        # self.mirostat/self.temperature/self.num_ctx/etc. are set (or from a caller-
        # supplied "options" kwarg it popped instead) -- overlay num_ctx onto that
        # dict rather than replacing it, so those other settings still reach the
        # request.
        options = dict(params.get("options") or {})
        options["num_ctx"] = num_ctx
        params["options"] = options
        return params

    def with_structured_output(self, schema, *, method=None, **kwargs):
        # Precedence, matching NormalizedChatOpenAI/OllamaChatOpenAI's contract
        # (issue #161): explicit method argument > config value (when not "auto") >
        # Ollama's own default ("json_schema", grammar-constrained at decode time --
        # ChatOllama.with_structured_output's own default, just made explicit here so
        # the config/method precedence chain is visible in one place).
        caps = get_capabilities(self.model)
        if caps.preferred_structured_method == "none":
            raise NotImplementedError(
                f"{self.model} has no structured-output method available; "
                f"agent factories will fall back to free-text generation."
            )

        if method is None:
            config_method = get_config().get("structured_output_method", "auto")
            method = config_method if config_method != "auto" else "json_schema"

        return super().with_structured_output(schema, method=method, **kwargs)


# Kwargs forwarded from user config straight through to a ChatOllama constructor field.
# Deliberately not the same list as OpenAIClient/AnthropicClient/GoogleClient's --
# ChatOllama has no `reasoning_effort`, `use_responses_api`, `api_key`, or (per-field)
# `max_retries`/`timeout`; forwarding those blindly would raise a pydantic validation
# error on construction (issue #169 edge case) instead of just being ignored.
_PASSTHROUGH_KWARGS = ("temperature", "callbacks")


class OllamaClient(BaseLLMClient):
    """Client for locally- or remotely-served Ollama models, via the native /api/chat endpoint."""

    def __init__(self, model: str, base_url: str | None = None, **kwargs):
        super().__init__(model, base_url, **kwargs)

    def get_llm(self) -> Any:
        """Return a configured ``NormalizedChatOllama`` instance."""
        self.warn_if_unknown_model()
        llm_kwargs: dict[str, Any] = {
            "model": self.model,
            "base_url": _resolve_base_url(self.base_url),
        }

        for key in _PASSTHROUGH_KWARGS:
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]

        # ChatOllama has no top-level `timeout` field; the underlying ollama-python
        # client accepts it as an httpx client kwarg instead (issue #169 edge case).
        timeout = self.kwargs.get("timeout")
        if timeout is not None:
            llm_kwargs["client_kwargs"] = {"timeout": timeout}

        # Explicit context-length override (issue #149) still wins outright: when
        # set, no OllamaNumCtxDerivation is ever forwarded (see
        # TradingAgentsGraph._get_provider_kwargs), so the num_ctx set here is the
        # only one ChatOllama's own `_chat_params` will ever see for this instance.
        num_ctx = self.kwargs.get("num_ctx")
        if num_ctx is not None:
            llm_kwargs["num_ctx"] = int(num_ctx)

        # Think mode (issue #155), re-pointed at ChatOllama's native `reasoning`
        # field by issue #169. Three states, distinguished by dict membership (not
        # truthiness) so an explicit False is never conflated with "unset":
        #   - key absent (direct construction, e.g. a script/test that never sets
        #     ollama_think) -> effective default False, same as the documented
        #     default_config.py value.
        #   - True/False present -> forwarded verbatim as `reasoning`.
        #   - None present -> `reasoning` is left at ChatOllama's own default
        #     (None), so no "think" field is sent at all (model decides).
        # dict.get's default only applies when the key is absent -- a present
        # value of None (explicit "no think field at all") is returned as-is,
        # not replaced by the False default, so this line alone implements
        # all three states described above.
        ollama_think = self.kwargs.get("ollama_think", False)
        if ollama_think is not None:
            llm_kwargs["reasoning"] = bool(ollama_think)

        llm = NormalizedChatOllama(**llm_kwargs)

        # Per-request num_ctx derivation (issue #154, re-pointed by #169): only
        # reached when no explicit num_ctx was set above. Not a ChatOllama/pydantic
        # field, so it's attached as a private instance attribute rather than a
        # constructor kwarg; NormalizedChatOllama._chat_params reads it back on
        # every request.
        derivation = self.kwargs.get("ollama_num_ctx_derivation")
        if derivation is not None:
            llm._ollama_num_ctx_derivation = derivation

        return llm

    def validate_model(self) -> bool:
        """Validate model for Ollama (any model name is accepted -- see validators.py)."""
        return validate_model("ollama", self.model)
