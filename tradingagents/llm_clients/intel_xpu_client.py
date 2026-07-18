"""Intel XPU client for local PyTorch/Transformers inference on Intel GPU hardware.

Loads the Mistral-3-3B-Reasoning model directly on torch.xpu (Intel Arc GPU)
and exposes it through the BaseLLMClient interface for drop-in compatibility
with the trading pipeline.

The model runs in-process with no HTTP roundtrip, but does NOT support
structured output or tool-calling — these raise NotImplementedError to signal
failure loudly to call sites (e.g., news_analyst.py) rather than silently
degrading.
"""

import logging
import threading
import warnings
from collections.abc import Callable
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, convert_to_openai_messages
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from .base_client import BaseLLMClient

logger = logging.getLogger(__name__)

# Locked model ID for v1: no changes without explicit decision.
_LOCKED_MODEL_ID = "mistralai/Ministral-3-3B-Reasoning-2512"


class _ModelCache:
    """Thread-safe, process-lifetime cache of loaded (model, tokenizer) pairs.

    Keyed by model id. Loading a model onto XPU is expensive (a full from-disk
    read plus VRAM residency), and both of ``TradingAgentsGraph``'s clients
    (deep + quick thinking) plus every repeat ``analyze_stock`` call in the
    long-lived MCP server request the *same* locked model id. Without sharing,
    each of those would re-load the identical weights and double VRAM for no
    benefit — so the first request for a given id loads it and every later
    request for that id reuses the same in-memory objects.

    Lifecycle (explicit, deliberate decision — not silent):
      * Entries are loaded lazily on first request and kept for the entire
        process lifetime. There is intentionally NO automatic eviction or
        invalidation in v1: the model id is locked to a single value and the
        weights never change under us, so there is nothing to invalidate.
      * ``clear()`` exists as an explicit, named seam for test isolation and
        deliberate teardown. It drops Python references from the cache but does
        NOT by itself free XPU memory — that additionally requires every
        ``IntelXPUChatModel`` holding the model to be released and
        ``torch.xpu.empty_cache()`` to run, which is out of scope for v1.

    Thread-safety: ``TradingAgentsGraph`` builds its two clients sequentially,
    but the MCP server dispatches the synchronous ``analyze_stock`` tool in
    worker threads, so two ``analyze_stock`` calls can construct clients
    concurrently. ``get_or_load`` therefore uses double-checked locking so a
    given id is loaded at most once even under concurrent first requests.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: dict[str, tuple[Any, Any]] = {}

    def get_or_load(
        self, model_id: str, loader: Callable[[], tuple[Any, Any]]
    ) -> tuple[Any, Any]:
        """Return the cached (model, tokenizer) for ``model_id``, loading once.

        ``loader`` is invoked at most once per distinct ``model_id`` for the
        life of the process, even if multiple threads request the same id
        simultaneously.
        """
        # Fast path: already loaded (dict reads are atomic under the GIL).
        entry = self._entries.get(model_id)
        if entry is not None:
            return entry
        with self._lock:
            # Re-check under the lock: another thread may have loaded it while
            # we waited to acquire.
            entry = self._entries.get(model_id)
            if entry is None:
                entry = loader()
                self._entries[model_id] = entry
            return entry

    def clear(self) -> None:
        """Drop all cached entries (test isolation / explicit teardown seam)."""
        with self._lock:
            self._entries.clear()

    def __contains__(self, model_id: str) -> bool:
        return model_id in self._entries


# Process-wide shared cache so IntelXPUClient instances requesting the same
# model id reuse one in-memory model/tokenizer. Use clear_model_cache() to reset.
_MODEL_CACHE = _ModelCache()


def clear_model_cache() -> None:
    """Reset the shared Intel XPU model/tokenizer cache.

    Named seam for test isolation and deliberate teardown. See ``_ModelCache``
    for why this does not, on its own, free XPU memory.
    """
    _MODEL_CACHE.clear()

# Sensible defaults applied only when not overridden by IntelXPUClient kwargs
# or per-call kwargs.
_GENERATION_DEFAULTS: dict[str, Any] = {
    "max_new_tokens": 2048,
    "temperature": 0.7,
    "do_sample": True,
}

# Kwargs forwarded from IntelXPUClient(**kwargs) / per-call invoke() kwargs
# into model.generate(), mirroring openai_client.py's _PASSTHROUGH_KWARGS
# pattern. Anything not listed here is dropped rather than passed straight
# through to transformers' generate() (which TypeErrors on unknown kwargs).
_GENERATION_PASSTHROUGH_KWARGS = (
    "max_new_tokens",
    "temperature",
    "do_sample",
    "top_p",
    "top_k",
    "repetition_penalty",
    "num_beams",
)


class IntelXPUChatModel(BaseChatModel):
    """LangChain-compatible chat model wrapping Mistral on Intel XPU.

    Formats the full message list (system + human + prior turns) via the
    tokenizer's chat template, runs model.generate(), and returns the result
    as an AIMessage.
    """

    model_name: str = _LOCKED_MODEL_ID

    # Pydantic config to allow arbitrary attributes (model, tokenizer)
    model_config = {"arbitrary_types_allowed": True}

    model: Any = None
    tokenizer: Any = None
    generation_kwargs: dict[str, Any] = Field(default_factory=dict)

    def __init__(self, model, tokenizer, generation_kwargs=None):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            generation_kwargs=dict(generation_kwargs or {}),
        )

    def _generate(
        self,
        messages,
        stop=None,
        run_manager=None,
        **kwargs,
    ) -> ChatResult:
        """Generate text from the full message history.

        Uses the tokenizer's chat template over ALL messages (not just the
        last one) so system prompts and prior turns survive — every analyst
        prompt in this pipeline is system+human, not a single bare message.
        """
        import torch

        chat_messages = convert_to_openai_messages(messages)
        inputs = self.tokenizer.apply_chat_template(
            chat_messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.model.device)

        # Defaults < client-level kwargs (self.generation_kwargs) < call-site
        # kwargs, so a per-invoke override always wins.
        gen_kwargs = dict(_GENERATION_DEFAULTS)
        gen_kwargs.update(self.generation_kwargs)
        gen_kwargs.update(
            {key: value for key, value in kwargs.items() if key in _GENERATION_PASSTHROUGH_KWARGS}
        )

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        # Decode only the newly generated tokens, not the echoed prompt.
        prompt_len = len(inputs["input_ids"][0])
        response_text = self.tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)

        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=response_text))])

    def bind_tools(self, tools, **kwargs):
        """Tool-calling is not supported on Intel XPU.

        Raises NotImplementedError so the caller (e.g., news_analyst.py) fails
        loudly rather than silently degrading.
        """
        warnings.warn(
            "Intel XPU does not support tool-calling; "
            "the caller will fail rather than fabricate results.",
            RuntimeWarning,
            stacklevel=2,
        )
        raise NotImplementedError(
            "Intel XPU chat model does not support tool-calling. "
            "News Analyst and other agents that require tools cannot run with this provider."
        )

    def with_structured_output(self, schema, *, method=None, **kwargs):
        """Structured output is not supported on Intel XPU.

        Raises NotImplementedError so the caller falls back to free-text generation.
        """
        warnings.warn(
            "Intel XPU does not support structured output; "
            "agent will use free-text generation instead.",
            RuntimeWarning,
            stacklevel=2,
        )
        raise NotImplementedError(
            "Intel XPU chat model does not support structured output. "
            "The agent factory will fall back to free-text generation."
        )

    @property
    def _llm_type(self) -> str:
        return "intel_xpu"


class IntelXPUClient(BaseLLMClient):
    """Client for local Mistral-3-3B-Reasoning inference on Intel XPU.

    Model loading and XPU availability are checked at construction, failing fast
    with a clear error if requirements are not met. The ``model`` argument
    actually gates what gets loaded: v1 supports exactly one locked model id,
    so a mismatch raises immediately instead of silently loading the locked
    model under a different name.

    The loaded model/tokenizer are held in a process-wide, model-id-keyed shared
    cache (``_MODEL_CACHE``), NOT on the instance: multiple ``IntelXPUClient``
    instances requesting the same model id (e.g. ``TradingAgentsGraph``'s deep +
    quick clients, and repeat ``analyze_stock`` calls in the long-lived MCP
    server) reuse one in-memory copy instead of each re-loading identical weights
    onto XPU. The cache is process-lifetime with no per-instance release path (an
    XPU model is a shared resource, not owned by one client); see ``_ModelCache``
    for the full lifecycle rationale and ``clear_model_cache()`` for the test/
    teardown seam.

    Structured output and tool-calling are explicitly unsupported (raise
    NotImplementedError) to prevent silent degradation in callers that rely on
    these capabilities.
    """

    provider = "intel_xpu"

    def __init__(self, model: str, base_url: str | None = None, **kwargs):
        super().__init__(model, base_url, **kwargs)
        if not self.validate_model():
            raise ValueError(
                f"Intel XPU client only supports the locked model id "
                f"'{_LOCKED_MODEL_ID}' (v1 scope is intentionally narrow, see #41); "
                f"got '{model}'. Pass '{_LOCKED_MODEL_ID}' as the model."
            )
        # Load at construction time so errors surface early (not deferred to the
        # first invoke call). The shared cache dedups by model id: the actual
        # disk/XPU load runs at most once per id per process, and every client
        # for that id — across TradingAgentsGraph's deep+quick clients and repeat
        # MCP runs — reuses the same in-memory model/tokenizer.
        self.model_obj, self.tokenizer = _MODEL_CACHE.get_or_load(
            self.model.lower(), self._load_model_and_tokenizer
        )

    def _load_model_and_tokenizer(self) -> tuple[Any, Any]:
        """Perform the actual disk/XPU load of the Mistral model and tokenizer.

        Called at most once per model id per process via ``_MODEL_CACHE`` — do
        not call directly if you want the shared/deduplicated copy. Uses the
        verified loading parameters from the reference scripts.

        Raises:
            ImportError: If torch or transformers are not installed.
            RuntimeError: If torch.xpu is not available.
        """
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "Intel XPU support requires 'torch'. "
                'Install with: pip install "tradingagents[xpu]"'
            ) from exc

        if not torch.xpu.is_available():
            raise RuntimeError(
                "torch.xpu is not available on this system. Intel XPU client requires "
                "an Intel Arc GPU and the Intel Extension for PyTorch installed."
            )

        try:
            from transformers import AutoTokenizer, Mistral3ForConditionalGeneration
        except ImportError as exc:
            raise ImportError(
                "Intel XPU support requires 'transformers'. "
                'Install with: pip install "tradingagents[xpu]"'
            ) from exc

        logger.info("Loading Mistral model on Intel XPU...")
        model = Mistral3ForConditionalGeneration.from_pretrained(
            _LOCKED_MODEL_ID,
            device_map="xpu",
            dtype=torch.bfloat16,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            _LOCKED_MODEL_ID,
            fix_mistral_regex=True,
        )
        logger.info("Model loaded successfully on Intel XPU")

        return model, tokenizer

    def get_llm(self) -> Any:
        """Return the configured IntelXPUChatModel instance.

        No warn_if_unknown_model() call here: __init__ already raises on a
        model mismatch, so by the time get_llm() runs validate_model() is
        guaranteed True.
        """
        generation_kwargs = {
            key: self.kwargs[key] for key in _GENERATION_PASSTHROUGH_KWARGS if key in self.kwargs
        }
        return IntelXPUChatModel(self.model_obj, self.tokenizer, generation_kwargs=generation_kwargs)

    def validate_model(self) -> bool:
        """Validate that the model is the locked Mistral ID.

        Only the one verified model (mistralai/Ministral-3-3B-Reasoning-2512)
        is supported in v1. This gates model loading in __init__: a mismatch
        raises immediately rather than warning-and-continuing with a
        different model than what was requested.
        """
        return self.model.lower() == _LOCKED_MODEL_ID.lower()
