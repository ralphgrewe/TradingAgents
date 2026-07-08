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
import warnings
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, convert_to_openai_messages
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from .base_client import BaseLLMClient

logger = logging.getLogger(__name__)

# Locked model ID for v1: no changes without explicit decision.
_LOCKED_MODEL_ID = "mistralai/Ministral-3-3B-Reasoning-2512"

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

    The loaded model/tokenizer are cached on the instance (not module-level
    globals) — one client owns, and can independently release, its own model.

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
        # Load at construction time so errors surface early (not deferred to
        # the first invoke call), and cache on this instance only.
        self.model_obj, self.tokenizer = self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(self) -> tuple[Any, Any]:
        """Load the Mistral model and tokenizer on XPU with the verified parameters.

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
