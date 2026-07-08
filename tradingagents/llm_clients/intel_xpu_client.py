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
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from .base_client import BaseLLMClient

logger = logging.getLogger(__name__)

# Locked model ID for v1: no changes without explicit decision.
_LOCKED_MODEL_ID = "mistralai/Ministral-3-3B-Reasoning-2512"

# Global cache for the loaded model and tokenizer (avoid reloading on each client instantiation)
_MODEL_CACHE = None
_TOKENIZER_CACHE = None


def _load_model_and_tokenizer():
    """Load the Mistral model and tokenizer on XPU with the verified parameters.

    Cached globally so repeated client instantiations don't reload.

    Raises:
        ImportError: If torch or transformers are not installed.
        RuntimeError: If torch.xpu is not available.
    """
    global _MODEL_CACHE, _TOKENIZER_CACHE
    if _MODEL_CACHE is not None and _TOKENIZER_CACHE is not None:
        return _MODEL_CACHE, _TOKENIZER_CACHE

    # Check torch.xpu availability FIRST before trying to import transformers
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
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "Intel XPU support requires 'transformers'. "
            'Install with: pip install "tradingagents[xpu]"'
        ) from exc

    logger.info("Loading Mistral model on Intel XPU...")
    model = AutoModelForCausalLM.from_pretrained(
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

    _MODEL_CACHE = model
    _TOKENIZER_CACHE = tokenizer
    return model, tokenizer


class IntelXPUChatModel(BaseChatModel):
    """LangChain-compatible chat model wrapping Mistral on Intel XPU.

    Tokenizes input, runs model.generate(), and returns LLM results as AIMessage.
    """

    model_name: str = _LOCKED_MODEL_ID

    # Pydantic config to allow arbitrary attributes (model, tokenizer)
    model_config = {"arbitrary_types_allowed": True}

    model: Any = None
    tokenizer: Any = None

    def __init__(self, model, tokenizer):
        super().__init__(model=model, tokenizer=tokenizer)

    def _generate(
        self,
        messages,
        stop=None,
        run_manager=None,
        **kwargs,
    ) -> ChatResult:
        """Generate text from messages."""
        # Convert messages to prompt text
        # For now, just extract the last message's content (simple case)
        if isinstance(messages, list):
            # Handle list of message objects
            text = ""
            for msg in messages:
                if hasattr(msg, "content"):
                    text = msg.content
        else:
            # Handle ChatPromptValue or other inputs
            text = str(messages)

        # Tokenize and generate
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with __import__("torch").no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=0.7,
                do_sample=True,
            )

        # Decode
        response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Return ChatResult with ChatGeneration
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=response_text))])

    def bind_tools(self, tools, **kwargs):
        """Tool-calling is not supported on Intel XPU.

        Raises NotImplementedError so the caller (e.g., news_analyst.py) fails
        loudly rather than silently degrading.
        """
        warnings.warn(
            f"Intel XPU does not support tool-calling; "
            f"the caller will fail rather than fabricate results.",
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
            f"Intel XPU does not support structured output; "
            f"agent will use free-text generation instead.",
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
    with a clear error if requirements are not met.

    Structured output and tool-calling are explicitly unsupported (raise
    NotImplementedError) to prevent silent degradation in callers that rely on
    these capabilities.
    """

    provider = "intel_xpu"

    def __init__(self, model: str, base_url: str | None = None, **kwargs):
        super().__init__(model, base_url, **kwargs)
        # Load the model at construction time so errors surface early
        # (not deferred to the first invoke call)
        self.model_obj, self.tokenizer = _load_model_and_tokenizer()

    def get_llm(self) -> Any:
        """Return the configured IntelXPUChatModel instance."""
        self.warn_if_unknown_model()
        return IntelXPUChatModel(self.model_obj, self.tokenizer)

    def validate_model(self) -> bool:
        """Validate that the model is the locked Mistral ID.

        Only the one verified model (mistralai/Ministral-3-3B-Reasoning-2512)
        is supported in v1. Anything else is unknown and triggers warn_if_unknown_model.
        """
        return self.model.lower() == _LOCKED_MODEL_ID.lower()
