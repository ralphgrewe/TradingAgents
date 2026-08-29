import os
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from tradingagents.dataflows.config import get_config
from tradingagents.llm_call_log import OllamaNumCtxDerivation, _count_prompt_tokens, _message_text

from .api_key_env import get_api_key_env
from .base_client import BaseLLMClient, normalize_content
from .capabilities import get_capabilities
from .validators import validate_model


class NormalizedChatOpenAI(ChatOpenAI):
    """ChatOpenAI with normalized content output and capability-aware binding.

    The Responses API returns content as a list of typed blocks
    (reasoning, text, etc.). ``invoke`` normalizes to string for
    consistent downstream handling.

    ``with_structured_output`` consults the per-model capability table
    (``capabilities.get_capabilities``) to pick the method and to decide
    whether ``tool_choice`` may be sent. Models that reject ``tool_choice``
    (e.g. DeepSeek V4 and reasoner — per their official tool-calling
    guide) still bind the schema as a tool, but no ``tool_choice``
    parameter is sent.

    Provider-specific quirks beyond structured-output (e.g. DeepSeek's
    reasoning_content roundtrip) live in subclasses so this base class
    stays small.
    """

    def invoke(self, input, config=None, **kwargs):
        return normalize_content(super().invoke(input, config, **kwargs))

    def with_structured_output(self, schema, *, method=None, **kwargs):
        caps = get_capabilities(self.model_name)
        if caps.preferred_structured_method == "none":
            raise NotImplementedError(
                f"{self.model_name} has no structured-output method available; "
                f"agent factories will fall back to free-text generation."
            )

        # Precedence for method resolution:
        # 1. Explicit argument from caller (already in ``method``)
        # 2. Config value if not "auto"
        # 3. Provider-level override (in subclasses like OllamaChatOpenAI)
        # 4. Capability table's preferred_structured_method
        # 5. Default (covered by caps.preferred_structured_method)

        if method is None:
            config_method = get_config().get("structured_output_method", "auto")
            # "auto" falls through to provider override (in subclasses) or capability table
            method = (
                config_method
                if config_method != "auto"
                else caps.preferred_structured_method
            )

        # When the method is function_calling and the model rejects tool_choice,
        # suppress langchain's hardcoded value. The schema is still bound as a
        # tool — exactly what DeepSeek's official tool-calling examples do.
        if method == "function_calling" and not caps.supports_tool_choice:
            kwargs.setdefault("tool_choice", None)
        return super().with_structured_output(schema, method=method, **kwargs)


class LocalCompatibleChatOpenAI(NormalizedChatOpenAI):
    """OpenAI-compatible client for arbitrary local servers (LM Studio, vLLM,
    llama.cpp via the generic ``openai_compatible`` provider).

    Their tool-calling support varies, and many reject the object-form
    ``tool_choice`` langchain sends for function-calling structured output. Bind
    the schema as a tool but don't force tool_choice, so structured output works
    across local servers regardless of the model ID's capabilities (#1057).
    """

    def with_structured_output(self, schema, *, method=None, **kwargs):
        resolved = method or get_capabilities(self.model_name).preferred_structured_method
        if resolved == "function_calling":
            kwargs.setdefault("tool_choice", None)
        return super().with_structured_output(schema, method=method, **kwargs)


def _input_to_messages(input_: Any) -> list:
    """Normalise a langchain LLM input to a list of message objects.

    Accepts a list of messages, a ``ChatPromptValue`` (from a
    ChatPromptTemplate), or anything else (treated as no messages).
    Used by providers that need to walk the outgoing message history;
    in particular DeepSeek thinking-mode propagation must work for
    both bare-list invocations and ChatPromptTemplate-driven ones, so
    treating only ``list`` here would silently skip half the call sites.
    """
    if isinstance(input_, list):
        return input_
    if hasattr(input_, "to_messages"):
        return input_.to_messages()
    return []


class DeepSeekChatOpenAI(NormalizedChatOpenAI):
    """DeepSeek-specific overrides on top of the OpenAI-compatible client.

    Thinking-mode round-trip is the only DeepSeek-specific behavior that
    stays here. When DeepSeek's thinking models return a response with
    ``reasoning_content``, that field must be echoed back as part of the
    assistant message on the next turn or the API fails with HTTP 400.
    ``_create_chat_result`` captures it on receive and
    ``_get_request_payload`` re-attaches it on send.

    Tool-choice handling for V4 and reasoner — those models reject the
    ``tool_choice`` parameter — is handled by the capability dispatch in
    ``NormalizedChatOpenAI.with_structured_output``, not here.
    """

    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        outgoing = payload.get("messages", [])
        for message_dict, message in zip(outgoing, _input_to_messages(input_), strict=False):
            if not isinstance(message, AIMessage):
                continue
            reasoning = message.additional_kwargs.get("reasoning_content")
            if reasoning is not None:
                message_dict["reasoning_content"] = reasoning
        return payload

    def _create_chat_result(self, response, generation_info=None):
        chat_result = super()._create_chat_result(response, generation_info)
        response_dict = (
            response
            if isinstance(response, dict)
            else response.model_dump(
                exclude={"choices": {"__all__": {"message": {"parsed"}}}}
            )
        )
        for generation, choice in zip(
            chat_result.generations, response_dict.get("choices", []), strict=False
        ):
            reasoning = choice.get("message", {}).get("reasoning_content")
            if reasoning is not None:
                generation.message.additional_kwargs["reasoning_content"] = reasoning
        return chat_result


class OllamaChatOpenAI(NormalizedChatOpenAI):
    """Ollama-specific overrides: per-request ``num_ctx`` derivation (issue #154) and think mode (issue #155).

    When an explicit ``ollama_num_ctx`` is configured, that value is baked
    into ``extra_body`` at construction time by ``OpenAIClient.get_llm`` (the
    #149 behaviour, unchanged) and this override does nothing. Otherwise,
    ``get_llm`` attaches an ``OllamaNumCtxDerivation`` policy to
    ``self._ollama_num_ctx_derivation``, and ``_get_request_payload`` (the
    same per-request hook ``DeepSeekChatOpenAI``/``MinimaxChatOpenAI`` above
    use) derives ``num_ctx`` from *this* request's own measured prompt size
    and attaches it to the outgoing request as the same non-standard
    ``extra_body.options.num_ctx`` field #149 already established.

    Think mode (issue #155) is set via ``extra_body.think`` at construction
    time by ``OpenAIClient.get_llm`` when ``ollama_think`` config is True,
    and this override preserves it through per-request derivation.

    This only ever *attaches* a value: whether the derived requirement fits
    within ``ollama_num_ctx_max`` was already decided by
    ``ContextWindowGuardHandler`` (``tradingagents/llm_call_log.py``), which
    runs from the callback path *before* ``_generate``/``_get_request_payload``
    are reached and raises ``PromptContextOverflowError`` to abort the run
    when it doesn't fit -- so by the time this method runs for a given
    request, the derived value is already known to be <= ``num_ctx_max``
    (the ``min()`` below is a defensive clamp, not live enforcement).

    Structured output method (issue #161): when structured_output_method is
    "auto", Ollama defaults to json_schema instead of function_calling, since
    Ollama's OpenAI-compatibility layer does not honor tool_choice directives
    and function_calling can silently return None when the model writes prose.
    """

    def with_structured_output(self, schema, *, method=None, **kwargs):
        # When method is not explicitly provided and config is "auto",
        # override the default from "function_calling" (the permissive default)
        # to "json_schema" (grammar-constrained at decode time).
        if method is None:
            config_method = get_config().get("structured_output_method", "auto")
            if config_method == "auto":
                # Provider-level override: Ollama → json_schema
                method = "json_schema"

        return super().with_structured_output(schema, method=method, **kwargs)

    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        derivation: OllamaNumCtxDerivation | None = getattr(
            self, "_ollama_num_ctx_derivation", None
        )
        if derivation is None or not derivation.applies_to(self.model_name):
            return payload

        messages = _input_to_messages(input_)
        prompt_text = "".join(_message_text(m) for m in messages)
        # Ollama is always served through ChatOpenAI (this class's base), so
        # the tiktoken-family estimator applies unconditionally here -- see
        # llm_call_log.py's "Tokenizer strategy" for why "openai-chat" is the
        # right llm_type for the whole OpenAI-compatible provider family.
        token_count, _method = _count_prompt_tokens(prompt_text, self.model_name, "openai-chat")
        needed = derivation.needed_tokens(token_count)
        num_ctx = min(needed, derivation.num_ctx_max)

        extra_body = dict(payload.get("extra_body") or {})
        options = dict(extra_body.get("options") or {})
        options["num_ctx"] = num_ctx
        extra_body["options"] = options
        payload["extra_body"] = extra_body
        return payload


class MinimaxChatOpenAI(NormalizedChatOpenAI):
    """MiniMax-specific overrides on top of the OpenAI-compatible client.

    M2.x reasoning models embed ``<think>...</think>`` blocks directly in
    ``message.content`` by default, which would pollute saved reports.
    Per platform.minimax.io/docs/api-reference/text-openai-api, setting
    ``reasoning_split=True`` in the request body redirects the thinking
    block into ``reasoning_details`` so ``content`` stays clean.

    The flag is gated by ``ModelCapabilities.requires_reasoning_split``
    because non-reasoning MiniMax endpoints (Coding Plan, MiniMax-Text-01)
    reject the parameter via the openai SDK's strict kwarg validation
    (#826).

    Tool-choice handling for M2.x — those models accept only the string
    enum ``{"none", "auto"}`` and reject langchain's function-spec dict —
    is handled by the capability dispatch in
    ``NormalizedChatOpenAI.with_structured_output``, not here.
    """

    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        if get_capabilities(self.model_name).requires_reasoning_split:
            payload.setdefault("reasoning_split", True)
        return payload


# Kwargs forwarded from user config to ChatOpenAI
_PASSTHROUGH_KWARGS = (
    "timeout", "max_retries", "reasoning_effort", "temperature",
    "api_key", "callbacks", "http_client", "http_async_client",
)

# OpenAI's ``reasoning_effort`` is only accepted by reasoning models — the GPT-5
# family and the o-series. Non-reasoning models (gpt-4.1, gpt-4o, ...) 400 with
# "Unsupported parameter: 'reasoning.effort' is not supported with this model".
# Drop the kwarg for those rather than crash the run.
_OPENAI_REASONING_MODEL = re.compile(r"^(gpt-5|o[1-9])")


def _supports_reasoning_effort(model: str) -> bool:
    """Whether the (native OpenAI) model accepts ``reasoning_effort``."""
    return bool(_OPENAI_REASONING_MODEL.match(model.lower().strip()))


@dataclass(frozen=True)
class ProviderSpec:
    """Declarative config for one OpenAI-compatible provider.

    The OpenAI-compatible family (OpenAI, xAI, DeepSeek, Qwen, GLM, MiniMax,
    OpenRouter, Ollama, and any user endpoint) all speak the same Chat
    Completions API and differ only by these fields — so one row here replaces
    the former per-provider base-URL dict, auth handling, and client-class
    branches. Native Anthropic / Google use their own clients (genuinely
    different APIs) and are intentionally NOT in this registry.

    The API-key env var stays in ``api_key_env.PROVIDER_API_KEY_ENV`` (the single
    source consulted by both this client and the CLI prompt); only behavior that
    is provider-specific (base URL, key optionality, wire-format quirks via
    ``chat_class``) lives here.
    """

    chat_class: type = NormalizedChatOpenAI   # provider quirks live in the subclass
    base_url: str | None = None            # default endpoint (None -> SDK default)
    base_url_env: str | None = None        # env var that overrides base_url (e.g. OLLAMA_BASE_URL)
    key_optional: bool = False                # don't require/prompt; send a placeholder if unset
    placeholder_key: str = "EMPTY"            # sent when no key is available (keyless local servers)
    require_base_url: bool = False            # error if no base_url is resolved (generic endpoint)
    use_responses_api: bool = False           # native OpenAI Responses API


# Single source of truth for the OpenAI-compatible provider family. Dual-region
# providers (qwen/glm/minimax) keep separate endpoints because international and
# China accounts cannot share credentials (#758).
OPENAI_COMPATIBLE_PROVIDERS: dict[str, ProviderSpec] = {
    "openai":     ProviderSpec(use_responses_api=True),
    "xai":        ProviderSpec(base_url="https://api.x.ai/v1"),
    "deepseek":   ProviderSpec(base_url="https://api.deepseek.com", chat_class=DeepSeekChatOpenAI),
    "qwen":       ProviderSpec(base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
    "qwen-cn":    ProviderSpec(base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"),
    "glm":        ProviderSpec(base_url="https://api.z.ai/api/paas/v4/"),
    "glm-cn":     ProviderSpec(base_url="https://open.bigmodel.cn/api/paas/v4/"),
    "minimax":    ProviderSpec(base_url="https://api.minimax.io/v1", chat_class=MinimaxChatOpenAI),
    "minimax-cn": ProviderSpec(base_url="https://api.minimaxi.com/v1", chat_class=MinimaxChatOpenAI),
    "openrouter": ProviderSpec(base_url="https://openrouter.ai/api/v1"),
    "mistral":    ProviderSpec(base_url="https://api.mistral.ai/v1"),
    "kimi":       ProviderSpec(base_url="https://api.moonshot.ai/v1"),
    "groq":       ProviderSpec(base_url="https://api.groq.com/openai/v1"),
    "nvidia":     ProviderSpec(base_url="https://integrate.api.nvidia.com/v1"),
    "ollama":     ProviderSpec(base_url="http://localhost:11434/v1", base_url_env="OLLAMA_BASE_URL",
                               key_optional=True, placeholder_key="ollama", chat_class=OllamaChatOpenAI),
    # Generic endpoint: user supplies base_url; key optional (keyless local).
    "openai_compatible": ProviderSpec(
        require_base_url=True, key_optional=True, chat_class=LocalCompatibleChatOpenAI
    ),
}


def is_openai_compatible(provider: str) -> bool:
    """Whether ``provider`` is served by the OpenAI-compatible registry."""
    return provider.lower() in OPENAI_COMPATIBLE_PROVIDERS


def _is_native_openai_base_url(base_url: str | None) -> bool:
    """True when ``base_url`` is unset or points at api.openai.com.

    The Responses API (/v1/responses) only exists on native OpenAI. A custom
    base_url on the ``openai`` provider (a proxy, gateway, or local server)
    speaks only Chat Completions, so the Responses API must stay off there even
    though the provider spec enables it (#1024).
    """
    if not base_url:
        return True
    if "://" not in base_url:
        base_url = "https://" + base_url
    host = urlparse(base_url).hostname or ""
    return host == "api.openai.com" or host.endswith(".openai.com")


class OpenAIClient(BaseLLMClient):
    """Client for OpenAI, Ollama, OpenRouter, and xAI providers.

    For native OpenAI models, uses the Responses API (/v1/responses) which
    supports reasoning_effort with function tools across all model families
    (GPT-4.1, GPT-5). Third-party compatible providers (xAI, OpenRouter,
    Ollama) use standard Chat Completions.
    """

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        provider: str = "openai",
        **kwargs,
    ):
        super().__init__(model, base_url, **kwargs)
        self.provider = provider.lower()

    def get_llm(self) -> Any:
        """Return a configured ChatOpenAI instance, driven by the provider registry."""
        self.warn_if_unknown_model()
        llm_kwargs = {"model": self.model}
        spec = OPENAI_COMPATIBLE_PROVIDERS.get(self.provider)
        chat_cls = NormalizedChatOpenAI

        if spec is not None:
            chat_cls = spec.chat_class

            # base_url precedence: explicit client base_url (carries the config /
            # TRADINGAGENTS_LLM_BACKEND_URL value) > provider env override (e.g.
            # OLLAMA_BASE_URL) > provider default. None means use the SDK default.
            env_base_url = os.environ.get(spec.base_url_env) if spec.base_url_env else None
            base_url = self.base_url or env_base_url or spec.base_url
            if spec.require_base_url and not base_url:
                raise ValueError(
                    f"Provider '{self.provider}' requires a base_url. Set it via "
                    "backend_url / TRADINGAGENTS_LLM_BACKEND_URL to your endpoint, "
                    "e.g. http://localhost:8000/v1 (vLLM) or http://localhost:1234/v1 "
                    "(LM Studio)."
                )
            if base_url:
                llm_kwargs["base_url"] = base_url

            # API key: required unless key_optional; keyless local servers get a
            # placeholder. The env-var name is the single source in api_key_env.
            api_key_env = get_api_key_env(self.provider)
            api_key = os.environ.get(api_key_env) if api_key_env else None
            if api_key:
                llm_kwargs["api_key"] = api_key
            elif spec.key_optional:
                llm_kwargs["api_key"] = spec.placeholder_key
            elif api_key_env:
                raise ValueError(
                    f"API key for provider '{self.provider}' is not set. "
                    f"Please set the {api_key_env} environment variable "
                    f"(e.g. add {api_key_env}=your_key to your .env file)."
                )

            # The Responses API only exists on native OpenAI; if the user points
            # the openai provider at a custom base_url (proxy/gateway/local), it
            # only speaks Chat Completions, so keep Responses off there (#1024).
            if spec.use_responses_api and _is_native_openai_base_url(base_url):
                llm_kwargs["use_responses_api"] = True
        elif self.base_url:
            llm_kwargs["base_url"] = self.base_url

        # Forward user-provided kwargs
        for key in _PASSTHROUGH_KWARGS:
            if key not in self.kwargs:
                continue
            if key == "reasoning_effort" and not _supports_reasoning_effort(self.model):
                continue
            llm_kwargs[key] = self.kwargs[key]

        # Ollama-native context-length override (issue #149, per the #148
        # diagnosis at docs/analysis/prompt-truncation-diagnosis.md): num_ctx
        # has no OpenAI chat-completions equivalent, so it can't go through
        # _PASSTHROUGH_KWARGS as a direct ChatOpenAI constructor kwarg.
        # Ollama's OpenAI-compatible endpoint accepts it via a non-standard
        # top-level "options" request field instead (confirmed live by
        # scripts/repro_ollama_token_anomaly.py) -- ChatOpenAI's own
        # ``extra_body`` field is serialized as extra top-level JSON on every
        # request, which is exactly the hook this needs. Only forwarded when
        # the caller actually set num_ctx (ollama_num_ctx config key,
        # TradingAgentsGraph._get_provider_kwargs) -- other OpenAI-compatible
        # providers don't get this kwarg passed to them at all, so they never
        # see an "options" field they might reject.
        # Ollama think mode (issue #155) follows the same pattern: sets
        # extra_body["think"] alongside the options.num_ctx field. Only for
        # ollama provider.
        if self.provider == "ollama":
            num_ctx = self.kwargs.get("num_ctx")
            ollama_think = self.kwargs.get("ollama_think")
            if num_ctx is not None or ollama_think:
                extra_body = {"options": {}}
                if num_ctx is not None:
                    extra_body["options"]["num_ctx"] = int(num_ctx)
                if ollama_think:
                    extra_body["think"] = True
                llm_kwargs["extra_body"] = extra_body

        # The subclass (provider quirks) comes from the registry spec.
        llm = chat_cls(**llm_kwargs)

        # Per-request num_ctx derivation (issue #154): only reached when no
        # explicit num_ctx was set above (TradingAgentsGraph._get_provider_kwargs
        # never forwards both at once -- num_ctx wins outright when present).
        # Not a ChatOpenAI/pydantic field (it has no OpenAI-API meaning of its
        # own), so it's attached as a private instance attribute rather than a
        # constructor kwarg; OllamaChatOpenAI._get_request_payload reads it
        # back per request. Harmless to set on any chat_cls -- only
        # OllamaChatOpenAI's _get_request_payload ever looks at it.
        derivation = self.kwargs.get("ollama_num_ctx_derivation")
        if derivation is not None:
            llm._ollama_num_ctx_derivation = derivation

        return llm

    def validate_model(self) -> bool:
        """Validate model for the provider."""
        return validate_model(self.provider, self.model)
