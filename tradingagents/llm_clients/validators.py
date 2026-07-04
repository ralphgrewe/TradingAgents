"""Model name validators for each provider."""

from .model_catalog import get_known_models

# Providers whose model names are user-defined (local servers, relays, hosted
# OpenAI-compatible endpoints serving many models), so any model string is
# accepted without warning. Mistral is deliberately NOT here: our fork
# maintains a curated model_catalog entry for it, so it stays strictly
# validated against VALID_MODELS like openai/anthropic/etc.
_ANY_MODEL_PROVIDERS = (
    "ollama", "openrouter", "openai_compatible",
    "kimi", "groq", "nvidia", "bedrock",
)

VALID_MODELS = {
    provider: models
    for provider, models in get_known_models().items()
    if provider not in _ANY_MODEL_PROVIDERS
}


def validate_model(provider: str, model: str) -> bool:
    """Check if model name is valid for the given provider.

    For providers in ``_ANY_MODEL_PROVIDERS`` (ollama, openrouter,
    openai_compatible, kimi, groq, nvidia, bedrock) - any model is accepted.
    """
    provider_lower = provider.lower()

    if provider_lower in _ANY_MODEL_PROVIDERS:
        return True

    if provider_lower not in VALID_MODELS:
        return True

    return model in VALID_MODELS[provider_lower]
