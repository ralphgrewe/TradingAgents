import unittest
import warnings

import pytest

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.llm_clients.base_client import BaseLLMClient
from tradingagents.llm_clients.model_catalog import get_known_models, get_model_options
from tradingagents.llm_clients.validators import _ANY_MODEL_PROVIDERS, validate_model


class DummyLLMClient(BaseLLMClient):
    def __init__(self, provider: str, model: str):
        self.provider = provider
        super().__init__(model)

    def get_llm(self):
        self.warn_if_unknown_model()
        return object()

    def validate_model(self) -> bool:
        return validate_model(self.provider, self.model)


@pytest.mark.unit
class ModelValidationTests(unittest.TestCase):
    def test_cli_catalog_models_are_all_validator_approved(self):
        for provider, models in get_known_models().items():
            if provider in ("ollama", "openrouter"):
                continue

            for model in models:
                with self.subTest(provider=provider, model=model):
                    self.assertTrue(validate_model(provider, model))

    def test_unknown_model_emits_warning_for_strict_provider(self):
        client = DummyLLMClient("openai", "not-a-real-openai-model")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            client.get_llm()

        self.assertEqual(len(caught), 1)
        self.assertIn("not-a-real-openai-model", str(caught[0].message))
        self.assertIn("openai", str(caught[0].message))

    def test_openrouter_and_ollama_accept_custom_models_without_warning(self):
        for provider in ("openrouter", "ollama"):
            client = DummyLLMClient(provider, "custom-model-name")

            with self.subTest(provider=provider):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    client.get_llm()

                self.assertEqual(caught, [])


@pytest.mark.unit
class ModelCatalogUpstreamSyncTests(unittest.TestCase):
    """Covers issue #25: model_catalog.py reconciled with upstream's refreshed
    per-provider lineup (upstream commits 03600f3 / 7bb16c5), except Mistral
    and Perplexity, which the fork deliberately keeps as-is."""

    def test_retired_model_ids_are_gone(self):
        known = get_known_models()
        retired_by_provider = {
            "openai": {"gpt-4.1"},
            "anthropic": {"claude-sonnet-4-5", "claude-opus-4-5"},
            "google": {
                "gemini-3-flash-preview", "gemini-2.5-flash",
                "gemini-2.5-flash-lite", "gemini-2.5-pro",
            },
            "xai": {
                "grok-4-fast-non-reasoning", "grok-4-fast-reasoning",
                "grok-4-0709", "grok-4.20", "grok-4.20-non-reasoning",
                "grok-4.20-reasoning",
            },
            "deepseek": {"deepseek-chat", "deepseek-reasoner"},
            "qwen": {"qwen3.6-flash", "qwen3.5-flash", "qwen3.5-plus", "qwen3-max"},
            "minimax": {"MiniMax-M2.1", "MiniMax-M2.1-highspeed", "MiniMax-M2"},
        }
        for provider, retired in retired_by_provider.items():
            with self.subTest(provider=provider):
                self.assertFalse(
                    retired & set(known[provider]),
                    f"{provider} catalog still lists retired models: "
                    f"{retired & set(known[provider])}",
                )

    def test_new_upstream_model_ids_are_present(self):
        known = get_known_models()
        added_by_provider = {
            "anthropic": {"claude-opus-4-8"},
            "google": {"gemini-3.5-flash"},
            "xai": {
                "grok-4.3", "grok-4.20-0309-non-reasoning",
                "grok-4.20-0309-reasoning", "grok-4.20-multi-agent-0309",
                "grok-build-0.1",
            },
            "qwen": {"qwen3.7-plus", "qwen3.7-max", "qwen3.6-max"},
            "glm": {"glm-5.2"},
            "minimax": {"MiniMax-M3"},
        }
        for provider, added in added_by_provider.items():
            with self.subTest(provider=provider):
                missing = added - set(known[provider])
                self.assertFalse(missing, f"{provider} catalog missing {missing}")

    def test_mistral_catalog_is_untouched_and_strictly_validated(self):
        self.assertNotIn("mistral", _ANY_MODEL_PROVIDERS)
        quick_values = {v for _, v in get_model_options("mistral", "quick")}
        deep_values = {v for _, v in get_model_options("mistral", "deep")}
        self.assertEqual(
            quick_values, {"mistral-large", "mistral-small", "mistral-tiny", "codestral"}
        )
        self.assertEqual(
            deep_values, {"mistral-large", "mistral-medium", "mistral-small", "codestral"}
        )

    def test_perplexity_catalog_is_untouched(self):
        quick_values = {v for _, v in get_model_options("perplexity", "quick")}
        self.assertEqual(quick_values, {"sonar-small-online", "sonar-online", "sonar-pro"})

    def test_ollama_retains_default_config_models_for_dropdown_visibility(self):
        for mode, config_key in (("quick", "quick_think_llm"), ("deep", "deep_think_llm")):
            with self.subTest(mode=mode):
                values = {v for _, v in get_model_options("ollama", mode)}
                self.assertIn(DEFAULT_CONFIG[config_key], values)
