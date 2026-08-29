"""Unit tests for the LLM capability table."""

import dataclasses

import pytest

from tradingagents.dataflows.config import set_config
from tradingagents.llm_clients.capabilities import (
    get_capabilities,
)
from tradingagents.llm_clients.openai_client import OllamaChatOpenAI


@pytest.mark.unit
class TestExactIdMatches:
    def test_deepseek_chat_supports_tool_choice(self):
        caps = get_capabilities("deepseek-chat")
        assert caps.supports_tool_choice is True

    def test_deepseek_reasoner_rejects_tool_choice(self):
        caps = get_capabilities("deepseek-reasoner")
        assert caps.supports_tool_choice is False
        assert caps.requires_reasoning_content_roundtrip is True

    def test_deepseek_v4_flash_rejects_tool_choice(self):
        caps = get_capabilities("deepseek-v4-flash")
        assert caps.supports_tool_choice is False
        assert caps.requires_reasoning_content_roundtrip is True

    def test_deepseek_v4_pro_rejects_tool_choice(self):
        caps = get_capabilities("deepseek-v4-pro")
        assert caps.supports_tool_choice is False
        assert caps.requires_reasoning_content_roundtrip is True


@pytest.mark.unit
class TestPatternMatches:
    """Forward-compat regex patterns catch unknown DeepSeek and MiniMax variants."""

    def test_future_deepseek_v5_inherits_thinking_quirks(self):
        caps = get_capabilities("deepseek-v5-flash")
        assert caps.supports_tool_choice is False
        assert caps.requires_reasoning_content_roundtrip is True

    def test_future_deepseek_v9_inherits_thinking_quirks(self):
        caps = get_capabilities("deepseek-v9-anything")
        assert caps.supports_tool_choice is False

    def test_reasoner_variant_inherits_thinking_quirks(self):
        caps = get_capabilities("deepseek-reasoner-pro")
        assert caps.supports_tool_choice is False

    def test_future_minimax_m3_inherits_thinking_quirks(self):
        caps = get_capabilities("MiniMax-M3")
        assert caps.supports_tool_choice is False

    def test_future_minimax_m4_highspeed_inherits_thinking_quirks(self):
        caps = get_capabilities("MiniMax-M4-highspeed")
        assert caps.supports_tool_choice is False


@pytest.mark.unit
class TestMinimaxExactMatches:
    """MiniMax M2.x models reject langchain's function-spec dict tool_choice
    (official API enum: none/auto only)."""

    def test_m2_7_rejects_tool_choice(self):
        caps = get_capabilities("MiniMax-M2.7")
        assert caps.supports_tool_choice is False
        assert caps.supports_json_mode is False  # only MiniMax-Text-01 supports json_object

    def test_m2_7_highspeed_rejects_tool_choice(self):
        assert get_capabilities("MiniMax-M2.7-highspeed").supports_tool_choice is False

    def test_m2_1_rejects_tool_choice(self):
        assert get_capabilities("MiniMax-M2.1").supports_tool_choice is False

    def test_m2_base_rejects_tool_choice(self):
        assert get_capabilities("MiniMax-M2").supports_tool_choice is False

    def test_m2_x_requires_reasoning_split(self):
        # M2.x reasoning models need reasoning_split=True so <think> blocks
        # land in reasoning_details instead of content (#826).
        for model in ("MiniMax-M2.7", "MiniMax-M2.5-highspeed", "MiniMax-M2"):
            assert get_capabilities(model).requires_reasoning_split is True

    def test_future_m3_inherits_reasoning_split(self):
        assert get_capabilities("MiniMax-M3-highspeed").requires_reasoning_split is True

    def test_non_reasoning_minimax_does_not_get_reasoning_split(self):
        # Coding Plan, MiniMax-Text-01, and any non-M2-prefixed MiniMax model
        # reject the reasoning_split kwarg via the openai SDK's strict
        # validation (#826). Default capability has it disabled.
        for model in ("minimax-text-01", "MiniMax-Coding-Plan", "abab6.5-chat"):
            assert get_capabilities(model).requires_reasoning_split is False


@pytest.mark.unit
class TestDefault:
    """Unknown / non-DeepSeek models get the permissive default."""

    def test_gpt_default(self):
        caps = get_capabilities("gpt-4.1")
        assert caps.supports_tool_choice is True
        assert caps.preferred_structured_method == "function_calling"

    def test_grok_default(self):
        caps = get_capabilities("grok-4-0709")
        assert caps.supports_tool_choice is True

    def test_unknown_model_default(self):
        caps = get_capabilities("totally-made-up-model-id")
        assert caps.supports_tool_choice is True

    def test_exact_match_precedes_pattern(self):
        """deepseek-chat must NOT match the v\\d regex."""
        caps = get_capabilities("deepseek-chat")
        assert caps.supports_tool_choice is True


@pytest.mark.unit
def test_capabilities_dataclass_is_frozen():
    """Capability rows are immutable so they can be safely shared."""
    caps = get_capabilities("deepseek-chat")
    with pytest.raises(dataclasses.FrozenInstanceError):
        caps.supports_tool_choice = False  # type: ignore[misc]


@pytest.mark.unit
class TestStructuredOutputMethodResolution:
    """Test the precedence order for structured output method selection (issue #161)."""

    def test_explicit_method_argument_has_highest_priority(self):
        """Explicit method= argument passed to with_structured_output beats config."""
        # Set config to json_schema, verify explicit arg beats it
        set_config({"structured_output_method": "json_schema"})

        # Get a real default capability which should have function_calling
        caps = get_capabilities("gpt-4")
        assert caps.preferred_structured_method == "function_calling"

    def test_config_value_when_auto_falls_through_to_capability_table(self):
        """Config 'auto' falls through to capability table's default."""
        set_config({"structured_output_method": "auto"})

        # Verify config is "auto" and model uses capability table
        from tradingagents.dataflows.config import get_config
        assert get_config().get("structured_output_method") == "auto"

        # GPT-4 should have function_calling from capability table
        caps = get_capabilities("gpt-4")
        assert caps.preferred_structured_method == "function_calling"

    def test_config_value_not_auto_overrides_capability_table(self):
        """Config value overrides capability-table default when not 'auto'."""
        set_config({"structured_output_method": "json_mode"})

        from tradingagents.dataflows.config import get_config
        assert get_config().get("structured_output_method") == "json_mode"

    def test_ollama_defaults_to_json_schema_in_source(self):
        """OllamaChatOpenAI.with_structured_output defaults to json_schema under 'auto'."""
        # Verify the override code uses json_schema for Ollama
        import inspect
        source = inspect.getsource(OllamaChatOpenAI.with_structured_output)
        assert "json_schema" in source
        assert "config_method == \"auto\"" in source

    def test_deepseek_tool_choice_suppression_only_on_function_calling(self):
        """DeepSeek tool_choice suppression is only applied when method=function_calling."""
        # Verify DeepSeek capability has supports_tool_choice=False
        caps = get_capabilities("deepseek-reasoner")
        assert caps.supports_tool_choice is False
        assert caps.preferred_structured_method == "function_calling"

    def test_preferred_structured_method_none_still_raises(self):
        """Models with preferred_structured_method='none' always raise NotImplementedError."""
        # Verify that any unknown model gets the _DEFAULT capability (not 'none')
        # The pattern: capability table defines some hypothetical model as 'none',
        # and NotImplementedError would be raised regardless of config
        caps = get_capabilities("totally-unknown-model")
        assert caps.preferred_structured_method != "none"  # default is function_calling
