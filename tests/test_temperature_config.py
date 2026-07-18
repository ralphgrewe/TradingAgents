"""Tests for the configurable sampling temperature reaching LLM clients (#178/#168).

Temperature is a cross-provider knob: when set it must reach the underlying
chat client; when unset the provider keeps its own default. ``TestTemperature
Forwarding`` covers only the llm_clients passthrough (issue #15's scope).

``TestGetProviderKwargsTemperature`` covers the config default/env override
(``DEFAULT_CONFIG["temperature"]``, ``TRADINGAGENTS_TEMPERATURE`` — see
tests/test_env_overrides.py) and ``TradingAgentsGraph._get_provider_kwargs``'s
float-casting/omission behavior, reconciled by issue #16: the fork now
defaults ``temperature`` to ``None`` (matching upstream) and
``_get_provider_kwargs`` casts a configured value through ``float()`` so a
string env var is tolerated.
"""

from unittest.mock import MagicMock

import pytest

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.llm_clients.factory import create_llm_client


@pytest.mark.unit
class TestTemperatureForwarding:
    @pytest.mark.parametrize(
        "provider,model",
        [
            ("openai", "gpt-4.1"),
            ("anthropic", "claude-sonnet-4-6"),
            ("google", "gemini-2.5-flash"),
            ("deepseek", "deepseek-chat"),
        ],
    )
    def test_temperature_reaches_client_when_set(self, provider, model):
        llm = create_llm_client(
            provider=provider, model=model, temperature=0.0, api_key="placeholder"
        ).get_llm()
        assert llm.temperature == 0.0

    def test_temperature_omitted_leaves_provider_default(self):
        # Not passing temperature must not force it to a value.
        llm = create_llm_client(
            provider="openai", model="gpt-4.1", api_key="placeholder"
        ).get_llm()
        # langchain's default is unset/None, not 0.0
        assert llm.temperature is None


@pytest.mark.unit
class TestGetProviderKwargsTemperature:
    """Unit tests for the temperature branch of ``_get_provider_kwargs``.

    Uses ``MagicMock(spec=TradingAgentsGraph)`` and calls the unbound method
    with the mock as ``self`` (same pattern as tests/test_memory_log.py) to
    exercise the method without a full graph build.
    """

    def test_float_config_value_passes_through(self):
        mock_graph = MagicMock(spec=TradingAgentsGraph)
        mock_graph.config = {"llm_provider": "ollama", "temperature": 0.3}
        kwargs = TradingAgentsGraph._get_provider_kwargs(mock_graph)
        assert kwargs["temperature"] == 0.3

    def test_string_config_value_is_float_cast(self):
        # Simulates TRADINGAGENTS_TEMPERATURE="0.3": default_config.py's
        # _coerce() can't infer float from a None default, so a string
        # reaches the config dict unchanged; _get_provider_kwargs must cast it.
        mock_graph = MagicMock(spec=TradingAgentsGraph)
        mock_graph.config = {"llm_provider": "ollama", "temperature": "0.3"}
        kwargs = TradingAgentsGraph._get_provider_kwargs(mock_graph)
        assert kwargs["temperature"] == pytest.approx(0.3)
        assert isinstance(kwargs["temperature"], float)

    def test_none_temperature_is_omitted(self):
        mock_graph = MagicMock(spec=TradingAgentsGraph)
        mock_graph.config = {"llm_provider": "ollama", "temperature": None}
        kwargs = TradingAgentsGraph._get_provider_kwargs(mock_graph)
        assert "temperature" not in kwargs
