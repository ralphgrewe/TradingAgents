"""Tests for the configurable sampling temperature reaching LLM clients (#178/#168).

Temperature is a cross-provider knob: when set it must reach the underlying
chat client; when unset the provider keeps its own default. This covers only
the llm_clients passthrough (issue #15's scope).

Upstream's companion coverage for the config default/env override
(``DEFAULT_CONFIG["temperature"]`` and ``TRADINGAGENTS_TEMPERATURE``) and for
``TradingAgentsGraph._get_provider_kwargs``'s float-casting/omission behavior
is intentionally NOT included here: those depend on default_config.py and
graph wiring, which are reconciled by issues #16 and #17 respectively. Our
fork currently defaults ``temperature`` to ``0.7`` (not upstream's ``None``)
and does not yet float-cast a string temperature — bringing those tests in
now would fail against still-pending #16/#17 work.
"""

import pytest

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
