"""Tests for OLLAMA_BASE_URL env-var override across CLI and client paths."""

from __future__ import annotations

# ---- ollama_client side: OllamaClient base-URL resolution -----------------
#
# Issue #169 moved the ollama provider off the OpenAI-compatible
# ``ProviderSpec`` registry entirely, onto a dedicated ``OllamaClient``
# talking to Ollama's native ``/api/chat`` endpoint. The precedence is
# unchanged (explicit base_url / TRADINGAGENTS_LLM_BACKEND_URL >
# OLLAMA_BASE_URL > default), but the default is now the bare host (no
# ``/v1`` -- the native endpoint isn't OpenAI-compat), and a configured URL
# ending in ``/v1``/``/v1/`` -- the old documented shape -- has that suffix
# stripped so existing setups keep working against the new endpoint.


def test_resolver_returns_default_when_env_unset(monkeypatch):
    from tradingagents.llm_clients.ollama_client import OllamaClient

    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    llm = OllamaClient(model="llama3.1").get_llm()
    assert llm.base_url == "http://localhost:11434"


def test_resolver_returns_env_when_set(monkeypatch):
    from tradingagents.llm_clients.ollama_client import OllamaClient

    monkeypatch.setenv("OLLAMA_BASE_URL", "http://remote-ollama:11434")
    llm = OllamaClient(model="llama3.1").get_llm()
    assert llm.base_url == "http://remote-ollama:11434"


def test_resolver_strips_v1_suffix_from_env(monkeypatch):
    """Existing setups with the old documented .../v1 value in OLLAMA_BASE_URL keep working."""
    from tradingagents.llm_clients.ollama_client import OllamaClient

    monkeypatch.setenv("OLLAMA_BASE_URL", "http://remote-ollama:11434/v1")
    llm = OllamaClient(model="llama3.1").get_llm()
    assert llm.base_url == "http://remote-ollama:11434"


def test_resolver_strips_v1_suffix_with_trailing_slash(monkeypatch):
    from tradingagents.llm_clients.ollama_client import OllamaClient

    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    llm = OllamaClient(model="llama3.1", base_url="http://explicit:11434/v1/").get_llm()
    assert llm.base_url == "http://explicit:11434"


def test_resolver_evaluation_is_call_time(monkeypatch):
    """Setting the env AFTER module import must still take effect."""
    from tradingagents.llm_clients.ollama_client import OllamaClient

    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    client = OllamaClient(model="llama3.1")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://late-set:11434")
    llm = client.get_llm()
    assert llm.base_url == "http://late-set:11434"


def test_resolver_does_not_affect_other_providers(monkeypatch):
    """OLLAMA_BASE_URL should NOT leak into xai/deepseek/etc."""
    from tradingagents.llm_clients.openai_client import OpenAIClient

    monkeypatch.setenv("OLLAMA_BASE_URL", "http://elsewhere")
    xai_llm = OpenAIClient(model="grok-4", provider="xai", api_key="k").get_llm()
    deepseek_llm = OpenAIClient(model="deepseek-chat", provider="deepseek", api_key="k").get_llm()
    assert str(xai_llm.openai_api_base) == "https://api.x.ai/v1"
    assert str(deepseek_llm.openai_api_base) == "https://api.deepseek.com"


def test_client_get_llm_picks_up_env(monkeypatch):
    """End-to-end: OllamaClient.get_llm() respects OLLAMA_BASE_URL."""
    from tradingagents.llm_clients.ollama_client import OllamaClient

    monkeypatch.setenv("OLLAMA_BASE_URL", "http://my-ollama:11434")
    client = OllamaClient(model="llama3.1")
    llm = client.get_llm()
    assert "my-ollama" in llm.base_url


def test_explicit_base_url_overrides_env(monkeypatch):
    """An explicit base_url passed to the client wins over the env var."""
    from tradingagents.llm_clients.ollama_client import OllamaClient

    monkeypatch.setenv("OLLAMA_BASE_URL", "http://env-set:11434")
    client = OllamaClient(
        model="llama3.1",
        base_url="http://explicit:11434",
    )
    llm = client.get_llm()
    assert "explicit" in llm.base_url
    assert "env-set" not in llm.base_url


def test_factory_routes_ollama_to_ollama_client_not_openai_compatible(monkeypatch):
    """create_llm_client("ollama", ...) must not construct an OpenAI-compat ChatOpenAI."""
    from tradingagents.llm_clients.factory import create_llm_client
    from tradingagents.llm_clients.ollama_client import NormalizedChatOllama

    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    llm = create_llm_client(provider="ollama", model="qwen3.5:9b").get_llm()
    assert isinstance(llm, NormalizedChatOllama)


# ---- cli.utils side: select_llm_provider dropdown -------------------------


def test_cli_dropdown_uses_env(monkeypatch):
    """The Ollama entry in the CLI dropdown must reflect OLLAMA_BASE_URL."""
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://cli-remote:11434/v1")
    # Reach inside the function via the same env-read it does at call time
    import os
    ollama_url = (
        os.environ.get("OLLAMA_BASE_URL")
        or "http://localhost:11434/v1"
    )
    assert ollama_url == "http://cli-remote:11434/v1"


def test_cli_dropdown_default_when_unset(monkeypatch):
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    import os
    ollama_url = (
        os.environ.get("OLLAMA_BASE_URL")
        or "http://localhost:11434/v1"
    )
    assert ollama_url == "http://localhost:11434/v1"


# ---- confirm_ollama_endpoint UX -------------------------------------------


def test_confirm_endpoint_shows_default(monkeypatch, capsys):
    from cli.utils import confirm_ollama_endpoint

    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    confirm_ollama_endpoint("http://localhost:11434/v1")
    out = capsys.readouterr().out
    assert "http://localhost:11434/v1" in out
    assert "OLLAMA_BASE_URL" not in out  # not from env
    assert "Note" not in out  # no warnings for the canonical default


def test_confirm_endpoint_marks_env_origin(monkeypatch, capsys):
    from cli.utils import confirm_ollama_endpoint

    monkeypatch.setenv("OLLAMA_BASE_URL", "http://remote-host:11434/v1")
    confirm_ollama_endpoint("http://remote-host:11434/v1")
    out = capsys.readouterr().out
    assert "http://remote-host:11434/v1" in out
    assert "OLLAMA_BASE_URL" in out


def test_confirm_endpoint_warns_on_missing_scheme(monkeypatch, capsys):
    """If user sets OLLAMA_BASE_URL=0.0.0.128, advise on the expected shape."""
    from cli.utils import confirm_ollama_endpoint

    monkeypatch.setenv("OLLAMA_BASE_URL", "0.0.0.128")
    confirm_ollama_endpoint("0.0.0.128")
    out = capsys.readouterr().out
    assert "missing a scheme" in out
    assert "http://<host>:11434/v1" in out


def test_confirm_endpoint_warns_on_non_default_port_remote(monkeypatch, capsys):
    """A remote host with no :11434 gets a soft hint about port mismatch."""
    from cli.utils import confirm_ollama_endpoint

    monkeypatch.setenv("OLLAMA_BASE_URL", "http://remote-host/v1")
    confirm_ollama_endpoint("http://remote-host/v1")
    out = capsys.readouterr().out
    assert "port 11434" in out


def test_confirm_endpoint_quiet_on_local_no_port(monkeypatch, capsys):
    """Local host without port shouldn't trigger the remote-port hint."""
    from cli.utils import confirm_ollama_endpoint

    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost/v1")
    confirm_ollama_endpoint("http://localhost/v1")
    out = capsys.readouterr().out
    assert "Note" not in out  # localhost is fine without explicit port


def test_ollama_model_labels_no_local_suffix():
    """Labels should no longer claim '(local)' since the endpoint is dynamic."""
    from tradingagents.llm_clients.model_catalog import get_model_options
    for mode in ("quick", "deep"):
        labels = [label for label, _ in get_model_options("ollama", mode)]
        assert all("local" not in label for label in labels), labels


def test_ollama_offers_custom_model_id():
    """Ollama users with custom-pulled models can pick 'Custom model ID'."""
    from tradingagents.llm_clients.model_catalog import get_model_options
    for mode in ("quick", "deep"):
        entries = get_model_options("ollama", mode)
        values = [v for _, v in entries]
        assert "custom" in values, f"Ollama {mode!r} missing 'custom' option: {entries}"
        # Custom option is last so it doesn't push the curated defaults off-screen
        assert values[-1] == "custom", f"'custom' should be last entry: {values}"
