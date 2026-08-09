"""Tests for TRADINGAGENTS_* env-var overlay onto DEFAULT_CONFIG."""

from __future__ import annotations

import importlib

import pytest

import tradingagents.dataflows.config as config_module
import tradingagents.default_config as default_config_module


def _reload_with_env(monkeypatch, **overrides):
    """Set/clear env vars then reload default_config to re-evaluate DEFAULT_CONFIG."""
    for key in list(default_config_module._ENV_OVERRIDES):
        monkeypatch.delenv(key, raising=False)
    for key, val in overrides.items():
        monkeypatch.setenv(key, val)
    return importlib.reload(default_config_module)


def test_no_env_uses_built_in_defaults(monkeypatch):
    dc = _reload_with_env(monkeypatch)
    # Fork deliberately defaults to a local-first provider rather than
    # upstream's openai/gpt-5.x (see issue #16).
    assert dc.DEFAULT_CONFIG["llm_provider"] == "ollama"
    assert dc.DEFAULT_CONFIG["deep_think_llm"] == "ministral-3:8b"
    assert dc.DEFAULT_CONFIG["quick_think_llm"] == "ministral-3:3b"
    assert dc.DEFAULT_CONFIG["backend_url"] is None
    assert dc.DEFAULT_CONFIG["max_debate_rounds"] == 1
    assert dc.DEFAULT_CONFIG["checkpoint_enabled"] is False
    assert dc.DEFAULT_CONFIG["temperature"] is None
    assert dc.DEFAULT_CONFIG["llm_timeout"] == 120
    assert dc.DEFAULT_CONFIG["memory_mcp_timeout"] == 30.0


def test_string_overrides(monkeypatch):
    dc = _reload_with_env(
        monkeypatch,
        TRADINGAGENTS_LLM_PROVIDER="google",
        TRADINGAGENTS_DEEP_THINK_LLM="gemini-3-pro-preview",
        TRADINGAGENTS_QUICK_THINK_LLM="gemini-3-flash-preview",
        TRADINGAGENTS_LLM_BACKEND_URL="https://example.invalid/v1",
        TRADINGAGENTS_OUTPUT_LANGUAGE="Chinese",
    )
    assert dc.DEFAULT_CONFIG["llm_provider"] == "google"
    assert dc.DEFAULT_CONFIG["deep_think_llm"] == "gemini-3-pro-preview"
    assert dc.DEFAULT_CONFIG["quick_think_llm"] == "gemini-3-flash-preview"
    assert dc.DEFAULT_CONFIG["backend_url"] == "https://example.invalid/v1"
    assert dc.DEFAULT_CONFIG["output_language"] == "Chinese"


def test_int_coercion(monkeypatch):
    dc = _reload_with_env(
        monkeypatch,
        TRADINGAGENTS_MAX_DEBATE_ROUNDS="3",
        TRADINGAGENTS_MAX_RISK_ROUNDS="2",
    )
    assert dc.DEFAULT_CONFIG["max_debate_rounds"] == 3
    assert isinstance(dc.DEFAULT_CONFIG["max_debate_rounds"], int)
    assert dc.DEFAULT_CONFIG["max_risk_discuss_rounds"] == 2
    assert isinstance(dc.DEFAULT_CONFIG["max_risk_discuss_rounds"], int)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("true", True), ("True", True), ("1", True), ("yes", True), ("on", True),
        ("false", False), ("False", False), ("0", False), ("no", False), ("off", False),
    ],
)
def test_bool_coercion(monkeypatch, raw, expected):
    dc = _reload_with_env(monkeypatch, TRADINGAGENTS_CHECKPOINT_ENABLED=raw)
    assert dc.DEFAULT_CONFIG["checkpoint_enabled"] is expected


def test_empty_env_value_is_passthrough(monkeypatch):
    """Empty TRADINGAGENTS_* values must not clobber the built-in default."""
    dc = _reload_with_env(
        monkeypatch,
        TRADINGAGENTS_LLM_PROVIDER="",
        TRADINGAGENTS_MAX_DEBATE_ROUNDS="",
    )
    assert dc.DEFAULT_CONFIG["llm_provider"] == "ollama"
    assert dc.DEFAULT_CONFIG["max_debate_rounds"] == 1


def test_temperature_override_passthrough(monkeypatch):
    """TRADINGAGENTS_TEMPERATURE overrides the None default.

    The default value (None) gives ``_coerce`` no type to coerce to, so the
    raw string passes through unchanged here; ``TradingAgentsGraph.
    _get_provider_kwargs`` is responsible for the float() cast at the point
    of use (see tests/test_temperature_config.py's docstring / issue #16).
    """
    dc = _reload_with_env(monkeypatch, TRADINGAGENTS_TEMPERATURE="0.3")
    assert dc.DEFAULT_CONFIG["temperature"] == "0.3"


def test_temperature_empty_env_is_passthrough(monkeypatch):
    dc = _reload_with_env(monkeypatch, TRADINGAGENTS_TEMPERATURE="")
    assert dc.DEFAULT_CONFIG["temperature"] is None


def test_llm_timeout_override(monkeypatch):
    """TRADINGAGENTS_LLM_TIMEOUT overrides the 120s default (#108)."""
    dc = _reload_with_env(monkeypatch, TRADINGAGENTS_LLM_TIMEOUT="45")
    assert dc.DEFAULT_CONFIG["llm_timeout"] == 45
    assert isinstance(dc.DEFAULT_CONFIG["llm_timeout"], int)


def test_memory_mcp_timeout_override(monkeypatch):
    """TRADINGAGENTS_MEMORY_MCP_TIMEOUT overrides the 30s default (#108)."""
    dc = _reload_with_env(monkeypatch, TRADINGAGENTS_MEMORY_MCP_TIMEOUT="5")
    assert dc.DEFAULT_CONFIG["memory_mcp_timeout"] == 5.0
    assert isinstance(dc.DEFAULT_CONFIG["memory_mcp_timeout"], float)


def test_invalid_int_raises(monkeypatch):
    """Garbage int values should surface a ValueError at import, not silently misconfigure."""
    monkeypatch.setenv("TRADINGAGENTS_MAX_DEBATE_ROUNDS", "not-a-number")
    with pytest.raises(ValueError):
        importlib.reload(default_config_module)
    # Restore module state for subsequent tests in this process
    monkeypatch.delenv("TRADINGAGENTS_MAX_DEBATE_ROUNDS", raising=False)
    importlib.reload(default_config_module)


def test_unknown_env_var_is_ignored(monkeypatch):
    """Env vars outside _ENV_OVERRIDES must not bleed into DEFAULT_CONFIG."""
    dc = _reload_with_env(
        monkeypatch,
        TRADINGAGENTS_NONEXISTENT_KEY="oops",
    )
    assert "nonexistent_key" not in dc.DEFAULT_CONFIG


# --- Nested (dot-notation) overrides, e.g. "data_vendors.knowledge_base" (#107) ---
#
# `_ENV_OVERRIDES["TRADINGAGENTS_DATA_VENDORS_KNOWLEDGE_BASE"]` maps to the
# dotted key path "data_vendors.knowledge_base" rather than a flat config key.
# These tests exercise that branch of `_apply_env_overrides` end-to-end
# through `get_config()` (the read path every agent/tool actually uses), and
# unit-test `_get_nested`/`_set_nested` directly.


def _sync_get_config_to_reloaded_module(dc):
    """Force ``tradingagents.dataflows.config``'s global to re-derive from the
    just-reloaded ``default_config`` module, then return ``get_config()``.

    ``get_config()`` normally reads a process-global singleton
    (``config_module._config``) that is only (re)seeded from
    ``DEFAULT_CONFIG`` lazily on first use. Since ``_reload_with_env`` mutates
    ``DEFAULT_CONFIG`` in place via ``importlib.reload``, the singleton must be
    invalidated for the reload to actually surface through ``get_config()``,
    the same way the ``_isolate_config`` autouse fixture in conftest.py does
    for every other test.
    """
    config_module._config = None
    return config_module.get_config()


def test_knowledge_base_vendor_override_applies_via_get_config(monkeypatch):
    """TRADINGAGENTS_DATA_VENDORS_KNOWLEDGE_BASE reaches
    data_vendors["knowledge_base"] through get_config(), not just DEFAULT_CONFIG."""
    dc = _reload_with_env(
        monkeypatch, TRADINGAGENTS_DATA_VENDORS_KNOWLEDGE_BASE="faiss"
    )
    cfg = _sync_get_config_to_reloaded_module(dc)
    assert cfg["data_vendors"]["knowledge_base"] == "faiss"

    # Leave module state clean for subsequent tests/files (mirrors
    # test_invalid_int_raises's restore-after-mutation pattern above).
    _reload_with_env(monkeypatch)
    config_module._config = None


def test_knowledge_base_override_leaves_other_data_vendors_keys_unaffected(
    monkeypatch,
):
    """Setting only the knowledge_base override must not disturb sibling
    data_vendors keys (web_search, core_stock_apis, ...)."""
    dc = _reload_with_env(
        monkeypatch, TRADINGAGENTS_DATA_VENDORS_KNOWLEDGE_BASE="faiss"
    )
    cfg = _sync_get_config_to_reloaded_module(dc)
    assert cfg["data_vendors"]["web_search"] == "tavily"
    assert cfg["data_vendors"]["core_stock_apis"] == "yfinance"
    assert cfg["data_vendors"]["technical_indicators"] == "yfinance"
    assert cfg["data_vendors"]["fundamental_data"] == "yfinance"
    assert cfg["data_vendors"]["news_data"] == "yfinance"
    assert cfg["data_vendors"]["macro_data"] == "fred"
    assert cfg["data_vendors"]["prediction_markets"] == "polymarket"

    _reload_with_env(monkeypatch)
    config_module._config = None


def test_knowledge_base_vendor_default_preserved_when_env_unset(monkeypatch):
    """With no TRADINGAGENTS_DATA_VENDORS_KNOWLEDGE_BASE set, the built-in
    "bm25" default must survive the nested-override code path unchanged."""
    dc = _reload_with_env(monkeypatch)
    cfg = _sync_get_config_to_reloaded_module(dc)
    assert cfg["data_vendors"]["knowledge_base"] == "bm25"

    config_module._config = None


class TestGetSetNestedHelpers:
    """Direct unit coverage for _get_nested / _set_nested (#107 finding 1)."""

    def test_get_nested_returns_leaf_value(self):
        d = {"a": {"b": "c"}}
        assert default_config_module._get_nested(d, ["a", "b"]) == "c"

    def test_get_nested_missing_leaf_key_returns_none(self):
        d = {"a": {}}
        assert default_config_module._get_nested(d, ["a", "b"]) is None

    def test_get_nested_missing_intermediate_key_returns_none(self):
        d = {}
        assert default_config_module._get_nested(d, ["a", "b"]) is None

    def test_get_nested_non_dict_intermediate_returns_none(self):
        d = {"a": "not-a-dict"}
        assert default_config_module._get_nested(d, ["a", "b"]) is None

    def test_get_nested_single_key_top_level(self):
        d = {"a": "value"}
        assert default_config_module._get_nested(d, ["a"]) == "value"

    def test_set_nested_creates_intermediate_dicts(self):
        d = {}
        default_config_module._set_nested(d, ["a", "b"], "value")
        assert d == {"a": {"b": "value"}}

    def test_set_nested_overwrites_existing_leaf_value(self):
        d = {"a": {"b": "old"}}
        default_config_module._set_nested(d, ["a", "b"], "new")
        assert d["a"]["b"] == "new"

    def test_set_nested_preserves_sibling_keys(self):
        d = {"a": {"b": "old", "c": "keep"}}
        default_config_module._set_nested(d, ["a", "b"], "new")
        assert d["a"] == {"b": "new", "c": "keep"}

    def test_set_nested_single_key_top_level(self):
        d = {}
        default_config_module._set_nested(d, ["a"], "value")
        assert d == {"a": "value"}
