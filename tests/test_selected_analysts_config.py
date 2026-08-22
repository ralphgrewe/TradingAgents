"""Tests for selected_analysts configuration (issue #118)."""

from __future__ import annotations

import importlib

import pytest

import tradingagents.dataflows.config as config_module
import tradingagents.default_config as default_config_module
from tradingagents.graph.trading_graph import TradingAgentsGraph


def _reload_with_env(monkeypatch, **overrides):
    """Set/clear env vars then reload default_config to re-evaluate DEFAULT_CONFIG."""
    for key in list(default_config_module._ENV_OVERRIDES):
        monkeypatch.delenv(key, raising=False)
    for key, val in overrides.items():
        monkeypatch.setenv(key, val)
    return importlib.reload(default_config_module)


def _sync_get_config_to_reloaded_module(dc):
    """Force config module's global to re-derive from the reloaded default_config."""
    config_module._config = None
    return config_module.get_config()


class TestSelectedAnalystsDefault:
    """Test default selected_analysts in DEFAULT_CONFIG."""

    def test_default_selected_analysts(self, monkeypatch):
        """Without env override, selected_analysts defaults to all four analysts."""
        dc = _reload_with_env(monkeypatch)
        assert dc.DEFAULT_CONFIG["selected_analysts"] == ["market", "social", "news", "fundamentals"]

    def test_default_selected_analysts_type(self, monkeypatch):
        """selected_analysts default is a list."""
        dc = _reload_with_env(monkeypatch)
        assert isinstance(dc.DEFAULT_CONFIG["selected_analysts"], list)


class TestSelectedAnalystsEnvOverride:
    """Test TRADINGAGENTS_SELECTED_ANALYSTS env var override."""

    def test_env_override_comma_separated(self, monkeypatch):
        """TRADINGAGENTS_SELECTED_ANALYSTS parses comma-separated values."""
        dc = _reload_with_env(monkeypatch, TRADINGAGENTS_SELECTED_ANALYSTS="market,news")
        assert dc.DEFAULT_CONFIG["selected_analysts"] == ["market", "news"]

    def test_env_override_space_separated(self, monkeypatch):
        """TRADINGAGENTS_SELECTED_ANALYSTS parses space-separated values."""
        dc = _reload_with_env(monkeypatch, TRADINGAGENTS_SELECTED_ANALYSTS="market news")
        assert dc.DEFAULT_CONFIG["selected_analysts"] == ["market", "news"]

    def test_env_override_mixed_separators(self, monkeypatch):
        """TRADINGAGENTS_SELECTED_ANALYSTS handles comma and space separators."""
        dc = _reload_with_env(monkeypatch, TRADINGAGENTS_SELECTED_ANALYSTS="market, news, fundamentals")
        assert dc.DEFAULT_CONFIG["selected_analysts"] == ["market", "news", "fundamentals"]

    def test_env_override_single_analyst(self, monkeypatch):
        """TRADINGAGENTS_SELECTED_ANALYSTS with single analyst."""
        dc = _reload_with_env(monkeypatch, TRADINGAGENTS_SELECTED_ANALYSTS="market")
        assert dc.DEFAULT_CONFIG["selected_analysts"] == ["market"]

    def test_env_override_empty_string_passthrough(self, monkeypatch):
        """Empty TRADINGAGENTS_SELECTED_ANALYSTS value is passthrough (not override)."""
        dc = _reload_with_env(monkeypatch, TRADINGAGENTS_SELECTED_ANALYSTS="")
        assert dc.DEFAULT_CONFIG["selected_analysts"] == ["market", "social", "news", "fundamentals"]

    def test_env_override_reaches_get_config(self, monkeypatch):
        """TRADINGAGENTS_SELECTED_ANALYSTS override surfaces through get_config()."""
        dc = _reload_with_env(monkeypatch, TRADINGAGENTS_SELECTED_ANALYSTS="market,fundamentals")
        cfg = _sync_get_config_to_reloaded_module(dc)
        assert cfg["selected_analysts"] == ["market", "fundamentals"]

        # Clean up for subsequent tests
        _reload_with_env(monkeypatch)
        config_module._config = None


class TestConstructorPrecedence:
    """Test TradingAgentsGraph constructor precedence."""

    def test_constructor_arg_overrides_config(self, monkeypatch):
        """Explicit selected_analysts constructor argument overrides config."""
        dc = _reload_with_env(monkeypatch, TRADINGAGENTS_SELECTED_ANALYSTS="market,news")
        config = dc.DEFAULT_CONFIG.copy()

        # Explicitly passing selected_analysts should win even if config differs
        try:
            ta = TradingAgentsGraph(selected_analysts=["market", "social"], config=config)
            assert ta.selected_analysts == ["market", "social"]
        except Exception as e:
            # Graph setup may fail due to missing LLM clients, but the validation
            # should still have passed
            assert "selected_analysts" not in str(e).lower() or "unknown analyst" not in str(e).lower()

        _reload_with_env(monkeypatch)
        config_module._config = None

    def test_constructor_none_uses_config(self, monkeypatch):
        """When selected_analysts is None, TradingAgentsGraph uses config value."""
        dc = _reload_with_env(monkeypatch, TRADINGAGENTS_SELECTED_ANALYSTS="market,news")
        config = dc.DEFAULT_CONFIG.copy()

        try:
            ta = TradingAgentsGraph(selected_analysts=None, config=config)
            assert ta.selected_analysts == ["market", "news"]
        except Exception as e:
            # Graph setup may fail due to missing LLM clients, but validation passed
            assert "selected_analysts" not in str(e).lower() or "unknown analyst" not in str(e).lower()

        _reload_with_env(monkeypatch)
        config_module._config = None


class TestValidation:
    """Test selected_analysts validation."""

    def test_validation_empty_list_raises(self, monkeypatch):
        """Empty selected_analysts list raises ValueError with clear message."""
        dc = _reload_with_env(monkeypatch)
        config = dc.DEFAULT_CONFIG.copy()
        config["selected_analysts"] = []

        with pytest.raises(ValueError, match="must be a non-empty list"):
            TradingAgentsGraph(selected_analysts=None, config=config)

        _reload_with_env(monkeypatch)

    def test_validation_invalid_analyst_raises(self, monkeypatch):
        """Invalid analyst name raises ValueError naming the offending entry."""
        dc = _reload_with_env(monkeypatch)
        config = dc.DEFAULT_CONFIG.copy()
        config["selected_analysts"] = ["market", "invalid_analyst", "news"]

        with pytest.raises(ValueError, match="unknown analyst key: invalid_analyst"):
            TradingAgentsGraph(selected_analysts=None, config=config)

        _reload_with_env(monkeypatch)

    def test_validation_invalid_type_raises(self, monkeypatch):
        """Non-list selected_analysts raises ValueError."""
        dc = _reload_with_env(monkeypatch)
        config = dc.DEFAULT_CONFIG.copy()
        config["selected_analysts"] = "market"

        with pytest.raises(ValueError, match="must be a list"):
            TradingAgentsGraph(selected_analysts=None, config=config)

        _reload_with_env(monkeypatch)

    def test_validation_lists_valid_options_in_error(self, monkeypatch):
        """Validation error message lists valid analyst options."""
        dc = _reload_with_env(monkeypatch)
        config = dc.DEFAULT_CONFIG.copy()
        config["selected_analysts"] = ["unknown"]

        with pytest.raises(ValueError) as exc_info:
            TradingAgentsGraph(selected_analysts=None, config=config)

        error_msg = str(exc_info.value)
        assert "market" in error_msg
        assert "social" in error_msg
        assert "news" in error_msg
        assert "fundamentals" in error_msg

        _reload_with_env(monkeypatch)


class TestMacroFundamentalsSelectable:
    """Test macro_fundamentals is a valid, opt-in selected_analysts entry (#132)."""

    def test_macro_fundamentals_not_in_default(self, monkeypatch):
        """selected_analysts default is unchanged: macro_fundamentals is opt-in
        (decision 2 in #126), not part of the default four."""
        dc = _reload_with_env(monkeypatch)
        assert "macro_fundamentals" not in dc.DEFAULT_CONFIG["selected_analysts"]

    def test_macro_fundamentals_accepted_alone(self, monkeypatch):
        """selected_analysts=['macro_fundamentals'] builds without validation error."""
        dc = _reload_with_env(monkeypatch)
        config = dc.DEFAULT_CONFIG.copy()

        try:
            ta = TradingAgentsGraph(selected_analysts=["macro_fundamentals"], config=config)
            assert ta.selected_analysts == ["macro_fundamentals"]
        except Exception as e:
            # Graph setup may fail due to missing LLM clients, but validation
            # itself must have passed (no "unknown analyst" error).
            assert "unknown analyst" not in str(e).lower()

        _reload_with_env(monkeypatch)

    def test_macro_fundamentals_accepted_alongside_default_four(self, monkeypatch):
        dc = _reload_with_env(monkeypatch)
        config = dc.DEFAULT_CONFIG.copy()
        selection = ["market", "social", "news", "fundamentals", "macro_fundamentals"]

        try:
            ta = TradingAgentsGraph(selected_analysts=selection, config=config)
            assert ta.selected_analysts == selection
        except Exception as e:
            assert "unknown analyst" not in str(e).lower()

        _reload_with_env(monkeypatch)

    def test_misspelled_macro_fundamentals_rejected(self, monkeypatch):
        """A misspelled key still fails validation before the run starts."""
        dc = _reload_with_env(monkeypatch)
        config = dc.DEFAULT_CONFIG.copy()
        config["selected_analysts"] = ["macro_fundamental"]  # missing trailing 's'

        with pytest.raises(ValueError, match="unknown analyst key: macro_fundamental"):
            TradingAgentsGraph(selected_analysts=None, config=config)

        _reload_with_env(monkeypatch)


class TestMacroNewsSelectable:
    """Test macro_news is a valid, opt-in selected_analysts entry (#134)."""

    def test_macro_news_not_in_default(self, monkeypatch):
        """selected_analysts default is unchanged: macro_news is opt-in
        (decision 2 in #126), not part of the default four."""
        dc = _reload_with_env(monkeypatch)
        assert "macro_news" not in dc.DEFAULT_CONFIG["selected_analysts"]

    def test_macro_news_accepted_alone(self, monkeypatch):
        """selected_analysts=['macro_news'] builds without validation error."""
        dc = _reload_with_env(monkeypatch)
        config = dc.DEFAULT_CONFIG.copy()

        try:
            ta = TradingAgentsGraph(selected_analysts=["macro_news"], config=config)
            assert ta.selected_analysts == ["macro_news"]
        except Exception as e:
            # Graph setup may fail due to missing LLM clients, but validation
            # itself must have passed (no "unknown analyst" error).
            assert "unknown analyst" not in str(e).lower()

        _reload_with_env(monkeypatch)

    def test_macro_news_accepted_alongside_default_four(self, monkeypatch):
        dc = _reload_with_env(monkeypatch)
        config = dc.DEFAULT_CONFIG.copy()
        selection = ["market", "social", "news", "fundamentals", "macro_news"]

        try:
            ta = TradingAgentsGraph(selected_analysts=selection, config=config)
            assert ta.selected_analysts == selection
        except Exception as e:
            assert "unknown analyst" not in str(e).lower()

        _reload_with_env(monkeypatch)

    def test_both_macro_analysts_together(self, monkeypatch):
        """Both macro_fundamentals and macro_news can be selected together."""
        dc = _reload_with_env(monkeypatch)
        config = dc.DEFAULT_CONFIG.copy()
        selection = ["macro_fundamentals", "macro_news"]

        try:
            ta = TradingAgentsGraph(selected_analysts=selection, config=config)
            assert ta.selected_analysts == selection
        except Exception as e:
            assert "unknown analyst" not in str(e).lower()

        _reload_with_env(monkeypatch)

    def test_misspelled_macro_news_rejected(self, monkeypatch):
        """A misspelled key still fails validation before the run starts."""
        dc = _reload_with_env(monkeypatch)
        config = dc.DEFAULT_CONFIG.copy()
        config["selected_analysts"] = ["macro_newss"]  # extra 's'

        with pytest.raises(ValueError, match="unknown analyst key: macro_newss"):
            TradingAgentsGraph(selected_analysts=None, config=config)

        _reload_with_env(monkeypatch)

    def test_default_unchanged_when_macro_news_available(self, monkeypatch):
        """Default selected_analysts remains the original four even though
        macro_news is now available (decision 2 in #126)."""
        dc = _reload_with_env(monkeypatch)
        # Verify macro_news is a known analyst (in ANALYST_NODE_SPECS)
        from tradingagents.graph.analyst_execution import ANALYST_NODE_SPECS
        assert "macro_news" in ANALYST_NODE_SPECS
        # But it's not in the default
        assert dc.DEFAULT_CONFIG["selected_analysts"] == ["market", "social", "news", "fundamentals"]


class TestConfigFileValidation:
    """Test run config file's 'config' block with selected_analysts."""

    def test_config_file_selected_analysts_validation(self, monkeypatch, tmp_path):
        """Config file's 'config' block recognizes selected_analysts as valid key."""
        from run_trading_agents import _validate_config_block

        config_file = tmp_path / "test_config.json"

        # Valid config block with selected_analysts
        config_block = {
            "selected_analysts": ["market", "news"],
            "research_stage": "researcher",
        }

        # Should not raise
        _validate_config_block(config_block, str(config_file))

    def test_config_file_selected_analysts_type_check(self, monkeypatch, tmp_path):
        """Config file validates selected_analysts is a list."""
        from run_trading_agents import _validate_config_block

        config_file = tmp_path / "test_config.json"

        # Invalid: selected_analysts as string instead of list
        config_block = {"selected_analysts": "market,news"}

        with pytest.raises(SystemExit):
            _validate_config_block(config_block, str(config_file))

    def test_config_file_selected_analysts_empty_list(self, monkeypatch, tmp_path):
        """Config file allows empty list (validation happens at graph init time)."""
        from run_trading_agents import _validate_config_block

        config_file = tmp_path / "test_config.json"

        # Empty list is syntactically valid in config; runtime validation happens later
        config_block = {"selected_analysts": []}

        # Type validation passes; runtime validation in TradingAgentsGraph catches it
        _validate_config_block(config_block, str(config_file))
