"""Tests for ablation stability-block run-config files (issue #120).

Verification of the prior implementation (commit 579cb37) found that while
scripts/run_ablation.py already supported a --stability flag and looked for
*.stability.json files, the actual stability-block deliverables (a
stability-stocks.json ticker list and one *.stability.json config per arm)
were never created — the acceptance criterion "Stability-block configs run
without portfolio mode and with isolated memory/report paths" was unmet.

This test module validates the real deliverable files (mirroring the
tests/test_ablation_arms_config.py pattern for the main arms):
1. All six stability configs load and parse cleanly
2. Each carries the same ablated setting as its corresponding main arm
3. Portfolio mode is off (no depot_id/style) on every stability config
4. memory_id/report_dir are isolated from both the main arms and each other
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from run_trading_agents import _validate_config_block, _validate_config_file
from tradingagents.memory.store import resolve_memory_id_to_db_path

pytestmark = pytest.mark.unit


# Stability arm definitions: (arm_name, ablated_key, ablated_value_or_none)
# Mirrors ARM_DEFINITIONS in test_ablation_arms_config.py — each stability
# config exercises the same ablation as its main-arm counterpart.
STABILITY_ARM_DEFINITIONS = [
    ("control", None, None),
    ("no-market", "selected_analysts", ["social", "news", "fundamentals"]),
    ("no-social", "selected_analysts", ["market", "news", "fundamentals"]),
    ("no-news", "selected_analysts", ["market", "social", "fundamentals"]),
    ("no-fundamentals", "selected_analysts", ["market", "social", "news"]),
    ("no-risk", "risk_stage", "none"),
]

ARMS_DIR = Path(__file__).parent.parent / "experiments" / "ablation-analysts"


def _stability_config_file(arm_name: str) -> Path:
    return ARMS_DIR / f"{arm_name}.stability.json"


class TestStabilityConfigsLoadAndValidate:
    """Test that all stability configs load cleanly without syntax/validation errors."""

    def test_stability_stocks_file_exists(self):
        """The small stability-block ticker universe should exist."""
        stocks_file = ARMS_DIR / "stability-stocks.json"
        assert stocks_file.exists(), "stability-stocks.json does not exist"

        with open(stocks_file) as f:
            stocks = json.load(f)
        assert isinstance(stocks, list), "stability-stocks.json should contain an array"
        assert len(stocks) == 3, (
            f"stability-stocks.json should have exactly 3 tickers per #81's cadence, "
            f"got {len(stocks)}"
        )
        for entry in stocks:
            assert "ticker" in entry, f"stability-stocks.json entry missing 'ticker': {entry}"

    def test_all_stability_configs_exist(self):
        """All six stability-block config files should exist."""
        for arm_name, _, _ in STABILITY_ARM_DEFINITIONS:
            config_file = _stability_config_file(arm_name)
            assert config_file.exists(), f"Config file {config_file} does not exist"

    def test_all_stability_configs_parse_as_valid_json(self):
        """All stability config files should parse as valid JSON objects."""
        for arm_name, _, _ in STABILITY_ARM_DEFINITIONS:
            config_file = _stability_config_file(arm_name)
            with open(config_file) as f:
                data = json.load(f)
            assert isinstance(data, dict), f"{config_file} is not a JSON object"

    def test_all_stability_configs_validate_without_errors(self):
        """All stability configs should validate through run_trading_agents's validators."""
        for arm_name, _, _ in STABILITY_ARM_DEFINITIONS:
            config_file = _stability_config_file(arm_name)
            with open(config_file) as f:
                config = json.load(f)

            # This should not raise
            _validate_config_file(config, str(config_file))
            _validate_config_block(config.get("config"), str(config_file))

    def test_all_stability_memory_ids_are_valid_format(self):
        """All stability memory_id values should be valid (no slashes, backslashes, .., etc.)."""
        for arm_name, _, _ in STABILITY_ARM_DEFINITIONS:
            config_file = _stability_config_file(arm_name)
            with open(config_file) as f:
                config = json.load(f)

            memory_id = config.get("memory_id")
            # This should not raise ValueError
            resolve_memory_id_to_db_path(memory_id)

    def test_stability_stocks_file_reference_is_relative(self):
        """All stability configs should reference stability-stocks.json (relative path)."""
        for arm_name, _, _ in STABILITY_ARM_DEFINITIONS:
            config_file = _stability_config_file(arm_name)
            with open(config_file) as f:
                config = json.load(f)

            assert config.get("stocks_file") == "stability-stocks.json", (
                f"{arm_name}.stability: stocks_file should be 'stability-stocks.json', "
                f"got '{config.get('stocks_file')}'"
            )


class TestStabilityConfigsPortfolioOff:
    """Acceptance criterion: stability-block configs run without portfolio mode."""

    def test_portfolio_mode_is_off_on_all_stability_configs(self):
        """No stability config should enable portfolio mode."""
        for arm_name, _, _ in STABILITY_ARM_DEFINITIONS:
            config_file = _stability_config_file(arm_name)
            with open(config_file) as f:
                config = json.load(f)

            assert config.get("portfolio") is False, (
                f"{arm_name}.stability: portfolio mode should be off (False), "
                f"got {config.get('portfolio')!r}"
            )

    def test_no_depot_id_or_style_on_stability_configs(self):
        """Stability configs must not set depot_id/style — no portfolio trades per #81."""
        for arm_name, _, _ in STABILITY_ARM_DEFINITIONS:
            config_file = _stability_config_file(arm_name)
            with open(config_file) as f:
                config = json.load(f)

            assert "depot_id" not in config, (
                f"{arm_name}.stability: should not set depot_id (no portfolio trades)"
            )
            assert "style" not in config, (
                f"{arm_name}.stability: should not set style (no portfolio trades)"
            )


class TestStabilityConfigsAblationSettings:
    """Test that each stability config carries the same ablated setting as its main arm."""

    def _get_config(self, arm_name: str) -> dict:
        config_file = _stability_config_file(arm_name)
        with open(config_file) as f:
            return json.load(f)

    def test_control_stability_has_no_ablation_config(self):
        """Stability control should have no selected_analysts or risk_stage override."""
        control = self._get_config("control")
        config_block = control.get("config", {})

        assert "selected_analysts" not in config_block, (
            "control.stability: should not override selected_analysts"
        )
        assert "risk_stage" not in config_block, (
            "control.stability: should not override risk_stage"
        )

    def test_analyst_stability_arms_match_main_arm_ablation(self):
        """Analyst-ablation stability configs should carry the same selected_analysts as the main arm."""
        for arm_name, _ablated_key, expected_value in STABILITY_ARM_DEFINITIONS[1:5]:
            stability_config = self._get_config(arm_name).get("config", {})
            main_config_file = ARMS_DIR / f"{arm_name}.json"
            with open(main_config_file) as f:
                main_config = json.load(f).get("config", {})

            assert "selected_analysts" in stability_config, (
                f"{arm_name}.stability: should override selected_analysts"
            )
            assert stability_config["selected_analysts"] == expected_value, (
                f"{arm_name}.stability: selected_analysts should be {expected_value}, "
                f"got {stability_config['selected_analysts']}"
            )
            assert stability_config["selected_analysts"] == main_config["selected_analysts"], (
                f"{arm_name}.stability: selected_analysts should match the main arm's "
                f"ablated setting so the repeat tests the same thing"
            )
            assert "risk_stage" not in stability_config, (
                f"{arm_name}.stability: should not override risk_stage"
            )

    def test_no_risk_stability_matches_main_arm_ablation(self):
        """The no-risk stability config should carry risk_stage=none, matching the main arm."""
        stability_config = self._get_config("no-risk").get("config", {})
        main_config_file = ARMS_DIR / "no-risk.json"
        with open(main_config_file) as f:
            main_config = json.load(f).get("config", {})

        assert stability_config.get("risk_stage") == "none", (
            "no-risk.stability: risk_stage should be 'none'"
        )
        assert stability_config["risk_stage"] == main_config["risk_stage"], (
            "no-risk.stability: risk_stage should match the main no-risk arm"
        )
        assert "selected_analysts" not in stability_config, (
            "no-risk.stability: should not override selected_analysts"
        )


class TestStabilityConfigsIsolation:
    """Test that stability configs are isolated from the main arms and from each other."""

    def _get_config(self, arm_name: str) -> dict:
        config_file = _stability_config_file(arm_name)
        with open(config_file) as f:
            return json.load(f)

    def test_stability_memory_ids_differ_from_main_arms(self):
        """Every stability memory_id must differ from every main-arm memory_id."""
        main_memory_ids = set()
        for arm_name, _, _ in STABILITY_ARM_DEFINITIONS:
            main_config_file = ARMS_DIR / f"{arm_name}.json"
            with open(main_config_file) as f:
                main_memory_ids.add(json.load(f).get("memory_id"))

        for arm_name, _, _ in STABILITY_ARM_DEFINITIONS:
            stability_memory_id = self._get_config(arm_name).get("memory_id")
            assert stability_memory_id not in main_memory_ids, (
                f"{arm_name}.stability: memory_id '{stability_memory_id}' must not collide "
                f"with a main-arm memory_id"
            )

    def test_stability_report_dirs_differ_from_main_arms(self):
        """Every stability report_dir must differ from every main-arm report_dir."""
        main_report_dirs = set()
        for arm_name, _, _ in STABILITY_ARM_DEFINITIONS:
            main_config_file = ARMS_DIR / f"{arm_name}.json"
            with open(main_config_file) as f:
                main_report_dirs.add(json.load(f).get("report_dir"))

        for arm_name, _, _ in STABILITY_ARM_DEFINITIONS:
            stability_report_dir = self._get_config(arm_name).get("report_dir")
            assert stability_report_dir not in main_report_dirs, (
                f"{arm_name}.stability: report_dir '{stability_report_dir}' must not collide "
                f"with a main-arm report_dir"
            )

    def test_stability_memory_ids_are_unique_across_stability_arms(self):
        """All six stability memory_id values should be unique."""
        memory_ids = {
            self._get_config(arm_name).get("memory_id")
            for arm_name, _, _ in STABILITY_ARM_DEFINITIONS
        }
        assert len(memory_ids) == len(STABILITY_ARM_DEFINITIONS), (
            f"Expected {len(STABILITY_ARM_DEFINITIONS)} unique stability memory_ids, "
            f"got {len(memory_ids)}"
        )

    def test_stability_report_dirs_are_unique_across_stability_arms(self):
        """All six stability report_dir values should be unique."""
        report_dirs = {
            self._get_config(arm_name).get("report_dir")
            for arm_name, _, _ in STABILITY_ARM_DEFINITIONS
        }
        assert len(report_dirs) == len(STABILITY_ARM_DEFINITIONS), (
            f"Expected {len(STABILITY_ARM_DEFINITIONS)} unique stability report_dirs, "
            f"got {len(report_dirs)}"
        )

    def test_stability_memory_log_paths_are_unique_and_isolated(self):
        """memory_log_path should be isolated per stability arm and from main arms."""
        main_memory_log_paths = set()
        for arm_name, _, _ in STABILITY_ARM_DEFINITIONS:
            main_config_file = ARMS_DIR / f"{arm_name}.json"
            with open(main_config_file) as f:
                main_config = json.load(f)
                main_memory_log_paths.add(main_config.get("config", {}).get("memory_log_path"))

        stability_memory_log_paths = set()
        for arm_name, _, _ in STABILITY_ARM_DEFINITIONS:
            path = self._get_config(arm_name).get("config", {}).get("memory_log_path")
            assert path is not None, f"{arm_name}.stability: missing config.memory_log_path"
            assert path not in main_memory_log_paths, (
                f"{arm_name}.stability: memory_log_path '{path}' must not collide with a "
                f"main-arm memory_log_path"
            )
            stability_memory_log_paths.add(path)

        assert len(stability_memory_log_paths) == len(STABILITY_ARM_DEFINITIONS), (
            "Stability memory_log_path values must be unique across all six stability arms"
        )

    def test_stability_arm_isolation_values_are_consistent(self):
        """Each stability arm's isolation values should follow the <arm>-stability convention."""
        for arm_name, _, _ in STABILITY_ARM_DEFINITIONS:
            config = self._get_config(arm_name)

            expected_memory_id = f"{arm_name}-stability"
            assert config.get("memory_id") == expected_memory_id, (
                f"{arm_name}.stability: memory_id should be '{expected_memory_id}', "
                f"got '{config.get('memory_id')}'"
            )

            expected_report_dir = f"reports/stability/{arm_name}"
            assert config.get("report_dir") == expected_report_dir, (
                f"{arm_name}.stability: report_dir should be '{expected_report_dir}', "
                f"got '{config.get('report_dir')}'"
            )

            memory_log_path = config.get("config", {}).get("memory_log_path")
            assert expected_memory_id in memory_log_path, (
                f"{arm_name}.stability: memory_log_path should include memory_id "
                f"'{expected_memory_id}'"
            )

            # results_dir isolation (issue #121) — same rationale as the main arms'
            # test_arm_isolation_values_are_consistent in test_ablation_arms_config.py.
            expected_results_dir = f"runs/results/{expected_memory_id}"
            assert config.get("config", {}).get("results_dir") == expected_results_dir, (
                f"{arm_name}.stability: results_dir should be '{expected_results_dir}', "
                f"got '{config.get('config', {}).get('results_dir')}'"
            )

    def test_stability_results_dirs_differ_from_main_arms_and_each_other(self):
        """Every stability results_dir must be unique and distinct from every main-arm results_dir."""
        main_results_dirs = set()
        for arm_name, _, _ in STABILITY_ARM_DEFINITIONS:
            main_config_file = ARMS_DIR / f"{arm_name}.json"
            with open(main_config_file) as f:
                main_config = json.load(f)
                main_results_dirs.add(main_config.get("config", {}).get("results_dir"))

        stability_results_dirs = set()
        for arm_name, _, _ in STABILITY_ARM_DEFINITIONS:
            path = self._get_config(arm_name).get("config", {}).get("results_dir")
            assert path is not None, f"{arm_name}.stability: missing config.results_dir"
            assert path not in main_results_dirs, (
                f"{arm_name}.stability: results_dir '{path}' must not collide with a "
                f"main-arm results_dir"
            )
            stability_results_dirs.add(path)

        assert len(stability_results_dirs) == len(STABILITY_ARM_DEFINITIONS), (
            "Stability results_dir values must be unique across all six stability arms"
        )


class TestStabilityConfigsDiscoveredByDriver:
    """Test that scripts/run_ablation.py's discovery logic picks up the real deliverables."""

    def test_get_arm_configs_discovers_all_six_stability_configs(self):
        """get_arm_configs(stability=True) should discover all six real stability configs."""
        from scripts.run_ablation import get_arm_configs

        configs = get_arm_configs(ARMS_DIR, stability=True)
        config_names = {f.stem for f in configs}

        expected_names = {f"{arm_name}.stability" for arm_name, _, _ in STABILITY_ARM_DEFINITIONS}
        assert expected_names.issubset(config_names), (
            f"Expected stability configs {expected_names} to be discovered, "
            f"got {config_names}"
        )

    def test_get_arm_configs_main_arms_excludes_stability_files(self):
        """get_arm_configs(stability=False) should not pick up *.stability.json or stability-stocks.json."""
        from scripts.run_ablation import get_arm_configs

        configs = get_arm_configs(ARMS_DIR, stability=False)
        config_names = {f.name for f in configs}

        for arm_name, _, _ in STABILITY_ARM_DEFINITIONS:
            assert f"{arm_name}.stability.json" not in config_names
        assert "stability-stocks.json" not in config_names
        assert "stocks.json" not in config_names
