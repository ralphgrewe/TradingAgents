"""Tests for ablation arm run-config files (issue #120).

Validates that:
1. All arm configs load and parse cleanly
2. Each arm differs from control in exactly the ablated setting + 4 isolation values
3. Arm-naming and isolation ID formats are valid
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from run_trading_agents import _validate_config_block, _validate_config_file
from tradingagents.memory.store import resolve_memory_id_to_db_path

pytestmark = pytest.mark.unit


# Arm definitions: (arm_name, ablated_key, ablated_value_or_none)
ARM_DEFINITIONS = [
    ("control", None, None),
    ("no-market", "selected_analysts", ["social", "news", "fundamentals"]),
    ("no-social", "selected_analysts", ["market", "news", "fundamentals"]),
    ("no-news", "selected_analysts", ["market", "social", "fundamentals"]),
    ("no-fundamentals", "selected_analysts", ["market", "social", "news"]),
    ("no-risk", "risk_stage", "none"),
]

ARMS_DIR = Path(__file__).parent.parent / "experiments" / "ablation-analysts"


class TestArmConfigsLoadAndValidate:
    """Test that all arm configs load cleanly without syntax/validation errors."""

    def test_all_arm_configs_exist(self):
        """All expected arm config files should exist."""
        for arm_name, _, _ in ARM_DEFINITIONS:
            config_file = ARMS_DIR / f"{arm_name}.json"
            assert config_file.exists(), f"Config file {config_file} does not exist"

    def test_all_arm_configs_parse_as_valid_json(self):
        """All arm config files should parse as valid JSON."""
        for arm_name, _, _ in ARM_DEFINITIONS:
            config_file = ARMS_DIR / f"{arm_name}.json"
            with open(config_file) as f:
                data = json.load(f)
            assert isinstance(data, dict), f"{config_file} is not a JSON object"

    def test_all_arm_configs_have_required_fields(self):
        """All arm configs should have stocks_file and core isolation fields."""
        for arm_name, _, _ in ARM_DEFINITIONS:
            config_file = ARMS_DIR / f"{arm_name}.json"
            with open(config_file) as f:
                config = json.load(f)

            # Required top-level keys
            assert "stocks_file" in config, f"{arm_name}: missing 'stocks_file'"
            assert "report_dir" in config, f"{arm_name}: missing 'report_dir'"
            assert "depot_id" in config, f"{arm_name}: missing 'depot_id'"
            assert "memory_id" in config, f"{arm_name}: missing 'memory_id'"
            assert "portfolio" in config, f"{arm_name}: missing 'portfolio'"
            assert "style" in config, f"{arm_name}: missing 'style'"

            # Required config block key
            assert "config" in config, f"{arm_name}: missing 'config' block"
            assert "memory_log_path" in config["config"], \
                f"{arm_name}: config block missing 'memory_log_path'"

    def test_all_arm_configs_validate_without_errors(self):
        """All arm configs should validate through run_trading_agents's validators."""
        for arm_name, _, _ in ARM_DEFINITIONS:
            config_file = ARMS_DIR / f"{arm_name}.json"
            with open(config_file) as f:
                config = json.load(f)

            # This should not raise
            _validate_config_file(config, str(config_file))
            _validate_config_block(config.get("config"), str(config_file))

    def test_all_memory_ids_are_valid_format(self):
        """All memory_id values should be valid (no slashes, backslashes, .., etc.)."""
        for arm_name, _, _ in ARM_DEFINITIONS:
            config_file = ARMS_DIR / f"{arm_name}.json"
            with open(config_file) as f:
                config = json.load(f)

            memory_id = config.get("memory_id")
            # This should not raise ValueError
            resolve_memory_id_to_db_path(memory_id)

    def test_portfolio_mode_is_enabled_on_all_arms(self):
        """All main arms should have portfolio mode enabled."""
        for arm_name, _, _ in ARM_DEFINITIONS:
            config_file = ARMS_DIR / f"{arm_name}.json"
            with open(config_file) as f:
                config = json.load(f)

            assert config.get("portfolio") is True, \
                f"{arm_name}: portfolio mode should be enabled"
            assert config.get("style") == "aggressive", \
                f"{arm_name}: style should be 'aggressive'"

    def test_stocks_file_is_relative_to_config(self):
        """All arm configs should have stocks_file as a relative path."""
        for arm_name, _, _ in ARM_DEFINITIONS:
            config_file = ARMS_DIR / f"{arm_name}.json"
            with open(config_file) as f:
                config = json.load(f)

            stocks_file = config.get("stocks_file")
            assert stocks_file == "stocks.json", \
                f"{arm_name}: stocks_file should be 'stocks.json' (relative path)"


class TestArmIsolationValues:
    """Test that each arm's isolation values (memory_id, depot_id, report_dir, memory_log_path) match."""

    def test_arm_isolation_values_are_consistent(self):
        """Each arm's isolation values (memory_id, depot_id, report_dir, memory_log_path) should be consistent."""
        for arm_name, _, _ in ARM_DEFINITIONS:
            config_file = ARMS_DIR / f"{arm_name}.json"
            with open(config_file) as f:
                config = json.load(f)

            memory_id = config.get("memory_id")
            depot_id = config.get("depot_id")
            report_dir = config.get("report_dir")
            memory_log_path = config.get("config", {}).get("memory_log_path")

            # Memory ID should match the arm name
            assert memory_id == arm_name, \
                f"{arm_name}: memory_id should be '{arm_name}', got '{memory_id}'"

            # Depot ID should be arm-name-depot
            expected_depot_id = f"{arm_name}-depot"
            assert depot_id == expected_depot_id, \
                f"{arm_name}: depot_id should be '{expected_depot_id}', got '{depot_id}'"

            # Report dir should be reports/arm-name
            expected_report_dir = f"reports/{arm_name}"
            assert report_dir == expected_report_dir, \
                f"{arm_name}: report_dir should be '{expected_report_dir}', got '{report_dir}'"

            # Memory log path should include the memory_id
            assert memory_id in memory_log_path, \
                f"{arm_name}: memory_log_path should include memory_id '{memory_id}'"


class TestArmAblationSettings:
    """Test that each arm differs from control in exactly the expected ablated setting."""

    def _get_arm_config(self, arm_name: str) -> dict:
        """Load an arm config by name."""
        config_file = ARMS_DIR / f"{arm_name}.json"
        with open(config_file) as f:
            return json.load(f)

    def test_control_has_no_ablation_config(self):
        """Control arm should have no selected_analysts or risk_stage override in config block."""
        control = self._get_arm_config("control")
        config_block = control.get("config", {})

        # Control should not override selected_analysts or risk_stage
        assert "selected_analysts" not in config_block, \
            "Control should not override selected_analysts"
        assert "risk_stage" not in config_block, \
            "Control should not override risk_stage"

    def test_no_analyst_arms_differ_only_in_selected_analysts(self):
        """Arms that ablate an analyst should differ from control only in selected_analysts."""
        for arm_name, _ablated_key, expected_value in ARM_DEFINITIONS[1:5]:
            arm = self._get_arm_config(arm_name)
            arm_config = arm.get("config", {})

            # Should have selected_analysts override
            assert "selected_analysts" in arm_config, \
                f"{arm_name}: should override selected_analysts"

            # Should not have risk_stage override
            assert "risk_stage" not in arm_config, \
                f"{arm_name}: should not override risk_stage"

            # Value should match expected
            assert arm_config["selected_analysts"] == expected_value, \
                f"{arm_name}: selected_analysts should be {expected_value}, got {arm_config['selected_analysts']}"

    def test_no_risk_arm_differs_only_in_risk_stage(self):
        """The no-risk arm should differ from control only in risk_stage."""
        no_risk = self._get_arm_config("no-risk")
        no_risk_config = no_risk.get("config", {})

        # Should have risk_stage override
        assert "risk_stage" in no_risk_config, \
            "no-risk: should override risk_stage"
        assert no_risk_config["risk_stage"] == "none", \
            f"no-risk: risk_stage should be 'none', got {no_risk_config['risk_stage']}"

        # Should not have selected_analysts override
        assert "selected_analysts" not in no_risk_config, \
            "no-risk: should not override selected_analysts"

    def test_isolation_values_differ_across_all_arms(self):
        """All arms should have different memory_id, depot_id, report_dir values."""
        memory_ids = set()
        depot_ids = set()
        report_dirs = set()

        for arm_name, _, _ in ARM_DEFINITIONS:
            config = self._get_arm_config(arm_name)
            memory_ids.add(config.get("memory_id"))
            depot_ids.add(config.get("depot_id"))
            report_dirs.add(config.get("report_dir"))

        # All should be unique
        assert len(memory_ids) == len(ARM_DEFINITIONS), \
            f"Expected {len(ARM_DEFINITIONS)} unique memory_ids, got {len(memory_ids)}"
        assert len(depot_ids) == len(ARM_DEFINITIONS), \
            f"Expected {len(ARM_DEFINITIONS)} unique depot_ids, got {len(depot_ids)}"
        assert len(report_dirs) == len(ARM_DEFINITIONS), \
            f"Expected {len(ARM_DEFINITIONS)} unique report_dirs, got {len(report_dirs)}"


class TestArmConfigCompleteness:
    """Test that all expected arms are defined and properly configured."""

    def test_all_expected_arms_are_defined(self):
        """All six expected arm names should map to existing configs."""
        expected_arms = [arm_name for arm_name, _, _ in ARM_DEFINITIONS]
        for expected_arm in expected_arms:
            config_file = ARMS_DIR / f"{expected_arm}.json"
            assert config_file.exists(), \
                f"Expected arm config file for '{expected_arm}' does not exist"

    def test_stocks_file_exists(self):
        """The shared stocks.json file should exist."""
        stocks_file = ARMS_DIR / "stocks.json"
        assert stocks_file.exists(), \
            "Shared stocks.json file does not exist"

        # Should parse as valid JSON array
        with open(stocks_file) as f:
            stocks = json.load(f)
        assert isinstance(stocks, list), \
            "stocks.json should contain an array"
