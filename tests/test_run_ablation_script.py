"""Tests for scripts/run_ablation.py driver script (issue #120).

Tests the subprocess-wrapper logic: sequential execution, failure handling,
summary reporting, and exit codes. Uses stubbed subprocess.run() to avoid
actually running run_trading_agents.py.
"""

from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit


@pytest.fixture
def temp_arms_dir(tmp_path):
    """Create a temporary arms directory with minimal test configs."""
    arms_dir = tmp_path / "ablation-analysts"
    arms_dir.mkdir()

    # Create test stocks.json
    stocks_file = arms_dir / "stocks.json"
    stocks_file.write_text(json.dumps([{"ticker": "AAPL"}]))

    # Create minimal arm configs
    for arm_name in ["control", "no-market"]:
        config = {
            "stocks_file": "stocks.json",
            "report_dir": f"reports/{arm_name}",
            "portfolio": True,
            "style": "aggressive",
            "depot_id": f"{arm_name}-depot",
            "memory_id": arm_name,
            "config": {"memory_log_path": f"~/.tradingagents/memory/{arm_name}/trading_memory.md"},
        }
        config_file = arms_dir / f"{arm_name}.json"
        config_file.write_text(json.dumps(config))

    return arms_dir


class TestGetArmConfigs:
    """Tests for discovering arm config files."""

    def test_discover_main_arm_configs(self, temp_arms_dir):
        """Should discover main arm configs (*.json, excluding stocks.json and stability)."""
        from scripts.run_ablation import get_arm_configs

        configs = get_arm_configs(temp_arms_dir, stability=False)
        config_names = [f.stem for f in configs]

        assert "control" in config_names
        assert "no-market" in config_names
        assert "stocks" not in config_names

    def test_discover_stability_configs(self, temp_arms_dir):
        """Should discover stability-block configs (*.stability.json)."""
        from scripts.run_ablation import get_arm_configs

        # Create a stability config
        config = {
            "stocks_file": "stability-stocks.json",
            "report_dir": "reports/stability/control",
            "portfolio": False,
            "memory_id": "control-stability",
            "config": {"memory_log_path": "~/.tradingagents/memory/control-stability/trading_memory.md"},
        }
        stability_file = temp_arms_dir / "control.stability.json"
        stability_file.write_text(json.dumps(config))

        configs = get_arm_configs(temp_arms_dir, stability=True)
        config_names = [f.stem for f in configs]

        assert "control.stability" in config_names
        assert "control" not in config_names  # Main config should be filtered out

    def test_directory_not_found_exits_with_error(self, tmp_path):
        """Should exit with error if arms directory does not exist."""
        from scripts.run_ablation import get_arm_configs

        nonexistent = tmp_path / "does-not-exist"
        with pytest.raises(SystemExit):
            get_arm_configs(nonexistent)


class TestRunArm:
    """Tests for running individual arm configurations."""

    def test_run_arm_success(self, temp_arms_dir):
        """Should return success=True when subprocess succeeds."""
        from scripts.run_ablation import run_arm

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")
            config_file = temp_arms_dir / "control.json"

            arm_name, success, message = run_arm(config_file)

            assert arm_name == "control"
            assert success is True
            assert message == "Success"

    def test_run_arm_failure(self, temp_arms_dir):
        """Should return success=False when subprocess fails."""
        from scripts.run_ablation import run_arm

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1, stderr="Error running arm", stdout=""
            )
            config_file = temp_arms_dir / "control.json"

            arm_name, success, message = run_arm(config_file)

            assert arm_name == "control"
            assert success is False
            assert "Error" in message

    def test_run_arm_timeout(self, temp_arms_dir):
        """Should handle timeout gracefully."""
        from scripts.run_ablation import run_arm

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired("cmd", 3600)
            config_file = temp_arms_dir / "control.json"

            arm_name, success, message = run_arm(config_file)

            assert arm_name == "control"
            assert success is False
            assert "Timeout" in message

    def test_run_arm_exception(self, temp_arms_dir):
        """Should handle unexpected exceptions gracefully."""
        from scripts.run_ablation import run_arm

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = RuntimeError("Unexpected error")
            config_file = temp_arms_dir / "control.json"

            arm_name, success, message = run_arm(config_file)

            assert arm_name == "control"
            assert success is False
            assert "Unexpected error" in message

    def test_run_arm_extracts_error_from_output(self, temp_arms_dir):
        """Should extract last few lines from stderr for error reporting."""
        from scripts.run_ablation import run_arm

        with patch("subprocess.run") as mock_run:
            error_text = "Line 1\nLine 2\nLine 3\nFinal error"
            mock_run.return_value = MagicMock(
                returncode=1, stderr=error_text, stdout=""
            )
            config_file = temp_arms_dir / "control.json"

            arm_name, success, message = run_arm(config_file)

            assert "Final error" in message
            # Last 3 lines should be included
            assert "Line 3" in message


class TestRunAblationMain:
    """Integration tests for the main() function."""

    def test_main_all_arms_pass(self, temp_arms_dir, capsys):
        """Should report success when all arms pass."""
        with patch("scripts.run_ablation.get_arm_configs") as mock_get, \
             patch("scripts.run_ablation.run_arm") as mock_run:
            # Mock discovering two configs
            mock_get.return_value = [
                temp_arms_dir / "control.json",
                temp_arms_dir / "no-market.json",
            ]
            # Mock both arms passing
            mock_run.side_effect = [
                ("control", True, "Success"),
                ("no-market", True, "Success"),
            ]

            with patch("sys.argv", ["run_ablation.py", "--arms-dir", str(temp_arms_dir)]):
                from scripts.run_ablation import main

                main()  # Should not raise

            captured = capsys.readouterr()
            assert "2 passed" in captured.out
            assert "0 failed" in captured.out

    def test_main_some_arms_fail_exits_nonzero(self, temp_arms_dir, capsys):
        """Should exit with non-zero code when any arm fails."""
        with patch("scripts.run_ablation.get_arm_configs") as mock_get, \
             patch("scripts.run_ablation.run_arm") as mock_run:
            # Mock discovering two configs
            mock_get.return_value = [
                temp_arms_dir / "control.json",
                temp_arms_dir / "no-market.json",
            ]
            # Mock first arm passing, second failing
            mock_run.side_effect = [
                ("control", True, "Success"),
                ("no-market", False, "Error occurred"),
            ]

            with patch("sys.argv", ["run_ablation.py", "--arms-dir", str(temp_arms_dir)]):
                from scripts.run_ablation import main

                with pytest.raises(SystemExit) as exc_info:
                    main()

                assert exc_info.value.code == 1

            captured = capsys.readouterr()
            assert "1 passed" in captured.out
            assert "1 failed" in captured.out

    def test_main_continues_on_arm_failure(self, temp_arms_dir, capsys):
        """Should continue to next arm when one fails (not abort)."""
        with patch("scripts.run_ablation.get_arm_configs") as mock_get, \
             patch("scripts.run_ablation.run_arm") as mock_run:
            # Mock three configs
            mock_get.return_value = [
                temp_arms_dir / "control.json",
                temp_arms_dir / "no-market.json",
                temp_arms_dir / "no-social.json",
            ]
            # Create a no-social config for mocking
            config = {
                "stocks_file": "stocks.json",
                "report_dir": "reports/no-social",
                "portfolio": True,
                "style": "aggressive",
                "depot_id": "no-social-depot",
                "memory_id": "no-social",
                "config": {"memory_log_path": "~/.tradingagents/memory/no-social/trading_memory.md"},
            }
            (temp_arms_dir / "no-social.json").write_text(json.dumps(config))

            # Mock middle arm failing, others passing
            mock_run.side_effect = [
                ("control", True, "Success"),
                ("no-market", False, "Error occurred"),
                ("no-social", True, "Success"),
            ]

            with patch("sys.argv", ["run_ablation.py", "--arms-dir", str(temp_arms_dir)]):
                from scripts.run_ablation import main

                with pytest.raises(SystemExit):
                    main()

            # Verify all three arms were run (not just the first two)
            assert mock_run.call_count == 3

    def test_main_prints_per_arm_summary(self, temp_arms_dir, capsys):
        """Should print a summary line for each arm."""
        with patch("scripts.run_ablation.get_arm_configs") as mock_get, \
             patch("scripts.run_ablation.run_arm") as mock_run:
            mock_get.return_value = [
                temp_arms_dir / "control.json",
                temp_arms_dir / "no-market.json",
            ]
            mock_run.side_effect = [
                ("control", True, "Success"),
                ("no-market", False, "Error"),
            ]

            with patch("sys.argv", ["run_ablation.py", "--arms-dir", str(temp_arms_dir)]):
                from scripts.run_ablation import main

                with pytest.raises(SystemExit):
                    main()

            captured = capsys.readouterr()
            # Should show both arms in output
            assert "control" in captured.out
            assert "no-market" in captured.out

    def test_main_no_configs_found_exits_with_error(self, tmp_path, capsys):
        """Should exit with error if no arm configs are found."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        with patch("sys.argv", ["run_ablation.py", "--arms-dir", str(empty_dir)]):
            from scripts.run_ablation import main

            with pytest.raises(SystemExit):
                main()

            captured = capsys.readouterr()
            assert "No arm configs found" in captured.out

    def test_main_supports_stability_flag(self, temp_arms_dir):
        """Should support --stability flag for stability-block configs."""
        with patch("scripts.run_ablation.get_arm_configs") as mock_get, \
             patch("scripts.run_ablation.run_arm") as _mock_run:
            mock_get.return_value = []

            with patch("sys.argv", ["run_ablation.py", "--stability"]):
                from scripts.run_ablation import main

                with pytest.raises(SystemExit):
                    main()

                # Verify get_arm_configs was called with stability=True
                mock_get.assert_called_once()
                args, kwargs = mock_get.call_args
                assert kwargs.get("stability") is True
