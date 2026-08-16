#!/usr/bin/env python3
"""Run all ablation arms sequentially, continuing on failure.

Usage:
    python scripts/run_ablation.py                           # Run main arms
    python scripts/run_ablation.py --arms-dir /path/to/dir  # Custom directory
    python scripts/run_ablation.py --stability               # Run stability blocks instead
"""

import argparse
import subprocess
import sys
from pathlib import Path


def get_arm_configs(arms_dir: Path, stability: bool = False) -> list[Path]:
    """Discover arm config files in the given directory.

    Args:
        arms_dir: Directory containing arm config files
        stability: If True, look for stability-block configs (*.stability.json);
                  otherwise look for main arm configs (*.json, excluding stability)

    Returns:
        Sorted list of config file paths
    """
    if not arms_dir.exists():
        print(f"Error: Arms directory '{arms_dir}' does not exist")
        sys.exit(1)

    if stability:
        # Look for *.stability.json files
        configs = sorted(arms_dir.glob("*.stability.json"))
    else:
        # Look for *.json files, excluding stability variants
        all_jsons = sorted(arms_dir.glob("*.json"))
        # Filter out *.stability.json and non-arm files like stocks.json, stability-stocks.json
        configs = [
            f for f in all_jsons
            if not f.name.endswith(".stability.json")
            and f.name not in ("stocks.json", "stability-stocks.json")
        ]

    return configs


def run_arm(config_path: Path) -> tuple[str, bool, str]:
    """Run a single arm config.

    Args:
        config_path: Path to the arm config JSON file

    Returns:
        Tuple of (arm_name, success: bool, output_or_error_msg: str)
    """
    arm_name = config_path.stem

    try:
        # Run run_trading_agents.py with the config file
        result = subprocess.run(
            ["./venv/bin/python", "run_trading_agents.py", str(config_path)],
            capture_output=True,
            text=True,
            timeout=3600,  # 1-hour timeout per arm
        )

        if result.returncode == 0:
            return arm_name, True, "Success"
        else:
            # Extract last few lines of stderr/stdout for error reporting
            error_output = (result.stderr or result.stdout or "Unknown error")
            error_lines = error_output.strip().split('\n')[-3:]
            return arm_name, False, '\n'.join(error_lines)

    except subprocess.TimeoutExpired:
        return arm_name, False, "Timeout (exceeded 1 hour)"
    except Exception as e:
        return arm_name, False, str(e)


def main():
    parser = argparse.ArgumentParser(
        description="Run all ablation arms sequentially, continuing on failure"
    )
    parser.add_argument(
        "--arms-dir",
        type=Path,
        default=Path("experiments/ablation-analysts"),
        help="Directory containing arm config files (default: experiments/ablation-analysts)",
    )
    parser.add_argument(
        "--stability",
        action="store_true",
        help="Run stability-block configs (*.stability.json) instead of main arms",
    )
    args = parser.parse_args()

    # Discover arm configs
    configs = get_arm_configs(args.arms_dir, stability=args.stability)

    if not configs:
        print(f"Error: No arm configs found in '{args.arms_dir}'")
        sys.exit(1)

    block_name = "stability-block" if args.stability else "main"
    print(f"Running {len(configs)} ablation arms ({block_name})...\n")

    # Run each arm
    results = []
    for config_path in configs:
        arm_name, success, message = run_arm(config_path)
        results.append((arm_name, success, message))

        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {arm_name}")
        if not success:
            # Print error details for failed arms
            for line in message.split('\n'):
                print(f"       {line}")

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    passed = sum(1 for _, success, _ in results if success)
    failed = len(results) - passed

    for arm_name, success, _message in results:
        status = "PASS" if success else "FAIL"
        print(f"  {arm_name:30s} {status}")

    print(f"\nTotal: {passed} passed, {failed} failed out of {len(results)}")
    print(f"{'='*60}\n")

    # Exit with non-zero if any arm failed
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
