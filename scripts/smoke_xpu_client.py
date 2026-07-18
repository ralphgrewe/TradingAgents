"""End-to-end smoke test for Intel XPU client against real hardware.

Constructs the Intel XPU client via create_llm_client, runs a single real
prompt through it, prints the response, and verifies the model is actually
running on the XPU device.

Usage:
    python scripts/smoke_xpu_client.py

The script requires:
  - Intel Arc GPU hardware (or compatible Intel GPU)
  - Intel Extension for PyTorch installed
  - Mistral-3-3B-Reasoning model available (will download on first run)
  - Install Intel XPU support: pip install "tradingagents[xpu]"

This is a manual smoke test for hardware-specific validation. It is NOT
part of the CI suite (no CI machine has Intel XPU hardware).
"""

from __future__ import annotations

import sys

from tradingagents.llm_clients import create_llm_client
from tradingagents.llm_clients.intel_xpu_client import _LOCKED_MODEL_ID


def main() -> int:
    """Run the Intel XPU smoke test."""
    print("=" * 70)
    print("Intel XPU Client Smoke Test")
    print("=" * 70)
    print(f"Model ID: {_LOCKED_MODEL_ID}")
    print()

    try:
        # 1) Construct the Intel XPU client via the public factory entry point.
        print("Creating Intel XPU client...")
        client = create_llm_client(provider="intel_xpu", model=_LOCKED_MODEL_ID)
        print("✓ Client created successfully")
        print()

        # 2) Get the LangChain-compatible chat model.
        print("Retrieving IntelXPUChatModel...")
        llm = client.get_llm()
        print("✓ IntelXPUChatModel retrieved successfully")
        print()

        # 3) Verify the model is on the XPU device.
        print("Verifying device placement...")
        device = client.model_obj.device
        device_str = str(device)
        print(f"  Model device: {device_str}")
        if "xpu" not in device_str.lower():
            print(
                f"⚠ WARNING: Model device is '{device_str}', expected XPU device. "
                f"Check Intel Extension for PyTorch installation."
            )
            return 1
        print("✓ Model confirmed on Intel XPU device")
        print()

        # 4) Run a single real prompt through the model.
        print("Running inference prompt...")
        system_prompt = (
            "You are a financial analyst. "
            "Provide a concise market analysis (2-3 sentences)."
        )
        user_prompt = "Analyze the recent performance of NVDA stock."
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        response = llm.invoke(messages)
        response_text = response.content
        print()
        print("=" * 70)
        print("Response from Intel XPU model:")
        print("=" * 70)
        print(response_text)
        print()

        # 5) Verify we got a reasonable response.
        if not response_text or len(response_text.strip()) < 10:
            print("⚠ ERROR: Response is too short or empty")
            return 1

        print("=" * 70)
        print("Smoke test checks")
        print("=" * 70)
        checks = [
            ("Client construction", True),
            ("Model on XPU device", "xpu" in device_str.lower()),
            ("Inference completed", bool(response_text)),
            ("Response length >= 10 chars", len(response_text.strip()) >= 10),
        ]

        failures = 0
        for check_name, passed in checks:
            status = "PASS" if passed else "FAIL"
            print(f"  {status}  {check_name}")
            failures += int(not passed)

        print()
        if failures:
            print(f"Smoke FAILED: {failures} check(s) failed.")
            return 1

        print("Smoke PASSED: Intel XPU client works correctly on this hardware.")
        return 0

    except ImportError as e:
        print(f"⚠ Import error: {e}")
        print()
        print("Intel XPU support requires:")
        print("  pip install \"tradingagents[xpu]\"")
        print()
        print("This also requires:")
        print("  - Intel Core Ultra or Arc GPU")
        print("  - Intel Extension for PyTorch")
        return 1

    except RuntimeError as e:
        print(f"⚠ Runtime error: {e}")
        print()
        print("torch.xpu is not available. Check:")
        print("  - Intel GPU hardware is present")
        print("  - Intel Extension for PyTorch is installed")
        print("  - Hardware drivers are up to date")
        return 1

    except NotImplementedError as e:
        print(f"⚠ Feature not supported: {e}")
        print()
        print("This usually means the model is trying to use an unsupported feature.")
        return 1

    except Exception as e:
        print(f"⚠ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
