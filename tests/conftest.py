"""Shared pytest fixtures that prevent CI hangs when API keys are absent."""

import logging
import os
from unittest.mock import MagicMock, patch

import pytest


def pytest_configure(config):
    for marker in ("unit", "integration", "smoke"):
        config.addinivalue_line("markers", f"{marker}: {marker}-level tests")


_API_KEY_ENV_VARS = (
    "OPENAI_API_KEY",
    "GOOGLE_API_KEY",
    "ANTHROPIC_API_KEY",
    "XAI_API_KEY",
    "DEEPSEEK_API_KEY",
    "DASHSCOPE_API_KEY",
    "DASHSCOPE_CN_API_KEY",
    "ZHIPU_API_KEY",
    "ZHIPU_CN_API_KEY",
    "MINIMAX_API_KEY",
    "MINIMAX_CN_API_KEY",
    "OPENROUTER_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "ALPHA_VANTAGE_API_KEY",
)


@pytest.fixture(autouse=True)
def _dummy_api_keys(monkeypatch):
    for env_var in _API_KEY_ENV_VARS:
        # `or` not a .get default: an env var present but empty (e.g. a key left
        # blank in a .env copied from .env.example) must still get the placeholder.
        monkeypatch.setenv(env_var, os.environ.get(env_var) or "placeholder")


@pytest.fixture(autouse=True)
def _isolate_config():
    """Reset the global dataflows config before and after each test.

    ``set_config`` merges (it never clears keys absent from the override), so a
    test that sets e.g. ``tool_vendors`` would otherwise leak into later tests
    and make routing behavior order-dependent. Replace the global outright so
    every test starts from a clean DEFAULT_CONFIG.
    """
    import copy

    import tradingagents.dataflows.config as config_module
    import tradingagents.default_config as default_config

    config_module._config = copy.deepcopy(default_config.DEFAULT_CONFIG)
    yield
    config_module._config = copy.deepcopy(default_config.DEFAULT_CONFIG)


@pytest.fixture(autouse=True)
def _restore_logging_disable():
    """Undo ``logging.disable(...)`` leakage across tests.

    ``mcp_server.py`` calls ``logging.disable(logging.CRITICAL)`` at import
    time (stdio transport: any stray log line would corrupt the JSON-RPC
    stream). That call is process-global and, because pytest imports every
    test module during collection (before any test runs), it takes effect
    before the first test even starts once anything pulls that module in
    (e.g. ``test_mcp_server_memory.py``). Reset it before every test so
    ``assertLogs``/``caplog`` keep working regardless of collection order.
    """
    logging.disable(logging.NOTSET)
    yield


@pytest.fixture()
def mock_llm_client():
    client = MagicMock()
    client.get_llm.return_value = MagicMock()
    with patch(
        "tradingagents.llm_clients.factory.create_llm_client",
        return_value=client,
    ):
        yield client


@pytest.fixture()
def mock_torch_xpu():
    """Mock torch and transformers modules for Intel XPU client tests.

    XPU is available in this fixture (for the "unavailable" scenario, tests
    build their own minimal mock — see test_xpu_unavailable_raises_at_construction).
    ``transformers.Mistral3ForConditionalGeneration`` / ``AutoTokenizer`` are
    MagicMocks so tests can assert on the real IntelXPUClient loading path
    (which class got called, with what args) without needing real weights.
    """
    import sys
    import types

    # Create mock modules
    mock_torch = types.ModuleType("torch")
    mock_xpu = types.ModuleType("xpu")
    mock_xpu.is_available = lambda: True
    mock_torch.xpu = mock_xpu
    mock_torch.bfloat16 = None

    class MockNoGrad:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    mock_torch.no_grad = lambda: MockNoGrad()

    # Create mock transformers module with stub loader classes so the real
    # IntelXPUClient._load_model_and_tokenizer() path is exercisable without
    # real weights. from_pretrained() returns a fresh MagicMock per call so
    # tests can inspect call args (which class, which kwargs) directly.
    mock_transformers = types.ModuleType("transformers")
    mock_transformers.Mistral3ForConditionalGeneration = MagicMock()
    mock_transformers.Mistral3ForConditionalGeneration.from_pretrained.return_value = MagicMock()
    mock_transformers.AutoTokenizer = MagicMock()
    mock_transformers.AutoTokenizer.from_pretrained.return_value = MagicMock()

    # Store original modules
    original_torch = sys.modules.get("torch")
    original_transformers = sys.modules.get("transformers")

    try:
        # Install mocks
        sys.modules["torch"] = mock_torch
        sys.modules["transformers"] = mock_transformers
        yield mock_torch, mock_transformers
    finally:
        # Restore originals
        if original_torch is not None:
            sys.modules["torch"] = original_torch
        elif "torch" in sys.modules:
            del sys.modules["torch"]

        if original_transformers is not None:
            sys.modules["transformers"] = original_transformers
        elif "transformers" in sys.modules:
            del sys.modules["transformers"]
