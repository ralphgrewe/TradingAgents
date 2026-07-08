"""Intel XPU client unit tests (mocked, no real hardware required).

Tests cover:
- torch.xpu.is_available() == False raises at construction
- Mocked model/tokenizer round-trip through invoke()
- with_structured_output logs and raises NotImplementedError
- bind_tools logs and raises NotImplementedError
- validate_model only accepts the locked model ID
- Factory routing
"""

import sys
import types
import warnings

import pytest

_LOCKED_MODEL_ID = "mistralai/Ministral-3-3B-Reasoning-2512"


def _make_mock_model_and_tokenizer():
    """Create mock model and tokenizer objects."""
    class MockModel:
        device = "xpu"

        def generate(self, **kwargs):
            return [[1, 2, 3, 4, 5]]

    class MockTokenizerOutput:
        def to(self, device):
            return {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}

    class MockTokenizer:
        def decode(self, x, skip_special_tokens=False):
            return "The stock will rise due to strong earnings."

        def __call__(self, text, return_tensors=None):
            return MockTokenizerOutput()

    return MockModel(), MockTokenizer()


@pytest.mark.unit
def test_factory_routes_intel_xpu(mock_torch_xpu, monkeypatch):
    """Factory correctly routes 'intel_xpu' provider to IntelXPUClient."""
    mock_torch, mock_transformers = mock_torch_xpu
    mock_model, mock_tokenizer = _make_mock_model_and_tokenizer()

    # Clear the module cache before importing
    if "tradingagents.llm_clients.intel_xpu_client" in sys.modules:
        del sys.modules["tradingagents.llm_clients.intel_xpu_client"]

    import tradingagents.llm_clients.intel_xpu_client as xpu_mod
    monkeypatch.setattr(xpu_mod, "_MODEL_CACHE", None)
    monkeypatch.setattr(xpu_mod, "_TOKENIZER_CACHE", None)
    monkeypatch.setattr(xpu_mod, "_load_model_and_tokenizer", lambda: (mock_model, mock_tokenizer))

    from tradingagents.llm_clients.factory import create_llm_client
    client = create_llm_client("intel_xpu", _LOCKED_MODEL_ID)
    assert type(client).__name__ == "IntelXPUClient"


@pytest.mark.unit
def test_xpu_unavailable_raises_at_construction(monkeypatch):
    """If torch.xpu.is_available() is False, construction raises RuntimeError."""
    import sys
    import types
    import importlib

    # Mock torch.xpu to be unavailable
    mock_torch = types.ModuleType("torch")
    mock_xpu = types.ModuleType("xpu")
    mock_xpu.is_available = lambda: False
    mock_torch.xpu = mock_xpu
    mock_torch.bfloat16 = None

    class MockNoGrad:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    mock_torch.no_grad = lambda: MockNoGrad()

    mock_transformers = types.ModuleType("transformers")

    monkeypatch.setitem(sys.modules, "torch", mock_torch)
    monkeypatch.setitem(sys.modules, "transformers", mock_transformers)

    # Clear the module cache to force a fresh import
    if "tradingagents.llm_clients.intel_xpu_client" in sys.modules:
        del sys.modules["tradingagents.llm_clients.intel_xpu_client"]

    # Import and clear the global caches directly
    import tradingagents.llm_clients.intel_xpu_client as xpu_mod
    xpu_mod._MODEL_CACHE = None
    xpu_mod._TOKENIZER_CACHE = None

    from tradingagents.llm_clients.intel_xpu_client import IntelXPUClient

    with pytest.raises(RuntimeError, match="torch.xpu is not available"):
        IntelXPUClient(_LOCKED_MODEL_ID)


@pytest.mark.unit
def test_validate_model_locked_to_single_model():
    """validate_model only returns True for the locked model ID."""
    assert _LOCKED_MODEL_ID.lower() == "mistralai/ministral-3-3b-reasoning-2512"

    # Any other model should not validate (triggers warn_if_unknown_model)
    other_models = [
        "mistralai/Mistral-7B",
        "meta-llama/Llama-2-7b",
        "unknown/model",
    ]
    for model in other_models:
        assert model.lower() != _LOCKED_MODEL_ID.lower()


@pytest.mark.unit
def test_with_structured_output_raises_and_logs(mock_torch_xpu, monkeypatch):
    """with_structured_output logs warning and raises NotImplementedError."""
    mock_torch, mock_transformers = mock_torch_xpu
    mock_model, mock_tokenizer = _make_mock_model_and_tokenizer()

    # Clear the module cache
    if "tradingagents.llm_clients.intel_xpu_client" in sys.modules:
        del sys.modules["tradingagents.llm_clients.intel_xpu_client"]

    import tradingagents.llm_clients.intel_xpu_client as xpu_mod
    monkeypatch.setattr(xpu_mod, "_MODEL_CACHE", None)
    monkeypatch.setattr(xpu_mod, "_TOKENIZER_CACHE", None)
    monkeypatch.setattr(xpu_mod, "_load_model_and_tokenizer", lambda: (mock_model, mock_tokenizer))

    from tradingagents.llm_clients.intel_xpu_client import IntelXPUClient

    client = IntelXPUClient(_LOCKED_MODEL_ID)
    llm = client.get_llm()

    # Should log warning and raise NotImplementedError on the chat model
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with pytest.raises(NotImplementedError, match="does not support structured output"):
            llm.with_structured_output({})
        assert len(w) == 1
        assert "does not support structured output" in str(w[0].message)


@pytest.mark.unit
def test_bind_tools_raises_and_logs(mock_torch_xpu, monkeypatch):
    """bind_tools logs warning and raises NotImplementedError."""
    mock_torch, mock_transformers = mock_torch_xpu
    mock_model, mock_tokenizer = _make_mock_model_and_tokenizer()

    # Clear the module cache
    if "tradingagents.llm_clients.intel_xpu_client" in sys.modules:
        del sys.modules["tradingagents.llm_clients.intel_xpu_client"]

    import tradingagents.llm_clients.intel_xpu_client as xpu_mod
    monkeypatch.setattr(xpu_mod, "_MODEL_CACHE", None)
    monkeypatch.setattr(xpu_mod, "_TOKENIZER_CACHE", None)
    monkeypatch.setattr(xpu_mod, "_load_model_and_tokenizer", lambda: (mock_model, mock_tokenizer))

    from tradingagents.llm_clients.intel_xpu_client import IntelXPUClient

    client = IntelXPUClient(_LOCKED_MODEL_ID)
    llm = client.get_llm()

    # Should log warning and raise NotImplementedError on the chat model
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with pytest.raises(NotImplementedError, match="does not support tool-calling"):
            llm.bind_tools([])
        assert len(w) == 1
        assert "does not support tool-calling" in str(w[0].message)


@pytest.mark.unit
def test_mocked_invoke_returns_aimessage(mock_torch_xpu, monkeypatch):
    """invoke() returns an AIMessage with the model's output."""
    from langchain_core.messages import AIMessage

    mock_torch, mock_transformers = mock_torch_xpu
    mock_model, mock_tokenizer = _make_mock_model_and_tokenizer()

    # Clear the module cache
    if "tradingagents.llm_clients.intel_xpu_client" in sys.modules:
        del sys.modules["tradingagents.llm_clients.intel_xpu_client"]

    import tradingagents.llm_clients.intel_xpu_client as xpu_mod
    monkeypatch.setattr(xpu_mod, "_MODEL_CACHE", None)
    monkeypatch.setattr(xpu_mod, "_TOKENIZER_CACHE", None)
    monkeypatch.setattr(xpu_mod, "_load_model_and_tokenizer", lambda: (mock_model, mock_tokenizer))

    from tradingagents.llm_clients.intel_xpu_client import IntelXPUClient

    client = IntelXPUClient(_LOCKED_MODEL_ID)
    llm = client.get_llm()

    # Invoke should return an AIMessage
    result = llm.invoke("What will the stock do?")
    assert isinstance(result, AIMessage)
    assert result.content == "The stock will rise due to strong earnings."


@pytest.mark.unit
def test_unknown_model_triggers_warning(mock_torch_xpu, monkeypatch):
    """Using a non-locked model ID triggers warn_if_unknown_model."""
    mock_torch, mock_transformers = mock_torch_xpu
    mock_model, mock_tokenizer = _make_mock_model_and_tokenizer()

    # Clear the module cache
    if "tradingagents.llm_clients.intel_xpu_client" in sys.modules:
        del sys.modules["tradingagents.llm_clients.intel_xpu_client"]

    import tradingagents.llm_clients.intel_xpu_client as xpu_mod
    monkeypatch.setattr(xpu_mod, "_MODEL_CACHE", None)
    monkeypatch.setattr(xpu_mod, "_TOKENIZER_CACHE", None)
    monkeypatch.setattr(xpu_mod, "_load_model_and_tokenizer", lambda: (mock_model, mock_tokenizer))

    from tradingagents.llm_clients.intel_xpu_client import IntelXPUClient

    # Try to use a different model
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        client = IntelXPUClient("mistralai/Mistral-7B")
        client.get_llm()  # warn_if_unknown_model is called in get_llm
        assert any("not in the known model list" in str(warn.message) for warn in w)
