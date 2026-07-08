"""Intel XPU client unit tests (mocked, no real hardware required).

Tests cover:
- torch.xpu.is_available() == False raises at construction
- model-id mismatch raises ValueError at construction, before any load is
  attempted (gates the load instead of warn-and-continue)
- the real load path calls Mistral3ForConditionalGeneration.from_pretrained
  (not AutoModelForCausalLM)
- model/tokenizer are cached on the instance, not module-level globals
- invoke() formats the FULL message list (system + human) via
  tokenizer.apply_chat_template(), not just the last message
- generation kwargs (max_new_tokens, temperature, ...) are threaded through
  from client kwargs and per-call kwargs, with defaults only when unset
- with_structured_output / bind_tools log and raise NotImplementedError
- Factory routing

No test monkeypatches module-level state: IntelXPUClient caches the loaded
model/tokenizer on ``self``, so isolation between tests only requires
monkeypatching ``IntelXPUClient._load_model_and_tokenizer`` (an instance
method) per-test via pytest's ``monkeypatch``, which is undone automatically.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from tradingagents.llm_clients.intel_xpu_client import _LOCKED_MODEL_ID, IntelXPUClient


class _MockBatchEncoding(dict):
    """Minimal stand-in for a transformers BatchEncoding: dict + .to(device)."""

    def to(self, device):
        self["input_ids"] = [[1, 2, 3]]
        self["attention_mask"] = [[1, 1, 1]]
        return self


class _RecordingModel:
    """Mock model recording every generate() call for assertions."""

    device = "xpu"

    def __init__(self, output_tokens=None):
        self.generate_calls: list[dict] = []
        self._output_tokens = output_tokens or [1, 2, 3, 4, 5]

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        return [self._output_tokens]


class _RecordingTokenizer:
    """Mock tokenizer recording every apply_chat_template() call for assertions."""

    def __init__(self, decoded_text="The stock will rise due to strong earnings."):
        self.chat_template_calls: list[list[dict]] = []
        self._decoded_text = decoded_text

    def apply_chat_template(self, messages, add_generation_prompt=True, return_tensors=None, return_dict=None):
        self.chat_template_calls.append(messages)
        return _MockBatchEncoding()

    def decode(self, token_ids, skip_special_tokens=False):
        return self._decoded_text


def _mock_loader(monkeypatch, model=None, tokenizer=None):
    """Patch IntelXPUClient._load_model_and_tokenizer to skip real loading.

    Patched on the class (an instance method), scoped to the test via
    pytest's monkeypatch — no module-global cache to reset between tests.
    """
    model = model or _RecordingModel()
    tokenizer = tokenizer or _RecordingTokenizer()
    monkeypatch.setattr(
        IntelXPUClient,
        "_load_model_and_tokenizer",
        lambda self: (model, tokenizer),
    )
    return model, tokenizer


@pytest.mark.unit
def test_factory_routes_intel_xpu(monkeypatch):
    """Factory correctly routes 'intel_xpu' provider to IntelXPUClient."""
    _mock_loader(monkeypatch)

    from tradingagents.llm_clients.factory import create_llm_client

    client = create_llm_client("intel_xpu", _LOCKED_MODEL_ID)
    assert type(client).__name__ == "IntelXPUClient"


@pytest.mark.unit
def test_xpu_unavailable_raises_at_construction(mock_torch_xpu):
    """If torch.xpu.is_available() is False, construction raises RuntimeError."""
    mock_torch, _mock_transformers = mock_torch_xpu
    mock_torch.xpu.is_available = lambda: False

    with pytest.raises(RuntimeError, match="torch.xpu is not available"):
        IntelXPUClient(_LOCKED_MODEL_ID)


@pytest.mark.unit
def test_model_mismatch_raises_valueerror_before_loading(monkeypatch):
    """A model id other than the locked one raises ValueError at construction,
    instead of silently loading the locked model under a different name."""

    def _fail_if_called(self):
        raise AssertionError("_load_model_and_tokenizer must not be called on mismatch")

    monkeypatch.setattr(IntelXPUClient, "_load_model_and_tokenizer", _fail_if_called)

    with pytest.raises(ValueError, match=_LOCKED_MODEL_ID):
        IntelXPUClient("mistralai/Mistral-7B")


@pytest.mark.unit
def test_validate_model_true_only_for_locked_id(monkeypatch):
    """validate_model() returns True for the locked id and gates loading."""
    _mock_loader(monkeypatch)

    client = IntelXPUClient(_LOCKED_MODEL_ID)
    assert client.validate_model() is True

    # Case-insensitive match, per validate_model()'s implementation.
    _mock_loader(monkeypatch)
    client_upper = IntelXPUClient(_LOCKED_MODEL_ID.upper())
    assert client_upper.validate_model() is True


@pytest.mark.unit
def test_uses_mistral3_for_conditional_generation(mock_torch_xpu):
    """The real load path calls Mistral3ForConditionalGeneration.from_pretrained,
    matching the AC's verified example scripts (not AutoModelForCausalLM)."""
    mock_torch, mock_transformers = mock_torch_xpu

    IntelXPUClient(_LOCKED_MODEL_ID)

    model_call = mock_transformers.Mistral3ForConditionalGeneration.from_pretrained.call_args
    assert model_call.args == (_LOCKED_MODEL_ID,)
    assert model_call.kwargs == {
        "device_map": "xpu",
        "dtype": mock_torch.bfloat16,
        "attn_implementation": "eager",
        "low_cpu_mem_usage": True,
    }

    tokenizer_call = mock_transformers.AutoTokenizer.from_pretrained.call_args
    assert tokenizer_call.args == (_LOCKED_MODEL_ID,)
    assert tokenizer_call.kwargs == {"fix_mistral_regex": True}


@pytest.mark.unit
def test_model_and_tokenizer_cached_on_instance_not_module(monkeypatch):
    """Each IntelXPUClient instance loads and caches its own model/tokenizer;
    there is no shared module-level cache to invalidate between instances."""
    import tradingagents.llm_clients.intel_xpu_client as xpu_mod

    assert not hasattr(xpu_mod, "_MODEL_CACHE")
    assert not hasattr(xpu_mod, "_TOKENIZER_CACHE")

    call_count = {"n": 0}
    models = [_RecordingModel(), _RecordingModel()]
    tokenizers = [_RecordingTokenizer(), _RecordingTokenizer()]

    def _loader(self):
        idx = call_count["n"]
        call_count["n"] += 1
        return models[idx], tokenizers[idx]

    monkeypatch.setattr(IntelXPUClient, "_load_model_and_tokenizer", _loader)

    client_a = IntelXPUClient(_LOCKED_MODEL_ID)
    client_b = IntelXPUClient(_LOCKED_MODEL_ID)

    assert call_count["n"] == 2  # loaded once per instance, not cached/shared
    assert client_a.model_obj is models[0]
    assert client_b.model_obj is models[1]
    assert client_a.model_obj is not client_b.model_obj


@pytest.mark.unit
def test_with_structured_output_raises_and_logs(monkeypatch):
    """with_structured_output logs warning and raises NotImplementedError."""
    _mock_loader(monkeypatch)
    client = IntelXPUClient(_LOCKED_MODEL_ID)
    llm = client.get_llm()

    with (
        pytest.warns(RuntimeWarning, match="does not support structured output"),
        pytest.raises(NotImplementedError, match="does not support structured output"),
    ):
        llm.with_structured_output({})


@pytest.mark.unit
def test_bind_tools_raises_and_logs(monkeypatch):
    """bind_tools logs warning and raises NotImplementedError."""
    _mock_loader(monkeypatch)
    client = IntelXPUClient(_LOCKED_MODEL_ID)
    llm = client.get_llm()

    with (
        pytest.warns(RuntimeWarning, match="does not support tool-calling"),
        pytest.raises(NotImplementedError, match="does not support tool-calling"),
    ):
        llm.bind_tools([])


@pytest.mark.unit
def test_mocked_invoke_returns_aimessage(mock_torch_xpu, monkeypatch):
    """invoke() returns an AIMessage with the model's decoded output."""
    _mock_loader(monkeypatch)
    client = IntelXPUClient(_LOCKED_MODEL_ID)
    llm = client.get_llm()

    result = llm.invoke("What will the stock do?")
    assert isinstance(result, AIMessage)
    assert result.content == "The stock will rise due to strong earnings."


@pytest.mark.unit
def test_invoke_uses_full_message_history_via_chat_template(mock_torch_xpu, monkeypatch):
    """The full message list (system + human) reaches apply_chat_template(),
    not just the last message's content."""
    model, tokenizer = _mock_loader(monkeypatch)
    client = IntelXPUClient(_LOCKED_MODEL_ID)
    llm = client.get_llm()

    llm.invoke(
        [
            SystemMessage(content="You are a helpful trading analyst."),
            HumanMessage(content="What will the stock do?"),
        ]
    )

    assert len(tokenizer.chat_template_calls) == 1
    formatted = tokenizer.chat_template_calls[0]
    roles = [m["role"] for m in formatted]
    contents = [m["content"] for m in formatted]
    assert roles == ["system", "user"]
    assert "You are a helpful trading analyst." in contents
    assert "What will the stock do?" in contents


@pytest.mark.unit
def test_generation_kwargs_use_defaults_when_not_overridden(mock_torch_xpu, monkeypatch):
    """Hardcoded-looking defaults still apply when no kwargs are supplied."""
    model, _tokenizer = _mock_loader(monkeypatch)
    client = IntelXPUClient(_LOCKED_MODEL_ID)
    llm = client.get_llm()

    llm.invoke("hello")

    assert len(model.generate_calls) == 1
    call_kwargs = model.generate_calls[0]
    assert call_kwargs["max_new_tokens"] == 2048
    assert call_kwargs["temperature"] == 0.7
    assert call_kwargs["do_sample"] is True


@pytest.mark.unit
def test_generation_kwargs_threaded_from_client_kwargs(mock_torch_xpu, monkeypatch):
    """Kwargs passed to IntelXPUClient(...) are forwarded into model.generate()."""
    model, _tokenizer = _mock_loader(monkeypatch)
    client = IntelXPUClient(
        _LOCKED_MODEL_ID, max_new_tokens=64, temperature=0.1, do_sample=False
    )
    llm = client.get_llm()

    llm.invoke("hello")

    call_kwargs = model.generate_calls[0]
    assert call_kwargs["max_new_tokens"] == 64
    assert call_kwargs["temperature"] == 0.1
    assert call_kwargs["do_sample"] is False


@pytest.mark.unit
def test_generation_kwargs_call_site_overrides_client_kwargs(mock_torch_xpu, monkeypatch):
    """Per-call kwargs to invoke() take precedence over client-level kwargs."""
    model, _tokenizer = _mock_loader(monkeypatch)
    client = IntelXPUClient(_LOCKED_MODEL_ID, max_new_tokens=64)
    llm = client.get_llm()

    llm.invoke("hello", max_new_tokens=999)

    assert model.generate_calls[0]["max_new_tokens"] == 999


@pytest.mark.unit
def test_generation_kwargs_unknown_keys_dropped(mock_torch_xpu, monkeypatch):
    """Kwargs unrelated to generation aren't forwarded to model.generate()."""
    model, _tokenizer = _mock_loader(monkeypatch)
    client = IntelXPUClient(_LOCKED_MODEL_ID, some_unrelated_kwarg="nope")
    llm = client.get_llm()

    llm.invoke("hello")

    assert "some_unrelated_kwarg" not in model.generate_calls[0]


@pytest.mark.unit
def test_unknown_model_raises_instead_of_warning(monkeypatch):
    """Using a non-locked model id fails construction outright (loud, not a
    warn-and-continue-with-the-locked-model degrade)."""
    with pytest.raises(ValueError):
        IntelXPUClient("mistralai/Mistral-7B")
