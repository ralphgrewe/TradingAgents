"""Intel XPU client unit tests (mocked, no real hardware required).

Tests cover:
- torch.xpu.is_available() == False raises at construction
- model-id mismatch raises ValueError at construction, before any load is
  attempted (gates the load instead of warn-and-continue)
- the real load path calls Mistral3ForConditionalGeneration.from_pretrained
  (not AutoModelForCausalLM)
- model/tokenizer for a given model id are loaded once and shared across
  IntelXPUClient instances via the process-wide _MODEL_CACHE (so the graph's
  deep+quick clients and repeat MCP runs reuse one in-memory copy)
- the shared cache is keyed by id (distinct ids stay isolated) and is
  thread-safe (concurrent first requests load exactly once)
- invoke() formats the FULL message list (system + human) via
  tokenizer.apply_chat_template(), not just the last message
- generation kwargs (max_new_tokens, temperature, ...) are threaded through
  from client kwargs and per-call kwargs, with defaults only when unset
- with_structured_output / bind_tools log and raise NotImplementedError
- Factory routing

Test isolation for the shared cache goes through a clean, named seam:
``clear_model_cache()`` (autouse fixture below), not by reaching into module
internals to patch cache state. Tests that don't want a real load still just
monkeypatch ``IntelXPUClient._load_model_and_tokenizer`` (an instance method),
which pytest undoes automatically.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from tradingagents.llm_clients.intel_xpu_client import (
    _LOCKED_MODEL_ID,
    IntelXPUClient,
    clear_model_cache,
)


@pytest.fixture(autouse=True)
def _clear_xpu_model_cache():
    """Reset the process-wide model cache around every test.

    The cache is process-lifetime by design, so without this a real (or mocked)
    load cached under the locked id in one test would leak into the next. This
    is the clean, named seam the design intends for isolation.
    """
    clear_model_cache()
    yield
    clear_model_cache()


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
def test_same_model_id_shares_one_loaded_model(monkeypatch):
    """Two IntelXPUClient instances for the same (locked) model id reuse one
    in-memory model/tokenizer via the shared process cache — the expensive XPU
    load runs once, not once per client. This is the deep+quick and repeat-MCP
    reuse the fix restores (without the old unmanaged module-global cache)."""
    call_count = {"n": 0}
    model = _RecordingModel()
    tokenizer = _RecordingTokenizer()

    def _loader(self):
        call_count["n"] += 1
        return model, tokenizer

    monkeypatch.setattr(IntelXPUClient, "_load_model_and_tokenizer", _loader)

    client_a = IntelXPUClient(_LOCKED_MODEL_ID)
    client_b = IntelXPUClient(_LOCKED_MODEL_ID)

    assert call_count["n"] == 1  # loaded once, shared across instances
    assert client_a.model_obj is client_b.model_obj is model
    assert client_a.tokenizer is client_b.tokenizer is tokenizer


@pytest.mark.unit
def test_clear_model_cache_forces_reload(monkeypatch):
    """clear_model_cache() is a real, working seam: after it, the next client
    re-loads instead of reusing the (now dropped) cached copy."""
    call_count = {"n": 0}

    def _loader(self):
        call_count["n"] += 1
        return _RecordingModel(), _RecordingTokenizer()

    monkeypatch.setattr(IntelXPUClient, "_load_model_and_tokenizer", _loader)

    IntelXPUClient(_LOCKED_MODEL_ID)
    IntelXPUClient(_LOCKED_MODEL_ID)
    assert call_count["n"] == 1  # second construction is a cache hit

    clear_model_cache()
    IntelXPUClient(_LOCKED_MODEL_ID)
    assert call_count["n"] == 2  # cache cleared -> reloaded


@pytest.mark.unit
def test_model_cache_isolates_distinct_ids():
    """The shared cache keys by model id: distinct ids load independently and
    don't clobber one another. Exercised through the cache's public interface,
    not by reaching into internals to patch state."""
    from tradingagents.llm_clients.intel_xpu_client import _ModelCache

    cache = _ModelCache()
    calls = []

    def make_loader(tag):
        def _loader():
            calls.append(tag)
            return (f"model-{tag}", f"tok-{tag}")

        return _loader

    a1 = cache.get_or_load("id-a", make_loader("a"))
    b1 = cache.get_or_load("id-b", make_loader("b"))
    a2 = cache.get_or_load("id-a", make_loader("a-again"))

    assert a1 == ("model-a", "tok-a")
    assert b1 == ("model-b", "tok-b")
    assert a2 is a1  # second request for id-a reuses; loader not re-run
    assert calls == ["a", "b"]  # exactly one load per distinct id


@pytest.mark.unit
def test_model_cache_loads_once_under_concurrency():
    """Concurrent first requests for the same id load exactly once (double-
    checked locking), never duplicating an expensive XPU load across threads —
    grounds the thread-safety claim, since the MCP server dispatches
    analyze_stock in worker threads."""
    import threading
    import time

    from tradingagents.llm_clients.intel_xpu_client import _ModelCache

    cache = _ModelCache()
    load_count = {"n": 0}
    barrier = threading.Barrier(8)
    sentinel = object()

    def _loader():
        time.sleep(0.01)  # widen the window for a racing thread
        load_count["n"] += 1
        return (sentinel, sentinel)

    results = []

    def worker():
        barrier.wait()  # release all threads into get_or_load together
        results.append(cache.get_or_load("shared-id", _loader))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert load_count["n"] == 1  # loaded once despite 8 concurrent requests
    assert all(r is results[0] for r in results)


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
