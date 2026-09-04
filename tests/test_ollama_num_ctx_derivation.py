"""Tests for the issue #154 per-request ``ollama_num_ctx`` derivation.

Issue #149 added ``ollama_num_ctx``: an explicit, fixed context length
forwarded to Ollama on every request. Issue #154 observed that a *single*
static value can't be big enough for every agent's prompt without being
wastefully large for the small ones -- so when ``ollama_num_ctx`` is left
unset (the default), ``num_ctx`` is instead derived **per request** from
that request's own measured prompt size:

    needed  = ceil(prompt_tokens_estimated * context_window_safety_margin)
              + ollama_num_ctx_response_headroom
    num_ctx = min(needed, ollama_num_ctx_max)

and the run aborts with ``PromptContextOverflowError`` when ``needed``
exceeds ``ollama_num_ctx_max``, before the call is dispatched.

These tests cover, in order:

1. ``OllamaNumCtxDerivation`` arithmetic (the formula itself).
2. ``ContextWindowGuardHandler``'s derivation branch: within-ceiling passes,
   over-ceiling raises with the right fields/message, and the abort reaches
   ``llm_calls.jsonl``.
3. ``LLMCallLogHandler`` recording the derived ``num_ctx`` on successful-call
   records.
4. ``TradingAgentsGraph._build_ollama_num_ctx_derivation`` /
   ``_get_provider_kwargs``: when derivation applies, when an explicit
   ``ollama_num_ctx`` bypasses it, and when a non-ollama provider is
   unaffected.
5. ``NormalizedChatOllama._chat_params`` (issue #169 re-pointed this from
   ``OllamaChatOpenAI._get_request_payload`` on the now-removed
   OpenAI-compatible client): the derived value actually reaching the
   outgoing request's ``options.num_ctx``, and other providers/chat classes
   being completely unaffected.
6. Ollama think mode (issue #155, re-pointed at ``ChatOllama``'s native
   ``reasoning``/``think`` field by issue #169): the True/False/None
   three-state contract.
"""

from __future__ import annotations

import json
import uuid
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import HumanMessage

from tradingagents.llm_call_log import (
    ContextWindowGuardHandler,
    LLMCallLogHandler,
    OllamaNumCtxDerivation,
    PromptContextOverflowError,
    _count_prompt_tokens,
)

pytestmark = pytest.mark.unit


def _messages(text: str) -> list[list[HumanMessage]]:
    return [[HumanMessage(content=text)]]


# ---------------------------------------------------------------------------
# 1. OllamaNumCtxDerivation arithmetic
# ---------------------------------------------------------------------------


class TestOllamaNumCtxDerivationArithmetic:
    def test_needed_tokens_applies_margin_then_headroom(self):
        derivation = OllamaNumCtxDerivation(
            models=frozenset({"ministral-3:8b"}),
            num_ctx_max=32768,
            response_headroom=2048,
            safety_margin=1.3,
        )
        # ceil(1000 * 1.3) + 2048 = 1300 + 2048 = 3348
        assert derivation.needed_tokens(1000) == 3348

    def test_needed_tokens_rounds_up_fractional_margin_product(self):
        derivation = OllamaNumCtxDerivation(
            models=frozenset({"m"}), num_ctx_max=100_000, response_headroom=0, safety_margin=1.3,
        )
        # 7 * 1.3 = 9.1 -> ceil to 10, not truncated to 9.
        assert derivation.needed_tokens(7) == 10

    def test_applies_to_checks_model_membership(self):
        derivation = OllamaNumCtxDerivation(
            models=frozenset({"ministral-3:3b", "ministral-3:8b"}),
            num_ctx_max=32768,
            response_headroom=0,
            safety_margin=1.0,
        )
        assert derivation.applies_to("ministral-3:3b") is True
        assert derivation.applies_to("ministral-3:8b") is True
        assert derivation.applies_to("some-other-model") is False


# ---------------------------------------------------------------------------
# 2. ContextWindowGuardHandler's derivation branch
# ---------------------------------------------------------------------------


class TestGuardDerivationWithinCeilingPasses:
    def test_prompt_within_derived_ceiling_does_not_raise(self):
        derivation = OllamaNumCtxDerivation(
            models=frozenset({"ministral-3:8b"}),
            num_ctx_max=100_000,
            response_headroom=100,
            safety_margin=1.0,
        )
        handler = ContextWindowGuardHandler(ollama_num_ctx_derivation=derivation)
        # Should not raise.
        handler.on_chat_model_start(
            {"kwargs": {"model": "ministral-3:8b"}},
            _messages("a short prompt"),
            run_id=uuid.uuid4(),
            metadata={"langgraph_node": "Market Analyst"},
        )

    def test_prompt_exactly_at_the_ceiling_does_not_raise(self):
        text = "x" * 40  # heuristic: 40 // 4 == 10 tokens
        derivation = OllamaNumCtxDerivation(
            models=frozenset({"custom-model"}),
            num_ctx_max=10,  # ceil(10 * 1.0) + 0 == 10 -> not > ceiling
            response_headroom=0,
            safety_margin=1.0,
        )
        handler = ContextWindowGuardHandler(ollama_num_ctx_derivation=derivation)
        handler.on_chat_model_start(
            {"kwargs": {"model": "custom-model"}},
            _messages(text),
            run_id=uuid.uuid4(),
            metadata={"langgraph_node": "Trader"},
        )


class TestGuardDerivationOverCeilingRaises:
    def test_over_ceiling_raises_dedicated_exception(self):
        derivation = OllamaNumCtxDerivation(
            models=frozenset({"ministral-3:8b"}),
            num_ctx_max=50,
            response_headroom=0,
            safety_margin=1.0,
        )
        handler = ContextWindowGuardHandler(ollama_num_ctx_derivation=derivation)
        long_prompt = "word " * 1000  # heuristic: well over 50 tokens

        with pytest.raises(PromptContextOverflowError) as exc_info:
            handler.on_chat_model_start(
                {"kwargs": {"model": "ministral-3:8b"}},
                _messages(long_prompt),
                run_id=uuid.uuid4(),
                metadata={"langgraph_node": "Portfolio Manager"},
            )

        error = exc_info.value
        assert error.agent == "Portfolio Manager"
        assert error.model == "ministral-3:8b"
        assert error.context_window == 50  # the ceiling, ollama_num_ctx_max
        assert error.response_headroom == 0
        assert error.needed_tokens == error.adjusted_tokens
        assert error.prompt_tokens_estimated > 50
        assert error.token_count_method == "heuristic_chars_per_token"

    def test_response_headroom_is_included_in_the_needed_figure(self):
        text = "x" * 400  # heuristic: 100 tokens
        derivation = OllamaNumCtxDerivation(
            models=frozenset({"custom-model"}),
            num_ctx_max=150,
            response_headroom=100,
            safety_margin=1.0,
        )
        handler = ContextWindowGuardHandler(ollama_num_ctx_derivation=derivation)
        with pytest.raises(PromptContextOverflowError) as exc_info:
            handler.on_chat_model_start(
                {"kwargs": {"model": "custom-model"}},
                _messages(text),
                run_id=uuid.uuid4(),
                metadata={"langgraph_node": "Researcher"},
            )
        error = exc_info.value
        assert error.adjusted_tokens == 100
        assert error.response_headroom == 100
        assert error.needed_tokens == 200  # 100 + 100 > 150 ceiling

    def test_error_message_is_actionable(self):
        derivation = OllamaNumCtxDerivation(
            models=frozenset({"ministral-3:8b"}),
            num_ctx_max=50,
            response_headroom=10,
            safety_margin=1.0,
        )
        handler = ContextWindowGuardHandler(ollama_num_ctx_derivation=derivation)
        long_prompt = "word " * 1000

        with pytest.raises(PromptContextOverflowError) as exc_info:
            handler.on_chat_model_start(
                {"kwargs": {"model": "ministral-3:8b"}},
                _messages(long_prompt),
                run_id=uuid.uuid4(),
                metadata={"langgraph_node": "Portfolio Manager"},
            )

        message = str(exc_info.value)
        assert "Portfolio Manager" in message
        assert "ministral-3:8b" in message
        assert "ollama_num_ctx_max" in message
        assert "50" in message  # the ceiling
        assert str(exc_info.value.prompt_tokens_estimated) in message
        assert "TRADINGAGENTS_OLLAMA_NUM_CTX_MAX" in message
        assert "TRADINGAGENTS_CONTEXT_WINDOW_CHECK_ENABLED" in message

    def test_model_not_covered_by_derivation_falls_back_to_context_windows(self):
        # A model not in derivation.models isn't derived-checked at all; it
        # falls through to the (empty here) static context_windows mapping
        # and passes through unchecked, exactly like the #149 "unknown
        # model" case.
        derivation = OllamaNumCtxDerivation(
            models=frozenset({"ministral-3:8b"}),  # only 8b is covered
            num_ctx_max=10,
            response_headroom=0,
            safety_margin=1.0,
        )
        handler = ContextWindowGuardHandler(ollama_num_ctx_derivation=derivation)
        long_prompt = "word " * 1000
        handler.on_chat_model_start(
            {"kwargs": {"model": "ministral-3:3b"}},
            _messages(long_prompt),
            run_id=uuid.uuid4(),
            metadata={"langgraph_node": "Trader"},
        )

    def test_disabled_guard_never_raises_even_over_ceiling(self):
        derivation = OllamaNumCtxDerivation(
            models=frozenset({"ministral-3:8b"}), num_ctx_max=10, response_headroom=0, safety_margin=1.0,
        )
        handler = ContextWindowGuardHandler(ollama_num_ctx_derivation=derivation, enabled=False)
        long_prompt = "word " * 1000
        handler.on_chat_model_start(
            {"kwargs": {"model": "ministral-3:8b"}},
            _messages(long_prompt),
            run_id=uuid.uuid4(),
            metadata={"langgraph_node": "Portfolio Manager"},
        )


class TestGuardDerivationErrorReachesJsonl:
    def test_oversize_derivation_call_writes_a_jsonl_error_record(self, tmp_path):
        log_path = tmp_path / "llm_calls.jsonl"
        log_handler = LLMCallLogHandler(log_path)
        log_handler.start_run(ticker="AAPL", date="2024-01-15")

        derivation = OllamaNumCtxDerivation(
            models=frozenset({"ministral-3:8b"}), num_ctx_max=50, response_headroom=0, safety_margin=1.0,
        )
        guard = ContextWindowGuardHandler(
            ollama_num_ctx_derivation=derivation, log_handler=log_handler,
        )

        long_prompt = "word " * 1000
        with pytest.raises(PromptContextOverflowError):
            guard.on_chat_model_start(
                {"kwargs": {"model": "ministral-3:8b"}},
                _messages(long_prompt),
                run_id=uuid.uuid4(),
                metadata={"langgraph_node": "Portfolio Manager"},
            )

        lines = log_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])

        assert record["agent"] == "Portfolio Manager"
        assert record["model"] == "ministral-3:8b"
        assert record["input_tokens"] is None
        assert record["output_tokens"] is None
        assert record["error"] is not None
        assert "ollama_num_ctx_max" in record["error"]
        assert record["ollama_num_ctx"] is None  # never dispatched


# ---------------------------------------------------------------------------
# 3. LLMCallLogHandler records the derived num_ctx on successful calls
# ---------------------------------------------------------------------------


class TestLLMCallLogHandlerRecordsDerivedNumCtx:
    def _end_the_call(self, handler: LLMCallLogHandler, run_id) -> None:
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, LLMResult

        response = LLMResult(generations=[[ChatGeneration(message=AIMessage(content="ok"))]])
        handler.on_llm_end(response, run_id=run_id)

    def test_ollama_num_ctx_field_present_for_a_derived_model(self, tmp_path):
        log_path = tmp_path / "llm_calls.jsonl"
        derivation = OllamaNumCtxDerivation(
            models=frozenset({"ministral-3:8b"}),
            num_ctx_max=100_000,
            response_headroom=100,
            safety_margin=1.0,
        )
        handler = LLMCallLogHandler(log_path, ollama_num_ctx_derivation=derivation)
        handler.start_run(ticker="AAPL", date="2024-01-15")

        run_id = uuid.uuid4()
        text = "x" * 400  # heuristic: 100 tokens
        handler.on_chat_model_start(
            {"kwargs": {"model": "ministral-3:8b"}},
            _messages(text),
            run_id=run_id,
            metadata={"langgraph_node": "Portfolio Manager"},
        )
        self._end_the_call(handler, run_id)

        records = handler.get_records()
        assert len(records) == 1
        # ceil(100 * 1.0) + 100 == 200, well under num_ctx_max.
        assert records[0]["ollama_num_ctx"] == 200

    def test_ollama_num_ctx_is_none_when_no_derivation_configured(self, tmp_path):
        log_path = tmp_path / "llm_calls.jsonl"
        handler = LLMCallLogHandler(log_path)
        handler.start_run(ticker="AAPL", date="2024-01-15")

        run_id = uuid.uuid4()
        handler.on_chat_model_start(
            {"kwargs": {"model": "gpt-5.5"}},
            _messages("hello"),
            run_id=run_id,
            metadata={"langgraph_node": "Trader"},
        )
        self._end_the_call(handler, run_id)

        assert handler.get_records()[0]["ollama_num_ctx"] is None

    def test_ollama_num_ctx_is_none_for_a_model_not_covered_by_derivation(self, tmp_path):
        log_path = tmp_path / "llm_calls.jsonl"
        derivation = OllamaNumCtxDerivation(
            models=frozenset({"ministral-3:8b"}),
            num_ctx_max=100_000,
            response_headroom=0,
            safety_margin=1.0,
        )
        handler = LLMCallLogHandler(log_path, ollama_num_ctx_derivation=derivation)
        handler.start_run(ticker="AAPL", date="2024-01-15")

        run_id = uuid.uuid4()
        handler.on_chat_model_start(
            {"kwargs": {"model": "some-other-model"}},
            _messages("hello"),
            run_id=run_id,
            metadata={"langgraph_node": "Trader"},
        )
        self._end_the_call(handler, run_id)

        assert handler.get_records()[0]["ollama_num_ctx"] is None


# ---------------------------------------------------------------------------
# 4. TradingAgentsGraph wiring
# ---------------------------------------------------------------------------


class TestBuildOllamaNumCtxDerivation:
    def _mock_graph(self, config):
        mock_graph = MagicMock()
        mock_graph.config = config
        return mock_graph

    def test_derives_from_config_when_ollama_num_ctx_unset(self):
        from tradingagents.graph.trading_graph import TradingAgentsGraph

        mock_graph = self._mock_graph({
            "llm_provider": "ollama",
            "quick_think_llm": "ministral-3:3b",
            "deep_think_llm": "ministral-3:8b",
            "ollama_num_ctx": None,
            "ollama_num_ctx_max": 16384,
            "ollama_num_ctx_response_headroom": 512,
            "context_window_safety_margin": 1.5,
        })
        derivation = TradingAgentsGraph._build_ollama_num_ctx_derivation(mock_graph)
        assert derivation is not None
        assert derivation.models == frozenset({"ministral-3:3b", "ministral-3:8b"})
        assert derivation.num_ctx_max == 16384
        assert derivation.response_headroom == 512
        assert derivation.safety_margin == 1.5

    def test_explicit_ollama_num_ctx_disables_derivation(self):
        from tradingagents.graph.trading_graph import TradingAgentsGraph

        mock_graph = self._mock_graph({
            "llm_provider": "ollama",
            "quick_think_llm": "ministral-3:3b",
            "deep_think_llm": "ministral-3:8b",
            "ollama_num_ctx": 8192,
        })
        assert TradingAgentsGraph._build_ollama_num_ctx_derivation(mock_graph) is None

    def test_non_ollama_provider_never_derives(self):
        from tradingagents.graph.trading_graph import TradingAgentsGraph

        mock_graph = self._mock_graph({
            "llm_provider": "openai",
            "quick_think_llm": "gpt-5.5",
            "deep_think_llm": "gpt-5.5",
            "ollama_num_ctx": None,
        })
        assert TradingAgentsGraph._build_ollama_num_ctx_derivation(mock_graph) is None

    def test_uses_default_config_values(self):
        from tradingagents.default_config import DEFAULT_CONFIG
        from tradingagents.graph.trading_graph import TradingAgentsGraph

        mock_graph = self._mock_graph({
            **DEFAULT_CONFIG,
            "llm_provider": "ollama",
        })
        derivation = TradingAgentsGraph._build_ollama_num_ctx_derivation(mock_graph)
        assert derivation is not None
        assert derivation.num_ctx_max == DEFAULT_CONFIG["ollama_num_ctx_max"]
        assert derivation.response_headroom == DEFAULT_CONFIG["ollama_num_ctx_response_headroom"]


class TestGetProviderKwargsDerivation:
    def test_forwards_derivation_when_ollama_num_ctx_unset(self):
        from tradingagents.graph.trading_graph import (
            OllamaNumCtxDerivation as _OD,
            TradingAgentsGraph,
        )

        mock_graph = MagicMock(spec=TradingAgentsGraph)
        mock_graph.config = {
            "llm_provider": "ollama",
            "ollama_num_ctx": None,
            "quick_think_llm": "ministral-3:3b",
            "deep_think_llm": "ministral-3:8b",
        }
        mock_graph._build_ollama_num_ctx_derivation.return_value = _OD(
            models=frozenset({"ministral-3:3b", "ministral-3:8b"}),
            num_ctx_max=32768,
            response_headroom=2048,
            safety_margin=1.3,
        )
        kwargs = TradingAgentsGraph._get_provider_kwargs(mock_graph)
        assert "num_ctx" not in kwargs
        assert kwargs["ollama_num_ctx_derivation"] is mock_graph._build_ollama_num_ctx_derivation.return_value

    def test_explicit_num_ctx_wins_and_omits_derivation(self):
        from tradingagents.graph.trading_graph import TradingAgentsGraph

        mock_graph = MagicMock(spec=TradingAgentsGraph)
        mock_graph.config = {"llm_provider": "ollama", "ollama_num_ctx": 16384}
        kwargs = TradingAgentsGraph._get_provider_kwargs(mock_graph)
        assert kwargs["num_ctx"] == 16384
        assert "ollama_num_ctx_derivation" not in kwargs

    def test_non_ollama_provider_gets_neither(self):
        from tradingagents.graph.trading_graph import TradingAgentsGraph

        mock_graph = MagicMock(spec=TradingAgentsGraph)
        mock_graph.config = {"llm_provider": "openai", "ollama_num_ctx": None}
        kwargs = TradingAgentsGraph._get_provider_kwargs(mock_graph)
        assert "num_ctx" not in kwargs
        assert "ollama_num_ctx_derivation" not in kwargs


# ---------------------------------------------------------------------------
# 5. NormalizedChatOllama._chat_params plumbing (issue #169)
# ---------------------------------------------------------------------------


class TestNormalizedChatOllamaChatParams:
    def test_derived_num_ctx_reaches_options(self):
        from tradingagents.llm_clients.factory import create_llm_client

        derivation = OllamaNumCtxDerivation(
            models=frozenset({"ministral-3:8b"}),
            num_ctx_max=100_000,
            response_headroom=100,
            safety_margin=1.0,
        )
        llm = create_llm_client(
            provider="ollama",
            model="ministral-3:8b",
            ollama_num_ctx_derivation=derivation,
        ).get_llm()

        # _chat_params always counts via the tiktoken path (ChatOllama's
        # _llm_type, "chat-ollama", is in _TIKTOKEN_LLM_TYPES -- issue #169
        # acceptance criterion 9), not the chars/4 heuristic, so derive the
        # expected token count the same way rather than assuming a heuristic
        # figure.
        from tradingagents.llm_call_log import _count_prompt_tokens

        text = "x" * 400
        token_count, _ = _count_prompt_tokens(text, "ministral-3:8b", "chat-ollama")
        params = llm._chat_params([HumanMessage(content=text)])
        assert params["options"]["num_ctx"] == token_count + 100

    def test_derived_num_ctx_is_clamped_to_the_ceiling(self):
        from tradingagents.llm_clients.factory import create_llm_client

        derivation = OllamaNumCtxDerivation(
            models=frozenset({"ministral-3:8b"}),
            num_ctx_max=50,
            response_headroom=0,
            safety_margin=1.0,
        )
        llm = create_llm_client(
            provider="ollama",
            model="ministral-3:8b",
            ollama_num_ctx_derivation=derivation,
        ).get_llm()

        long_text = "word " * 1000  # well over 50 tokens
        params = llm._chat_params([HumanMessage(content=long_text)])
        assert params["options"]["num_ctx"] == 50

    def test_no_derivation_leaves_options_without_num_ctx(self):
        from tradingagents.llm_clients.factory import create_llm_client

        llm = create_llm_client(provider="ollama", model="ministral-3:8b").get_llm()
        params = llm._chat_params([HumanMessage(content="hi")])
        assert "num_ctx" not in params.get("options", {})

    def test_explicit_num_ctx_and_derivation_are_mutually_exclusive_in_practice(self):
        # TradingAgentsGraph._get_provider_kwargs never sets both at once (see
        # TestGetProviderKwargsDerivation above), but this documents that if
        # only num_ctx is set, the static value survives untouched since no
        # derivation attribute is attached.
        from tradingagents.llm_clients.factory import create_llm_client

        llm = create_llm_client(
            provider="ollama", model="ministral-3:8b", num_ctx=16384
        ).get_llm()
        params = llm._chat_params([HumanMessage(content="word " * 1000)])
        assert params["options"]["num_ctx"] == 16384

    def test_model_not_covered_by_derivation_is_unaffected(self):
        from tradingagents.llm_clients.factory import create_llm_client

        derivation = OllamaNumCtxDerivation(
            models=frozenset({"some-other-model"}),
            num_ctx_max=50,
            response_headroom=0,
            safety_margin=1.0,
        )
        llm = create_llm_client(
            provider="ollama",
            model="ministral-3:8b",
            ollama_num_ctx_derivation=derivation,
        ).get_llm()
        params = llm._chat_params([HumanMessage(content="word " * 1000)])
        assert "num_ctx" not in params.get("options", {})

    def test_other_providers_never_route_through_ollama_client(self):
        from tradingagents.llm_clients.factory import create_llm_client
        from tradingagents.llm_clients.ollama_client import NormalizedChatOllama

        # Even if a derivation object were (incorrectly) forwarded to a
        # non-ollama provider, that provider's client never constructs a
        # NormalizedChatOllama, so there is no _chat_params override to act
        # on it.
        llm = create_llm_client(
            provider="openai", model="gpt-5.5", api_key="placeholder"
        ).get_llm()
        assert not isinstance(llm, NormalizedChatOllama)


# ---------------------------------------------------------------------------
# 6. Ollama think mode (issue #155, re-pointed at ChatOllama's native
#    `reasoning`/`think` field by issue #169's three-state contract)
# ---------------------------------------------------------------------------


class TestOllamaThinkMode:
    def test_think_key_absent_defaults_to_reasoning_false(self):
        """A caller that never sets ollama_think gets the documented effective default: False."""
        from tradingagents.llm_clients.factory import create_llm_client

        llm = create_llm_client(provider="ollama", model="ministral-3:8b").get_llm()
        assert llm.reasoning is False

    def test_think_true(self):
        from tradingagents.llm_clients.factory import create_llm_client

        llm = create_llm_client(
            provider="ollama", model="ministral-3:8b", ollama_think=True,
        ).get_llm()
        assert llm.reasoning is True

    def test_think_false_explicit(self):
        from tradingagents.llm_clients.factory import create_llm_client

        llm = create_llm_client(
            provider="ollama", model="ministral-3:8b", ollama_think=False,
        ).get_llm()
        assert llm.reasoning is False

    def test_think_none_leaves_reasoning_unset(self):
        """ollama_think=None must be distinguishable from False -- no think field at all."""
        from tradingagents.llm_clients.factory import create_llm_client

        llm = create_llm_client(
            provider="ollama", model="ministral-3:8b", ollama_think=None,
        ).get_llm()
        assert llm.reasoning is None

    def test_think_field_dropped_from_chat_params_when_reasoning_none(self):
        from tradingagents.llm_clients.factory import create_llm_client

        llm = create_llm_client(
            provider="ollama", model="ministral-3:8b", ollama_think=None,
        ).get_llm()
        params = llm._chat_params([HumanMessage(content="hi")])
        # ChatOllama._chat_params always includes a "think" key (from
        # self.reasoning); the underlying ollama-python client drops it from
        # the actual request body via model_dump(exclude_none=True) -- see
        # ollama_client.py's module docstring. None here is what makes that
        # downstream stripping happen.
        assert params["think"] is None

    def test_think_field_explicit_false_in_chat_params(self):
        from tradingagents.llm_clients.factory import create_llm_client

        llm = create_llm_client(
            provider="ollama", model="ministral-3:8b", ollama_think=False,
        ).get_llm()
        params = llm._chat_params([HumanMessage(content="hi")])
        assert params["think"] is False

    def test_think_and_num_ctx_compose(self):
        from tradingagents.llm_clients.factory import create_llm_client

        llm = create_llm_client(
            provider="ollama",
            model="ministral-3:8b",
            num_ctx=16384,
            ollama_think=True,
        ).get_llm()
        params = llm._chat_params([HumanMessage(content="hi")])
        assert params["think"] is True
        assert params["options"]["num_ctx"] == 16384

    def test_think_preserved_through_derivation(self):
        from tradingagents.llm_clients.factory import create_llm_client

        derivation = OllamaNumCtxDerivation(
            models=frozenset({"ministral-3:8b"}),
            num_ctx_max=100_000,
            response_headroom=100,
            safety_margin=1.0,
        )
        llm = create_llm_client(
            provider="ollama",
            model="ministral-3:8b",
            ollama_think=True,
            ollama_num_ctx_derivation=derivation,
        ).get_llm()
        params = llm._chat_params([HumanMessage(content="word " * 100)])
        assert params["think"] is True
        assert "num_ctx" in params.get("options", {})

    def test_non_ollama_provider_never_gets_reasoning(self):
        """ollama_think must not leak into another provider's client as a truthy `reasoning`.

        ChatOpenAI happens to have its own unrelated `reasoning` field (for
        OpenAI's reasoning-effort config); the assertion here is that
        `ollama_think=True` never sets it, not that the attribute is absent.
        """
        from tradingagents.llm_clients.factory import create_llm_client

        llm = create_llm_client(
            provider="openai",
            model="gpt-4o",
            ollama_think=True,  # ignored for non-ollama
            api_key="placeholder",
        ).get_llm()
        assert getattr(llm, "reasoning", None) is not True


# ---------------------------------------------------------------------------
# 7. Per-agent response-headroom overrides (issue #170)
# ---------------------------------------------------------------------------


class TestPerAgentResponseHeadroomOverrides:
    def test_new_default_headroom_is_4096(self):
        """Issue #170: raised from 2048 to 4096."""
        from tradingagents.default_config import DEFAULT_CONFIG

        assert DEFAULT_CONFIG["ollama_num_ctx_response_headroom"] == 4096

    def test_new_overrides_config_key_exists_and_defaults_to_empty_dict(self):
        """Issue #170: new config key with empty dict default."""
        from tradingagents.default_config import DEFAULT_CONFIG

        assert "ollama_num_ctx_response_headroom_overrides" in DEFAULT_CONFIG
        assert DEFAULT_CONFIG["ollama_num_ctx_response_headroom_overrides"] == {}

    def test_needed_tokens_uses_global_headroom_when_agent_not_in_overrides(self):
        """Override miss: agent not in overrides dict, falls back to global."""
        derivation = OllamaNumCtxDerivation(
            models=frozenset({"ministral-3:8b"}),
            num_ctx_max=100_000,
            response_headroom=1000,  # global headroom
            safety_margin=1.0,
            response_headroom_overrides={"Fundamentals Analyst": 5000},
        )
        # "Market Analyst" not in overrides, should use global 1000
        assert derivation.needed_tokens(100, agent="Market Analyst") == 1100

    def test_needed_tokens_uses_override_when_agent_found(self):
        """Override hit: agent in overrides dict uses the override value."""
        derivation = OllamaNumCtxDerivation(
            models=frozenset({"ministral-3:8b"}),
            num_ctx_max=100_000,
            response_headroom=1000,  # global headroom
            safety_margin=1.0,
            response_headroom_overrides={"Fundamentals Analyst": 5000},
        )
        # "Fundamentals Analyst" in overrides, should use 5000
        assert derivation.needed_tokens(100, agent="Fundamentals Analyst") == 5100

    def test_unknown_agent_name_is_silently_ignored(self):
        """Unknown agent name: ignored silently, falls back to global headroom."""
        derivation = OllamaNumCtxDerivation(
            models=frozenset({"ministral-3:8b"}),
            num_ctx_max=100_000,
            response_headroom=2000,
            safety_margin=1.0,
            response_headroom_overrides={"Market Analyst": 3000},
        )
        # Nonexistent agent should not raise, just use global
        result = derivation.needed_tokens(100, agent="Nonexistent Agent")
        assert result == 2100  # 100 + 2000, using global

    def test_none_agent_uses_global_headroom(self):
        """When agent is None, uses global headroom regardless of overrides."""
        derivation = OllamaNumCtxDerivation(
            models=frozenset({"ministral-3:8b"}),
            num_ctx_max=100_000,
            response_headroom=1000,
            safety_margin=1.0,
            response_headroom_overrides={"Market Analyst": 5000},
        )
        # No agent specified: use global even though Market Analyst is in overrides
        assert derivation.needed_tokens(100, agent=None) == 1100

    def test_overrides_build_from_config(self):
        """OllamaNumCtxDerivation built from config includes overrides."""
        from tradingagents.graph.trading_graph import TradingAgentsGraph

        mock_graph = MagicMock()
        mock_graph.config = {
            "llm_provider": "ollama",
            "quick_think_llm": "ministral-3:3b",
            "deep_think_llm": "ministral-3:8b",
            "ollama_num_ctx": None,
            "ollama_num_ctx_max": 32768,
            "ollama_num_ctx_response_headroom": 4096,
            "ollama_num_ctx_response_headroom_overrides": {
                "Fundamentals Analyst": 5000,
                "Market Analyst": 4500,
            },
            "context_window_safety_margin": 1.3,
        }
        derivation = TradingAgentsGraph._build_ollama_num_ctx_derivation(mock_graph)
        assert derivation is not None
        assert derivation.response_headroom_overrides == {
            "Fundamentals Analyst": 5000,
            "Market Analyst": 4500,
        }

    def test_override_applied_in_guard_handler(self):
        """ContextWindowGuardHandler uses override when computing needed tokens."""
        text = "x" * 800  # heuristic: 200 tokens
        derivation = OllamaNumCtxDerivation(
            models=frozenset({"custom-model"}),
            num_ctx_max=300,
            response_headroom=50,
            safety_margin=1.0,
            response_headroom_overrides={"Portfolio Manager": 150},
        )
        handler = ContextWindowGuardHandler(ollama_num_ctx_derivation=derivation)

        # With override, Portfolio Manager needs 200 + 150 = 350 > 300 -> raises
        with pytest.raises(PromptContextOverflowError) as exc_info:
            handler.on_chat_model_start(
                {"kwargs": {"model": "custom-model"}},
                _messages(text),
                run_id=uuid.uuid4(),
                metadata={"langgraph_node": "Portfolio Manager"},
            )
        error = exc_info.value
        assert error.agent == "Portfolio Manager"
        assert error.response_headroom == 150  # the override

    def test_override_not_applied_for_different_agent(self):
        """Override for one agent doesn't affect another agent."""
        text = "x" * 800  # heuristic: 200 tokens
        derivation = OllamaNumCtxDerivation(
            models=frozenset({"custom-model"}),
            num_ctx_max=300,
            response_headroom=50,
            safety_margin=1.0,
            response_headroom_overrides={"Portfolio Manager": 150},
        )
        handler = ContextWindowGuardHandler(ollama_num_ctx_derivation=derivation)

        # Trader uses global headroom, so 200 + 50 = 250 <= 300 -> no raise
        handler.on_chat_model_start(
            {"kwargs": {"model": "custom-model"}},
            _messages(text),
            run_id=uuid.uuid4(),
            metadata={"langgraph_node": "Trader"},
        )
        # Should not raise

    def test_override_applied_in_log_handler(self, tmp_path):
        """LLMCallLogHandler uses override when recording num_ctx."""
        log_path = tmp_path / "llm_calls.jsonl"
        derivation = OllamaNumCtxDerivation(
            models=frozenset({"ministral-3:8b"}),
            num_ctx_max=100_000,
            response_headroom=100,
            safety_margin=1.0,
            response_headroom_overrides={"Portfolio Manager": 1000},
        )
        handler = LLMCallLogHandler(log_path, ollama_num_ctx_derivation=derivation)
        handler.start_run(ticker="AAPL", date="2024-01-15")

        run_id = uuid.uuid4()
        text = "x" * 400  # heuristic: 100 tokens
        handler.on_chat_model_start(
            {"kwargs": {"model": "ministral-3:8b"}},
            _messages(text),
            run_id=run_id,
            metadata={"langgraph_node": "Portfolio Manager"},
        )

        # Manually end the call
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, LLMResult

        response = LLMResult(generations=[[ChatGeneration(message=AIMessage(content="ok"))]])
        handler.on_llm_end(response, run_id=run_id)

        records = handler.get_records()
        assert len(records) == 1
        # Portfolio Manager override is 1000, so 100 + 1000 = 1100
        assert records[0]["ollama_num_ctx"] == 1100

    def test_headroom_for_is_the_single_lookup_shared_by_needed_tokens(self):
        """headroom_for is a hit/miss/unknown/None-agent lookup, and needed_tokens

        (issue #170 re-dispatch) delegates to it rather than re-implementing the
        dict lookup inline, so this table doubles as coverage for needed_tokens'
        headroom resolution too.
        """
        derivation = OllamaNumCtxDerivation(
            models=frozenset({"ministral-3:8b"}),
            num_ctx_max=100_000,
            response_headroom=1000,
            safety_margin=1.0,
            response_headroom_overrides={"Fundamentals Analyst": 5000},
        )
        assert derivation.headroom_for("Fundamentals Analyst") == 5000  # hit
        assert derivation.headroom_for("Market Analyst") == 1000  # miss -> global
        assert derivation.headroom_for("Nonexistent Agent") == 1000  # unknown -> global
        assert derivation.headroom_for(None) == 1000  # no agent -> global
        assert derivation.headroom_for("") == 1000  # falsy agent -> global

        assert derivation.needed_tokens(100, agent="Fundamentals Analyst") == 5100
        assert derivation.needed_tokens(100, agent="Market Analyst") == 1100


# ---------------------------------------------------------------------------
# 8. NormalizedChatOllama._chat_params per-agent override (issue #170 re-dispatch)
#
# The original #170 implementation wired the override into LLMCallLogHandler and
# ContextWindowGuardHandler but never into _chat_params -- the one call site that
# actually attaches num_ctx to the outgoing Ollama request. So a configured
# override changed what got logged/aborted-on but not what was actually sent.
# These tests cover the previously-missing path directly, plus an end-to-end
# check that all three consumers agree for the same agent.
# ---------------------------------------------------------------------------


class TestChatParamsPerAgentOverride:
    def _build_llm(self, derivation):
        from tradingagents.llm_clients.factory import create_llm_client

        return create_llm_client(
            provider="ollama",
            model="ministral-3:8b",
            ollama_num_ctx_derivation=derivation,
        ).get_llm()

    def test_chat_params_uses_override_when_langgraph_node_metadata_present(self):
        """A per-agent override actually reaches the outgoing request's num_ctx.

        The LangGraph node name reaches _chat_params via the ambient
        RunnableConfig (langchain_core.runnables.config.var_child_runnable_config)
        rather than an explicit argument -- see NormalizedChatOllama's class
        docstring. `set_config_context` + `ctx.run(...)` is the same mechanism
        LangGraph itself uses to invoke a node's callable, so driving it directly
        here exercises the real propagation path without needing a full compiled
        graph.
        """
        from langchain_core.runnables.config import set_config_context

        derivation = OllamaNumCtxDerivation(
            models=frozenset({"ministral-3:8b"}),
            num_ctx_max=100_000,
            response_headroom=100,
            safety_margin=1.0,
            response_headroom_overrides={"Fundamentals Analyst": 5000},
        )
        llm = self._build_llm(derivation)
        text = "x" * 400
        token_count, _ = _count_prompt_tokens(text, "ministral-3:8b", "chat-ollama")

        with set_config_context({"metadata": {"langgraph_node": "Fundamentals Analyst"}}) as ctx:
            params = ctx.run(llm._chat_params, [HumanMessage(content=text)])

        assert params["options"]["num_ctx"] == token_count + 5000

    def test_chat_params_falls_back_to_global_for_agent_without_override(self):
        from langchain_core.runnables.config import set_config_context

        derivation = OllamaNumCtxDerivation(
            models=frozenset({"ministral-3:8b"}),
            num_ctx_max=100_000,
            response_headroom=100,
            safety_margin=1.0,
            response_headroom_overrides={"Fundamentals Analyst": 5000},
        )
        llm = self._build_llm(derivation)
        text = "x" * 400
        token_count, _ = _count_prompt_tokens(text, "ministral-3:8b", "chat-ollama")

        with set_config_context({"metadata": {"langgraph_node": "Market Analyst"}}) as ctx:
            params = ctx.run(llm._chat_params, [HumanMessage(content=text)])

        assert params["options"]["num_ctx"] == token_count + 100

    def test_chat_params_falls_back_to_global_with_no_ambient_config(self):
        """No LangGraph context at all (e.g. a direct/test invocation): global headroom."""
        derivation = OllamaNumCtxDerivation(
            models=frozenset({"ministral-3:8b"}),
            num_ctx_max=100_000,
            response_headroom=100,
            safety_margin=1.0,
            response_headroom_overrides={"Fundamentals Analyst": 5000},
        )
        llm = self._build_llm(derivation)
        text = "x" * 400
        token_count, _ = _count_prompt_tokens(text, "ministral-3:8b", "chat-ollama")

        params = llm._chat_params([HumanMessage(content=text)])
        assert params["options"]["num_ctx"] == token_count + 100

    def test_all_three_consumers_agree_for_an_overridden_agent(self, tmp_path):
        """Log handler, guard handler, and _chat_params must compute the same
        num_ctx for the same agent + prompt -- the exact drift issue #170's
        re-dispatch was about (the override reached two of three consumers but
        not the third).
        """
        from langchain_core.runnables.config import set_config_context

        agent = "Fundamentals Analyst"
        derivation = OllamaNumCtxDerivation(
            models=frozenset({"ministral-3:8b"}),
            num_ctx_max=100_000,
            response_headroom=100,
            safety_margin=1.0,
            response_headroom_overrides={agent: 5000},
        )
        text = "x" * 400
        token_count, _ = _count_prompt_tokens(text, "ministral-3:8b", "chat-ollama")

        # 1. _chat_params (the actual outgoing request)
        llm = self._build_llm(derivation)
        with set_config_context({"metadata": {"langgraph_node": agent}}) as ctx:
            params = ctx.run(llm._chat_params, [HumanMessage(content=text)])
        chat_params_num_ctx = params["options"]["num_ctx"]

        # 2. LLMCallLogHandler (the audit record). invocation_params carries
        # "_type": "chat-ollama" here (as BaseChatModel._get_invocation_params
        # does for a real call) so this path's tokenizer selection matches
        # _chat_params' hardcoded OLLAMA_LLM_TYPE above -- otherwise the two
        # would legitimately disagree on token_count via a different mechanism
        # (tiktoken vs. the chars/4 heuristic) that has nothing to do with #170.
        log_path = tmp_path / "llm_calls.jsonl"
        log_handler = LLMCallLogHandler(log_path, ollama_num_ctx_derivation=derivation)
        log_handler.start_run(ticker="AAPL", date="2024-01-15")
        run_id = uuid.uuid4()
        log_handler.on_chat_model_start(
            {"kwargs": {"model": "ministral-3:8b"}},
            _messages(text),
            run_id=run_id,
            metadata={"langgraph_node": agent},
            invocation_params={"_type": "chat-ollama"},
        )
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, LLMResult

        log_handler.on_llm_end(
            LLMResult(generations=[[ChatGeneration(message=AIMessage(content="ok"))]]),
            run_id=run_id,
        )
        logged_num_ctx = log_handler.get_records()[0]["ollama_num_ctx"]

        # 3. ContextWindowGuardHandler's abort threshold: same `needed` figure,
        # just compared against num_ctx_max instead of clamped/sent. Verify it
        # doesn't raise when num_ctx_max comfortably covers the shared figure,
        # and that shrinking num_ctx_max to just below it raises with the same
        # headroom value reported in the error.
        guard = ContextWindowGuardHandler(ollama_num_ctx_derivation=derivation)
        guard.on_chat_model_start(
            {"kwargs": {"model": "ministral-3:8b"}},
            _messages(text),
            run_id=uuid.uuid4(),
            metadata={"langgraph_node": agent},
            invocation_params={"_type": "chat-ollama"},
        )  # does not raise: num_ctx_max (100_000) >> needed

        tight_derivation = OllamaNumCtxDerivation(
            models=frozenset({"ministral-3:8b"}),
            num_ctx_max=chat_params_num_ctx - 1,
            response_headroom=100,
            safety_margin=1.0,
            response_headroom_overrides={agent: 5000},
        )
        tight_guard = ContextWindowGuardHandler(ollama_num_ctx_derivation=tight_derivation)
        with pytest.raises(PromptContextOverflowError) as exc_info:
            tight_guard.on_chat_model_start(
                {"kwargs": {"model": "ministral-3:8b"}},
                _messages(text),
                run_id=uuid.uuid4(),
                metadata={"langgraph_node": agent},
                invocation_params={"_type": "chat-ollama"},
            )

        assert chat_params_num_ctx == logged_num_ctx == token_count + 5000
        assert exc_info.value.response_headroom == 5000

    def test_all_three_consumers_agree_for_an_agent_without_override(self, tmp_path):
        """Same as above, but for an agent with no configured override: all three
        must land on the global headroom, not silently diverge.
        """
        from langchain_core.runnables.config import set_config_context

        agent = "Trader"
        derivation = OllamaNumCtxDerivation(
            models=frozenset({"ministral-3:8b"}),
            num_ctx_max=100_000,
            response_headroom=100,
            safety_margin=1.0,
            response_headroom_overrides={"Fundamentals Analyst": 5000},
        )
        text = "x" * 400
        token_count, _ = _count_prompt_tokens(text, "ministral-3:8b", "chat-ollama")

        llm = self._build_llm(derivation)
        with set_config_context({"metadata": {"langgraph_node": agent}}) as ctx:
            params = ctx.run(llm._chat_params, [HumanMessage(content=text)])
        chat_params_num_ctx = params["options"]["num_ctx"]

        log_path = tmp_path / "llm_calls.jsonl"
        log_handler = LLMCallLogHandler(log_path, ollama_num_ctx_derivation=derivation)
        log_handler.start_run(ticker="AAPL", date="2024-01-15")
        run_id = uuid.uuid4()
        log_handler.on_chat_model_start(
            {"kwargs": {"model": "ministral-3:8b"}},
            _messages(text),
            run_id=run_id,
            metadata={"langgraph_node": agent},
            invocation_params={"_type": "chat-ollama"},
        )
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, LLMResult

        log_handler.on_llm_end(
            LLMResult(generations=[[ChatGeneration(message=AIMessage(content="ok"))]]),
            run_id=run_id,
        )
        logged_num_ctx = log_handler.get_records()[0]["ollama_num_ctx"]

        assert chat_params_num_ctx == logged_num_ctx == token_count + 100
