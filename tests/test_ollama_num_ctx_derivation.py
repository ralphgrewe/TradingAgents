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
5. ``OllamaChatOpenAI._get_request_payload``: the derived value actually
   reaching the outgoing request's ``extra_body.options.num_ctx``, and other
   providers/chat classes being completely unaffected.
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
# 5. OllamaChatOpenAI._get_request_payload plumbing
# ---------------------------------------------------------------------------


class TestOllamaChatOpenAIRequestPayload:
    def test_derived_num_ctx_reaches_extra_body(self):
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
            api_key="placeholder",
        ).get_llm()

        # _get_request_payload always counts via the tiktoken path (this
        # class is always ChatOpenAI-family / "openai-chat"), not the
        # chars/4 heuristic, so derive the expected token count the same way
        # rather than assuming a heuristic figure.
        from tradingagents.llm_call_log import _count_prompt_tokens

        text = "x" * 400
        token_count, _ = _count_prompt_tokens(text, "ministral-3:8b", "openai-chat")
        payload = llm._get_request_payload([HumanMessage(content=text)])
        assert payload["extra_body"]["options"]["num_ctx"] == token_count + 100

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
            api_key="placeholder",
        ).get_llm()

        long_text = "word " * 1000  # well over 50 tokens
        payload = llm._get_request_payload([HumanMessage(content=long_text)])
        assert payload["extra_body"]["options"]["num_ctx"] == 50

    def test_no_derivation_leaves_payload_unaffected(self):
        from tradingagents.llm_clients.factory import create_llm_client

        llm = create_llm_client(
            provider="ollama", model="ministral-3:8b", api_key="placeholder"
        ).get_llm()
        payload = llm._get_request_payload([HumanMessage(content="hi")])
        assert "extra_body" not in payload

    def test_explicit_num_ctx_and_derivation_are_mutually_exclusive_in_practice(self):
        # TradingAgentsGraph._get_provider_kwargs never sets both at once (see
        # TestGetProviderKwargsDerivation above), but this documents that if
        # only num_ctx is set, the static value survives untouched since no
        # derivation attribute is attached.
        from tradingagents.llm_clients.factory import create_llm_client

        llm = create_llm_client(
            provider="ollama", model="ministral-3:8b", num_ctx=16384, api_key="placeholder"
        ).get_llm()
        payload = llm._get_request_payload([HumanMessage(content="word " * 1000)])
        assert payload["extra_body"]["options"]["num_ctx"] == 16384

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
            api_key="placeholder",
        ).get_llm()
        payload = llm._get_request_payload([HumanMessage(content="word " * 1000)])
        assert "extra_body" not in payload

    def test_other_providers_never_see_an_options_field(self):
        from tradingagents.llm_clients.factory import create_llm_client

        derivation = OllamaNumCtxDerivation(
            models=frozenset({"gpt-5.5"}), num_ctx_max=50, response_headroom=0, safety_margin=1.0,
        )
        # Even if a derivation object were (incorrectly) forwarded to a
        # non-ollama provider, its chat_class isn't OllamaChatOpenAI, so
        # there is no _get_request_payload override to act on it.
        llm = create_llm_client(
            provider="openai",
            model="gpt-5.5",
            ollama_num_ctx_derivation=derivation,
            api_key="placeholder",
        ).get_llm()
        payload = llm._get_request_payload([HumanMessage(content="word " * 1000)])
        assert "extra_body" not in payload
