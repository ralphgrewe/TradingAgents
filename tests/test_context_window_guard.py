"""Tests for the issue #149 oversize-prompt enforcement.

``docs/analysis/prompt-truncation-diagnosis.md`` (issue #148) found that a
prompt exceeding a local Ollama model's actual (VRAM-tiered, invisible)
context window can be silently truncated with no error anywhere in the run,
and that the provider-reported ``input_tokens`` cannot be used to detect this
(#148's Anomaly A: it locks to a wrong constant past a size threshold). This
issue's fix is ``ContextWindowGuardHandler`` (``tradingagents/llm_call_log.py``):
a callback handler that compares the #147 tiktoken/heuristic prompt-size
estimate — never ``input_tokens`` — against a *known* context window and
aborts the run with ``PromptContextOverflowError`` before the call is
dispatched when it doesn't fit.

These tests feed synthetic callback events directly (no real LLM calls, per
``tests/conftest.py`` conventions) and cover the acceptance criteria from
issue #149: within-limit passes, over-limit raises with the required fields,
an unknown model passes unchecked, the escape hatch disables enforcement, and
the oversize event reaches ``llm_calls.jsonl``. A second test class covers
``TradingAgentsGraph._build_context_window_guard``'s resolution of
``ollama_num_ctx`` / ``context_window_overrides`` into the ``context_windows``
mapping, and a third covers the ``num_ctx`` -> ``extra_body`` plumbing in
``OpenAIClient``.
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
    PromptContextOverflowError,
)

pytestmark = pytest.mark.unit


def _messages(text: str) -> list[list[HumanMessage]]:
    return [[HumanMessage(content=text)]]


class TestWithinLimitPasses:
    def test_prompt_within_limit_does_not_raise(self):
        handler = ContextWindowGuardHandler(
            context_windows={"ministral-3:3b": 100_000},
            safety_margin=1.3,
        )
        # Should not raise.
        handler.on_chat_model_start(
            {"kwargs": {"model": "ministral-3:3b"}},
            _messages("a short prompt"),
            run_id=uuid.uuid4(),
            metadata={"langgraph_node": "Market Analyst"},
        )

    def test_prompt_exactly_at_the_margin_adjusted_limit_does_not_raise(self):
        # 10 chars/4 ~= 2 tokens (heuristic path, no tiktoken llm_type), so
        # pick a window that exactly matches token_count * safety_margin.
        text = "x" * 40  # heuristic: 40 // 4 = 10 tokens
        handler = ContextWindowGuardHandler(
            context_windows={"custom-model": 10},  # 10 * 1.0 == 10 -> not > window
            safety_margin=1.0,
        )
        handler.on_chat_model_start(
            {"kwargs": {"model": "custom-model"}},
            _messages(text),
            run_id=uuid.uuid4(),
            metadata={"langgraph_node": "Trader"},
        )


class TestOverLimitRaises:
    def test_over_limit_raises_dedicated_exception(self):
        handler = ContextWindowGuardHandler(
            context_windows={"ministral-3:8b": 50},
            safety_margin=1.0,
        )
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
        assert error.context_window == 50
        assert error.prompt_tokens_estimated > 50
        assert error.token_count_method == "heuristic_chars_per_token"

    def test_error_message_is_actionable(self):
        """Required fields per #149's acceptance criteria: agent, model, measured
        prompt size, the limit, and a pointer to the remedy."""
        handler = ContextWindowGuardHandler(
            context_windows={"ministral-3:8b": 50},
            safety_margin=1.0,
        )
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
        assert "50" in message  # the context window
        assert str(exc_info.value.prompt_tokens_estimated) in message
        assert "docs/local-models.md" in message
        assert "TRADINGAGENTS_CONTEXT_WINDOW_CHECK_ENABLED" in message

    def test_over_limit_via_on_llm_start_legacy_path(self):
        handler = ContextWindowGuardHandler(
            context_windows={"custom-model": 10},
            safety_margin=1.0,
        )
        with pytest.raises(PromptContextOverflowError):
            handler.on_llm_start(
                {"kwargs": {"model": "custom-model"}},
                ["word " * 1000],
                run_id=uuid.uuid4(),
                metadata={"langgraph_node": "Trader"},
            )

    def test_safety_margin_pushes_a_call_over_the_limit(self):
        # 100 tokens (heuristic) * 2.0 margin = 200 > a 150-token window.
        text = "x" * 400  # 400 // 4 == 100 tokens
        handler = ContextWindowGuardHandler(
            context_windows={"custom-model": 150},
            safety_margin=2.0,
        )
        with pytest.raises(PromptContextOverflowError) as exc_info:
            handler.on_chat_model_start(
                {"kwargs": {"model": "custom-model"}},
                _messages(text),
                run_id=uuid.uuid4(),
                metadata={"langgraph_node": "Researcher"},
            )
        assert exc_info.value.prompt_tokens_estimated == 100
        assert exc_info.value.safety_margin == 2.0


class TestUnknownModelPassesUnchecked:
    def test_model_not_in_context_windows_is_never_checked(self):
        handler = ContextWindowGuardHandler(
            context_windows={"ministral-3:8b": 10},  # only 8b is known
            safety_margin=1.0,
        )
        long_prompt = "word " * 1000
        # "ministral-3:3b" has no entry -> pass through unchecked even though
        # it's a huge prompt and 8b's window would have rejected it.
        handler.on_chat_model_start(
            {"kwargs": {"model": "ministral-3:3b"}},
            _messages(long_prompt),
            run_id=uuid.uuid4(),
            metadata={"langgraph_node": "Trader"},
        )

    def test_empty_context_windows_is_a_no_op(self):
        handler = ContextWindowGuardHandler(context_windows={}, safety_margin=1.0)
        long_prompt = "word " * 1000
        handler.on_chat_model_start(
            {"kwargs": {"model": "anything"}},
            _messages(long_prompt),
            run_id=uuid.uuid4(),
            metadata={"langgraph_node": "Trader"},
        )


class TestEscapeHatchDisablesEnforcement:
    def test_disabled_guard_never_raises_even_over_limit(self):
        handler = ContextWindowGuardHandler(
            context_windows={"ministral-3:8b": 10},
            safety_margin=1.0,
            enabled=False,
        )
        long_prompt = "word " * 1000
        handler.on_chat_model_start(
            {"kwargs": {"model": "ministral-3:8b"}},
            _messages(long_prompt),
            run_id=uuid.uuid4(),
            metadata={"langgraph_node": "Portfolio Manager"},
        )

    def test_default_config_ships_enforcement_enabled(self):
        from tradingagents.default_config import DEFAULT_CONFIG

        assert DEFAULT_CONFIG["context_window_check_enabled"] is True

    def test_default_config_ships_no_known_context_windows(self):
        # Out of the box (no ollama_num_ctx, no overrides) nothing is
        # actually enforced -- an unknown model always passes. This is the
        # "only enforce where known" acceptance criterion, verified at the
        # config level.
        from tradingagents.default_config import DEFAULT_CONFIG

        assert DEFAULT_CONFIG["ollama_num_ctx"] is None
        assert DEFAULT_CONFIG["context_window_overrides"] == {}


class TestErrorRecordReachesJsonl:
    def test_oversize_call_writes_a_jsonl_error_record(self, tmp_path):
        log_path = tmp_path / "llm_calls.jsonl"
        log_handler = LLMCallLogHandler(log_path)
        log_handler.start_run(ticker="AAPL", date="2024-01-15")

        guard = ContextWindowGuardHandler(
            context_windows={"ministral-3:8b": 50},
            safety_margin=1.0,
            log_handler=log_handler,
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
        assert record["ticker"] == "AAPL"
        assert record["date"] == "2024-01-15"
        assert record["input_tokens"] is None
        assert record["output_tokens"] is None
        assert record["error"] is not None
        assert "Portfolio Manager" in record["error"]
        assert record["prompt_tokens_estimated"] > 50

    def test_no_log_handler_still_raises_without_writing_anything(self, tmp_path):
        # log_handler=None is a supported configuration (e.g. a caller that
        # doesn't wire up LLMCallLogHandler at all) -- the guard's primary
        # job (aborting the run) must still work.
        guard = ContextWindowGuardHandler(
            context_windows={"ministral-3:8b": 50},
            safety_margin=1.0,
            log_handler=None,
        )
        with pytest.raises(PromptContextOverflowError):
            guard.on_chat_model_start(
                {"kwargs": {"model": "ministral-3:8b"}},
                _messages("word " * 1000),
                run_id=uuid.uuid4(),
                metadata={"langgraph_node": "Portfolio Manager"},
            )

    def test_disabled_log_handler_is_a_no_op_and_does_not_prevent_the_raise(self, tmp_path):
        log_path = tmp_path / "llm_calls.jsonl"
        log_handler = LLMCallLogHandler(log_path, enabled=False)
        guard = ContextWindowGuardHandler(
            context_windows={"ministral-3:8b": 50},
            safety_margin=1.0,
            log_handler=log_handler,
        )
        with pytest.raises(PromptContextOverflowError):
            guard.on_chat_model_start(
                {"kwargs": {"model": "ministral-3:8b"}},
                _messages("word " * 1000),
                run_id=uuid.uuid4(),
                metadata={"langgraph_node": "Portfolio Manager"},
            )
        assert not log_path.exists()


class TestRaiseErrorFlag:
    def test_guard_handler_sets_raise_error_so_langchain_propagates_it(self):
        # langchain_core.callbacks.manager.handle_event only re-raises a
        # handler's exception when that handler's raise_error is True (every
        # other handler's exceptions are logged and swallowed) -- this is
        # what makes "abort before dispatch" actually work end-to-end.
        handler = ContextWindowGuardHandler(context_windows={})
        assert handler.raise_error is True

    def test_llm_call_log_handler_keeps_its_never_raise_guarantee(self):
        # #147's design: LLMCallLogHandler must never break a run.
        # ContextWindowGuardHandler is a deliberately separate class so this
        # guarantee is untouched.
        handler = LLMCallLogHandler(None)
        assert handler.raise_error is False


class TestBuildContextWindowGuardWiring:
    """Tests for TradingAgentsGraph._build_context_window_guard's mapping resolution."""

    def _mock_graph(self, config, callbacks=None):
        mock_graph = MagicMock()
        mock_graph.config = config
        mock_graph.callbacks = callbacks or []
        return mock_graph

    def test_ollama_num_ctx_populates_both_model_names(self):
        from tradingagents.graph.trading_graph import TradingAgentsGraph

        mock_graph = self._mock_graph({
            "llm_provider": "ollama",
            "quick_think_llm": "ministral-3:3b",
            "deep_think_llm": "ministral-3:8b",
            "ollama_num_ctx": 8192,
        })
        guard = TradingAgentsGraph._build_context_window_guard(mock_graph)
        assert guard.context_windows == {
            "ministral-3:3b": 8192,
            "ministral-3:8b": 8192,
        }

    def test_non_ollama_provider_does_not_derive_a_window(self):
        from tradingagents.graph.trading_graph import TradingAgentsGraph

        mock_graph = self._mock_graph({
            "llm_provider": "openai",
            "quick_think_llm": "gpt-5.4-mini",
            "deep_think_llm": "gpt-5.5",
            "ollama_num_ctx": 8192,  # irrelevant for a non-ollama provider
        })
        guard = TradingAgentsGraph._build_context_window_guard(mock_graph)
        assert guard.context_windows == {}

    def test_overrides_take_precedence_over_ollama_derived_value(self):
        from tradingagents.graph.trading_graph import TradingAgentsGraph

        mock_graph = self._mock_graph({
            "llm_provider": "ollama",
            "quick_think_llm": "ministral-3:3b",
            "deep_think_llm": "ministral-3:8b",
            "ollama_num_ctx": 8192,
            "context_window_overrides": {"ministral-3:3b": 32768},
        })
        guard = TradingAgentsGraph._build_context_window_guard(mock_graph)
        assert guard.context_windows["ministral-3:3b"] == 32768
        assert guard.context_windows["ministral-3:8b"] == 8192

    def test_no_ollama_num_ctx_and_no_overrides_yields_empty_mapping(self):
        from tradingagents.graph.trading_graph import TradingAgentsGraph

        mock_graph = self._mock_graph({
            "llm_provider": "ollama",
            "quick_think_llm": "ministral-3:3b",
            "deep_think_llm": "ministral-3:8b",
            "ollama_num_ctx": None,
        })
        guard = TradingAgentsGraph._build_context_window_guard(mock_graph)
        assert guard.context_windows == {}

    def test_finds_the_llm_call_log_handler_among_callbacks(self):
        from tradingagents.graph.trading_graph import TradingAgentsGraph

        log_handler = LLMCallLogHandler(None)
        mock_graph = self._mock_graph(
            {"llm_provider": "ollama", "quick_think_llm": "a", "deep_think_llm": "b"},
            callbacks=[object(), log_handler],
        )
        guard = TradingAgentsGraph._build_context_window_guard(mock_graph)
        assert guard.log_handler is log_handler

    def test_escape_hatch_and_safety_margin_reach_the_guard(self):
        from tradingagents.graph.trading_graph import TradingAgentsGraph

        mock_graph = self._mock_graph({
            "llm_provider": "ollama",
            "quick_think_llm": "a",
            "deep_think_llm": "b",
            "context_window_check_enabled": False,
            "context_window_safety_margin": 1.7,
        })
        guard = TradingAgentsGraph._build_context_window_guard(mock_graph)
        assert guard.enabled is False
        assert guard.safety_margin == 1.7


class TestOllamaNumCtxPlumbing:
    """Tests for the num_ctx -> extra_body plumbing (issue #149)."""

    def test_get_provider_kwargs_forwards_ollama_num_ctx(self):
        from tradingagents.graph.trading_graph import TradingAgentsGraph

        mock_graph = MagicMock(spec=TradingAgentsGraph)
        mock_graph.config = {"llm_provider": "ollama", "ollama_num_ctx": 16384}
        kwargs = TradingAgentsGraph._get_provider_kwargs(mock_graph)
        assert kwargs["num_ctx"] == 16384

    def test_get_provider_kwargs_omits_num_ctx_when_unset(self):
        from tradingagents.graph.trading_graph import TradingAgentsGraph

        mock_graph = MagicMock(spec=TradingAgentsGraph)
        mock_graph.config = {"llm_provider": "ollama", "ollama_num_ctx": None}
        kwargs = TradingAgentsGraph._get_provider_kwargs(mock_graph)
        assert "num_ctx" not in kwargs

    def test_get_provider_kwargs_ignores_num_ctx_for_other_providers(self):
        from tradingagents.graph.trading_graph import TradingAgentsGraph

        mock_graph = MagicMock(spec=TradingAgentsGraph)
        mock_graph.config = {"llm_provider": "openai", "ollama_num_ctx": 16384}
        kwargs = TradingAgentsGraph._get_provider_kwargs(mock_graph)
        assert "num_ctx" not in kwargs

    def test_num_ctx_reaches_chat_openai_as_extra_body(self):
        from tradingagents.llm_clients.factory import create_llm_client

        llm = create_llm_client(
            provider="ollama", model="ministral-3:8b", num_ctx=16384, api_key="placeholder"
        ).get_llm()
        assert llm.extra_body == {"options": {"num_ctx": 16384}}

    def test_no_num_ctx_leaves_extra_body_unset(self):
        from tradingagents.llm_clients.factory import create_llm_client

        llm = create_llm_client(
            provider="ollama", model="ministral-3:8b", api_key="placeholder"
        ).get_llm()
        assert llm.extra_body is None
