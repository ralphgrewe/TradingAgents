"""Shared helpers for invoking an agent with structured output and a graceful fallback.

The Portfolio Manager, Trader, and Research Manager all follow the same
canonical pattern:

1. At agent creation, wrap the LLM with ``with_structured_output(Schema)``
   so the model returns a typed Pydantic instance. If the provider does
   not support structured output (rare; mostly older Ollama models), the
   wrap is skipped and the agent uses free-text generation instead.
2. At invocation, run the structured call and render the result back to
   markdown. If the structured call itself fails for any reason
   (malformed JSON from a weak model, transient provider issue), fall
   back to a plain ``llm.invoke`` so the pipeline never blocks.

Centralising the pattern here keeps the agent factories small and ensures
all three agents log the same warnings when fallback fires.

Part of issue #104: the ``run_structured_with_tools`` helper provides the
shared machinery for running a bounded tool-loop that terminates with
structured output, so consumers (portfolio manager, swing trader) don't
each reinvent and diverge on that logic.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, TypeVar

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def _normalize_content(content: Any) -> str:
    """Normalize message content (string or list of blocks) to a string.

    Mirrors the logic from llm_call_log._message_text to handle both
    plain strings and multimodal content blocks consistently.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(content)


def bind_structured(llm: Any, schema: type[T], agent_name: str) -> Any | None:
    """Return ``llm.with_structured_output(schema)`` or ``None`` if unsupported.

    Logs a warning when the binding fails so the user understands the agent
    will use free-text generation for every call instead of one-shot fallback.
    """
    try:
        return llm.with_structured_output(schema)
    except (NotImplementedError, AttributeError) as exc:
        logger.warning(
            "%s: provider does not support with_structured_output (%s); "
            "falling back to free-text generation",
            agent_name, exc,
        )
        return None


def invoke_structured_or_freetext(
    structured_llm: Any | None,
    plain_llm: Any,
    prompt: Any,
    render: Callable[[T], str],
    agent_name: str,
) -> str:
    """Run the structured call and render to markdown; fall back to free-text on any failure.

    ``prompt`` is whatever the underlying LLM accepts (a string for chat
    invocations, a list of message dicts for chat models that take that
    shape). The same value is forwarded to the free-text path so the
    fallback sees the same input the structured call did.
    """
    if structured_llm is not None:
        try:
            result = structured_llm.invoke(prompt)
            return render(result)
        except Exception as exc:
            logger.warning(
                "%s: structured-output invocation failed (%s); retrying once as free text",
                agent_name, exc,
            )

    response = plain_llm.invoke(prompt)
    return response.content


def run_structured_with_tools(
    llm: Any,
    messages: list[BaseMessage],
    tools: list[Any],
    response_model: type[T],
    *,
    max_rounds: int = 2,
    agent_name: str = "Agent",
) -> tuple[T | None, str | None, list[BaseMessage]]:
    """Run a bounded tool-calling loop that terminates with structured output.

    Single-shot structured-output nodes (Portfolio Manager, Swing Trader) can
    consult tools during execution and then emit their final decision. This helper
    provides the shared machinery: bind tools, run a bounded loop where the LLM
    can invoke tools and receive results, then attempt structured output on the
    final response.

    Preserves the same structured-then-fallback contract as
    ``invoke_structured_or_freetext``: the final step is *always* attempted
    (structured output if supported, otherwise/on-failure a plain free-text
    ``llm.invoke``), so the caller does not have to reimplement the free-text
    fallback itself.

    Contract, stated precisely: **if this function returns, exactly one of
    ``structured_result`` / ``fallback_text`` is non-``None``.** There is no
    return path that yields both as ``None``, so callers may branch on
    ``structured_result is None`` and use ``fallback_text`` unconditionally.
    The price of that guarantee is that a true double failure — the structured
    call fails (or is unsupported) *and* the free-text fallback call also
    raises, e.g. a provider outage hitting both calls — propagates the
    fallback's exception uncaught, exactly as ``invoke_structured_or_freetext``
    does. A hard provider failure is surfaced as a real exception rather than
    being swallowed into a silent, signal-free "no output" return.

    Args:
        llm: The LLM instance (will be wrapped with tools and structured output).
        messages: Initial message history (list of BaseMessage objects).
        tools: List of LangChain tool objects to bind to the LLM.
        response_model: Pydantic model for structured output.
        max_rounds: Maximum tool-calling loop iterations (default 2, typically
                   set via config["knowledge_base_tool_max_rounds"]).
        agent_name: Name for logging purposes (e.g., "PortfolioManager", "SwingTrader").

    Returns:
        Tuple of (structured_result, fallback_text, message_trace):
        - structured_result: Parsed Pydantic instance, or None if structured output
                            failed or is unsupported.
        - fallback_text: Free-text content on fallback, always populated (never None) when
                        ``structured_result`` is None. Sources are prioritized:
                        1. If the trace ends in an ``AIMessage`` whose content, once
                           normalized and stripped of surrounding whitespace, is
                           non-empty, that normalized content is reused (no extra
                           LLM call).
                        2. Otherwise, a plain ``llm.invoke`` call on the final trace
                           (mirrors ``invoke_structured_or_freetext``'s fallback).
                        When the trace ends in a ``ToolMessage`` (e.g., tool loop exhausted
                        its rounds) or an empty/whitespace-only ``AIMessage``, path 2 is
                        used. None when the structured call succeeded.
        - message_trace: The full message history including tool calls and results,
                        useful for prompt logging (e.g., via record_agent_prompt).

    Raises:
        Exception: Whatever the free-text fallback ``llm.invoke`` raises, when the
            structured path already failed or was unsupported. This is the single
            documented failure mode, and it replaces what would otherwise be a
            silent ``(None, None, trace)`` return carrying no usable output and no
            signal that the provider actually failed. Failures *inside* the tool
            loop (a tool-call round raising, or an individual tool raising) are
            still caught and logged, because the final call can — and does — still
            recover from them.

    Example:
        >>> result, fallback_text, trace = run_structured_with_tools(
        ...     llm, messages, [search_wiki, get_market_data],
        ...     PortfolioDecision, max_rounds=2, agent_name="PM"
        ... )
        >>> if result:
        ...     decision = result.decision  # access structured fields
        >>> else:
        ...     decision_text = fallback_text  # guaranteed usable free text
        >>> # trace can be logged for debugging/analysis
    """
    # Bind tools to the LLM
    llm_with_tools = llm.bind_tools(tools)

    # Try to bind structured output; graceful None if unsupported
    structured_llm = bind_structured(llm, response_model, agent_name)

    # Tool-calling loop
    message_trace = list(messages)
    round_count = 0

    while round_count < max_rounds:
        round_count += 1

        # Invoke LLM with current message history
        try:
            response = llm_with_tools.invoke(message_trace)
        except Exception as exc:
            logger.warning(
                "%s: LLM invocation failed in tool loop round %d (%s)",
                agent_name, round_count, exc,
            )
            break

        # Append LLM response to trace
        message_trace.append(response)

        # Check if LLM requested tool calls
        tool_calls = getattr(response, "tool_calls", [])
        if not tool_calls:
            # No tool calls: LLM is done with the loop
            logger.debug(
                "%s: LLM returned no tool calls in round %d; ending loop",
                agent_name, round_count,
            )
            break

        # Execute requested tools and append results
        for tool_call in tool_calls:
            tool_name = tool_call.get("name") or tool_call.get("type")
            # "args" may legitimately be {} for a zero-arg tool call, which is
            # falsy in Python -- use explicit key presence, not truthiness, to
            # decide whether to fall back to the "input" key.
            tool_input = (
                tool_call["args"] if "args" in tool_call else tool_call.get("input", {})
            )
            tool_id = tool_call.get("id") or tool_call.get("tool_call_id")

            # Find the tool by name
            tool_impl = None
            for t in tools:
                if hasattr(t, "name") and t.name == tool_name:
                    tool_impl = t
                    break

            if tool_impl is None:
                logger.warning(
                    "%s: tool '%s' not found in tool list (round %d); skipping",
                    agent_name, tool_name, round_count,
                )
                result_text = f"Tool '{tool_name}' not found"
            else:
                try:
                    # Invoke the tool with its input
                    if isinstance(tool_input, dict):
                        result = tool_impl.invoke(tool_input)
                    else:
                        result = tool_impl.invoke({"input": tool_input})
                    result_text = str(result)
                except Exception as exc:
                    logger.warning(
                        "%s: tool '%s' invocation failed (round %d): %s",
                        agent_name, tool_name, round_count, exc,
                    )
                    result_text = f"Tool execution failed: {exc}"

            # Append tool result to message trace
            tool_message = ToolMessage(content=result_text, tool_call_id=tool_id)
            message_trace.append(tool_message)

        logger.debug(
            "%s: completed round %d of %d; executed %d tool calls",
            agent_name, round_count, max_rounds, len(tool_calls),
        )

    # Tool loop is done (whether it ended because the LLM stopped requesting
    # tools, the round budget was exhausted, or an invocation raised). Always
    # attempt the final structured call on this synthesized trace -- no gate
    # on what's already in the trace, so max_rounds=0 or a first-round
    # failure still gets a real attempt instead of a silent None.
    structured_result: T | None = None
    fallback_text: str | None = None

    if structured_llm is not None:
        try:
            structured_result = structured_llm.invoke(message_trace)
            logger.debug(
                "%s: structured output succeeded after %d loop rounds",
                agent_name, round_count,
            )
        except Exception as exc:
            logger.warning(
                "%s: structured output failed after tool loop (%s); "
                "falling back to free text on the final trace",
                agent_name, exc,
            )
    else:
        logger.debug(
            "%s: structured output not supported by LLM; falling back to free text",
            agent_name,
        )

    if structured_result is None:
        # Guarantee usable output. Try to reuse the model's real answer if it's
        # already sitting in the trace, then fall back to a fresh llm.invoke if needed.
        #
        # Priority 1: if the trace ends in an AIMessage with non-empty content, that
        # is the model's final substantive response from the tool loop. Reuse it
        # directly (no extra LLM call). This avoids the problem where invoking the
        # LLM again on a trace ending in an assistant turn makes it emit a continuation
        # of its own prior message, discarding the real answer.
        #
        # Priority 2 (fallback): if the trace ends in a ToolMessage (tool loop exhausted
        # max_rounds while still receiving tool calls) or an empty AIMessage, invoke
        # the plain LLM on the final trace for a fresh response. This path is NOT
        # wrapped in try/except: if it also raises, the structured path *and* the
        # fallback path have both failed. Swallowing that exception would hand the
        # caller a silent (None, None, trace) indistinguishable from "provider was
        # fine, it just said nothing". Letting it propagate matches invoke_structured_or_freetext,
        # matches how the graph nodes treat a dead provider elsewhere (abort, don't
        # guess), and keeps the documented invariant that exactly one of
        # structured_result / fallback_text is non-None on every return.
        normalized_last_content = (
            _normalize_content(message_trace[-1].content)
            if message_trace and isinstance(message_trace[-1], AIMessage)
            else None
        )
        if normalized_last_content is not None and normalized_last_content.strip():
            # Trace ends in an AIMessage with non-empty, non-whitespace-only
            # content: reuse it. Checking after normalization (and stripping
            # whitespace) -- rather than plain Python truthiness on the raw
            # ``.content`` -- ensures a whitespace-only string (e.g. " " or
            # "\n") is treated the same as empty content and falls through to
            # the fresh-invoke fallback below, instead of being reused as if
            # it were a real answer.
            fallback_text = normalized_last_content
            logger.debug(
                "%s: reusing model's final AIMessage from tool loop as fallback "
                "(no extra invoke)",
                agent_name,
            )
        else:
            # Trace ends in ToolMessage, empty AIMessage, or HumanMessage: invoke
            # the LLM for a fresh response.
            fallback_response = llm.invoke(message_trace)
            fallback_text = _normalize_content(fallback_response.content)
            logger.debug(
                "%s: fallback invoked fresh LLM response (trace did not end in "
                "content-bearing AIMessage)",
                agent_name,
            )

    return structured_result, fallback_text, message_trace
