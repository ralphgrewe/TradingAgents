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

import json
import logging
from collections.abc import Callable
from typing import Any, TypeVar

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from pydantic import BaseModel

from tradingagents.dataflows.config import get_config

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


def _resolve_schema_node(node: dict[str, Any], defs: dict[str, Any]) -> dict[str, Any]:
    """Inline a ``$ref`` into the node that references it.

    Pydantic renders an enum-typed field as ``{"$ref": "#/$defs/Foo",
    "description": ...}`` with the enum's ``"enum": [...]`` list living in
    ``$defs``. Merging the definition into the referencing node (the node's own
    keys winning, so the *field* description beats the enum class docstring)
    gives one flat dict carrying both the legal values and the field's
    instruction text.
    """
    ref = node.get("$ref")
    if not isinstance(ref, str):
        return node
    definition = defs.get(ref.rsplit("/", 1)[-1])
    if not isinstance(definition, dict):
        return node
    merged = dict(definition)
    merged.update({key: value for key, value in node.items() if key != "$ref"})
    return merged


_MAX_SCHEMA_DESCRIPTION_DEPTH = 2

_BOUND_PHRASES = {
    "minimum": ">= {}",
    "maximum": "<= {}",
    "minLength": "at least {} character(s)",
    "maxLength": "at most {} character(s)",
    "minItems": "at least {} item(s)",
    "maxItems": "at most {} item(s)",
}


def _render_bounds(node: dict[str, Any], keys: tuple[str, ...]) -> str:
    """Render the JSON-schema constraint keywords present on ``node``."""
    return ", ".join(
        _BOUND_PHRASES[key].format(node[key]) for key in keys if key in node
    )


def _describe_schema_type(
    node: dict[str, Any], defs: dict[str, Any], *, depth: int = 0
) -> str:
    """Render one JSON-schema node as a short, human-readable type phrase.

    Deliberately produces prose an LLM can act on ('one of: "Buy", "Sell"')
    rather than a Python repr ("<enum 'SwingAction'>"), because this text is
    fed straight back to the model as a repair instruction (issue #153).
    ``depth`` bounds how far nested object/array shapes are expanded, so a
    self-referential schema cannot send this into infinite recursion.
    """
    node = _resolve_schema_node(node, defs)

    enum_values = node.get("enum")
    if isinstance(enum_values, list) and enum_values:
        rendered = ", ".join(json.dumps(value) for value in enum_values)
        return f"one of: {rendered}"

    variants = node.get("anyOf") or node.get("oneOf")
    if isinstance(variants, list) and variants:
        parts: list[str] = []
        for variant in variants:
            part = _describe_schema_type(variant, defs, depth=depth)
            if part not in parts:
                parts.append(part)
        return " or ".join(parts) if parts else "any"

    node_type = node.get("type")

    if node_type == "array":
        items = node.get("items")
        inner = (
            _describe_schema_type(items, defs, depth=depth + 1)
            if isinstance(items, dict) and depth < _MAX_SCHEMA_DESCRIPTION_DEPTH
            else "object" if isinstance(items, dict) else "any"
        )
        counts = _render_bounds(node, ("minItems", "maxItems"))
        prefix = f"array ({counts})" if counts else "array"
        return f"{prefix} of {inner}"

    if node_type == "object":
        properties = node.get("properties")
        if (
            isinstance(properties, dict)
            and properties
            and depth < _MAX_SCHEMA_DESCRIPTION_DEPTH
        ):
            inner_fields = ", ".join(
                f"{name} ({_describe_schema_type(sub, defs, depth=depth + 1)})"
                for name, sub in properties.items()
                if isinstance(sub, dict)
            )
            if inner_fields:
                return f"object with fields: {inner_fields}"
        return "object"

    if not isinstance(node_type, str):
        return "any"

    bounds = _render_bounds(node, ("minimum", "maximum", "minLength", "maxLength"))
    return f"{node_type}, {bounds}" if bounds else node_type


def _generate_repair_instruction(response_model: type[T]) -> str:
    """Generate a self-contained repair instruction from a Pydantic model.

    Derived from ``response_model.model_json_schema()`` rather than from the
    raw ``model_fields`` annotations (issue #153): the JSON schema is what
    renders enum members as an explicit list of legal values and carries each
    ``Field(description=...)`` -- which, per ``agents/schemas.py``, *are* the
    model's output instructions. Reading ``.annotation`` instead produced
    unusable lines like ``rating: <enum 'PortfolioRating'>``, telling the model
    nothing about the values it is allowed to emit on precisely the fields
    (``rating``/``action``) most likely to be malformed.
    """
    schema = response_model.model_json_schema()
    defs = schema.get("$defs", {}) or {}
    properties = schema.get("properties", {}) or {}
    required = set(schema.get("required", []) or [])

    field_specs: list[str] = []
    for field_name, raw_prop in properties.items():
        prop = raw_prop if isinstance(raw_prop, dict) else {}
        requiredness = "required" if field_name in required else "optional"
        type_phrase = _describe_schema_type(prop, defs)
        spec = f"- {field_name} ({requiredness}, {type_phrase})"

        description = _resolve_schema_node(prop, defs).get("description")
        if isinstance(description, str) and description.strip():
            # Collapse whitespace so a multi-line docstring stays on one line.
            spec += f": {' '.join(description.split())}"
        field_specs.append(spec)

    fields_list = "\n".join(field_specs)
    return (
        f"Your previous reply could not be parsed into the required structure.\n"
        f"Reply again with a JSON object matching this specification exactly:\n"
        f"{fields_list}\n"
        f"\n"
        f"Reply with ONLY valid JSON, no prose or explanation."
    )


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

    Callers with no tools to offer (e.g. the Portfolio Manager and Swing Trader
    when ``knowledge_base_enabled`` is False) pass ``tools=[]`` and
    ``max_rounds=0``: nothing is bound, the loop body never runs, and the helper
    degenerates to a single structured call on ``messages`` plus the shared
    fallback and schema-repair-retry logic. That is deliberate -- it is what
    keeps the no-tools configuration from re-growing its own divergent copy of
    the structured-output contract (issue #153).

    Args:
        llm: The LLM instance (will be wrapped with tools and structured output).
        messages: Initial message history (list of BaseMessage objects).
        tools: List of LangChain tool objects to bind to the LLM. Empty means no
              tool binding happens at all.
        response_model: Pydantic model for structured output.
        max_rounds: Maximum tool-calling loop iterations (default 2, typically
                   set via config["knowledge_base_tool_max_rounds"]). 0 skips the
                   loop entirely.
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
        - message_trace: The full message history including tool calls, tool
                        results and -- when a schema-repair retry was dispatched
                        (issue #153) -- the repair instruction that was sent,
                        useful for prompt logging (e.g., via record_agent_prompt).
                        It records what was *sent*; the free-text fallback's own
                        response is not appended, as before.

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
    # Bind tools to the LLM. With an empty ``tools`` list there is nothing to
    # bind and some providers reject a zero-length tool list outright, so skip
    # ``bind_tools`` entirely -- callers that have no tools (e.g. the Portfolio
    # Manager / Swing Trader with ``knowledge_base_enabled=False``) then behave
    # exactly like a plain ``llm``.
    llm_with_tools = llm.bind_tools(tools) if tools else llm

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
    # Set when a repair retry was dispatched, so the returned trace can record
    # the repair round-trip without disturbing the fallback logic below (which
    # deliberately reasons about the *tool-loop* trace's final message).
    retry_trace: list[BaseMessage] | None = None

    if structured_llm is not None:
        try:
            structured_result = structured_llm.invoke(message_trace)
            logger.debug(
                "%s: structured output succeeded after %d loop rounds",
                agent_name, round_count,
            )
        except Exception as exc:
            # First attempt failed; attempt retry if enabled in config
            first_failure = exc
            should_retry = bool(
                get_config().get("structured_output_repair_retry", True)
            )

            if should_retry:
                logger.warning(
                    "%s: structured output failed after tool loop (%s); "
                    "retrying once with schema-repair instruction",
                    agent_name, first_failure,
                )

                # Append repair instruction to trace
                repair_instruction = _generate_repair_instruction(response_model)
                retry_trace = message_trace + [HumanMessage(content=repair_instruction)]

                try:
                    structured_result = structured_llm.invoke(retry_trace)
                    logger.warning(
                        "%s: structured output retry succeeded",
                        agent_name,
                    )
                except Exception as retry_exc:
                    logger.warning(
                        "%s: structured output retry also failed (%s); "
                        "falling back to free text",
                        agent_name, retry_exc,
                    )
            else:
                logger.warning(
                    "%s: structured output failed after tool loop (%s); "
                    "falling back to free text on the final trace (retry disabled)",
                    agent_name, first_failure,
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

    if retry_trace is not None:
        # Surface the repair round-trip in the returned history (the docstring
        # promises the *full* message history). Deliberately done after the
        # fallback block: the #152 "reuse the model's final AIMessage" check
        # reasons about how the tool loop ended, and appending a trailing
        # HumanMessage before that check would silently force the extra
        # free-text invoke it exists to avoid.
        message_trace = retry_trace

    return structured_result, fallback_text, message_trace
