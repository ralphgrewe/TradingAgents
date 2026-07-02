# TradingAgents/graph/reflection.py

from typing import Any, Optional


class Reflector:
    """Handles reflection on trading decisions."""

    def __init__(self, quick_thinking_llm: Any):
        """Initialize the reflector with an LLM."""
        self.quick_thinking_llm = quick_thinking_llm
        self.log_reflection_prompt = self._get_log_reflection_prompt()
        self.decision_reflection_prompt = self._get_decision_reflection_prompt()

    def _get_log_reflection_prompt(self) -> str:
        """Concise prompt for reflect_on_final_decision (Phase B log entries).

        Produces 2-4 sentences of plain prose — compact enough to be re-injected
        into future agent prompts without bloating the context window.
        """
        return (
            "You are a trading analyst reviewing your own past decision now that the outcome is known.\n"
            "Write exactly 2-4 sentences of plain prose (no bullets, no headers, no markdown).\n\n"
            "Cover in order:\n"
            "1. Was the directional call correct? (cite the alpha figure)\n"
            "2. Which part of the investment thesis held or failed?\n"
            "3. One concrete lesson to apply to the next similar analysis.\n\n"
            "Be specific and terse. Your output will be stored verbatim in a decision log "
            "and re-read by future analysts, so every word must earn its place."
        )

    def _get_decision_reflection_prompt(self) -> str:
        """Prompt for ``reflect_on_decision`` (memory-core resolve path, issue #6).

        Adapted from ``_get_log_reflection_prompt`` for a single stored
        ``decisions`` row (signal/confidence/key_drivers/thesis) rather than
        a free-text final trade decision, and without benchmark/alpha
        context — the resolve path only has a raw forward return available
        (``benchmark_return`` is reserved for the deferred Tier-3 depot
        entries, see ``tradingagents/memory/resolve.py``).
        """
        return (
            "You are a trading analyst reviewing your own past decision now that the outcome is known.\n"
            "Write exactly 2-4 sentences of plain prose (no bullets, no headers, no markdown).\n\n"
            "Cover in order:\n"
            "1. Was the directional call correct? (cite the realized return)\n"
            "2. Which key driver(s) or thesis point(s) held or failed?\n"
            "3. One concrete lesson to apply to the next similar analysis.\n\n"
            "Be specific and terse. Your output will be stored verbatim in a decision log "
            "and re-read by future analysts, so every word must earn its place."
        )

    def reflect_on_decision(
        self,
        signal: str,
        confidence: Optional[float],
        key_drivers: Optional[Any],
        thesis: Optional[str],
        forward_return: float,
    ) -> str:
        """Single reflection call on one resolved memory-core decision row (issue #6).

        Unlike ``reflect_on_final_decision`` (legacy log, has alpha vs a
        benchmark), this reflects on a raw forward return only — the
        memory-core ``decisions`` schema leaves ``benchmark_return`` NULL
        for analyst/trader entries (see ``tradingagents/memory/resolve.py``).
        """
        if isinstance(key_drivers, dict):
            drivers_str = ", ".join(str(v) for v in key_drivers.values()) or "none recorded"
        elif key_drivers:
            drivers_str = ", ".join(str(d) for d in key_drivers)
        else:
            drivers_str = "none recorded"
        confidence_str = f"{confidence:.0%}" if confidence is not None else "n/a"

        messages = [
            ("system", self.decision_reflection_prompt),
            (
                "human",
                (
                    f"Signal: {signal}\n"
                    f"Confidence: {confidence_str}\n"
                    f"Key drivers: {drivers_str}\n"
                    f"Thesis: {thesis or 'none recorded'}\n\n"
                    f"Realized forward return: {forward_return:+.1%}"
                ),
            ),
        ]
        return self.quick_thinking_llm.invoke(messages).content

    def reflect_on_final_decision(
        self,
        final_decision: str,
        raw_return: float,
        alpha_return: float,
        benchmark_name: str = "SPY",
    ) -> str:
        """Single reflection call on the final trade decision with outcome context.

        Used by Phase B deferred reflection. The final_trade_decision already
        synthesises all analyst insights, so no separate market context is needed.
        ``benchmark_name`` is the label used for the alpha line (e.g. ``"SPY"``
        for US tickers, ``"^N225"`` for ``.T`` listings); defaults to SPY for
        callers that haven't been updated to thread the benchmark through.
        """
        messages = [
            ("system", self.log_reflection_prompt),
            (
                "human",
                (
                    f"Raw return: {raw_return:+.1%}\n"
                    f"Alpha vs {benchmark_name}: {alpha_return:+.1%}\n\n"
                    f"Final Decision:\n{final_decision}"
                ),
            ),
        ]
        return self.quick_thinking_llm.invoke(messages).content
