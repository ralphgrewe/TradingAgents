"""Exceptions for portfolio manager and related agent nodes."""

from __future__ import annotations


class PortfolioDecisionError(RuntimeError):
    """Raised when the Portfolio Manager produces no usable structured decision.

    This exception indicates a hard failure in decision-making rather than a
    graceful fallback scenario. It is deliberately a dedicated exception type
    (not a bare ``RuntimeError`` or ``ValueError``) so callers — and tests —
    can distinguish "this run was aborted because a structured decision could
    not be produced" from any other failure.

    Follows the same hard-fail precedent as ``MemoryMCPConnectionError`` and
    ``PromptContextOverflowError``.
    """

    pass
