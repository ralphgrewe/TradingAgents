"""SQLite-backed shared memory core — retrieval path (issue #7).

Renders resolved ``decisions`` rows (written by ``store_decision``, issue
#5, and resolved by ``resolve_pending``, issue #6) into a compact markdown
block suitable for injection into an agent prompt. This is the "read" half
of the core API — a view generated from SQLite at call time rather than a
markdown file maintained natively (that native-markdown behavior lived in
the legacy ``TradingMemoryLog.get_past_context``,
``tradingagents/agents/utils/memory.py``, which this supersedes for the new
SQLite-backed pipeline).

Design choices (for future readers — in particular issue #23, per-agent/
per-ticker statistics, and issue #24, the MCP wrapper — both of which may
want to reuse or extend this shape):

- **Resolved rows only**: every query filters on ``resolved_at IS NOT NULL``.
  Pending decisions (no realized outcome yet) never appear in the output —
  there is no lesson to report yet, and surfacing an unresolved decision as
  "past context" would be misleading to the consuming agent.
- **Ticker matching is exact-string, not normalized**: the queried
  ``ticker`` is matched against the stored ``decisions.ticker`` column
  as-is, with no call to ``dataflows.symbol_utils.normalize_symbol``. This
  mirrors ``resolve_pending``'s own ``ticker=`` filter (issue #6), which is
  also exact-match with no normalization applied to the filter argument —
  keeping the read path's matching semantics consistent with the resolve
  path's rather than introducing a third normalization behavior. It also
  means "cross-ticker" is decided by simple string inequality against the
  same un-normalized value. Practical implication: if the same underlying
  instrument was ever stored under two different spellings (e.g. a broker
  alias vs. its canonical form — see the ``store.py``/``resolve.py``
  docstrings), rows under the "other" spelling will show up in the
  cross-ticker section rather than the same-ticker section for a query
  using only one of those spellings. Callers that need alias-aware matching
  should normalize their own ``ticker`` argument before calling.
- **Ordering: ``decision_date`` DESC, ``id`` DESC as a tiebreaker.** Ordering
  by the trading decision's own date (rather than ``resolved_at``, which is
  just "decision_date + horizon_days" plus write-time jitter) is the more
  meaningful "most recent" for what an agent is being told: the underlying
  call being described. ``resolved_at`` is monotonic with ``decision_date``
  in the vast majority of cases (the resolve job processes eligible rows in
  batches) but is not guaranteed to be, since it depends on when
  ``resolve_pending`` happened to run for that row. The ``id`` tiebreaker
  (rows are inserted in increasing id order) makes results deterministic
  when two rows share a ``decision_date``.
- **Same-ticker vs. cross-ticker are two independent queries, not one query
  split in Python.** Each is its own ``SELECT ... LIMIT n`` so that
  ``n_same``/``n_cross`` are enforced by SQL, not by slicing a
  larger-than-needed Python list — cheaper for a large history and avoids
  ever having to guess how many extra rows to over-fetch.
- **Cross-ticker excludes the queried ticker entirely** (``ticker != ?``),
  so the two sections never show overlapping information — same-ticker
  history already covers that ticker in full detail; the cross-ticker
  section exists purely to surface *other* lessons from the same agent.
- **Markdown shape** (stable, documented here since issue #24's MCP wrapper
  and issue #23's statistics module may depend on parsing or reproducing
  it):

  ```
  ## Past context: {ticker}

  ### Same-ticker lessons ({ticker})
  - {decision_date} | {signal} (confidence {confidence}): {lesson}
  - ...

  ### Cross-ticker lessons ({agent})
  - {decision_date} | {ticker} | {signal} (confidence {confidence}): {lesson}
  - ...
  ```

  - The top-level heading is always present and always ``## Past context:
    {ticker}``, with ``{ticker}`` substituted verbatim (the queried value,
    not normalized).
  - Each subsection heading (``### Same-ticker lessons (...)`` / ``###
    Cross-ticker lessons (...)``) is only emitted if that section has at
    least one entry — an empty section is omitted entirely rather than
    printed with a "none" placeholder, to keep the block compact for
    prompt injection.
  - Each entry is a single markdown bullet (``- ...``), one line, in the
    documented order (see "Ordering" above). Same-ticker bullets omit the
    ticker (it's already named in the section heading); cross-ticker
    bullets include it (``{ticker}``) since that section mixes tickers.
  - ``confidence`` is omitted from a bullet when ``NULL`` (rendered as just
    ``{signal}: {lesson}`` with no parenthetical). ``key_drivers`` and
    ``thesis`` are intentionally *not* included in the bullet — the
    ``lesson`` field is the distilled takeaway (see ``resolve.py``); this
    keeps each bullet to roughly one line so the whole block stays compact.
  - If there are no matching rows in *either* section (including the
    degenerate case of no resolved rows at all in the DB), the return value
    is still a well-formed, non-empty markdown block: the heading plus a
    single italic "no prior lessons" line. Callers can always inject the
    return value directly into a prompt without checking for emptiness
    first.
- **Determinism**: this module makes no LLM calls and does no network I/O —
  purely a SQL read plus deterministic string formatting, per the issue #7
  spec ("compact markdown block", not an LLM summarization task).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Union

from tradingagents.memory.store import get_connection

# Module-level defaults for the "no matching rows at all" message, kept as a
# constant so tests/future callers can assert on it without hardcoding the
# string in multiple places.
_NO_LESSONS_MESSAGE = "*No prior resolved lessons yet.*"


def _format_confidence(confidence: Any) -> str:
    """Render a ``confidence`` value for inline bullet display, or ``""``."""
    if confidence is None:
        return ""
    return f" (confidence {confidence:g})"


def _format_same_ticker_entry(row: Any) -> str:
    """One markdown bullet for a same-ticker entry (ticker omitted — implied by the section heading)."""
    return (
        f"- {row['decision_date']} | {row['signal']}"
        f"{_format_confidence(row['confidence'])}: {row['lesson']}"
    )


def _format_cross_ticker_entry(row: Any) -> str:
    """One markdown bullet for a cross-ticker entry (ticker included — section mixes tickers)."""
    return (
        f"- {row['decision_date']} | {row['ticker']} | {row['signal']}"
        f"{_format_confidence(row['confidence'])}: {row['lesson']}"
    )


def get_past_context(
    agent: str,
    ticker: str,
    n_same: int = 5,
    n_cross: int = 3,
    db_path: Union[str, Path, None] = None,
) -> str:
    """Render past resolved lessons for ``agent``/``ticker`` as prompt-ready markdown.

    Retrieves up to ``n_same`` of the most recent resolved decisions for the
    exact same ``(agent, ticker)`` pair, plus up to ``n_cross`` of the most
    recent resolved decisions for the same ``agent`` on any *other* ticker,
    and renders both as a single compact markdown block. See the module
    docstring for the exact markdown shape, the ordering rule
    (``decision_date`` DESC, ``id`` DESC tiebreak), and the ticker-matching
    rule (exact string match, no normalization — consistent with
    ``resolve_pending``'s own filter semantics).

    Only resolved rows (``resolved_at IS NOT NULL``) are ever considered;
    pending rows are always excluded.

    Args:
        agent: Exact agent identifier to match (e.g. ``"trader"``).
        ticker: Exact ticker string to match for the same-ticker section,
            and to exclude for the cross-ticker section. Matched verbatim
            against the stored ``ticker`` column — not normalized.
        n_same: Maximum number of same-ticker entries to include. Must be
            respected strictly (never more rows than this).
        n_cross: Maximum number of cross-ticker entries to include. Must be
            respected strictly (never more rows than this).
        db_path: Optional override for the DB path (see
            ``tradingagents.memory.store.resolve_db_path``).

    Returns:
        A non-empty markdown string. If no matching resolved rows exist in
        either section (including an empty/nonexistent database), a
        well-formed markdown block with a short "no prior lessons" message
        is returned instead of an empty string — safe to inject into a
        prompt unconditionally.
    """
    conn = get_connection(db_path)
    try:
        same_rows = conn.execute(
            """
            SELECT decision_date, signal, confidence, lesson
            FROM decisions
            WHERE resolved_at IS NOT NULL AND agent = ? AND ticker = ?
            ORDER BY decision_date DESC, id DESC
            LIMIT ?
            """,
            (agent, ticker, n_same),
        ).fetchall()

        cross_rows = conn.execute(
            """
            SELECT decision_date, ticker, signal, confidence, lesson
            FROM decisions
            WHERE resolved_at IS NOT NULL AND agent = ? AND ticker != ?
            ORDER BY decision_date DESC, id DESC
            LIMIT ?
            """,
            (agent, ticker, n_cross),
        ).fetchall()
    finally:
        conn.close()

    if not same_rows and not cross_rows:
        return f"## Past context: {ticker}\n\n{_NO_LESSONS_MESSAGE}"

    parts: List[str] = [f"## Past context: {ticker}"]

    if same_rows:
        parts.append(f"### Same-ticker lessons ({ticker})")
        parts.append("\n".join(_format_same_ticker_entry(row) for row in same_rows))

    if cross_rows:
        parts.append(f"### Cross-ticker lessons ({agent})")
        parts.append("\n".join(_format_cross_ticker_entry(row) for row in cross_rows))

    return "\n\n".join(parts)
