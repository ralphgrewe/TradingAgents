"""SQLite-backed shared memory core — inventory/health-check script (issue #65).

Provides a simple standalone CLI to inspect the memory store's contents: total count
of all entries, list of distinct tickers with their entry counts, and optionally a
detailed list of all entries for a selected ticker across all agents.

This module is intentionally kept fully separate from stats.py (issue #23) — it reports
raw counts/inventory (pending + resolved rows), not performance statistics (hit-rate /
average return / calibration, which operate on resolved rows only).

Design choices:

- **All rows included**: unlike stats.py and query.py, which filter on
  ``resolved_at IS NOT NULL``, the inventory script includes both pending
  (``resolved_at IS NULL``) and resolved rows in every count and listing. This
  is an inventory/health-check, not a statistical analysis — we want to know
  what actually exists in the database.
- **Ticker matching is exact-string**: ``--symbol TICKER`` matches the stored
  ``decisions.ticker`` column verbatim, with no call to
  ``dataflows.symbol_utils.normalize_symbol``, consistent with the rest of the
  memory module.
- **Output is plain terminal text/table**: uses simple markdown-style tables and
  bullet lists, no charting library or image output.
- **Determinism**: no LLM calls and no network I/O — a SQL read plus deterministic
  string formatting, suitable for use in CI/test environments.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from tradingagents.memory.store import get_connection, resolve_db_path


def get_inventory(db_path: str | Path | None = None) -> dict[str, Any]:
    """Fetch inventory statistics: total count and per-ticker breakdown.

    Returns a plain, JSON-serializable dict:
    ``{"total": <int>, "by_ticker": [{"ticker", "n"}, ...]}``

    The ``by_ticker`` list is sorted alphabetically by ticker for deterministic
    output (same convention as stats.py's ``per_agent``/``per_agent_ticker``
    sorting). Includes both pending and resolved rows.

    Args:
        db_path: Optional override for the DB path (see
            ``tradingagents.memory.store.resolve_db_path``).
    """
    conn = get_connection(db_path)
    try:
        # Get total count
        total_result = conn.execute("SELECT COUNT(*) as total FROM decisions").fetchone()
        total = total_result["total"] if total_result else 0

        # Get per-ticker breakdown
        ticker_rows = conn.execute(
            "SELECT ticker, COUNT(*) as n FROM decisions GROUP BY ticker ORDER BY ticker"
        ).fetchall()

        by_ticker = [{"ticker": row["ticker"], "n": row["n"]} for row in ticker_rows]
    finally:
        conn.close()

    return {"total": total, "by_ticker": by_ticker}


def get_ticker_entries(
    ticker: str, db_path: str | Path | None = None
) -> list[dict[str, Any]]:
    """Fetch all entries for a given ticker across all agents.

    Args:
        ticker: Exact (un-normalized) ticker string to match against
            ``decisions.ticker`` — matched verbatim.
        db_path: Optional override for the DB path (see
            ``tradingagents.memory.store.resolve_db_path``).

    Returns:
        A list of dicts, one per row, ordered ``decision_date`` DESC, ``id`` DESC
        (same ordering convention as query.py). Each dict contains:
        ``{"id", "agent", "ticker", "decision_date", "signal", "confidence", "thesis",
           "resolved_at", "forward_return", "lesson"}``.

        Returns an empty list if no rows match the ticker (no error).
    """
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            """
            SELECT id, agent, ticker, decision_date, signal, confidence, thesis,
                   resolved_at, forward_return, lesson
            FROM decisions
            WHERE ticker = ?
            ORDER BY decision_date DESC, id DESC
            """,
            (ticker,),
        ).fetchall()
    finally:
        conn.close()

    return [dict(row) for row in rows]


# ---------------------------------------------------------------------------
# Terminal formatting
# ---------------------------------------------------------------------------


def _format_inventory_text(inventory: dict[str, Any]) -> str:
    """Render the inventory dict as a plain-text report."""
    total = inventory["total"]
    by_ticker = inventory["by_ticker"]

    lines = [f"Total entries: {total}"]

    if by_ticker:
        lines.append("")
        lines.append("Entries by ticker:")
        lines.append("")
        for item in by_ticker:
            lines.append(f"  {item['ticker']}: {item['n']}")
    else:
        lines.append("(No entries yet)")

    return "\n".join(lines)


def _format_confidence(confidence: Any) -> str:
    """Render confidence as a string or 'n/a' if None."""
    if confidence is None:
        return "n/a"
    return f"{confidence:g}"


def _format_resolved_status(resolved_at: Any) -> str:
    """Render resolved status as 'resolved' or 'pending'."""
    return "resolved" if resolved_at is not None else "pending"


def _format_ticker_entries_text(ticker: str, entries: list[dict[str, Any]]) -> str:
    """Render detailed entries for a ticker as a plain-text table."""
    if not entries:
        return f"No entries for ticker {ticker}."

    lines = [f"Entries for {ticker} ({len(entries)} total):"]
    lines.append("")

    # Column widths (estimated for readability)
    col_specs = [
        ("Agent", "agent", 15),
        ("Date", "decision_date", 12),
        ("Signal", "signal", 12),
        ("Conf.", "confidence", 8),
        ("Thesis", "thesis", 40),
        ("Status", None, 10),  # computed from resolved_at
        ("Forward Return", "forward_return", 14),
        ("Lesson", "lesson", 50),
    ]

    # Print header
    header_parts = [name.ljust(width) for name, _, width in col_specs]
    lines.append("  ".join(header_parts))
    lines.append("  ".join("-" * width for _, _, width in col_specs))

    # Print rows
    for entry in entries:
        row_parts = []
        for col_name, field, width in col_specs:
            if field is None:
                # Special handling for status
                value = _format_resolved_status(entry.get("resolved_at"))
            else:
                value = entry.get(field)
                if field == "confidence":
                    value = _format_confidence(value)
                elif field == "forward_return":
                    if value is None:
                        value = "n/a"
                    else:
                        value = f"{value:+.2%}"
                elif value is None:
                    value = ""
                else:
                    value = str(value)

            # Truncate long strings
            if len(str(value)) > width:
                value = str(value)[: width - 3] + "..."
            row_parts.append(str(value).ljust(width))

        lines.append("  ".join(row_parts))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point: python -m tradingagents.memory.inventory
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tradingagents.memory.inventory",
        description="Inspect memory store contents: total count and per-ticker breakdown.",
    )
    parser.add_argument(
        "--symbol",
        default=None,
        help="Show detailed entries for this exact ticker (un-normalized).",
    )
    parser.add_argument("--db-path", default=None, help="Override the memory DB path.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)

    if args.symbol:
        # Detail view for a specific symbol
        entries = get_ticker_entries(args.symbol, db_path=args.db_path)
        print(_format_ticker_entries_text(args.symbol, entries))
    else:
        # Default: inventory overview
        inventory = get_inventory(db_path=args.db_path)
        print(_format_inventory_text(inventory))


if __name__ == "__main__":
    main()
