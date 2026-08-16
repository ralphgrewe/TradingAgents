#!/usr/bin/env python3
"""KPI report script for the #81 ablation-study harness (issue #121).

Reads the per-arm artifacts produced by running the run-config files in
``experiments/ablation-analysts/`` (issue #120: isolated memory DBs, depots,
and full-state-log trees, one set per arm) and writes a single markdown
report computing the five #81 KPIs, each ablation arm compared against the
``control`` arm. This script encodes **no** keep/merge/drop decision rule —
that call is explicitly human judgment made on top of the report (see #81's
"Confirmed design decisions").

Usage::

    ./venv/bin/python scripts/ablation_report.py
    ./venv/bin/python scripts/ablation_report.py --arms-dir /path/to/arms --out /path/to/report.md
    ./venv/bin/python scripts/ablation_report.py --sim-data-dir /path/to/McpTradingSimulation/data

Run from the repo root (or pass explicit paths) — every artifact path below is
resolved the same way ``run_trading_agents.py``/``TradingAgentsGraph`` resolve
them: relative to the current working directory, not this script's location.

Artifact access paths (one per KPI; discovered by reading the actual writers,
not just the arm config field names)
------------------------------------------------------------------------------
- **KPI 1/2 (decisions, forward returns)**: each arm's SQLite memory DB at
  ``runs/memory/<memory_id>/memory.db`` (``tradingagents.memory.store
  .resolve_memory_id_to_db_path``), read directly with ``sqlite3`` — this
  script never calls the memory MCP server (unlike ``trading_graph.py``,
  for which the MCP server is a hard dependency; this is an offline batch
  report over already-written data, so a live server is unnecessary
  overhead). Only the ``decisions`` table's ``portfolio_manager``-agent rows
  are used (the Portfolio Manager's ``final_trade_decision`` is the run's
  bottom-line 5-tier call — see ``tradingagents/graph/trading_graph.py``'s
  ``agent="portfolio_manager"`` store_decision call). A DB file that doesn't
  exist yet (arm never run) is treated as "no decisions", not an error.
- **KPI 3 (depot performance)**: each arm's ``McpTradingSimulation`` depot is
  a **separate SQLite file** the simulator manages directly on disk at
  ``<sim-data-dir>/depots/<depot_id>.db`` (confirmed by reading the sibling
  ``McpTradingSimulation`` checkout's ``trading_sim/config.py``
  ``get_db_path``/``trading_sim/models.py``), *not* something this script
  reaches over the ``McpTradingSimulation`` MCP server (``SimulationClient``
  in ``tradingagents/simulation/__init__.py``) — a live server is neither
  necessary nor desirable for an offline report over historical data, so
  this script reads the depot's ``trade``/``depot``/``configrow`` tables
  directly, mirroring the KPI 1/2 direct-SQLite-read choice above.
  ``<sim-data-dir>`` defaults to the sibling-checkout heuristic
  ``tradingagents/simulation/__init__.py``'s ``_get_server_config`` already
  uses (``<repo-root>/../McpTradingSimulation/data``); override with
  ``--sim-data-dir`` if the checkout lives elsewhere. **Equity-curve
  approximation**: the simulator has no historical daily mark-to-market
  snapshot table — ``get_portfolio`` (live quotes) only tells you *today's*
  equity. This script instead replays the append-only ``trade`` ledger in
  timestamp order, marking each held symbol at *its own last observed trade
  price* (falling back to nothing held = no contribution) and using each
  trade's own ``cash_after`` for cash — i.e. equity is marked to the last
  price this depot actually traded at, not a live quote. This is a
  documented approximation, not true daily mark-to-market; it is exact at
  every point a trade occurs and only drifts from "true" equity between
  trades if the market price moved and this depot didn't trade that symbol
  that day. "Daily" equity path buckets to one point per calendar day (the
  last trade of that day).
- **KPI 4 (token cost proxy)**: ``full_states_log_<date>.json`` files under
  ``<results_dir>/<TICKER>/TradingAgentsStrategy_logs/`` (written by
  ``TradingAgentsGraph._log_state``, ``tradingagents/graph/trading_graph.py``).
  Important: ``results_dir`` is a **separate** ``DEFAULT_CONFIG`` key from
  the run-config file's top-level ``report_dir`` (``report_dir`` only
  controls ``save_report_to_disk``'s per-ticker markdown/JSON report tree
  and the portfolio-mode run report; full-state logs are written under
  ``results_dir``, which defaults to the *shared* ``~/.tradingagents/logs``
  if never overridden). Every arm config under ``experiments/ablation-analysts/``
  now sets its own ``config.results_dir`` (``runs/results/<arm>``) precisely
  so this KPI's input is actually isolated per arm instead of every arm's
  full-state logs colliding in one shared directory (a gap in the original
  #120 artifact layout fixed alongside this script — see the arm JSON files
  and README "Isolation Guarantees" for the isolated paths). Token count is
  the char/4 proxy specified by #81/#121: total characters across a run's
  ``full_states_log_*.json`` file, divided by 4.
- **KPI 5 (repeat-run rating variance)**: the stability-block memory DBs
  (``runs/memory/<arm>-stability/memory.db``), same direct-SQLite read as
  KPI 1/2, grouped by ticker across every stored ``portfolio_manager``
  decision for that ticker in that arm's stability DB (each row is one
  repeat run — see ``experiments/ablation-analysts/README.md``'s "Stability
  Block" section for why repeats land on distinct ``decision_date`` values:
  ``store_decision``'s ``UNIQUE(agent, ticker, decision_date)`` constraint
  makes same-day re-runs a no-op, so a real repeat requires a different
  trading day).

Partial-data handling
----------------------
Every KPI section degrades gracefully: a missing arm config, a memory DB or
depot DB that doesn't exist yet, an unresolved (pending) forward return, or a
ticker-day present in one arm but not another is reported as an explicit gap
in the relevant section rather than raising or silently being dropped.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ``scripts`` is not an installed package (pyproject.toml's packages.find only
# includes tradingagents*/cli*), so `python scripts/ablation_report.py` alone
# leaves the repo root off sys.path and `from scripts.run_ablation import
# get_arm_configs` below would fail with ModuleNotFoundError. Insert the repo
# root explicitly so this script works both invoked directly and under pytest
# (which already puts the rootdir on sys.path, making this a no-op there).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.run_ablation import get_arm_configs  # noqa: E402
from tradingagents.agents.utils.rating import RATINGS_5_TIER  # noqa: E402
from tradingagents.memory.store import resolve_memory_id_to_db_path  # noqa: E402

# Verbatim from issue #81's "Known limitation to state up front" (unchanged
# from the #69 proposal) — #121 requires this caveat be stated up front in
# the report, quoted exactly rather than paraphrased.
CAVEAT_TEXT = (
    "Known limitation to state up front (unchanged from the #69 proposal): "
    "with ~weeks of data on a handful of tickers, return-based KPIs cannot "
    "statistically separate arms — flip rate, cost, and stability are the "
    "decisive measures; returns are directional evidence only."
)

DEFAULT_ARMS_DIR = Path("experiments/ablation-analysts")
CONTROL_ARM_NAME = "control"

_TIER_INDEX = {rating.lower(): i for i, rating in enumerate(RATINGS_5_TIER)}


def tier_index(signal: str | None) -> int | None:
    """Map a 5-tier rating string to its ordinal index (Buy=0 .. Sell=4).

    Returns ``None`` for an unrecognized/empty signal so callers can exclude
    it from stats rather than guess.
    """
    if not signal:
        return None
    return _TIER_INDEX.get(signal.strip().lower())


def _fmt_keys(keys: list[tuple[str, str]], limit: int = 5) -> str:
    shown = ", ".join(f"{ticker}/{date}" for ticker, date in keys[:limit])
    if len(keys) > limit:
        shown += f", ... ({len(keys) - limit} more)"
    return shown


# ─── Arm discovery ──────────────────────────────────────────────────────────


@dataclass
class Arm:
    name: str
    config: dict[str, Any]
    config_path: Path
    is_stability: bool = False

    @property
    def memory_id(self) -> str | None:
        return self.config.get("memory_id")

    @property
    def depot_id(self) -> str | None:
        return self.config.get("depot_id")

    @property
    def results_dir(self) -> str | None:
        return self.config.get("config", {}).get("results_dir")

    @property
    def memory_db_path(self) -> Path:
        return resolve_memory_id_to_db_path(self.memory_id)


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def discover_main_arms(arms_dir: Path) -> dict[str, Arm]:
    """Discover the six main arm configs (portfolio-mode, vs.-control KPIs).

    Reuses ``scripts.run_ablation.get_arm_configs`` (issue #120) for
    discovery so the two scripts can never drift on what counts as a "main
    arm" config file.
    """
    arms: dict[str, Arm] = {}
    if not arms_dir.exists():
        return arms
    for config_path in get_arm_configs(arms_dir, stability=False):
        name = config_path.stem
        arms[name] = Arm(name=name, config=_load_json(config_path), config_path=config_path)
    return arms


def discover_stability_arms(arms_dir: Path) -> dict[str, Arm]:
    """Discover the six stability-block configs (``*.stability.json``)."""
    arms: dict[str, Arm] = {}
    if not arms_dir.exists():
        return arms
    for config_path in get_arm_configs(arms_dir, stability=True):
        # config_path.stem for "no-market.stability.json" is "no-market.stability"
        name = config_path.stem.removesuffix(".stability")
        arms[name] = Arm(
            name=name, config=_load_json(config_path), config_path=config_path, is_stability=True
        )
    return arms


# ─── KPI 1/2 shared: reading decisions ─────────────────────────────────────


def load_agent_decisions(db_path: Path, agent: str = "portfolio_manager") -> dict[tuple[str, str], dict]:
    """Read ``(ticker, decision_date) -> decision row`` for one agent.

    Reads directly with ``sqlite3`` rather than
    ``tradingagents.memory.store.get_connection`` deliberately: that helper
    creates the DB file/schema as a side effect of merely opening a
    connection, which would litter ``runs/memory/`` with empty DB files for
    arms that haven't run yet. A report script should never have that side
    effect — a missing DB is legitimately "no decisions yet", not an error.
    """
    if not db_path.exists():
        return {}
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT ticker, decision_date, signal, confidence, forward_return, resolved_at "
            "FROM decisions WHERE agent = ? ORDER BY ticker, decision_date",
            (agent,),
        ).fetchall()
    finally:
        conn.close()
    return {
        (row["ticker"], row["decision_date"]): {
            "signal": row["signal"],
            "confidence": row["confidence"],
            "forward_return": row["forward_return"],
            "resolved": row["resolved_at"] is not None,
        }
        for row in rows
    }


# ─── KPI 1: decision agreement / flip rate ─────────────────────────────────


@dataclass
class AgreementResult:
    arm: str
    n_compared: int = 0
    n_exact: int = 0
    n_within1: int = 0
    n_flips: int = 0
    disagreements: list[dict] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)


def compute_agreement(
    arm_name: str,
    arm_decisions: dict[tuple[str, str], dict],
    control_decisions: dict[tuple[str, str], dict],
) -> AgreementResult:
    gaps: list[str] = []
    arm_keys = set(arm_decisions)
    control_keys = set(control_decisions)
    common = sorted(arm_keys & control_keys)
    arm_only = sorted(arm_keys - control_keys)
    control_only = sorted(control_keys - arm_keys)

    if not arm_decisions:
        gaps.append(f"'{arm_name}' has no recorded portfolio_manager decisions yet (arm not run?)")
    if not control_decisions:
        gaps.append("control has no recorded portfolio_manager decisions yet (control not run?)")
    if arm_only:
        gaps.append(
            f"{len(arm_only)} ticker-day(s) in '{arm_name}' missing from control: {_fmt_keys(arm_only)}"
        )
    if control_only:
        gaps.append(
            f"{len(control_only)} ticker-day(s) in control missing from '{arm_name}': {_fmt_keys(control_only)}"
        )

    n_exact = n_within1 = 0
    disagreements: list[dict] = []
    for key in common:
        arm_signal = arm_decisions[key]["signal"]
        control_signal = control_decisions[key]["signal"]
        arm_idx = tier_index(arm_signal)
        control_idx = tier_index(control_signal)
        if arm_idx is None or control_idx is None:
            gaps.append(
                f"Unrecognized signal for {key[0]}/{key[1]} "
                f"(arm={arm_signal!r}, control={control_signal!r}) — excluded from agreement stats"
            )
            continue
        diff = abs(arm_idx - control_idx)
        if diff == 0:
            n_exact += 1
        else:
            disagreements.append(
                {
                    "ticker": key[0],
                    "date": key[1],
                    "arm_signal": arm_signal,
                    "control_signal": control_signal,
                }
            )
        if diff <= 1:
            n_within1 += 1

    n_compared = n_exact + len(disagreements)
    return AgreementResult(
        arm=arm_name,
        n_compared=n_compared,
        n_exact=n_exact,
        n_within1=n_within1,
        n_flips=len(disagreements),
        disagreements=disagreements,
        gaps=gaps,
    )


# ─── KPI 2: decision-aligned forward return, paired on disagreements ───────


@dataclass
class ForwardReturnResult:
    arm: str
    n_resolved: int = 0
    n_pending: int = 0
    arm_mean_return: float | None = None
    control_mean_return: float | None = None
    mean_paired_diff: float | None = None
    rows: list[dict] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)


def compute_paired_forward_returns(
    arm_name: str,
    disagreements: list[dict],
    arm_decisions: dict[tuple[str, str], dict],
    control_decisions: dict[tuple[str, str], dict],
) -> ForwardReturnResult:
    gaps: list[str] = []
    rows: list[dict] = []
    resolved_pairs: list[tuple[float, float]] = []
    n_pending = 0

    for disagreement in disagreements:
        key = (disagreement["ticker"], disagreement["date"])
        arm_row = arm_decisions[key]
        control_row = control_decisions[key]
        arm_return = arm_row["forward_return"]
        control_return = control_row["forward_return"]
        resolved = arm_row["resolved"] and control_row["resolved"] and arm_return is not None and control_return is not None
        rows.append(
            {
                **disagreement,
                "arm_return": arm_return,
                "control_return": control_return,
                "resolved": resolved,
            }
        )
        if resolved:
            resolved_pairs.append((arm_return, control_return))
        else:
            n_pending += 1

    arm_mean = control_mean = mean_diff = None
    if not disagreements:
        gaps.append(f"No disagreement ticker-days for '{arm_name}' vs. control — nothing to pair")
    elif resolved_pairs:
        arm_mean = statistics.mean(pair[0] for pair in resolved_pairs)
        control_mean = statistics.mean(pair[1] for pair in resolved_pairs)
        mean_diff = statistics.mean(a - c for a, c in resolved_pairs)
    else:
        gaps.append(
            f"All {n_pending} disagreement ticker-day(s) for '{arm_name}' are still pending "
            f"resolution (forward_return not yet available)"
        )

    return ForwardReturnResult(
        arm=arm_name,
        n_resolved=len(resolved_pairs),
        n_pending=n_pending,
        arm_mean_return=arm_mean,
        control_mean_return=control_mean,
        mean_paired_diff=mean_diff,
        rows=rows,
        gaps=gaps,
    )


# ─── KPI 3: depot performance ───────────────────────────────────────────────


@dataclass
class DepotResult:
    arm: str
    depot_id: str | None = None
    db_path: Path | None = None
    found: bool = False
    initial_cash: float | None = None
    final_equity: float | None = None
    max_drawdown: float | None = None
    daily_equity: list[tuple[str, float]] = field(default_factory=list)
    n_trades: int = 0
    gaps: list[str] = field(default_factory=list)


def resolve_depot_db_path(depot_id: str, sim_data_dir: Path | None = None) -> Path:
    """Resolve a depot_id to its on-disk SQLite path.

    Mirrors the sibling-checkout heuristic ``tradingagents/simulation/__init__.py``'s
    ``_get_server_config`` already uses for the MCP server binary, applied to the
    simulator's ``data/`` directory instead: ``<repo-root>/../McpTradingSimulation/data``.
    Pass ``sim_data_dir`` to override (e.g. if the checkout lives elsewhere).
    """
    if sim_data_dir is not None:
        return Path(sim_data_dir) / "depots" / f"{depot_id}.db"
    repo_root = Path(__file__).resolve().parent.parent
    sibling_data_dir = repo_root.parent / "McpTradingSimulation" / "data"
    return sibling_data_dir / "depots" / f"{depot_id}.db"


def _replay_equity_curve(conn: sqlite3.Connection, initial_cash: float) -> list[tuple[str, float]]:
    """Replay the ``trade`` ledger to approximate an equity curve.

    See the module docstring's "Equity-curve approximation" for the exact
    semantics: cash tracks each trade's own ``cash_after``; each held
    symbol's contribution to equity is marked at its own last observed trade
    price. Returns a list of ``(timestamp, equity)`` points, one per trade,
    in ascending ``ts`` order (bucketing to one-per-day happens in the
    caller).
    """
    rows = conn.execute(
        "SELECT ts, symbol, side, quantity, price, cash_after FROM trade ORDER BY ts, id"
    ).fetchall()

    positions: dict[str, int] = {}
    last_price: dict[str, float] = {}
    curve: list[tuple[str, float]] = []
    for row in rows:
        symbol, side, quantity, price, cash_after = (
            row["symbol"],
            row["side"],
            row["quantity"],
            row["price"],
            row["cash_after"],
        )
        positions[symbol] = positions.get(symbol, 0) + (quantity if side == "buy" else -quantity)
        last_price[symbol] = price
        positions_value = sum(
            qty * last_price[sym] for sym, qty in positions.items() if qty != 0 and sym in last_price
        )
        curve.append((row["ts"], cash_after + positions_value))
    return curve


def _bucket_daily(curve: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """Collapse a per-trade equity curve to one point per calendar day (the day's last trade)."""
    daily: dict[str, float] = {}
    for ts, equity in curve:
        day = ts.split(" ")[0].split("T")[0]
        daily[day] = equity  # later entries (curve is chronological) overwrite earlier same-day ones
    return sorted(daily.items())


def _max_drawdown(daily_equity: list[tuple[str, float]]) -> float | None:
    if not daily_equity:
        return None
    peak = daily_equity[0][1]
    max_dd = 0.0
    for _, equity in daily_equity:
        peak = max(peak, equity)
        if peak > 0:
            max_dd = max(max_dd, (peak - equity) / peak)
    return max_dd


def compute_depot_performance(
    arm_name: str, depot_id: str | None, sim_data_dir: Path | None = None
) -> DepotResult:
    if not depot_id:
        return DepotResult(
            arm=arm_name,
            gaps=[f"'{arm_name}' has no depot_id configured (stability/no-portfolio arm) — KPI 3 not applicable"],
        )

    db_path = resolve_depot_db_path(depot_id, sim_data_dir)
    if not db_path.exists():
        return DepotResult(
            arm=arm_name,
            depot_id=depot_id,
            db_path=db_path,
            gaps=[f"Depot DB not found for '{arm_name}' at {db_path} — arm may not have run in portfolio mode yet"],
        )

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        config_row = conn.execute("SELECT initial_cash FROM configrow WHERE id = 1").fetchone()
        depot_row = conn.execute("SELECT cash_balance FROM depot WHERE id = 1").fetchone()
        initial_cash = config_row["initial_cash"] if config_row else None
        curve = _replay_equity_curve(conn, initial_cash or 0.0)
        n_trades = len(curve)
    finally:
        conn.close()

    gaps: list[str] = []
    daily_equity = _bucket_daily(curve)
    if curve:
        final_equity = curve[-1][1]
    elif depot_row is not None:
        final_equity = depot_row["cash_balance"]
        gaps.append(f"'{arm_name}' depot has no trades yet — final equity is just cash balance")
    else:
        final_equity = None
        gaps.append(f"'{arm_name}' depot DB has no trade or depot rows — cannot determine equity")

    return DepotResult(
        arm=arm_name,
        depot_id=depot_id,
        db_path=db_path,
        found=True,
        initial_cash=initial_cash,
        final_equity=final_equity,
        max_drawdown=_max_drawdown(daily_equity),
        daily_equity=daily_equity,
        n_trades=n_trades,
        gaps=gaps,
    )


# ─── KPI 4: token cost proxy ────────────────────────────────────────────────


@dataclass
class TokenCostResult:
    arm: str
    results_dir: Path | None = None
    found: bool = False
    n_files: int = 0
    total_chars: int = 0
    total_token_proxy: float = 0.0
    per_ticker: dict[str, float] = field(default_factory=dict)
    gaps: list[str] = field(default_factory=list)


def compute_token_cost(arm_name: str, results_dir: str | None) -> TokenCostResult:
    if not results_dir:
        return TokenCostResult(
            arm=arm_name,
            gaps=[f"'{arm_name}' has no config.results_dir set — cannot locate full-state logs"],
        )

    resolved_dir = Path(results_dir).expanduser()
    if not resolved_dir.exists():
        return TokenCostResult(
            arm=arm_name,
            results_dir=resolved_dir,
            gaps=[f"results_dir not found for '{arm_name}' at {resolved_dir} — arm may not have run yet"],
        )

    files = sorted(resolved_dir.glob("*/TradingAgentsStrategy_logs/full_states_log_*.json"))
    gaps: list[str] = []
    per_ticker: dict[str, float] = {}
    total_chars = 0
    for log_file in files:
        text = log_file.read_text(encoding="utf-8")
        total_chars += len(text)
        ticker = log_file.parent.parent.name
        try:
            data = json.loads(text)
            ticker = data.get("company_of_interest", ticker)
        except json.JSONDecodeError:
            gaps.append(f"{log_file} is not valid JSON — chars counted, ticker attribution may be off")
        per_ticker[ticker] = per_ticker.get(ticker, 0.0) + len(text) / 4

    if not files:
        gaps.append(f"No full-state log files found under {resolved_dir} for '{arm_name}'")

    return TokenCostResult(
        arm=arm_name,
        results_dir=resolved_dir,
        found=True,
        n_files=len(files),
        total_chars=total_chars,
        total_token_proxy=total_chars / 4,
        per_ticker=per_ticker,
        gaps=gaps,
    )


# ─── KPI 5: repeat-run rating variance (stability block) ──────────────────


@dataclass
class StabilityResult:
    arm: str
    db_path: Path | None = None
    found: bool = False
    per_ticker: dict[str, dict] = field(default_factory=dict)
    gaps: list[str] = field(default_factory=list)


def compute_stability_variance(arm_name: str, db_path: Path) -> StabilityResult:
    if not db_path.exists():
        return StabilityResult(
            arm=arm_name,
            db_path=db_path,
            gaps=[f"Stability memory DB not found for '{arm_name}' at {db_path} — stability block may not have run yet"],
        )

    decisions = load_agent_decisions(db_path)
    by_ticker: dict[str, list[dict]] = {}
    for (ticker, date), row in decisions.items():
        by_ticker.setdefault(ticker, []).append({"date": date, **row})

    gaps: list[str] = []
    per_ticker: dict[str, dict] = {}
    for ticker, rows in sorted(by_ticker.items()):
        rows.sort(key=lambda r: r["date"])
        tier_indices = [idx for idx in (tier_index(r["signal"]) for r in rows) if idx is not None]
        confidences = [r["confidence"] for r in rows if r["confidence"] is not None]

        signal_tier_variance = statistics.pvariance(tier_indices) if len(tier_indices) >= 2 else None
        confidence_variance = statistics.pvariance(confidences) if len(confidences) >= 2 else None

        if len(rows) < 2:
            gaps.append(f"'{arm_name}'/{ticker}: only {len(rows)} repeat(s) recorded — variance needs >= 2")

        per_ticker[ticker] = {
            "n": len(rows),
            "decisions": rows,
            "signal_tier_variance": signal_tier_variance,
            "confidence_variance": confidence_variance,
        }

    if not per_ticker:
        gaps.append(f"No portfolio_manager decisions found in stability DB for '{arm_name}'")

    return StabilityResult(arm=arm_name, db_path=db_path, found=True, per_ticker=per_ticker, gaps=gaps)


# ─── Report rendering ───────────────────────────────────────────────────────


def _fmt_pct(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "n/a"
    return f"{100 * numerator / denominator:.1f}%"


def _fmt_num(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def render_report(
    generated_at: str,
    arms_dir: Path,
    agreements: dict[str, AgreementResult],
    forward_returns: dict[str, ForwardReturnResult],
    depots: dict[str, DepotResult],
    token_costs: dict[str, TokenCostResult],
    stability: dict[str, StabilityResult],
) -> str:
    lines: list[str] = []
    lines.append("# Ablation KPI Report")
    lines.append("")
    lines.append(f"Generated: {generated_at}")
    lines.append(f"Arms directory: `{arms_dir}`")
    lines.append("")
    lines.append("> " + CAVEAT_TEXT)
    lines.append("")
    lines.append(
        "The keep/merge/drop call for each ablated agent is human judgment made on top of "
        "this report — this script encodes no decision rule (see #81)."
    )
    lines.append("")

    # KPI 1
    lines.append("## KPI 1: Decision agreement / flip rate (vs. control)")
    lines.append("")
    lines.append("| Arm | Compared | Exact agreement | Within ±1 tier | Flips |")
    lines.append("|---|---:|---:|---:|---:|")
    for arm_name, result in agreements.items():
        lines.append(
            f"| {arm_name} | {result.n_compared} | "
            f"{result.n_exact} ({_fmt_pct(result.n_exact, result.n_compared)}) | "
            f"{result.n_within1} ({_fmt_pct(result.n_within1, result.n_compared)}) | "
            f"{result.n_flips} |"
        )
    lines.append("")
    for arm_name, result in agreements.items():
        if result.disagreements:
            lines.append(f"**{arm_name} disagreeing ticker-days:**")
            for d in result.disagreements:
                lines.append(
                    f"- {d['ticker']}/{d['date']}: arm={d['arm_signal']} vs. control={d['control_signal']}"
                )
            lines.append("")
        if result.gaps:
            lines.append(f"**{arm_name} gaps:**")
            for gap in result.gaps:
                lines.append(f"- {gap}")
            lines.append("")

    # KPI 2
    lines.append("## KPI 2: Decision-aligned 10-trading-day forward return (paired, disagreement cases only)")
    lines.append("")
    lines.append("| Arm | Resolved pairs | Pending | Arm mean return | Control mean return | Mean paired diff |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for arm_name, result in forward_returns.items():
        lines.append(
            f"| {arm_name} | {result.n_resolved} | {result.n_pending} | "
            f"{_fmt_num(result.arm_mean_return)} | {_fmt_num(result.control_mean_return)} | "
            f"{_fmt_num(result.mean_paired_diff)} |"
        )
    lines.append("")
    for arm_name, result in forward_returns.items():
        if result.gaps:
            lines.append(f"**{arm_name} gaps:**")
            for gap in result.gaps:
                lines.append(f"- {gap}")
            lines.append("")

    # KPI 3
    lines.append("## KPI 3: Depot performance")
    lines.append("")
    lines.append("| Arm | Depot ID | Trades | Initial cash | Final equity | Max drawdown |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for arm_name, result in depots.items():
        if not result.found:
            lines.append(f"| {arm_name} | {result.depot_id or 'n/a'} | - | - | - | - |")
            continue
        lines.append(
            f"| {arm_name} | {result.depot_id} | {result.n_trades} | "
            f"{_fmt_num(result.initial_cash, 2)} | {_fmt_num(result.final_equity, 2)} | "
            f"{_fmt_num(result.max_drawdown, 4)} |"
        )
    lines.append("")
    for arm_name, result in depots.items():
        if result.gaps:
            lines.append(f"**{arm_name} gaps:**")
            for gap in result.gaps:
                lines.append(f"- {gap}")
            lines.append("")

    # KPI 4
    lines.append("## KPI 4: Token cost per run (char/4 proxy over full-state logs)")
    lines.append("")
    lines.append("| Arm | Runs (ticker-days) | Total token proxy |")
    lines.append("|---|---:|---:|")
    for arm_name, result in token_costs.items():
        lines.append(f"| {arm_name} | {result.n_files} | {_fmt_num(result.total_token_proxy, 0)} |")
    lines.append("")
    for arm_name, result in token_costs.items():
        if result.gaps:
            lines.append(f"**{arm_name} gaps:**")
            for gap in result.gaps:
                lines.append(f"- {gap}")
            lines.append("")

    # KPI 5
    lines.append("## KPI 5: Repeat-run rating variance (stability block)")
    lines.append("")
    for arm_name, result in stability.items():
        lines.append(f"### {arm_name}")
        if result.per_ticker:
            lines.append("| Ticker | Repeats | Signal tier variance | Confidence variance |")
            lines.append("|---|---:|---:|---:|")
            for ticker, stats in result.per_ticker.items():
                lines.append(
                    f"| {ticker} | {stats['n']} | {_fmt_num(stats['signal_tier_variance'])} | "
                    f"{_fmt_num(stats['confidence_variance'])} |"
                )
            lines.append("")
        if result.gaps:
            for gap in result.gaps:
                lines.append(f"- {gap}")
            lines.append("")

    return "\n".join(lines) + "\n"


# ─── Main ────────────────────────────────────────────────────────────────


def build_report(arms_dir: Path, sim_data_dir: Path | None = None) -> str:
    """Discover arms, compute all five KPIs, and render the markdown report."""
    main_arms = discover_main_arms(arms_dir)
    stability_arms = discover_stability_arms(arms_dir)

    generated_at = datetime.now(timezone.utc).isoformat()

    control = main_arms.get(CONTROL_ARM_NAME)
    control_decisions = load_agent_decisions(control.memory_db_path) if control else {}

    agreements: dict[str, AgreementResult] = {}
    forward_returns: dict[str, ForwardReturnResult] = {}
    depots: dict[str, DepotResult] = {}
    token_costs: dict[str, TokenCostResult] = {}

    # Report control's own depot/token-cost numbers as a baseline reference row.
    if control:
        depots[CONTROL_ARM_NAME] = compute_depot_performance(
            CONTROL_ARM_NAME, control.depot_id, sim_data_dir
        )
        token_costs[CONTROL_ARM_NAME] = compute_token_cost(CONTROL_ARM_NAME, control.results_dir)

    for arm_name, arm in main_arms.items():
        if arm_name == CONTROL_ARM_NAME:
            continue
        arm_decisions = load_agent_decisions(arm.memory_db_path)
        agreement = compute_agreement(arm_name, arm_decisions, control_decisions)
        agreements[arm_name] = agreement
        forward_returns[arm_name] = compute_paired_forward_returns(
            arm_name, agreement.disagreements, arm_decisions, control_decisions
        )
        depots[arm_name] = compute_depot_performance(arm_name, arm.depot_id, sim_data_dir)
        token_costs[arm_name] = compute_token_cost(arm_name, arm.results_dir)

    if control is None:
        agreements[CONTROL_ARM_NAME] = AgreementResult(
            arm=CONTROL_ARM_NAME, gaps=["control arm config not found under --arms-dir"]
        )

    stability: dict[str, StabilityResult] = {}
    for arm_name, arm in stability_arms.items():
        stability[arm_name] = compute_stability_variance(arm_name, arm.memory_db_path)

    return render_report(
        generated_at=generated_at,
        arms_dir=arms_dir,
        agreements=agreements,
        forward_returns=forward_returns,
        depots=depots,
        token_costs=token_costs,
        stability=stability,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute the #81 ablation-study KPIs and write a markdown report."
    )
    parser.add_argument(
        "--arms-dir",
        type=Path,
        default=DEFAULT_ARMS_DIR,
        help=f"Directory containing arm config files (default: {DEFAULT_ARMS_DIR})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Path to write the markdown report (default: <arms-dir>/ablation_report.md)",
    )
    parser.add_argument(
        "--sim-data-dir",
        type=Path,
        default=None,
        help="Override the McpTradingSimulation data/ directory (default: sibling checkout)",
    )
    args = parser.parse_args()

    if not args.arms_dir.exists():
        print(f"Error: arms directory '{args.arms_dir}' does not exist")
        sys.exit(1)

    out_path = args.out or (args.arms_dir / "ablation_report.md")
    report = build_report(args.arms_dir, sim_data_dir=args.sim_data_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"Ablation KPI report written to: {out_path}")


if __name__ == "__main__":
    main()
