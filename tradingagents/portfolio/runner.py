"""Portfolio-mode orchestration for ``run_trading_agents.py``.

Given per-ticker ratings already produced by a full run of the five-stage
pipeline (see ``tradingagents.graph.trading_graph.TradingAgentsGraph``),
this module:

1. Maps each ticker's final 5-tier rating to a style-table signal
   (:func:`tradingagents.portfolio.rebalance.rating_to_signal`).
2. Fetches current quotes for the universe from the simulator, falling
   back to the depot's last recorded price if a quote is unavailable, and
   dropping the ticker (with a warning) if neither is available.
3. Computes the target allocation
   (:func:`tradingagents.portfolio.rebalance.compute_allocation`).
4. Executes rebalancing trades against the simulator: sells first (to
   free cash), then buys sorted by conviction (``raw_weight``, descending).
   Rejected orders are logged but never abort the run.
5. Writes a JSON run report to the report directory, shaped like the
   ``skills/portfolio-manager`` reference skill's envelope ``details``.

Tickers held in the depot but absent from the universe are never touched —
this module only ever computes allocation/orders for tickers it has a
signal for.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any

from tradingagents.agents.utils.rating import RATINGS_5_TIER
from tradingagents.graph.signal_processing import SignalProcessor
from tradingagents.portfolio.rebalance import STYLE_PARAMS, compute_allocation, rating_to_signal
from tradingagents.simulation import SimulationClient, SimulationClientError

DEFAULT_INITIAL_CASH = 100_000.0


def extract_rating(final_state: dict[str, Any]) -> str:
    """Extract the final 5-tier rating from a completed pipeline run's state.

    Prefers the Portfolio Manager's structured output
    (``final_state["portfolio_structured_data"]["rating"]``); falls back to
    the deterministic heuristic parser over ``final_trade_decision`` markdown
    when structured data isn't available (e.g. the provider doesn't support
    structured output and the agent fell back to free text).

    Note: As of issue #156, a completely failed Portfolio Manager decision
    (no structured output or invalid rating when
    ``portfolio_manager_require_structured_decision=True``) aborts the ticker
    with ``PortfolioDecisionError`` and never reaches this function. The
    ``SignalProcessor`` fallback below therefore only handles legitimate
    free-text fallback scenarios, not failed decisions.
    """
    structured = final_state.get("portfolio_structured_data") or {}
    rating = structured.get("rating")
    if rating:
        rating_str = rating.value if hasattr(rating, "value") else str(rating)
        if rating_str in RATINGS_5_TIER:
            return rating_str

    return SignalProcessor().process_signal(final_state.get("final_trade_decision", "") or "")


def build_signals(ratings: dict[str, str]) -> dict[str, dict[str, str]]:
    """Map ``{ticker: rating}`` to ``{ticker: {signal, confidence}}`` via the style tables."""
    return {ticker: rating_to_signal(rating) for ticker, rating in ratings.items()}


def _fetch_prices(
    client: SimulationClient,
    universe: list[str],
    pre_positions: dict[str, Any],
) -> tuple[dict[str, float], list[str]]:
    """Fetch current quotes for ``universe``, falling back to depot price.

    Returns ``(prices, dropped)`` where ``dropped`` lists tickers with no
    usable price (neither a fresh quote nor a recorded position price).
    """
    prices: dict[str, float] = {}
    dropped: list[str] = []

    for ticker in universe:
        price = None
        try:
            quote = client.get_quote(ticker)
            if isinstance(quote, dict) and not quote.get("error"):
                price = quote.get("price")
        except SimulationClientError as exc:
            print(f"Warning: get_quote failed for {ticker}: {exc}")

        if price is None:
            fallback = pre_positions.get(ticker, {}).get("price")
            if fallback:
                price = fallback

        if price is None:
            print(f"Warning: no quote available for {ticker} — dropping from this run")
            dropped.append(ticker)
            continue

        prices[ticker] = price

    return prices, dropped


def _execute_trades(
    client: SimulationClient,
    depot_id: str,
    allocation: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Execute sells first, then buys sorted by conviction (raw_weight) descending."""
    sells = [(t, a) for t, a in allocation.items() if a["delta"] < 0]
    buys = [(t, a) for t, a in allocation.items() if a["delta"] > 0]
    buys.sort(key=lambda kv: kv[1]["raw_weight"], reverse=True)

    trades_executed: list[dict[str, Any]] = []
    for ticker, alloc in [*sells, *buys]:
        side = "sell" if alloc["delta"] < 0 else "buy"
        quantity = abs(alloc["delta"])
        try:
            result = client.place_order(ticker, side, quantity, depot_id)
        except SimulationClientError as exc:
            trades_executed.append({
                "symbol": ticker,
                "side": side,
                "quantity": quantity,
                "status": "rejected",
                "message": str(exc),
                "fill_price": None,
            })
            continue

        trades_executed.append({
            "symbol": ticker,
            "side": side,
            "quantity": quantity,
            "status": result.get("status", "unknown"),
            "message": result.get("message", ""),
            "fill_price": result.get("price"),
        })

    return trades_executed


def run_portfolio_mode(
    universe: list[str],
    ratings: dict[str, str],
    style: str,
    depot_id: str,
    report_dir: Path,
) -> tuple[dict[str, Any], Path]:
    """Run the portfolio-level rebalance and write the run report to disk.

    Args:
        universe: the full stock-list universe (order preserved for tickers
            that make it through to allocation).
        ratings: ``{ticker: rating}`` for tickers whose pipeline run
            succeeded (see :func:`extract_rating`). Tickers missing from
            this map are dropped from the run with a warning (their
            pipeline run failed upstream).
        style: "aggressive" or "conservative".
        depot_id: named simulated depot to get-or-create and trade in.
        report_dir: directory the JSON run report is written into
            (alongside the existing per-ticker reports).

    Returns:
        ``(envelope, report_file)`` — the envelope dict that was written
        and the path it was written to.
    """
    if style not in STYLE_PARAMS:
        raise ValueError(f"Unknown portfolio style: {style!r} (expected one of {list(STYLE_PARAMS)})")

    missing = [t for t in universe if t not in ratings]
    for ticker in missing:
        print(f"Warning: no rating for {ticker} (pipeline run failed?) — dropping from portfolio run")
    signal_universe = [t for t in universe if t in ratings]
    signals = build_signals({t: ratings[t] for t in signal_universe})

    with SimulationClient() as client:
        depots = client.list_depots()
        depot_ids = {d.get("id") for d in depots} if isinstance(depots, list) else set()
        if depot_id not in depot_ids:
            client.create_depot(depot_id, initial_cash=DEFAULT_INITIAL_CASH)

        pre_portfolio = client.get_portfolio(depot_id)
        pre_positions = pre_portfolio.get("positions", {}) or {}
        pre_snapshot = {
            "equity": pre_portfolio.get("total_equity", 0.0),
            "cash": pre_portfolio.get("cash", 0.0),
            "positions": pre_positions,
        }

        prices, dropped = _fetch_prices(client, signal_universe, pre_positions)
        effective_universe = [t for t in signal_universe if t not in dropped]
        signals = {t: signals[t] for t in effective_universe}

        allocation = compute_allocation(
            style, effective_universe, signals, prices, pre_snapshot["equity"], pre_positions
        )

        trades_executed = _execute_trades(client, depot_id, allocation)

        post_portfolio = client.get_portfolio(depot_id)
        post_snapshot = {
            "equity": post_portfolio.get("total_equity", 0.0),
            "cash": post_portfolio.get("cash", 0.0),
            "positions": post_portfolio.get("positions", {}) or {},
        }

    rejected_orders = [t for t in trades_executed if t["status"] == "rejected"]
    n_buys = sum(1 for t in trades_executed if t["side"] == "buy" and t["status"] != "rejected")
    n_sells = sum(1 for t in trades_executed if t["side"] == "sell" and t["status"] != "rejected")
    n_holds = len(effective_universe) - len(trades_executed)

    details = {
        "style": style,
        "depot_id": depot_id,
        "universe": effective_universe,
        "style_params": STYLE_PARAMS[style],
        "signals": signals,
        "allocation": allocation,
        "trades_executed": trades_executed,
        "rejected_orders": rejected_orders,
        "pre_snapshot": pre_snapshot,
        "post_snapshot": post_snapshot,
        "equity_change": round(post_snapshot["equity"] - pre_snapshot["equity"], 2),
    }

    envelope = {
        "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "style": style,
        "depot_id": depot_id,
        "summary": (
            f"{style} rebalance of {depot_id}: {n_buys} buys, {n_sells} sells, "
            f"{n_holds} holds — equity ${post_snapshot['equity']:,.0f}"
        ),
        "details": details,
    }

    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_file = report_dir / f"portfolio-manager-{depot_id}.json"
    report_file.write_text(json.dumps(envelope, indent=2), encoding="utf-8")

    return envelope, report_file
