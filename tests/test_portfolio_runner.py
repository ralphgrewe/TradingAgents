"""Tests for tradingagents.portfolio.runner: rating extraction and the
end-to-end portfolio-mode orchestration (allocation -> execution -> report).

The simulation client is faked (no real MCP subprocess); pipeline results are
supplied directly as final_state dicts, standing in for a mocked
TradingAgentsGraph.propagate() run.
"""

import json

import pytest

from tradingagents.portfolio import runner as runner_module
from tradingagents.portfolio.runner import build_signals, extract_rating, run_portfolio_mode


class FakeSimulationClient:
    """In-memory stand-in for tradingagents.simulation.SimulationClient.

    Tracks depots/positions/prices/orders so tests can assert on execution
    order and rejected-order handling without a real MCP subprocess.
    """

    def __init__(self, *, existing_depots=None, prices=None, portfolios=None, reject_symbols=None):
        self.existing_depots = existing_depots or []
        self.prices = prices or {}
        self.reject_symbols = reject_symbols or set()
        self.created_depot = None
        self.orders = []
        # portfolios: mutable dict {depot_id: {"cash":..., "total_equity":..., "positions": {...}}}
        self.portfolios = portfolios or {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def list_depots(self):
        return [{"id": d} for d in self.existing_depots]

    def create_depot(self, depot_id, initial_cash=10_000.0):
        self.created_depot = (depot_id, initial_cash)
        self.portfolios[depot_id] = {"cash": initial_cash, "total_equity": initial_cash, "positions": {}}
        return {"status": "created", "depot_id": depot_id, "initial_cash": initial_cash}

    def get_portfolio(self, depot_id="default"):
        return self.portfolios[depot_id]

    def get_quote(self, symbol):
        if symbol not in self.prices:
            return {"error": f"Symbol '{symbol}' not found"}
        return {"symbol": symbol, "price": self.prices[symbol], "is_fresh": True}

    def place_order(self, symbol, side, quantity, depot_id="default"):
        self.orders.append((symbol, side, quantity))
        if symbol in self.reject_symbols:
            return {"status": "rejected", "message": "Insufficient cash", "symbol": symbol, "side": side}

        # Apply the trade to the in-memory portfolio so post_snapshot reflects it.
        portfolio = self.portfolios[depot_id]
        positions = portfolio["positions"]
        price = self.prices.get(symbol) or positions.get(symbol, {}).get("price")
        current = positions.get(symbol, {"shares": 0, "price": price, "market_value": 0.0})
        delta_shares = quantity if side == "buy" else -quantity
        new_shares = current["shares"] + delta_shares
        cash_delta = -delta_shares * price
        portfolio["cash"] += cash_delta
        if new_shares > 0:
            positions[symbol] = {"shares": new_shares, "price": price, "market_value": new_shares * price}
        else:
            positions.pop(symbol, None)
        portfolio["total_equity"] = portfolio["cash"] + sum(
            p["market_value"] for p in positions.values()
        )
        return {
            "status": "executed",
            "message": "Order executed at market price",
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price,
        }

    def get_trades(self, limit=50, symbol="", depot_id="default"):
        return []


class TestExtractRating:
    def test_prefers_structured_data(self):
        final_state = {
            "portfolio_structured_data": {"rating": "Overweight"},
            "final_trade_decision": "**Rating**: Sell\n",
        }
        assert extract_rating(final_state) == "Overweight"

    def test_falls_back_to_markdown_parsing(self):
        final_state = {"final_trade_decision": "**Rating**: Sell\n\nSome text."}
        assert extract_rating(final_state) == "Sell"

    def test_falls_back_to_hold_default_when_nothing_present(self):
        assert extract_rating({}) == "Hold"

    def test_ignores_unrecognised_structured_rating(self):
        final_state = {
            "portfolio_structured_data": {"rating": "Not A Real Rating"},
            "final_trade_decision": "**Rating**: Buy\n",
        }
        assert extract_rating(final_state) == "Buy"


class TestBuildSignals:
    def test_maps_ratings_to_signals(self):
        signals = build_signals({"AAA": "Buy", "BBB": "Sell"})
        assert signals == {
            "AAA": {"signal": "BUY", "confidence": "HIGH"},
            "BBB": {"signal": "SELL", "confidence": "HIGH"},
        }


class TestRunPortfolioMode:
    def test_end_to_end_new_depot(self, tmp_path, monkeypatch):
        fake = FakeSimulationClient(
            existing_depots=[],
            prices={"AAA": 100.0, "BBB": 50.0},
        )
        monkeypatch.setattr(runner_module, "SimulationClient", lambda: fake)

        envelope, report_file, missing_ratings, missing_prices = run_portfolio_mode(
            universe=["AAA", "BBB"],
            ratings={"AAA": "Buy", "BBB": "Sell"},
            style="aggressive",
            depot_id="new-depot",
            report_dir=tmp_path,
        )

        # Depot was created with the fixed initial cash (100_000, not the
        # SimulationClient default of 10_000).
        assert fake.created_depot == ("new-depot", 100_000.0)

        # AAA (Buy/HIGH) is bought; BBB has no position so the HOLD/SELL exit
        # is a no-op (nothing to sell).
        assert ("AAA", "buy", allocation_shares(envelope, "AAA")) in fake.orders
        assert envelope["details"]["allocation"]["AAA"]["delta"] > 0
        assert envelope["details"]["allocation"]["BBB"]["delta"] == 0

        assert report_file == tmp_path / "portfolio-manager-new-depot.json"
        assert report_file.exists()
        on_disk = json.loads(report_file.read_text())
        assert on_disk == envelope

        # Verify missing_ratings and missing_prices are empty (all tickers succeeded)
        assert missing_ratings == []
        assert missing_prices == []

    def test_sells_execute_before_buys(self, tmp_path, monkeypatch):
        fake = FakeSimulationClient(
            existing_depots=["existing-depot"],
            prices={"AAA": 100.0, "BBB": 50.0},
            portfolios={
                "existing-depot": {
                    "cash": 5_000.0,
                    "total_equity": 10_000.0,
                    "positions": {"BBB": {"shares": 100, "price": 50.0, "market_value": 5_000.0}},
                }
            },
        )
        monkeypatch.setattr(runner_module, "SimulationClient", lambda: fake)

        run_portfolio_mode(
            universe=["AAA", "BBB"],
            ratings={"AAA": "Buy", "BBB": "Sell"},
            style="aggressive",
            depot_id="existing-depot",
            report_dir=tmp_path,
        )

        # Existing depot is reused, not recreated.
        assert fake.created_depot is None
        # BBB (SELL -> exit) is executed before AAA (BUY) is placed.
        order_symbols = [o[0] for o in fake.orders]
        assert order_symbols.index("BBB") < order_symbols.index("AAA")
        assert fake.orders[0][1] == "sell"

    def test_rejected_order_is_logged_not_fatal(self, tmp_path, monkeypatch):
        fake = FakeSimulationClient(
            existing_depots=["d"],
            prices={"AAA": 100.0},
            portfolios={"d": {"cash": 100_000.0, "total_equity": 100_000.0, "positions": {}}},
            reject_symbols={"AAA"},
        )
        monkeypatch.setattr(runner_module, "SimulationClient", lambda: fake)

        envelope, _, missing_ratings, missing_prices = run_portfolio_mode(
            universe=["AAA"],
            ratings={"AAA": "Buy"},
            style="aggressive",
            depot_id="d",
            report_dir=tmp_path,
        )

        assert len(envelope["details"]["rejected_orders"]) == 1
        assert envelope["details"]["rejected_orders"][0]["symbol"] == "AAA"
        assert envelope["details"]["rejected_orders"][0]["status"] == "rejected"

    def test_no_quote_drops_ticker_with_fallback_to_position_price(self, tmp_path, monkeypatch):
        # AAA has no quote available but does have an existing position with a
        # recorded price -> fallback price is used instead of dropping it.
        # BBB has neither a quote nor a position -> dropped.
        fake = FakeSimulationClient(
            existing_depots=["d"],
            prices={},  # no quotes available at all
            portfolios={
                "d": {
                    "cash": 1_000.0,
                    "total_equity": 6_000.0,
                    "positions": {"AAA": {"shares": 50, "price": 100.0, "market_value": 5_000.0}},
                }
            },
        )
        monkeypatch.setattr(runner_module, "SimulationClient", lambda: fake)

        envelope, _, missing_ratings, missing_prices = run_portfolio_mode(
            universe=["AAA", "BBB"],
            ratings={"AAA": "Hold", "BBB": "Buy"},
            style="aggressive",
            depot_id="d",
            report_dir=tmp_path,
        )

        assert "BBB" not in envelope["details"]["universe"]
        assert "AAA" in envelope["details"]["universe"]
        assert envelope["details"]["allocation"]["AAA"]["price"] == 100.0

        # Verify missing_prices correctly reports BBB as dropped
        assert missing_ratings == []
        assert missing_prices == ["BBB"]

    def test_missing_rating_drops_ticker(self, tmp_path, monkeypatch):
        # CCC's pipeline run failed upstream (no rating supplied) -> dropped
        # from the portfolio run entirely, without touching the simulator for it.
        fake = FakeSimulationClient(
            existing_depots=["d"],
            prices={"AAA": 100.0, "CCC": 10.0},
            portfolios={"d": {"cash": 100_000.0, "total_equity": 100_000.0, "positions": {}}},
        )
        monkeypatch.setattr(runner_module, "SimulationClient", lambda: fake)

        envelope, _, missing_ratings, missing_prices = run_portfolio_mode(
            universe=["AAA", "CCC"],
            ratings={"AAA": "Buy"},
            style="aggressive",
            depot_id="d",
            report_dir=tmp_path,
        )

        assert envelope["details"]["universe"] == ["AAA"]
        assert all(o[0] != "CCC" for o in fake.orders)

        # Verify missing_ratings correctly reports CCC as dropped
        assert missing_ratings == ["CCC"]
        assert missing_prices == []

    def test_missing_rating_from_portfolio_decision_error(self, tmp_path, monkeypatch, capsys):
        # Simulate the behavior when a ticker's pipeline fails with
        # PortfolioDecisionError (issue #156) — it won't be in the ratings dict
        # passed to run_portfolio_mode, so it's dropped as "pipeline run failed".
        fake = FakeSimulationClient(
            existing_depots=["d"],
            prices={"AAA": 100.0, "FAILED_TICKER": 50.0},
            portfolios={"d": {"cash": 100_000.0, "total_equity": 100_000.0, "positions": {}}},
        )
        monkeypatch.setattr(runner_module, "SimulationClient", lambda: fake)

        envelope, _, missing_ratings, missing_prices = run_portfolio_mode(
            universe=["AAA", "FAILED_TICKER"],
            ratings={"AAA": "Buy"},  # FAILED_TICKER is not in ratings
            style="aggressive",
            depot_id="d",
            report_dir=tmp_path,
        )

        # FAILED_TICKER is dropped from the portfolio run
        assert envelope["details"]["universe"] == ["AAA"]
        assert all(o[0] != "FAILED_TICKER" for o in fake.orders)

        # The warning was printed
        captured = capsys.readouterr()
        assert "no rating for FAILED_TICKER" in captured.out
        assert "pipeline run failed" in captured.out
        assert "dropping from portfolio run" in captured.out

        # Verify missing_ratings correctly reports FAILED_TICKER as dropped
        assert missing_ratings == ["FAILED_TICKER"]
        assert missing_prices == []

    def test_unknown_style_raises(self, tmp_path, monkeypatch):
        fake = FakeSimulationClient()
        monkeypatch.setattr(runner_module, "SimulationClient", lambda: fake)
        with pytest.raises(ValueError):
            run_portfolio_mode(
                universe=["AAA"],
                ratings={"AAA": "Buy"},
                style="not-a-style",
                depot_id="d",
                report_dir=tmp_path,
            )


def allocation_shares(envelope, ticker):
    return abs(envelope["details"]["allocation"][ticker]["delta"])
