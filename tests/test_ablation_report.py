"""Tests for scripts/ablation_report.py — the #81 ablation KPI report script (issue #121).

Covers each KPI computation and the partial-data paths against small,
fabricated fixture artifacts (synthetic memory DBs, full-state logs, depot
DBs) — no live LLM/MCP/simulator server is used anywhere in this file.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from scripts.ablation_report import (
    CAVEAT_TEXT,
    CONTROL_ARM_NAME,
    Arm,
    build_report,
    compute_agreement,
    compute_depot_performance,
    compute_paired_forward_returns,
    compute_stability_variance,
    compute_token_cost,
    discover_main_arms,
    discover_stability_arms,
    resolve_depot_db_path,
    tier_index,
)
from tradingagents.memory.store import get_connection

pytestmark = pytest.mark.unit


# ─── Fixture helpers ────────────────────────────────────────────────────────


def _make_memory_db(db_path: Path, rows: list[dict]) -> None:
    """Write portfolio_manager decision rows directly (bypassing store_decision's
    idempotency) — get_connection() creates the schema, then we insert freely so
    tests can construct rows with pre-set forward_return/resolved_at.
    """
    conn = get_connection(db_path)
    for row in rows:
        conn.execute(
            """
            INSERT INTO decisions (
                agent, ticker, decision_date, signal, confidence,
                created_at, forward_return, resolved_at
            ) VALUES (?, ?, ?, ?, ?, 'x', ?, ?)
            """,
            (
                row.get("agent", "portfolio_manager"),
                row["ticker"],
                row["date"],
                row["signal"],
                row.get("confidence"),
                row.get("forward_return"),
                row.get("resolved_at"),
            ),
        )
    conn.commit()
    conn.close()


def _make_depot_db(db_path: Path, initial_cash: float, trades: list[dict]) -> None:
    """Build a synthetic McpTradingSimulation depot DB (configrow/depot/trade tables),
    matching the schema in the sibling McpTradingSimulation checkout's trading_sim/models.py.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE configrow (
            id INTEGER PRIMARY KEY, fixed_fee REAL, percentage_fee REAL,
            max_fee REAL, initial_cash REAL, currency TEXT
        );
        CREATE TABLE depot (
            id INTEGER PRIMARY KEY, cash_balance REAL, created_at TEXT
        );
        CREATE TABLE trade (
            id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, symbol TEXT, side TEXT,
            quantity INTEGER, price REAL, fee REAL, gross REAL, net REAL,
            cash_after REAL, pending_order_id INTEGER, note TEXT
        );
        """
    )
    conn.execute(
        "INSERT INTO configrow (id, fixed_fee, percentage_fee, max_fee, initial_cash, currency) "
        "VALUES (1, 4.95, 0.0025, 69.0, ?, 'USD')",
        (initial_cash,),
    )
    conn.execute(
        "INSERT INTO depot (id, cash_balance, created_at) VALUES (1, ?, '2026-08-01 00:00:00')",
        (trades[-1]["cash_after"] if trades else initial_cash,),
    )
    for trade in trades:
        conn.execute(
            "INSERT INTO trade (ts, symbol, side, quantity, price, fee, gross, net, cash_after) "
            "VALUES (?, ?, ?, ?, ?, 0, 0, 0, ?)",
            (
                trade["ts"],
                trade["symbol"],
                trade["side"],
                trade["quantity"],
                trade["price"],
                trade["cash_after"],
            ),
        )
    conn.commit()
    conn.close()


def _make_full_state_log(results_dir: Path, ticker: str, date: str, text: str) -> Path:
    log_dir = results_dir / ticker / "TradingAgentsStrategy_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"full_states_log_{date}.json"
    log_path.write_text(
        json.dumps({"company_of_interest": ticker, "trade_date": date, "final_trade_decision": text}),
        encoding="utf-8",
    )
    return log_path


# ─── tier_index ──────────────────────────────────────────────────────────


class TestTierIndex:
    def test_recognizes_all_five_tiers(self):
        assert tier_index("Buy") == 0
        assert tier_index("Overweight") == 1
        assert tier_index("Hold") == 2
        assert tier_index("Underweight") == 3
        assert tier_index("Sell") == 4

    def test_is_case_insensitive(self):
        assert tier_index("bUY") == 0
        assert tier_index("SELL") == 4

    def test_unrecognized_or_empty_returns_none(self):
        assert tier_index("Nonsense") is None
        assert tier_index("") is None
        assert tier_index(None) is None


# ─── KPI 1: decision agreement ──────────────────────────────────────────


class TestComputeAgreement:
    def test_exact_agreement(self):
        arm = {("AAPL", "2026-08-01"): {"signal": "Buy", "confidence": None, "forward_return": None, "resolved": False}}
        control = {("AAPL", "2026-08-01"): {"signal": "Buy", "confidence": None, "forward_return": None, "resolved": False}}
        result = compute_agreement("no-market", arm, control)
        assert result.n_compared == 1
        assert result.n_exact == 1
        assert result.n_within1 == 1
        assert result.n_flips == 0
        assert result.disagreements == []

    def test_within_one_tier_but_not_exact(self):
        arm = {("AAPL", "2026-08-01"): {"signal": "Overweight", "confidence": None, "forward_return": None, "resolved": False}}
        control = {("AAPL", "2026-08-01"): {"signal": "Buy", "confidence": None, "forward_return": None, "resolved": False}}
        result = compute_agreement("no-market", arm, control)
        assert result.n_exact == 0
        assert result.n_within1 == 1
        assert result.n_flips == 1
        assert result.disagreements[0]["ticker"] == "AAPL"

    def test_full_flip_beyond_one_tier(self):
        arm = {("AAPL", "2026-08-01"): {"signal": "Sell", "confidence": None, "forward_return": None, "resolved": False}}
        control = {("AAPL", "2026-08-01"): {"signal": "Buy", "confidence": None, "forward_return": None, "resolved": False}}
        result = compute_agreement("no-market", arm, control)
        assert result.n_exact == 0
        assert result.n_within1 == 0
        assert result.n_flips == 1

    def test_mismatched_ticker_days_reported_as_gaps_not_counted(self):
        arm = {
            ("AAPL", "2026-08-01"): {"signal": "Buy", "confidence": None, "forward_return": None, "resolved": False},
            ("MSFT", "2026-08-01"): {"signal": "Hold", "confidence": None, "forward_return": None, "resolved": False},
        }
        control = {("AAPL", "2026-08-01"): {"signal": "Buy", "confidence": None, "forward_return": None, "resolved": False}}
        result = compute_agreement("no-market", arm, control)
        assert result.n_compared == 1
        assert any("MSFT" in gap for gap in result.gaps)

    def test_empty_decisions_reported_as_gap_not_crash(self):
        result = compute_agreement("no-market", {}, {})
        assert result.n_compared == 0
        assert any("no recorded portfolio_manager decisions" in gap for gap in result.gaps)

    def test_unrecognized_signal_excluded_and_reported(self):
        arm = {("AAPL", "2026-08-01"): {"signal": "Weird", "confidence": None, "forward_return": None, "resolved": False}}
        control = {("AAPL", "2026-08-01"): {"signal": "Buy", "confidence": None, "forward_return": None, "resolved": False}}
        result = compute_agreement("no-market", arm, control)
        assert result.n_compared == 0
        assert any("Unrecognized signal" in gap for gap in result.gaps)


# ─── KPI 2: paired forward returns ──────────────────────────────────────


class TestComputePairedForwardReturns:
    def test_no_disagreements_reports_gap(self):
        result = compute_paired_forward_returns("no-market", [], {}, {})
        assert result.n_resolved == 0
        assert any("nothing to pair" in gap for gap in result.gaps)

    def test_all_pending_reports_gap_and_no_means(self):
        disagreements = [{"ticker": "AAPL", "date": "2026-08-01", "arm_signal": "Sell", "control_signal": "Buy"}]
        arm = {("AAPL", "2026-08-01"): {"signal": "Sell", "confidence": None, "forward_return": None, "resolved": False}}
        control = {("AAPL", "2026-08-01"): {"signal": "Buy", "confidence": None, "forward_return": None, "resolved": False}}
        result = compute_paired_forward_returns("no-market", disagreements, arm, control)
        assert result.n_resolved == 0
        assert result.n_pending == 1
        assert result.arm_mean_return is None
        assert any("pending" in gap for gap in result.gaps)

    def test_resolved_pairs_compute_means_and_diff(self):
        disagreements = [
            {"ticker": "AAPL", "date": "2026-08-01", "arm_signal": "Sell", "control_signal": "Buy"},
            {"ticker": "MSFT", "date": "2026-08-01", "arm_signal": "Sell", "control_signal": "Buy"},
        ]
        arm = {
            ("AAPL", "2026-08-01"): {"signal": "Sell", "confidence": None, "forward_return": -0.02, "resolved": True},
            ("MSFT", "2026-08-01"): {"signal": "Sell", "confidence": None, "forward_return": 0.02, "resolved": True},
        }
        control = {
            ("AAPL", "2026-08-01"): {"signal": "Buy", "confidence": None, "forward_return": 0.05, "resolved": True},
            ("MSFT", "2026-08-01"): {"signal": "Buy", "confidence": None, "forward_return": 0.03, "resolved": True},
        }
        result = compute_paired_forward_returns("no-market", disagreements, arm, control)
        assert result.n_resolved == 2
        assert result.n_pending == 0
        assert result.arm_mean_return == pytest.approx(0.0)
        assert result.control_mean_return == pytest.approx(0.04)
        assert result.mean_paired_diff == pytest.approx(-0.04)
        assert result.gaps == []

    def test_partial_resolution_mixes_resolved_and_pending(self):
        disagreements = [
            {"ticker": "AAPL", "date": "2026-08-01", "arm_signal": "Sell", "control_signal": "Buy"},
            {"ticker": "MSFT", "date": "2026-08-01", "arm_signal": "Sell", "control_signal": "Buy"},
        ]
        arm = {
            ("AAPL", "2026-08-01"): {"signal": "Sell", "confidence": None, "forward_return": -0.02, "resolved": True},
            ("MSFT", "2026-08-01"): {"signal": "Sell", "confidence": None, "forward_return": None, "resolved": False},
        }
        control = {
            ("AAPL", "2026-08-01"): {"signal": "Buy", "confidence": None, "forward_return": 0.05, "resolved": True},
            ("MSFT", "2026-08-01"): {"signal": "Buy", "confidence": None, "forward_return": None, "resolved": False},
        }
        result = compute_paired_forward_returns("no-market", disagreements, arm, control)
        assert result.n_resolved == 1
        assert result.n_pending == 1


# ─── KPI 3: depot performance ────────────────────────────────────────────


class TestComputeDepotPerformance:
    def test_no_depot_id_reports_not_applicable(self):
        result = compute_depot_performance("control-stability", None)
        assert not result.found
        assert any("not applicable" in gap for gap in result.gaps)

    def test_missing_depot_db_reports_gap_not_crash(self, tmp_path):
        result = compute_depot_performance("control", "control-depot", sim_data_dir=tmp_path / "nope")
        assert not result.found
        assert any("Depot DB not found" in gap for gap in result.gaps)

    def test_no_trades_yet_uses_cash_balance(self, tmp_path):
        sim_data_dir = tmp_path / "sim-data"
        _make_depot_db(sim_data_dir / "depots" / "control-depot.db", initial_cash=100_000.0, trades=[])
        result = compute_depot_performance("control", "control-depot", sim_data_dir=sim_data_dir)
        assert result.found
        assert result.final_equity == 100_000.0
        assert result.n_trades == 0
        assert any("no trades yet" in gap for gap in result.gaps)

    def test_equity_curve_and_drawdown_from_trade_replay(self, tmp_path):
        sim_data_dir = tmp_path / "sim-data"
        trades = [
            {"ts": "2026-08-01 10:00:00", "symbol": "AAPL", "side": "buy", "quantity": 100, "price": 100.0, "cash_after": 90_000.0},
            {"ts": "2026-08-02 10:00:00", "symbol": "AAPL", "side": "sell", "quantity": 50, "price": 80.0, "cash_after": 94_000.0},
        ]
        # After trade 1: cash 90k + 100*100 = 100k equity.
        # After trade 2: cash 94k + 50*80 = 98k equity (drawdown from 100k peak).
        _make_depot_db(sim_data_dir / "depots" / "control-depot.db", initial_cash=100_000.0, trades=trades)
        result = compute_depot_performance("control", "control-depot", sim_data_dir=sim_data_dir)
        assert result.found
        assert result.n_trades == 2
        assert result.final_equity == pytest.approx(98_000.0)
        assert result.max_drawdown == pytest.approx(0.02)
        assert len(result.daily_equity) == 2

    def test_resolve_depot_db_path_uses_override(self, tmp_path):
        path = resolve_depot_db_path("my-depot", sim_data_dir=tmp_path)
        assert path == tmp_path / "depots" / "my-depot.db"

    def test_resolve_depot_db_path_defaults_to_sibling_checkout(self):
        path = resolve_depot_db_path("my-depot")
        assert path.name == "my-depot.db"
        assert "McpTradingSimulation" in str(path)


# ─── KPI 4: token cost proxy ─────────────────────────────────────────────


class TestComputeTokenCost:
    def test_no_results_dir_configured_reports_gap(self):
        result = compute_token_cost("no-market", None)
        assert not result.found
        assert any("no config.results_dir" in gap for gap in result.gaps)

    def test_missing_results_dir_reports_gap_not_crash(self, tmp_path):
        result = compute_token_cost("no-market", str(tmp_path / "does-not-exist"))
        assert not result.found
        assert any("not found" in gap for gap in result.gaps)

    def test_no_files_found_reports_gap(self, tmp_path):
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        result = compute_token_cost("no-market", str(results_dir))
        assert result.found
        assert result.n_files == 0
        assert any("No full-state log files found" in gap for gap in result.gaps)

    def test_sums_chars_over_four_across_files(self, tmp_path):
        results_dir = tmp_path / "results"
        _make_full_state_log(results_dir, "AAPL", "2026-08-01", "x" * 400)
        _make_full_state_log(results_dir, "MSFT", "2026-08-01", "y" * 400)
        result = compute_token_cost("control", str(results_dir))
        assert result.n_files == 2
        assert result.total_token_proxy == pytest.approx(result.total_chars / 4)
        assert set(result.per_ticker) == {"AAPL", "MSFT"}
        assert result.gaps == []

    def test_unparseable_json_still_counted_with_gap(self, tmp_path):
        results_dir = tmp_path / "results"
        log_dir = results_dir / "AAPL" / "TradingAgentsStrategy_logs"
        log_dir.mkdir(parents=True)
        (log_dir / "full_states_log_2026-08-01.json").write_text("not json", encoding="utf-8")
        result = compute_token_cost("control", str(results_dir))
        assert result.n_files == 1
        assert result.total_chars == len("not json")
        assert any("not valid JSON" in gap for gap in result.gaps)


# ─── KPI 5: repeat-run rating variance ──────────────────────────────────


class TestComputeStabilityVariance:
    def test_missing_db_reports_gap_not_crash(self, tmp_path):
        result = compute_stability_variance("control", tmp_path / "does-not-exist.db")
        assert not result.found
        assert any("not found" in gap for gap in result.gaps)

    def test_single_repeat_reports_insufficient_data_gap(self, tmp_path):
        db_path = tmp_path / "memory.db"
        _make_memory_db(
            db_path,
            [{"ticker": "AAPL", "date": "2026-08-01", "signal": "Buy", "confidence": 0.7}],
        )
        result = compute_stability_variance("control-stability", db_path)
        assert result.per_ticker["AAPL"]["n"] == 1
        assert result.per_ticker["AAPL"]["signal_tier_variance"] is None
        assert any(">= 2" in gap for gap in result.gaps)

    def test_three_repeats_compute_variance(self, tmp_path):
        db_path = tmp_path / "memory.db"
        _make_memory_db(
            db_path,
            [
                {"ticker": "AAPL", "date": "2026-08-01", "signal": "Buy", "confidence": 0.9},
                {"ticker": "AAPL", "date": "2026-08-08", "signal": "Hold", "confidence": 0.5},
                {"ticker": "AAPL", "date": "2026-08-15", "signal": "Buy", "confidence": 0.8},
            ],
        )
        result = compute_stability_variance("control-stability", db_path)
        stats = result.per_ticker["AAPL"]
        assert stats["n"] == 3
        assert stats["signal_tier_variance"] is not None
        assert stats["signal_tier_variance"] > 0  # Buy(0), Hold(2), Buy(0) -> not all identical
        assert stats["confidence_variance"] is not None
        assert result.gaps == []

    def test_identical_repeats_have_zero_variance(self, tmp_path):
        db_path = tmp_path / "memory.db"
        _make_memory_db(
            db_path,
            [
                {"ticker": "AAPL", "date": "2026-08-01", "signal": "Hold", "confidence": 0.5},
                {"ticker": "AAPL", "date": "2026-08-08", "signal": "Hold", "confidence": 0.5},
            ],
        )
        result = compute_stability_variance("control-stability", db_path)
        stats = result.per_ticker["AAPL"]
        assert stats["signal_tier_variance"] == 0.0
        assert stats["confidence_variance"] == 0.0


# ─── Arm discovery ───────────────────────────────────────────────────────


def _write_arm_config(arms_dir: Path, name: str, *, stability: bool = False, **extra) -> Path:
    suffix = ".stability.json" if stability else ".json"
    config = {
        "stocks_file": "stability-stocks.json" if stability else "stocks.json",
        "report_dir": f"reports/{'stability/' if stability else ''}{name}",
        "memory_id": f"{name}-stability" if stability else name,
        "config": {"memory_log_path": f"~/.tradingagents/memory/{name}/trading_memory.md"},
    }
    if not stability:
        config.update({"portfolio": True, "style": "aggressive", "depot_id": f"{name}-depot"})
    config["config"].update(extra)
    path = arms_dir / f"{name}{suffix}"
    path.write_text(json.dumps(config))
    return path


class TestArmDiscovery:
    def test_discover_main_arms_excludes_stability_and_stock_lists(self, tmp_path):
        arms_dir = tmp_path / "arms"
        arms_dir.mkdir()
        (arms_dir / "stocks.json").write_text("[]")
        (arms_dir / "stability-stocks.json").write_text("[]")
        _write_arm_config(arms_dir, "control")
        _write_arm_config(arms_dir, "control", stability=True)

        arms = discover_main_arms(arms_dir)
        assert set(arms) == {"control"}
        assert isinstance(arms["control"], Arm)
        assert not arms["control"].is_stability

    def test_discover_stability_arms_strips_suffix(self, tmp_path):
        arms_dir = tmp_path / "arms"
        arms_dir.mkdir()
        _write_arm_config(arms_dir, "no-market", stability=True)

        arms = discover_stability_arms(arms_dir)
        assert set(arms) == {"no-market"}
        assert arms["no-market"].is_stability

    def test_discover_on_missing_dir_returns_empty(self, tmp_path):
        assert discover_main_arms(tmp_path / "nope") == {}
        assert discover_stability_arms(tmp_path / "nope") == {}

    def test_real_experiments_dir_discovers_six_main_arms(self):
        """Sanity check against the actual experiments/ablation-analysts/ layout (#120)."""
        arms_dir = Path(__file__).parent.parent / "experiments" / "ablation-analysts"
        arms = discover_main_arms(arms_dir)
        assert set(arms) == {"control", "no-market", "no-social", "no-news", "no-fundamentals", "no-risk"}
        for arm in arms.values():
            assert arm.results_dir is not None, f"{arm.name}: results_dir should be isolated per arm"

    def test_real_experiments_dir_discovers_six_stability_arms(self):
        arms_dir = Path(__file__).parent.parent / "experiments" / "ablation-analysts"
        arms = discover_stability_arms(arms_dir)
        assert set(arms) == {"control", "no-market", "no-social", "no-news", "no-fundamentals", "no-risk"}


# ─── End-to-end: build_report ────────────────────────────────────────────


class TestBuildReportEndToEnd:
    def _setup_arms(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        arms_dir = tmp_path / "arms"
        arms_dir.mkdir()
        (arms_dir / "stocks.json").write_text(json.dumps([{"ticker": "AAPL"}]))

        for name in (CONTROL_ARM_NAME, "no-market"):
            results_dir = tmp_path / "results" / name
            _write_arm_config(arms_dir, name, results_dir=str(results_dir))
            _make_full_state_log(results_dir, "AAPL", "2026-08-01", "z" * 40)

        _make_memory_db(
            Path("runs/memory/control/memory.db"),
            [{"ticker": "AAPL", "date": "2026-08-01", "signal": "Buy", "confidence": 0.8}],
        )
        _make_memory_db(
            Path("runs/memory/no-market/memory.db"),
            [{"ticker": "AAPL", "date": "2026-08-01", "signal": "Hold", "confidence": 0.6}],
        )
        return arms_dir

    def test_report_states_caveat_and_all_five_kpi_sections(self, tmp_path, monkeypatch):
        arms_dir = self._setup_arms(tmp_path, monkeypatch)
        report = build_report(arms_dir, sim_data_dir=tmp_path / "no-sim-checkout")

        assert CAVEAT_TEXT in report
        assert "## KPI 1: Decision agreement" in report
        assert "## KPI 2: Decision-aligned" in report
        assert "## KPI 3: Depot performance" in report
        assert "## KPI 4: Token cost" in report
        assert "## KPI 5: Repeat-run rating variance" in report
        assert "keep/merge/drop" in report

    def test_report_reflects_flip_between_arm_and_control(self, tmp_path, monkeypatch):
        arms_dir = self._setup_arms(tmp_path, monkeypatch)
        report = build_report(arms_dir, sim_data_dir=tmp_path / "no-sim-checkout")
        assert "AAPL/2026-08-01: arm=Hold vs. control=Buy" in report

    def test_report_handles_missing_control_gracefully(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        arms_dir = tmp_path / "arms"
        arms_dir.mkdir()
        _write_arm_config(arms_dir, "no-market", results_dir=str(tmp_path / "results" / "no-market"))

        # Should not raise even with zero arms run and no control present.
        report = build_report(arms_dir, sim_data_dir=tmp_path / "no-sim-checkout")
        assert CAVEAT_TEXT in report

    def test_report_on_completely_empty_arms_dir_does_not_crash(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        arms_dir = tmp_path / "arms"
        arms_dir.mkdir()
        report = build_report(arms_dir, sim_data_dir=tmp_path / "no-sim-checkout")
        assert CAVEAT_TEXT in report
