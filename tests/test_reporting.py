"""Tests for the shared report-tree writer and its structured-JSON callers.

Covers issue #19: adopting upstream's ``write_report_tree`` for the on-disk
subfolder tree while layering our fork's ``trading_recommendation.json``
structured export on top, in both `cli/main.py` and
`tradingagents/report_generator.py`.

Includes tests for issue #157: backfilling per-ticker reports with portfolio
adjustments and executed orders after portfolio mode completes.

Issue #165 switched the consolidated report from a `complete_report.md` text
file to a `complete_report.pdf` (rendered via
`tradingagents.report_pdf.render_complete_report_pdf`, #164) plus a
`complete_report.sections.json` structured-data sidecar; assertions against
the old markdown text are replaced with pypdf text extraction against the PDF
and direct JSON checks against the sidecar. The per-agent files
(`market.json`, `trader.md`, `decision.md`, etc.) are unaffected and keep
their pre-#165 assertions unchanged.
"""

import json

import pytest
from pypdf import PdfReader

from tradingagents.reporting import backfill_portfolio_adjustments, write_report_tree

pytestmark = pytest.mark.unit


def _extract_pdf_text(pdf_path) -> str:
    """Extract plain text from a rendered report PDF for assertions."""
    reader = PdfReader(str(pdf_path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _final_state(**overrides):
    state = {
        "market_report": "Market looks bullish.",
        "sentiment_report": "Sentiment is positive.",
        "news_report": "No major news.",
        "fundamentals_report": "Fundamentals are solid.",
        "investment_debate_state": {
            "bull_history": "Bull case text.",
            "bear_history": "Bear case text.",
            "judge_decision": "Research manager verdict.",
        },
        "trader_investment_plan": "Buy 10 shares.",
        "risk_debate_state": {
            "aggressive_history": "Aggressive take.",
            "conservative_history": "Conservative take.",
            "neutral_history": "Neutral take.",
            "judge_decision": "Final portfolio decision.",
        },
        "final_trade_decision": "Final portfolio decision.",
    }
    state.update(overrides)
    return state


class TestWriteReportTree:
    def test_writes_full_section_tree(self, tmp_path):
        save_path = tmp_path / "report"
        report_file = write_report_tree(_final_state(), "AAPL", save_path)

        assert report_file == save_path / "complete_report.pdf"
        assert report_file.exists()
        assert (save_path / "1_analysts" / "market.json").read_text() == "Market looks bullish."
        assert (save_path / "1_analysts" / "sentiment.json").read_text() == "Sentiment is positive."
        assert (save_path / "1_analysts" / "news.json").read_text() == "No major news."
        assert (save_path / "1_analysts" / "fundamentals.json").read_text() == "Fundamentals are solid."
        assert not (save_path / "1_analysts" / "macro.json").exists()
        assert (save_path / "2_research" / "bull.md").exists()
        assert (save_path / "2_research" / "bear.md").exists()
        assert (save_path / "2_research" / "manager.md").exists()
        assert (save_path / "3_trading" / "trader.md").read_text() == "Buy 10 shares."
        assert (save_path / "4_risk" / "aggressive.md").exists()
        assert (save_path / "4_risk" / "conservative.md").exists()
        assert (save_path / "4_risk" / "neutral.md").exists()
        assert (save_path / "5_portfolio" / "decision.md").read_text() == "Final portfolio decision."

        complete_text = _extract_pdf_text(report_file)
        assert "Trading Analysis Report: AAPL" in complete_text
        assert "I. Analyst Team Reports" in complete_text
        assert "II. Research Team Decision" in complete_text
        assert "III. Trading Team Plan" in complete_text
        assert "IV. Risk Management Team Decision" in complete_text
        assert "V. Portfolio Manager Decision" in complete_text
        for expected in [
            "Market looks bullish.",
            "Sentiment is positive.",
            "No major news.",
            "Fundamentals are solid.",
            "Bull case text.",
            "Bear case text.",
            "Research manager verdict.",
            "Buy 10 shares.",
            "Aggressive take.",
            "Conservative take.",
            "Neutral take.",
            "Final portfolio decision.",
        ]:
            assert expected in complete_text

        # Structured sidecar mirrors the same section/subsection data rendered.
        sidecar_file = save_path / "complete_report.sections.json"
        assert sidecar_file.exists()
        sidecar = json.loads(sidecar_file.read_text(encoding="utf-8"))
        assert sidecar["ticker"] == "AAPL"
        assert [s["title"] for s in sidecar["sections"]] == [
            "I. Analyst Team Reports",
            "II. Research Team Decision",
            "III. Trading Team Plan",
            "IV. Risk Management Team Decision",
            "V. Portfolio Manager Decision",
        ]
        pm_section = sidecar["sections"][-1]
        assert pm_section["subsections"] == [
            {"agent_name": "Portfolio Manager", "content": "Final portfolio decision."}
        ]

    def test_skips_missing_sections(self, tmp_path):
        save_path = tmp_path / "report"
        state = {"market_report": "Only market report available."}
        report_file = write_report_tree(state, "MSFT", save_path)

        assert (save_path / "1_analysts" / "market.json").exists()
        assert not (save_path / "2_research").exists()
        assert not (save_path / "3_trading").exists()
        assert not (save_path / "4_risk").exists()
        assert not (save_path / "5_portfolio").exists()

        assert report_file.exists()
        sidecar = json.loads((save_path / "complete_report.sections.json").read_text())
        assert [s["title"] for s in sidecar["sections"]] == ["I. Analyst Team Reports"]

    def test_accepts_string_save_path(self, tmp_path):
        save_path = str(tmp_path / "report")
        report_file = write_report_tree(_final_state(), "AAPL", save_path)
        assert report_file.exists()
        assert report_file.name == "complete_report.pdf"

    def test_writes_macro_report_when_present(self, tmp_path):
        """macro_report (#132, opt-in) is written as .json when present, and
        included in the consolidated Analyst Team Reports section."""
        save_path = tmp_path / "report"
        state = _final_state(macro_report='{"skill": "macro-fundamentals-analyst"}')
        report_file = write_report_tree(state, "AAPL", save_path)

        assert (save_path / "1_analysts" / "macro.json").read_text() == (
            '{"skill": "macro-fundamentals-analyst"}'
        )
        complete_text = _extract_pdf_text(report_file)
        assert "Macro Fundamentals Analyst" in complete_text

    def test_non_portfolio_run_report_is_unchanged_by_backfill_support(self, tmp_path):
        """AC (issue #157): runs without --portfolio must be unaffected — no
        adjustments section, no backfill pass, no new files.

        ``write_report_tree`` itself is untouched by #157's diff (the backfill
        pass lives entirely in ``backfill_portfolio_adjustments``, which is
        opt-in and only ever called from ``run_trading_agents.py``'s
        ``if args.portfolio:`` block). This test pins that: as long as
        ``backfill_portfolio_adjustments`` is never invoked, neither the
        rendered PDF, the decision.md file, nor the sections sidecar carry any
        backfill markers, and no `portfolio-rebalance-*.md` file appears.
        """
        save_path = tmp_path / "report"
        report_file = write_report_tree(_final_state(), "AAPL", save_path)
        complete_text = _extract_pdf_text(report_file)
        decision = (save_path / "5_portfolio" / "decision.md").read_text(encoding="utf-8")

        # No backfill markers anywhere — backfill_portfolio_adjustments was
        # never called, matching a non-portfolio run.
        for marker in ("### Proposed adjustments", "### Executed", "### Dropped"):
            assert marker not in complete_text
            assert marker not in decision

        sidecar = json.loads((save_path / "complete_report.sections.json").read_text())
        pm_section = next(
            s for s in sidecar["sections"] if s["title"] == "V. Portfolio Manager Decision"
        )
        assert "portfolio_backfilled" not in pm_section

        # No run-level rebalance summary is produced by write_report_tree
        # itself — that file only exists when --portfolio triggers it.
        assert not list(save_path.parent.glob("portfolio-rebalance-*.md"))
        assert decision == "Final portfolio decision."


class TestCliSaveReportToDisk:
    def test_delegates_tree_and_adds_json(self, tmp_path):
        from cli.main import save_report_to_disk

        state = _final_state(
            portfolio_structured_data={
                "rating": "BUY",
                "executive_summary": "Strong buy.",
                "investment_thesis": "Growth ahead.",
                "price_target": 250.0,
                "time_horizon": "6 months",
            },
            trader_structured_data={
                "action": "BUY",
                "reasoning": "Momentum is strong.",
                "entry_price": 200.0,
                "stop_loss": 180.0,
                "position_sizing": "5%",
            },
        )
        save_path = tmp_path / "cli_report"
        report_file, structured_data = save_report_to_disk(state, "AAPL", save_path)

        # PDF report tree came from write_report_tree.
        assert report_file == save_path / "complete_report.pdf"
        assert (save_path / "1_analysts" / "market.json").exists()

        # JSON export layered on top.
        json_file = save_path / "trading_recommendation.json"
        assert json_file.exists()
        on_disk = json.loads(json_file.read_text())
        assert on_disk == structured_data
        assert structured_data["ticker"] == "AAPL"
        assert structured_data["rating"] == "BUY"
        assert structured_data["action"] == "BUY"
        assert structured_data["entry_price"] == 200.0

    def test_defaults_to_na_without_structured_data(self, tmp_path):
        from cli.main import save_report_to_disk

        save_path = tmp_path / "cli_report_no_structured"
        _, structured_data = save_report_to_disk(_final_state(), "TSLA", save_path)

        assert structured_data["rating"] == "N/A"
        assert structured_data["action"] == "N/A"
        assert structured_data["entry_price"] is None


class TestReportGeneratorSaveReportToDisk:
    def test_structured_data_path_writes_summary_and_delegates_tree(self, tmp_path):
        from tradingagents.report_generator import save_report_to_disk

        state = _final_state(
            portfolio_structured_data={
                "rating": "HOLD",
                "executive_summary": "Wait and see.",
                "investment_thesis": "Uncertain macro.",
                "price_target": None,
                "time_horizon": None,
            },
            trader_structured_data={
                "action": "HOLD",
                "reasoning": "No clear edge.",
                "entry_price": None,
                "stop_loss": None,
                "position_sizing": None,
            },
        )
        save_path = tmp_path / "gen_report"
        report_file, structured_data = save_report_to_disk(state, "NVDA", save_path)

        assert report_file == save_path / "complete_report.pdf"
        assert (save_path / "1_analysts" / "market.json").exists()
        assert (save_path / "5_portfolio" / "decision.md").exists()

        assert structured_data["rating"] == "HOLD"
        assert structured_data["action"] == "HOLD"

        summary = (save_path / "summary.txt").read_text()
        assert "HOLD" in summary
        json_file = save_path / "trading_recommendation.json"
        assert json.loads(json_file.read_text())["ticker"] == "NVDA"

    def test_fallback_regex_parsing_without_structured_data(self, tmp_path):
        from tradingagents.report_generator import save_report_to_disk

        state = _final_state(
            trader_investment_plan=(
                "**Action**: SELL\n"
                "**Entry Price**: 123.45\n"
                "**Stop Loss**: 110.0\n"
                "**Position Sizing**: 2%\n"
                "**Reasoning**: Overbought conditions.\n"
            ),
            final_trade_decision=(
                "**Rating**: SELL\n"
                "**Executive Summary**: Take profits.\n"
                "**Investment Thesis**: Momentum fading.\n"
            ),
        )
        save_path = tmp_path / "gen_report_fallback"
        report_file, structured_data = save_report_to_disk(state, "GME", save_path)

        assert report_file == save_path / "complete_report.pdf"
        assert structured_data["rating"] == "SELL"
        assert structured_data["action"] == "SELL"
        assert structured_data["entry_price"] == 123.45
        assert structured_data["stop_loss"] == 110.0
        assert (save_path / "summary.txt").exists()
        assert (save_path / "trading_recommendation.json").exists()


class TestTradingGraphSaveReports:
    def test_delegates_to_write_report_tree_with_default_path(self, tmp_path):
        from tradingagents.graph.trading_graph import TradingAgentsGraph

        graph = object.__new__(TradingAgentsGraph)
        graph.config = {"results_dir": str(tmp_path)}

        report_file = graph.save_reports(_final_state(), "AAPL")

        assert report_file.exists()
        assert report_file.name == "complete_report.pdf"
        assert report_file.is_relative_to(tmp_path / "reports")

    def test_honors_explicit_save_path(self, tmp_path):
        from tradingagents.graph.trading_graph import TradingAgentsGraph

        graph = object.__new__(TradingAgentsGraph)
        graph.config = {"results_dir": str(tmp_path)}
        explicit_path = tmp_path / "custom" / "location"

        report_file = graph.save_reports(_final_state(), "AAPL", save_path=explicit_path)

        assert report_file == explicit_path / "complete_report.pdf"


class TestBackfillPortfolioAdjustments:
    """Tests for backfilling per-ticker reports with portfolio adjustments (issue
    #157), reworked for issue #165's PDF + sections-sidecar output: assertions
    against `decision.md` are unchanged (still a plain text file), while
    assertions against the former `complete_report.md` are now split between
    the `complete_report.sections.json` sidecar (exact string checks) and the
    re-rendered `complete_report.pdf` (checked via pypdf text extraction).
    """

    def _make_envelope(self, **details_overrides):
        """Create a sample portfolio envelope like run_portfolio_mode returns."""
        details = {
            "depot_id": "test-depot",
            "universe": ["AAPL", "MSFT"],
            "signals": {
                "AAPL": {"signal": "BUY", "confidence": "HIGH"},
                "MSFT": {"signal": "HOLD", "confidence": "MEDIUM"},
            },
            "allocation": {
                "AAPL": {
                    "raw_weight": 0.5,
                    "target_weight": 0.50,
                    "target_shares": 10,
                    "current_shares": 0,
                    "delta": 10,
                    "price": 150.0,
                },
                "MSFT": {
                    "raw_weight": 0.5,
                    "target_weight": 0.50,
                    "target_shares": 5,
                    "current_shares": 5,
                    "delta": 0,
                    "price": 300.0,
                },
            },
            "trades_executed": [
                {
                    "symbol": "AAPL",
                    "side": "buy",
                    "quantity": 10,
                    "status": "executed",
                    "message": "Order executed at market price",
                    "fill_price": 149.5,
                },
                {
                    "symbol": "MSFT",
                    "side": "buy",
                    "quantity": 0,
                    "status": "rejected",
                    "message": "Min trade size 2 not met",
                    "fill_price": None,
                },
            ],
            "rejected_orders": [
                {
                    "symbol": "MSFT",
                    "side": "buy",
                    "quantity": 0,
                    "status": "rejected",
                    "message": "Min trade size 2 not met",
                    "fill_price": None,
                },
            ],
            "pre_snapshot": {
                "equity": 10000.0,
                "cash": 10000.0,
                "positions": {},
            },
            "post_snapshot": {
                "equity": 11495.0,
                "cash": 8505.0,
                "positions": {"AAPL": {"shares": 10, "price": 149.5, "market_value": 1495.0}},
            },
            "equity_change": 1495.0,
        }
        details.update(details_overrides)
        return {"details": details}

    def _write_ticker_reports(self, tmp_path, ticker, pm_decision="**Rating**: X"):
        """Create a ticker's report tree via write_report_tree, exactly like a
        real run would: `5_portfolio/decision.md` + `complete_report.pdf` +
        `complete_report.sections.json`, all containing just a V section built
        from `pm_decision` (mirroring the pre-#165 fixtures' minimal shape).
        """
        report_dir = tmp_path / f"{ticker}_2024-01-01_20240101_120000"
        write_report_tree(
            {"risk_debate_state": {"judge_decision": pm_decision}},
            ticker,
            report_dir,
        )
        return report_dir

    def _sidecar(self, report_dir):
        return json.loads((report_dir / "complete_report.sections.json").read_text(encoding="utf-8"))

    def _pm_section(self, report_dir):
        sidecar = self._sidecar(report_dir)
        return next(s for s in sidecar["sections"] if s["title"] == "V. Portfolio Manager Decision")

    def test_backfill_renders_proposed_adjustments_table(self, tmp_path):
        """Backfill adds the proposed adjustments table to per-ticker reports."""
        report_dir = self._write_ticker_reports(
            tmp_path, "AAPL", "**Rating**: Buy\n**Executive Summary**: Strong buy signal."
        )

        envelope = self._make_envelope()
        backfill_portfolio_adjustments(
            envelope=envelope,
            per_ticker_report_dirs={"AAPL": str(report_dir)},
        )

        # decision.md (unchanged text-append logic)
        decision_text = (report_dir / "5_portfolio" / "decision.md").read_text(encoding="utf-8")
        assert "### Proposed adjustments (depot test-depot)" in decision_text
        assert "| AAPL | BUY | 0.50 | 10 |" in decision_text
        assert "| MSFT |" not in decision_text

        # Sidecar: structured, ticker-scoped content plus the backfill marker.
        pm_section = self._pm_section(report_dir)
        assert pm_section["portfolio_backfilled"] is True
        pm_content = pm_section["subsections"][-1]["content"]
        assert "### Proposed adjustments (depot test-depot)" in pm_content
        assert "| AAPL | BUY | 0.50 | 10 |" in pm_content
        assert "| MSFT |" not in pm_content

        # Re-rendered PDF reflects the same content.
        pdf_text = _extract_pdf_text(report_dir / "complete_report.pdf")
        assert "### Proposed adjustments (depot test-depot)" in pdf_text
        assert "| AAPL | BUY | 0.50 | 10 |" in pdf_text

    def test_backfill_shows_executed_trades_with_fill_price(self, tmp_path):
        """Backfill shows executed trades with fill price in the Executed section."""
        report_dir = self._write_ticker_reports(tmp_path, "AAPL", "**Rating**: Buy")

        envelope = self._make_envelope()
        backfill_portfolio_adjustments(
            envelope=envelope,
            per_ticker_report_dirs={"AAPL": str(report_dir)},
        )

        decision_text = (report_dir / "5_portfolio" / "decision.md").read_text(encoding="utf-8")
        assert "### Executed" in decision_text
        assert "- AAPL: BUY 10 shares @ $149.50 — executed" in decision_text

        pm_content = self._pm_section(report_dir)["subsections"][-1]["content"]
        assert "### Executed" in pm_content
        assert "- AAPL: BUY 10 shares @ $149.50 — executed" in pm_content

    def test_backfill_shows_rejected_orders_with_message(self, tmp_path):
        """Backfill shows rejected orders with their rejection message."""
        # Use MSFT which has a rejected order in the envelope
        report_dir = self._write_ticker_reports(tmp_path, "MSFT", "**Rating**: Hold")

        envelope = self._make_envelope()
        backfill_portfolio_adjustments(
            envelope=envelope,
            per_ticker_report_dirs={"MSFT": str(report_dir)},
        )

        decision_text = (report_dir / "5_portfolio" / "decision.md").read_text(encoding="utf-8")
        assert "REJECTED: Min trade size 2 not met" in decision_text

        pm_content = self._pm_section(report_dir)["subsections"][-1]["content"]
        assert "REJECTED: Min trade size 2 not met" in pm_content

    def test_backfill_is_idempotent(self, tmp_path):
        """Running backfill twice doesn't duplicate the portfolio section."""
        report_dir = self._write_ticker_reports(tmp_path, "AAPL", "**Rating**: Buy")

        envelope = self._make_envelope()
        backfill_portfolio_adjustments(
            envelope=envelope,
            per_ticker_report_dirs={"AAPL": str(report_dir)},
        )

        first_decision = (report_dir / "5_portfolio" / "decision.md").read_text(encoding="utf-8")
        first_pm_content = self._pm_section(report_dir)["subsections"][-1]["content"]

        # Backfill again
        backfill_portfolio_adjustments(
            envelope=envelope,
            per_ticker_report_dirs={"AAPL": str(report_dir)},
        )

        second_decision = (report_dir / "5_portfolio" / "decision.md").read_text(encoding="utf-8")
        second_pm_content = self._pm_section(report_dir)["subsections"][-1]["content"]

        # Should be identical (no duplicate append), in both decision.md and
        # the sidecar's structured content.
        assert first_decision == second_decision
        assert first_decision.count("### Proposed adjustments") == 1
        assert first_pm_content == second_pm_content
        assert first_pm_content.count("### Proposed adjustments") == 1
        assert self._pm_section(report_dir)["portfolio_backfilled"] is True

    def test_backfill_shows_dropped_tickers_with_explanation(self, tmp_path):
        """Backfill shows tickers dropped due to missing rating or price in their report."""
        report_dir_aapl = self._write_ticker_reports(tmp_path, "AAPL", "**Rating**: Buy")
        report_dir_tsla = self._write_ticker_reports(tmp_path, "TSLA", "**Rating**: Buy")

        envelope = self._make_envelope()
        backfill_portfolio_adjustments(
            envelope=envelope,
            per_ticker_report_dirs={"AAPL": str(report_dir_aapl), "TSLA": str(report_dir_tsla)},
            missing_ratings=[],
            missing_prices=["TSLA"],
        )

        # AAPL (successful) should have the normal proposed adjustments section
        decision_aapl = (report_dir_aapl / "5_portfolio" / "decision.md").read_text(encoding="utf-8")
        assert "### Proposed adjustments" in decision_aapl

        # TSLA (dropped) should have the dropped explanation
        decision_tsla = (report_dir_tsla / "5_portfolio" / "decision.md").read_text(encoding="utf-8")
        assert "### Dropped" in decision_tsla
        assert "no usable price" in decision_tsla

        tsla_pm_content = self._pm_section(report_dir_tsla)["subsections"][-1]["content"]
        assert "### Dropped" in tsla_pm_content
        assert "no usable price" in tsla_pm_content

    def test_backfill_only_updates_tickers_in_per_ticker_dirs(self, tmp_path):
        """Backfill only processes tickers that have report directories."""
        report_dir_aapl = self._write_ticker_reports(tmp_path, "AAPL", "**Rating**: Buy")

        envelope = self._make_envelope()
        # Only pass AAPL in per_ticker_report_dirs, even though envelope has MSFT
        backfill_portfolio_adjustments(
            envelope=envelope,
            per_ticker_report_dirs={"AAPL": str(report_dir_aapl)},
        )

        # AAPL should have been updated
        decision_text = (report_dir_aapl / "5_portfolio" / "decision.md").read_text(encoding="utf-8")
        assert "### Proposed adjustments" in decision_text

        # MSFT's report directory wasn't provided, so it wasn't updated
        # (this is the expected behavior — per-ticker backfill happens
        # only for tickers that have report directories in per_ticker_report_dirs)

    def test_backfill_sections_sidecar_is_scoped_to_its_own_ticker(self, tmp_path):
        """Regression coverage (issue #157 escalation, carried forward into
        #165's structured sidecar): each ticker's sidecar/PDF must show ONLY
        its own row/trade, not the whole universe's.
        """
        report_dir_aapl = self._write_ticker_reports(tmp_path, "AAPL")
        report_dir_msft = self._write_ticker_reports(tmp_path, "MSFT")

        envelope = self._make_envelope()
        backfill_portfolio_adjustments(
            envelope=envelope,
            per_ticker_report_dirs={
                "AAPL": str(report_dir_aapl),
                "MSFT": str(report_dir_msft),
            },
        )

        aapl_content = self._pm_section(report_dir_aapl)["subsections"][-1]["content"]
        msft_content = self._pm_section(report_dir_msft)["subsections"][-1]["content"]

        # AAPL's sidecar shows AAPL's row and trade only.
        assert "### Proposed adjustments (depot test-depot)" in aapl_content
        assert "| AAPL | BUY | 0.50 | 10 |" in aapl_content
        assert "| MSFT |" not in aapl_content
        assert "- AAPL: BUY 10 shares @ $149.50 — executed" in aapl_content
        assert "MSFT" not in aapl_content.split("### Executed", 1)[1]

        # MSFT's sidecar shows MSFT's row and (rejected) trade only.
        assert "### Proposed adjustments (depot test-depot)" in msft_content
        assert "| MSFT | HOLD | 0.50 | 0 |" in msft_content
        assert "| AAPL |" not in msft_content
        assert "REJECTED: Min trade size 2 not met" in msft_content
        assert "AAPL" not in msft_content.split("### Executed", 1)[1]

    def test_backfill_sections_sidecar_shows_dropped_ticker_explanation_only(self, tmp_path):
        """Regression coverage (issue #157 escalation): a dropped ticker's
        structured content must show the drop-explanation line IN PLACE OF the
        adjustments table, not the full table plus a "### Dropped" note.
        """
        report_dir_aapl = self._write_ticker_reports(tmp_path, "AAPL")
        report_dir_tsla = self._write_ticker_reports(tmp_path, "TSLA")

        envelope = self._make_envelope()
        backfill_portfolio_adjustments(
            envelope=envelope,
            per_ticker_report_dirs={
                "AAPL": str(report_dir_aapl),
                "TSLA": str(report_dir_tsla),
            },
            missing_ratings=[],
            missing_prices=["TSLA"],
        )

        aapl_content = self._pm_section(report_dir_aapl)["subsections"][-1]["content"]
        tsla_content = self._pm_section(report_dir_tsla)["subsections"][-1]["content"]

        # AAPL (not dropped) still gets its normal, filtered table.
        assert "### Proposed adjustments" in aapl_content
        assert "### Dropped" not in aapl_content

        # TSLA (dropped) gets ONLY the drop explanation — no table at all,
        # and no leaked data from AAPL's proposed adjustments/executed trades.
        assert "### Dropped" in tsla_content
        assert "no usable price" in tsla_content
        assert "### Proposed adjustments" not in tsla_content
        assert "### Executed" not in tsla_content
        assert "AAPL" not in tsla_content

    def test_backfill_skips_ticker_without_sections_sidecar(self, tmp_path):
        """A report directory missing the sidecar (e.g. a pre-#165 report tree,
        or one where write_report_tree never ran) is skipped for the sidecar/PDF
        step without raising — decision.md backfilling still proceeds.
        """
        report_dir = tmp_path / "AAPL_2024-01-01_20240101_120000"
        portfolio_subdir = report_dir / "5_portfolio"
        portfolio_subdir.mkdir(parents=True)
        (portfolio_subdir / "decision.md").write_text("**Rating**: Buy", encoding="utf-8")

        envelope = self._make_envelope()
        backfill_portfolio_adjustments(
            envelope=envelope,
            per_ticker_report_dirs={"AAPL": str(report_dir)},
        )

        decision_text = portfolio_subdir.joinpath("decision.md").read_text(encoding="utf-8")
        assert "### Proposed adjustments" in decision_text
        assert not (report_dir / "complete_report.sections.json").exists()
        assert not (report_dir / "complete_report.pdf").exists()
