"""Tests for the shared report-tree writer and its structured-JSON callers.

Covers issue #19: adopting upstream's ``write_report_tree`` for the markdown
subfolder tree while layering our fork's ``trading_recommendation.json``
structured export on top, in both `cli/main.py` and
`tradingagents/report_generator.py`.
"""

import json

import pytest

from tradingagents.reporting import write_report_tree

pytestmark = pytest.mark.unit


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

        assert report_file == save_path / "complete_report.md"
        assert (save_path / "1_analysts" / "market.json").read_text() == "Market looks bullish."
        assert (save_path / "1_analysts" / "sentiment.md").read_text() == "Sentiment is positive."
        assert (save_path / "1_analysts" / "news.json").read_text() == "No major news."
        assert (save_path / "1_analysts" / "fundamentals.json").read_text() == "Fundamentals are solid."
        assert (save_path / "2_research" / "bull.md").exists()
        assert (save_path / "2_research" / "bear.md").exists()
        assert (save_path / "2_research" / "manager.md").exists()
        assert (save_path / "3_trading" / "trader.md").read_text() == "Buy 10 shares."
        assert (save_path / "4_risk" / "aggressive.md").exists()
        assert (save_path / "4_risk" / "conservative.md").exists()
        assert (save_path / "4_risk" / "neutral.md").exists()
        assert (save_path / "5_portfolio" / "decision.md").read_text() == "Final portfolio decision."

        complete = report_file.read_text()
        assert "Trading Analysis Report: AAPL" in complete
        assert "I. Analyst Team Reports" in complete
        assert "V. Portfolio Manager Decision" in complete

    def test_skips_missing_sections(self, tmp_path):
        save_path = tmp_path / "report"
        state = {"market_report": "Only market report available."}
        write_report_tree(state, "MSFT", save_path)

        assert (save_path / "1_analysts" / "market.json").exists()
        assert not (save_path / "2_research").exists()
        assert not (save_path / "3_trading").exists()
        assert not (save_path / "4_risk").exists()
        assert not (save_path / "5_portfolio").exists()

    def test_accepts_string_save_path(self, tmp_path):
        save_path = str(tmp_path / "report")
        report_file = write_report_tree(_final_state(), "AAPL", save_path)
        assert report_file.exists()


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

        # Markdown tree came from write_report_tree.
        assert report_file == save_path / "complete_report.md"
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

        assert report_file == save_path / "complete_report.md"
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

        assert report_file == save_path / "complete_report.md"
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
        assert report_file.name == "complete_report.md"
        assert report_file.is_relative_to(tmp_path / "reports")

    def test_honors_explicit_save_path(self, tmp_path):
        from tradingagents.graph.trading_graph import TradingAgentsGraph

        graph = object.__new__(TradingAgentsGraph)
        graph.config = {"results_dir": str(tmp_path)}
        explicit_path = tmp_path / "custom" / "location"

        report_file = graph.save_reports(_final_state(), "AAPL", save_path=explicit_path)

        assert report_file == explicit_path / "complete_report.md"
