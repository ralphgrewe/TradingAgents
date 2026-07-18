"""Tests for injecting analyst JSON envelopes into trader and portfolio manager prompts (issue #77).

Tests verify that:
1. Trader prompt receives all four analyst reports with reading instructions
2. Portfolio manager prompt receives all four analyst reports with reading instructions
3. Missing/empty reports are handled gracefully (omitted from output)
4. JSON envelope reading instructions are included in both prompts
5. The fundamentals report label is asset-type aware (crypto phrasing), matching
   the pattern already used by bull_researcher.py / bear_researcher.py

``_format_analyst_reports_section`` originally lived duplicated in both
``trader.py`` and ``portfolio_manager.py``; it now lives once as
``format_analyst_reports_section`` in ``tradingagents.agents.utils.agent_utils``
and both modules import it from there (design-review follow-up on cf49861).
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager
from tradingagents.agents.trader.trader import create_trader
from tradingagents.agents.utils.agent_utils import (
    format_analyst_reports_section as shared_format,
)

# Both trader.py and portfolio_manager.py call the same shared helper, so the
# section-formatting tests exercise it directly once instead of duplicating
# per-module.
trader_format = shared_format
pm_format = shared_format


def _make_sample_market_report():
    """Create a sample market analyst JSON envelope."""
    return json.dumps({
        "skill": "quant-indicator-analyst",
        "ticker": "AAPL",
        "date": "2026-07-16",
        "signal": "BUY",
        "confidence": "HIGH",
        "summary": "Technical setup looks bullish with SMA-50 breakout confirmed",
        "details": {
            "trade_setup": {
                "bias": "BUY",
                "entry_trigger": "breakout",
                "stop_loss": 190.0,
                "take_profit": 210.0,
                "risk_reward": 2.0
            },
            "indicators": [
                {"name": "sma_50", "signal": "Bullish"},
                {"name": "macdh", "signal": "Bullish"}
            ]
        }
    })


def _make_sample_sentiment_report():
    """Create a sample sentiment analyst JSON envelope."""
    return json.dumps({
        "skill": "sentiment-analyst",
        "ticker": "AAPL",
        "date": "2026-07-16",
        "signal": "BUY",
        "confidence": "MEDIUM",
        "summary": "Social sentiment trending positive across platforms",
        "details": {
            "sources": {
                "twitter": {"signal": "BUY", "confidence": 0.65},
                "reddit": {"signal": "HOLD", "confidence": 0.5}
            }
        }
    })


def _make_sample_news_report():
    """Create a sample news analyst JSON envelope."""
    return json.dumps({
        "skill": "financial-news-analyst",
        "ticker": "AAPL",
        "date": "2026-07-16",
        "signal": "BUY",
        "confidence": "HIGH",
        "summary": "Recent news highlights strong product announcements and analyst upgrades",
        "details": {
            "articles_analyzed": 14,
            "conservative": {"rating": "BUY", "confidence": 0.65},
            "risky": {"rating": "BUY", "confidence": 0.8}
        }
    })


def _make_sample_fundamentals_report():
    """Create a sample fundamentals analyst JSON envelope."""
    return json.dumps({
        "skill": "fundamental-analyst",
        "ticker": "AAPL",
        "date": "2026-07-16",
        "signal": "BUY",
        "confidence": "HIGH",
        "summary": "Strong balance sheet and positive cash flow generation",
        "details": {
            "value": {"signal": "BUY", "confidence": "HIGH", "pe_ratio": 28.5},
            "growth": {"signal": "BUY", "confidence": "MEDIUM", "fcf_growth": 0.12}
        }
    })


@pytest.mark.unit
class TestTraderAnalystReportsSection:
    """Tests for trader's _format_analyst_reports_section function."""

    def test_all_four_reports_included_when_present(self):
        """All four reports should be included with proper formatting."""
        market = _make_sample_market_report()
        sentiment = _make_sample_sentiment_report()
        news = _make_sample_news_report()
        fundamentals = _make_sample_fundamentals_report()

        result = trader_format(market, sentiment, news, fundamentals)

        assert "Market research report (JSON envelope):" in result
        assert "Social media sentiment report (JSON envelope):" in result
        assert "Latest world affairs news (JSON envelope):" in result
        assert "Company fundamentals report (JSON envelope):" in result

    def test_reading_instructions_present_when_reports_exist(self):
        """Reading instructions for JSON envelopes should be in the output."""
        market = _make_sample_market_report()
        sentiment = _make_sample_sentiment_report()
        news = _make_sample_news_report()
        fundamentals = _make_sample_fundamentals_report()

        result = trader_format(market, sentiment, news, fundamentals)

        assert "structured JSON envelopes" in result
        assert "`signal`" in result
        assert "`confidence`" in result
        assert "`summary`" in result
        assert "`details`" in result

    def test_graceful_degradation_with_missing_reports(self):
        """Missing reports should be omitted, not cause errors."""
        market = _make_sample_market_report()
        sentiment = None  # Missing
        news = _make_sample_news_report()
        fundamentals = ""  # Empty string

        result = trader_format(market, sentiment, news, fundamentals)

        assert "Market research report (JSON envelope):" in result
        assert "Latest world affairs news (JSON envelope):" in result
        # These should NOT be in the output
        assert "Social media sentiment report (JSON envelope):" not in result
        assert "Company fundamentals report (JSON envelope):" not in result
        # Instructions should still be there because we have reports
        assert "structured JSON envelopes" in result

    def test_empty_string_returned_when_all_reports_missing(self):
        """Should return empty string when all reports are missing."""
        result = trader_format(None, None, None, None)
        assert result == ""

        result = trader_format("", "", "", "")
        assert result == ""

    def test_report_content_preserved_verbatim(self):
        """Report JSON content should be preserved exactly as provided."""
        market = _make_sample_market_report()
        market_data = json.loads(market)

        result = trader_format(market, None, None, None)

        # The market report JSON should appear verbatim in the output
        assert market in result
        assert str(market_data["details"]["trade_setup"]["entry_trigger"]) in result

    def test_single_report_works(self):
        """Function should work with just one report."""
        market = _make_sample_market_report()

        result = trader_format(market, None, None, None)

        assert len(result) > 0
        assert "Market research report (JSON envelope):" in result
        assert "Social media sentiment report (JSON envelope):" not in result

    def test_non_string_reports_ignored(self):
        """Non-string reports (e.g., dict, int) should be ignored."""
        market = _make_sample_market_report()
        sentiment = {"invalid": "dict"}  # Not a string
        news = 42  # Not a string

        result = trader_format(market, sentiment, news, None)

        assert "Market research report (JSON envelope):" in result
        # Non-string reports should be ignored
        assert "Social media sentiment report (JSON envelope):" not in result
        assert "Latest world affairs news (JSON envelope):" not in result


@pytest.mark.unit
class TestPortfolioManagerAnalystReportsSection:
    """Tests for portfolio manager's _format_analyst_reports_section function."""

    def test_all_four_reports_included_when_present(self):
        """All four reports should be included with proper formatting."""
        market = _make_sample_market_report()
        sentiment = _make_sample_sentiment_report()
        news = _make_sample_news_report()
        fundamentals = _make_sample_fundamentals_report()

        result = pm_format(market, sentiment, news, fundamentals)

        assert "Market research report (JSON envelope):" in result
        assert "Social media sentiment report (JSON envelope):" in result
        assert "Latest world affairs news (JSON envelope):" in result
        assert "Company fundamentals report (JSON envelope):" in result

    def test_reading_instructions_present(self):
        """Reading instructions for JSON envelopes should be included."""
        market = _make_sample_market_report()

        result = pm_format(market, None, None, None)

        assert "structured JSON envelopes" in result
        assert "`signal`" in result
        assert "`confidence`" in result
        assert "`summary`" in result
        assert "`details`" in result

    def test_graceful_degradation_with_missing_reports(self):
        """Missing reports should be gracefully omitted."""
        sentiment = _make_sample_sentiment_report()
        news = _make_sample_news_report()

        result = pm_format(None, sentiment, news, None)

        assert "Social media sentiment report (JSON envelope):" in result
        assert "Latest world affairs news (JSON envelope):" in result
        assert "Market research report (JSON envelope):" not in result
        assert "Company fundamentals report (JSON envelope):" not in result

    def test_empty_string_when_all_missing(self):
        """Should return empty string when all reports are missing."""
        result = pm_format(None, None, None, None)
        assert result == ""

        result = pm_format("", "", "", "")
        assert result == ""


@pytest.mark.unit
class TestFormatAnalystReportsSectionAssetTypeLabel:
    """format_analyst_reports_section's fundamentals label should be asset-type
    aware, matching the pattern bull_researcher.py / bear_researcher.py already
    use (issue #77 design-review follow-up)."""

    def test_stock_uses_company_fundamentals_label_by_default(self):
        fundamentals = _make_sample_fundamentals_report()

        result = shared_format(None, None, None, fundamentals)

        assert "Company fundamentals report (JSON envelope):" in result
        assert "Asset fundamentals report" not in result

    def test_stock_asset_type_uses_company_fundamentals_label(self):
        fundamentals = _make_sample_fundamentals_report()

        result = shared_format(None, None, None, fundamentals, asset_type="stock")

        assert "Company fundamentals report (JSON envelope):" in result

    def test_crypto_asset_type_uses_crypto_aware_label(self):
        fundamentals = _make_sample_fundamentals_report()

        result = shared_format(None, None, None, fundamentals, asset_type="crypto")

        assert "Asset fundamentals report (may be unavailable for crypto) (JSON envelope):" in result
        assert "Company fundamentals report" not in result


def _make_trader_state(**overrides):
    """Create a minimal state for trader testing."""
    state = {
        "company_of_interest": "AAPL",
        "asset_type": "stock",
        "investment_plan": "Strong bullish outlook based on technical and fundamental analysis.",
        "market_report": _make_sample_market_report(),
        "sentiment_report": _make_sample_sentiment_report(),
        "news_report": _make_sample_news_report(),
        "fundamentals_report": _make_sample_fundamentals_report(),
        "messages": [],
    }
    state.update(overrides)
    return state


@pytest.mark.unit
class TestTraderPromptAssembly:
    """Tests for trader node prompt assembly with analyst reports."""

    def test_trader_includes_analyst_reports_in_prompt(self):
        """Trader prompt should include all four analyst reports."""
        llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "BUY AAPL at $195 with stop loss at $190"
        llm.invoke.return_value = mock_response

        with patch("tradingagents.agents.trader.trader.bind_structured", return_value=None):
            trader_node = create_trader(llm)
            state = _make_trader_state()
            trader_node(state)

            # Check that llm.invoke was called (fallback path)
            assert llm.invoke.called
            call_args = llm.invoke.call_args
            messages = call_args[0][0]
            user_message = messages[1]["content"]

            # Verify all four reports are in the prompt
            assert "Market research report (JSON envelope):" in user_message
            assert "Social media sentiment report (JSON envelope):" in user_message
            assert "Latest world affairs news (JSON envelope):" in user_message
            assert "Company fundamentals report (JSON envelope):" in user_message

    def test_trader_includes_reading_instructions(self):
        """Trader prompt should include JSON envelope reading instructions."""
        llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "BUY AAPL"
        llm.invoke.return_value = mock_response

        with patch("tradingagents.agents.trader.trader.bind_structured", return_value=None):
            trader_node = create_trader(llm)
            state = _make_trader_state()
            trader_node(state)

            call_args = llm.invoke.call_args
            messages = call_args[0][0]
            user_message = messages[1]["content"]

            assert "structured JSON envelopes" in user_message
            assert "`signal`" in user_message
            assert "`confidence`" in user_message
            assert "`summary`" in user_message
            assert "`details`" in user_message

    def test_trader_handles_missing_reports_gracefully(self):
        """Trader should work when some reports are missing."""
        llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "BUY AAPL"
        llm.invoke.return_value = mock_response

        with patch("tradingagents.agents.trader.trader.bind_structured", return_value=None):
            trader_node = create_trader(llm)
            state = _make_trader_state(sentiment_report=None, fundamentals_report="")
            trader_node(state)

            call_args = llm.invoke.call_args
            messages = call_args[0][0]
            user_message = messages[1]["content"]

            # Only market and news should be in the prompt
            assert "Market research report (JSON envelope):" in user_message
            assert "Latest world affairs news (JSON envelope):" in user_message
            # These should not be present
            assert "Social media sentiment report (JSON envelope):" not in user_message
            assert "Company fundamentals report (JSON envelope):" not in user_message

    def test_trader_uses_crypto_fundamentals_label_for_crypto_asset_type(self):
        """Trader prompt should use the crypto-aware fundamentals label when
        asset_type is crypto, matching bull_researcher.py's pattern (issue #77
        design-review follow-up)."""
        llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "BUY BTC-USD"
        llm.invoke.return_value = mock_response

        with patch("tradingagents.agents.trader.trader.bind_structured", return_value=None):
            trader_node = create_trader(llm)
            state = _make_trader_state(asset_type="crypto")
            trader_node(state)

            call_args = llm.invoke.call_args
            messages = call_args[0][0]
            user_message = messages[1]["content"]

            assert (
                "Asset fundamentals report (may be unavailable for crypto) (JSON envelope):"
                in user_message
            )
            assert "Company fundamentals report (JSON envelope):" not in user_message

    def test_trader_uses_company_fundamentals_label_for_stock_asset_type(self):
        """Explicit stock asset_type should keep the company-fundamentals label."""
        llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "BUY AAPL"
        llm.invoke.return_value = mock_response

        with patch("tradingagents.agents.trader.trader.bind_structured", return_value=None):
            trader_node = create_trader(llm)
            state = _make_trader_state(asset_type="stock")
            trader_node(state)

            call_args = llm.invoke.call_args
            messages = call_args[0][0]
            user_message = messages[1]["content"]

            assert "Company fundamentals report (JSON envelope):" in user_message


def _make_pm_state(**overrides):
    """Create a minimal state for portfolio manager testing."""
    state = {
        "company_of_interest": "AAPL",
        "investment_plan": "Bullish outlook from research team",
        "trader_investment_plan": "BUY 100 shares at $195",
        "market_report": _make_sample_market_report(),
        "sentiment_report": _make_sample_sentiment_report(),
        "news_report": _make_sample_news_report(),
        "fundamentals_report": _make_sample_fundamentals_report(),
        "risk_debate_state": {
            "history": "Aggressive analyst: High growth potential. Conservative analyst: Valuation risk.",
            "aggressive_history": "Growth potential is strong",
            "conservative_history": "Valuation might be high",
            "neutral_history": "Balanced view",
            "latest_speaker": "Conservative",
            "current_aggressive_response": "Latest aggressive",
            "current_conservative_response": "Latest conservative",
            "current_neutral_response": "Latest neutral",
            "judge_decision": "",
            "count": 3,
        },
    }
    state.update(overrides)
    return state


@pytest.mark.unit
class TestPortfolioManagerPromptAssembly:
    """Tests for portfolio manager node prompt assembly with analyst reports."""

    def test_pm_includes_analyst_reports_in_prompt(self):
        """Portfolio manager prompt should include all four analyst reports."""
        llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "BUY"
        llm.invoke.return_value = mock_response

        with patch("tradingagents.agents.managers.portfolio_manager.bind_structured", return_value=None):
            pm_node = create_portfolio_manager(llm)
            state = _make_pm_state()
            pm_node(state)

            # Check that llm.invoke was called (fallback path)
            assert llm.invoke.called
            call_args = llm.invoke.call_args
            prompt = call_args[0][0]

            # Verify all four reports are in the prompt
            assert "Market research report (JSON envelope):" in prompt
            assert "Social media sentiment report (JSON envelope):" in prompt
            assert "Latest world affairs news (JSON envelope):" in prompt
            assert "Company fundamentals report (JSON envelope):" in prompt

    def test_pm_includes_reading_instructions(self):
        """Portfolio manager prompt should include JSON envelope reading instructions."""
        llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "BUY"
        llm.invoke.return_value = mock_response

        with patch("tradingagents.agents.managers.portfolio_manager.bind_structured", return_value=None):
            pm_node = create_portfolio_manager(llm)
            state = _make_pm_state()
            pm_node(state)

            call_args = llm.invoke.call_args
            prompt = call_args[0][0]

            assert "structured JSON envelopes" in prompt
            assert "`signal`" in prompt
            assert "`confidence`" in prompt
            assert "`summary`" in prompt
            assert "`details`" in prompt

    def test_pm_handles_missing_reports_gracefully(self):
        """Portfolio manager should work when some reports are missing."""
        llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "BUY"
        llm.invoke.return_value = mock_response

        with patch("tradingagents.agents.managers.portfolio_manager.bind_structured", return_value=None):
            pm_node = create_portfolio_manager(llm)
            state = _make_pm_state(news_report=None, fundamentals_report="")
            pm_node(state)

            call_args = llm.invoke.call_args
            prompt = call_args[0][0]

            # Market and sentiment should be in the prompt
            assert "Market research report (JSON envelope):" in prompt
            assert "Social media sentiment report (JSON envelope):" in prompt
            # These should not be present
            assert "Latest world affairs news (JSON envelope):" not in prompt
            assert "Company fundamentals report (JSON envelope):" not in prompt

    def test_pm_includes_investment_plan_and_trader_plan(self):
        """Portfolio manager prompt should still include investment and trader plans."""
        llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "BUY"
        llm.invoke.return_value = mock_response

        with patch("tradingagents.agents.managers.portfolio_manager.bind_structured", return_value=None):
            pm_node = create_portfolio_manager(llm)
            state = _make_pm_state()
            pm_node(state)

            call_args = llm.invoke.call_args
            prompt = call_args[0][0]

            # These should still be present (requirement: keep investment_plan)
            assert "investment plan: **Bullish outlook from research team**" in prompt
            assert "Trader's transaction proposal: **BUY 100 shares at $195**" in prompt

    def test_pm_uses_crypto_fundamentals_label_for_crypto_asset_type(self):
        """PM prompt should use the crypto-aware fundamentals label when
        asset_type is crypto, matching bull_researcher.py's pattern (issue #77
        design-review follow-up)."""
        llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "BUY"
        llm.invoke.return_value = mock_response

        with patch("tradingagents.agents.managers.portfolio_manager.bind_structured", return_value=None):
            pm_node = create_portfolio_manager(llm)
            state = _make_pm_state(asset_type="crypto")
            pm_node(state)

            call_args = llm.invoke.call_args
            prompt = call_args[0][0]

            assert (
                "Asset fundamentals report (may be unavailable for crypto) (JSON envelope):"
                in prompt
            )
            assert "Company fundamentals report (JSON envelope):" not in prompt

    def test_pm_uses_company_fundamentals_label_when_asset_type_missing(self):
        """PM should default to the stock-style label when asset_type is absent
        from state, matching the trader's default and bull_researcher.py."""
        llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "BUY"
        llm.invoke.return_value = mock_response

        with patch("tradingagents.agents.managers.portfolio_manager.bind_structured", return_value=None):
            pm_node = create_portfolio_manager(llm)
            state = _make_pm_state()
            pm_node(state)

            call_args = llm.invoke.call_args
            prompt = call_args[0][0]

            assert "Company fundamentals report (JSON envelope):" in prompt
