"""Tests for issue #156: Portfolio Manager structured decision requirement.

Covers PortfolioDecisionError abort when structured decision is missing or invalid.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from tradingagents.agents.managers.exceptions import PortfolioDecisionError
from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager
from tradingagents.agents.schemas import PortfolioDecision, PortfolioRating
from tradingagents.dataflows.config import set_config

pytestmark = pytest.mark.unit


def _create_mock_llm(structured_decision=None, fallback_text=None, model_name="gpt-4o-mini"):
    """Create a mock LLM for testing portfolio manager.

    ``_llm_type`` is deliberately left at MagicMock's generic auto-attribute
    (rather than set to something like "openai-chat") because production code
    must not read it for the model name — ``ChatOpenAI._llm_type`` always
    returns the constant "openai-chat" regardless of the configured model
    (this was issue #156's bug 2). ``model_name`` mirrors the real
    ``ChatOpenAI``/``AzureChatOpenAI`` pydantic field that actually carries the
    configured model.
    """
    llm = MagicMock()
    llm.model_name = model_name

    def mock_with_structured_output(schema):
        if schema is PortfolioDecision:
            structured = MagicMock()
            if structured_decision is not None:
                structured.invoke.return_value = structured_decision
            else:
                structured.invoke.return_value = None
            return structured
        return None

    llm.with_structured_output.side_effect = mock_with_structured_output

    # Mock bind_tools to return the llm itself
    llm.bind_tools = MagicMock(return_value=llm)

    # For free-text fallback responses
    llm.invoke.return_value = None

    return llm


def _make_base_state():
    """Create a minimal AgentState for testing."""
    return {
        "date": "2026-08-29",
        "company_of_interest": "NVDA",
        "asset_type": "stock",
        "market_report": json.dumps({"signal": "BUY", "confidence": "HIGH"}),
        "sentiment_report": json.dumps({"signal": "BUY", "confidence": "MEDIUM"}),
        "news_report": json.dumps({"signal": "BUY", "confidence": "MEDIUM"}),
        "fundamentals_report": json.dumps({"signal": "BUY", "confidence": "HIGH"}),
        "investment_plan": "Bullish recommendation",
        "trader_investment_plan": "Buy entry",
        "past_context": "",
        "risk_debate_state": {
            "history": "Risk team debated, concluded bullish.",
            "aggressive_history": "Aggressive debator supports buy.",
            "conservative_history": "Conservative debator cautiously supports.",
            "neutral_history": "Neutral debator agrees.",
            "latest_speaker": "Neutral",
            "current_aggressive_response": "",
            "current_conservative_response": "",
            "current_neutral_response": "",
            "judge_decision": "",
            "count": 1,
        },
        "messages": [],
    }


class TestPortfolioManagerStructuredDecisionRequirement:
    """Tests for issue #156: Portfolio Manager structured decision requirement."""

    def test_valid_structured_decision_passes_through(self):
        """When structured_result is valid with proper rating, it passes through."""
        valid_decision = PortfolioDecision(
            rating=PortfolioRating.BUY,
            executive_summary="Buy on strength.",
            investment_thesis="Technical and fundamental support.",
            price_target=125.0,
        )
        llm = _create_mock_llm(structured_decision=valid_decision)

        # Patch fallback to never be used
        with patch(
            "tradingagents.agents.managers.portfolio_manager.run_structured_with_tools"
        ) as mock_run:
            mock_run.return_value = (valid_decision, None, [])

            pm = create_portfolio_manager(llm)
            state = _make_base_state()
            set_config({"portfolio_manager_require_structured_decision": True})

            result = pm(state)

            # Should complete successfully with structured data
            assert result["portfolio_structured_data"] is not None
            assert result["portfolio_structured_data"]["rating"] == "Buy"
            assert result["final_trade_decision"]
            assert "Buy" in result["final_trade_decision"]

    def test_missing_structured_data_raises_portfolio_decision_error(self):
        """When structured_result is None, PortfolioDecisionError is raised."""
        llm = _create_mock_llm(structured_decision=None)

        with patch(
            "tradingagents.agents.managers.portfolio_manager.run_structured_with_tools"
        ) as mock_run:
            mock_run.return_value = (None, "Fallback text (bad decision)", [])

            pm = create_portfolio_manager(llm)
            state = _make_base_state()
            set_config({"portfolio_manager_require_structured_decision": True})

            with pytest.raises(PortfolioDecisionError) as exc_info:
                pm(state)

            error_msg = str(exc_info.value)
            assert "NVDA" in error_msg
            assert "gpt-4o-mini" in error_msg
            assert "openai-chat" not in error_msg
            assert "structured decision" in error_msg
            assert "aborting ticker" in error_msg

    def test_invalid_rating_raises_portfolio_decision_error(self):
        """When structured result has invalid rating, PortfolioDecisionError is raised."""
        # Create a decision with an invalid rating
        invalid_decision = MagicMock(spec=PortfolioDecision)
        invalid_decision.dict.return_value = {
            "rating": "InvalidRating",  # Not in RATINGS_5_TIER
            "executive_summary": "Bad rating.",
            "investment_thesis": "This has an invalid rating.",
        }

        llm = _create_mock_llm(structured_decision=invalid_decision)

        with patch(
            "tradingagents.agents.managers.portfolio_manager.run_structured_with_tools"
        ) as mock_run:
            mock_run.return_value = (invalid_decision, None, [])

            pm = create_portfolio_manager(llm)
            state = _make_base_state()
            set_config({"portfolio_manager_require_structured_decision": True})

            with pytest.raises(PortfolioDecisionError) as exc_info:
                pm(state)

            error_msg = str(exc_info.value)
            assert "NVDA" in error_msg
            assert "gpt-4o-mini" in error_msg
            assert "openai-chat" not in error_msg
            assert "invalid rating" in error_msg
            assert "InvalidRating" in error_msg
            assert "aborting ticker" in error_msg

    def test_missing_rating_field_raises_portfolio_decision_error(self):
        """When structured result is missing rating field, PortfolioDecisionError is raised."""
        # Create a decision with missing rating
        invalid_decision = MagicMock(spec=PortfolioDecision)
        invalid_decision.dict.return_value = {
            # No "rating" key
            "executive_summary": "No rating.",
            "investment_thesis": "This is missing a rating.",
        }

        llm = _create_mock_llm(structured_decision=invalid_decision)

        with patch(
            "tradingagents.agents.managers.portfolio_manager.run_structured_with_tools"
        ) as mock_run:
            mock_run.return_value = (invalid_decision, None, [])

            pm = create_portfolio_manager(llm)
            state = _make_base_state()
            set_config({"portfolio_manager_require_structured_decision": True})

            with pytest.raises(PortfolioDecisionError) as exc_info:
                pm(state)

            error_msg = str(exc_info.value)
            assert "invalid rating" in error_msg
            assert "None" in error_msg  # rating is None

    def test_error_message_uses_company_of_interest_as_ticker(self):
        """The error message names the real ticker (state["company_of_interest"]),
        not a "ticker" key -- AgentState never has one (issue #156 bug 1)."""
        llm = _create_mock_llm(structured_decision=None)

        with patch(
            "tradingagents.agents.managers.portfolio_manager.run_structured_with_tools"
        ) as mock_run:
            mock_run.return_value = (None, "Fallback text (bad decision)", [])

            pm = create_portfolio_manager(llm)
            state = _make_base_state()
            state["company_of_interest"] = "TSLA"
            assert "ticker" not in state
            set_config({"portfolio_manager_require_structured_decision": True})

            with pytest.raises(PortfolioDecisionError) as exc_info:
                pm(state)

            error_msg = str(exc_info.value)
            assert "TSLA" in error_msg
            assert "unknown" not in error_msg

    def test_error_message_falls_back_to_model_attribute(self):
        """When the llm exposes only ``model`` (e.g. Anthropic/Google-style
        clients) rather than ``model_name`` (ChatOpenAI/Azure-style), the
        error message still names the real model instead of "unknown"."""
        llm = MagicMock()
        del llm.model_name  # simulate a client with no model_name attribute
        llm.model = "claude-sonnet-4-5"

        def mock_with_structured_output(schema):
            structured = MagicMock()
            structured.invoke.return_value = None
            return structured

        llm.with_structured_output.side_effect = mock_with_structured_output
        llm.bind_tools = MagicMock(return_value=llm)

        with patch(
            "tradingagents.agents.managers.portfolio_manager.run_structured_with_tools"
        ) as mock_run:
            mock_run.return_value = (None, "Fallback text (bad decision)", [])

            pm = create_portfolio_manager(llm)
            state = _make_base_state()
            set_config({"portfolio_manager_require_structured_decision": True})

            with pytest.raises(PortfolioDecisionError) as exc_info:
                pm(state)

            error_msg = str(exc_info.value)
            assert "claude-sonnet-4-5" in error_msg

    def test_opt_out_restores_previous_behavior(self):
        """When portfolio_manager_require_structured_decision=False, no error is raised."""
        llm = _create_mock_llm(structured_decision=None)

        with patch(
            "tradingagents.agents.managers.portfolio_manager.run_structured_with_tools"
        ) as mock_run:
            mock_run.return_value = (None, "Fallback text without error", [])

            pm = create_portfolio_manager(llm)
            state = _make_base_state()
            # Opt out of the requirement
            set_config({"portfolio_manager_require_structured_decision": False})

            # Should not raise, should use fallback
            result = pm(state)

            assert result["final_trade_decision"] == "Fallback text without error"
            assert "portfolio_structured_data" not in result or result["portfolio_structured_data"] is None

    def test_all_valid_ratings_pass(self):
        """All ratings in RATINGS_5_TIER pass validation."""
        from tradingagents.agents.utils.rating import RATINGS_5_TIER

        for rating_str in RATINGS_5_TIER:
            rating_enum = PortfolioRating(rating_str)
            valid_decision = PortfolioDecision(
                rating=rating_enum,
                executive_summary=f"Decision for {rating_str}.",
                investment_thesis=f"This is a {rating_str} decision.",
            )
            llm = _create_mock_llm(structured_decision=valid_decision)

            with patch(
                "tradingagents.agents.managers.portfolio_manager.run_structured_with_tools"
            ) as mock_run:
                mock_run.return_value = (valid_decision, None, [])

                pm = create_portfolio_manager(llm)
                state = _make_base_state()
                set_config({"portfolio_manager_require_structured_decision": True})

                result = pm(state)

                assert result["portfolio_structured_data"] is not None
                assert result["portfolio_structured_data"]["rating"] == rating_str

    def test_portfolio_structured_data_only_in_result_when_valid(self):
        """portfolio_structured_data is only included in result when not None."""
        # Test with None
        llm = _create_mock_llm(structured_decision=None)

        with patch(
            "tradingagents.agents.managers.portfolio_manager.run_structured_with_tools"
        ) as mock_run:
            mock_run.return_value = (None, "Fallback", [])

            pm = create_portfolio_manager(llm)
            state = _make_base_state()
            set_config({"portfolio_manager_require_structured_decision": False})

            result = pm(state)

            # When None and opt-out is True, portfolio_structured_data should not be in result
            # (or be None)
            assert "portfolio_structured_data" not in result or result.get("portfolio_structured_data") is None

        # Test with valid decision
        valid_decision = PortfolioDecision(
            rating=PortfolioRating.HOLD,
            executive_summary="Hold position.",
            investment_thesis="No change needed.",
        )
        llm = _create_mock_llm(structured_decision=valid_decision)

        with patch(
            "tradingagents.agents.managers.portfolio_manager.run_structured_with_tools"
        ) as mock_run:
            mock_run.return_value = (valid_decision, None, [])

            pm = create_portfolio_manager(llm)
            state = _make_base_state()
            set_config({"portfolio_manager_require_structured_decision": True})

            result = pm(state)

            # When valid, portfolio_structured_data should be in result
            assert "portfolio_structured_data" in result
            assert result["portfolio_structured_data"] is not None
