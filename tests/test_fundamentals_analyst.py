"""
Tests for the fundamentals analyst node (issue #31): pre-computed ratio tables
with JSON envelope output.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from tradingagents.agents.analysts.fundamentals_analyst import create_fundamentals_analyst


def _make_state(**overrides):
    state = {
        "trade_date": "2026-07-05",
        "company_of_interest": "AAPL",
        "messages": [],
    }
    state.update(overrides)
    return state


def _make_sample_ratios():
    """Create sample computed ratios for testing."""
    return {
        "context": {
            "market_cap": 3000000000000,
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "52w_high": 200.0,
            "52w_low": 150.0,
            "analyst_consensus": "buy",
        },
        "annual": {
            "2026": {
                "valuation": {
                    "pe": 28.5,
                    "pb": 45.2,
                    "ev_ebitda": 20.1,
                    "pcf": 35.2,
                    "peg": 2.1,
                    "ps": 8.5,
                },
                "profitability": {
                    "gross_margin": 0.46,
                    "op_margin": 0.30,
                    "net_margin": 0.25,
                    "roe": 0.85,
                    "roic": 0.72,
                    "roa": 0.18,
                },
                "balance_sheet": {
                    "debt_to_equity": 0.15,
                    "current_ratio": 1.2,
                    "quick_ratio": 1.1,
                    "equity_ratio": 0.65,
                    "interest_coverage": 45.0,
                },
                "cashflow": {
                    "fcf": 75000000000,
                    "op_cf": 110000000000,
                    "fcf_yield": 0.025,
                    "fcf_margin": 0.18,
                    "capex_to_revenue": 0.08,
                },
                "dividends": {
                    "yield": 0.005,
                    "payout_ratio": 0.15,
                },
            },
            "2025": {
                "valuation": {
                    "pe": 26.0,
                    "pb": 42.0,
                    "ev_ebitda": 19.5,
                    "pcf": 33.0,
                    "peg": 2.0,
                    "ps": 8.0,
                },
                "profitability": {
                    "gross_margin": 0.44,
                    "op_margin": 0.28,
                    "net_margin": 0.23,
                    "roe": 0.80,
                    "roic": 0.68,
                    "roa": 0.17,
                },
                "balance_sheet": {
                    "debt_to_equity": 0.16,
                    "current_ratio": 1.18,
                    "quick_ratio": 1.08,
                    "equity_ratio": 0.64,
                    "interest_coverage": 42.0,
                },
                "cashflow": {
                    "fcf": 70000000000,
                    "op_cf": 105000000000,
                    "fcf_yield": 0.023,
                    "fcf_margin": 0.17,
                    "capex_to_revenue": 0.08,
                },
                "dividends": {
                    "yield": 0.005,
                    "payout_ratio": 0.14,
                },
            },
        },
        "insider_sentiment": "BULLISH",
        "forecast": {},
    }


@pytest.mark.unit
class TestFundamentalsAnalystJsonEnvelope:
    """Tests for JSON envelope output per issue #31."""

    def test_fundamentals_report_is_json_string(self):
        """fundamentals_report should contain JSON envelope, not markdown."""
        llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '{"value": {"signal": "BUY", "confidence": "HIGH", "data_confidence": "HIGH", "key_ratios": ["pe"]}, "growth": {"signal": "BUY", "confidence": "HIGH", "data_confidence": "HIGH", "key_ratios": ["gross_margin"]}}'
        mock_llm = MagicMock(return_value=mock_response)
        llm.get_llm.return_value = mock_llm

        with patch(
            "tradingagents.agents.analysts.fundamentals_analyst.compute",
            return_value=_make_sample_ratios(),
        ):
            node = create_fundamentals_analyst(llm)
            result = node(_make_state())

        assert "fundamentals_report" in result
        report_str = result["fundamentals_report"]

        # Should be valid JSON
        envelope = json.loads(report_str)
        assert isinstance(envelope, dict)

    def test_envelope_has_required_fields(self):
        """JSON envelope must have skill, ticker, date, signal, confidence, summary, details."""
        llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '{"value": {"signal": "BUY", "confidence": "HIGH", "data_confidence": "HIGH", "key_ratios": []}, "growth": {"signal": "BUY", "confidence": "HIGH", "data_confidence": "HIGH", "key_ratios": []}}'
        mock_llm = MagicMock(return_value=mock_response)
        llm.get_llm.return_value = mock_llm

        with patch(
            "tradingagents.agents.analysts.fundamentals_analyst.compute",
            return_value=_make_sample_ratios(),
        ):
            node = create_fundamentals_analyst(llm)
            result = node(_make_state())

        envelope = json.loads(result["fundamentals_report"])
        assert envelope["skill"] == "fundamental-analyst"
        assert envelope["ticker"] == "AAPL"
        assert envelope["date"] == "2026-07-05"
        assert "signal" in envelope
        assert "confidence" in envelope
        assert "summary" in envelope
        assert "details" in envelope

    def test_envelope_details_has_required_sections(self):
        """details should contain context, annual, insider_sentiment, value/growth."""
        llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '{"value": {"signal": "BUY", "confidence": "HIGH", "data_confidence": "HIGH", "key_ratios": ["pe"]}, "growth": {"signal": "BUY", "confidence": "HIGH", "data_confidence": "HIGH", "key_ratios": ["gross_margin"]}}'
        mock_llm = MagicMock(return_value=mock_response)
        llm.get_llm.return_value = mock_llm

        with patch(
            "tradingagents.agents.analysts.fundamentals_analyst.compute",
            return_value=_make_sample_ratios(),
        ):
            node = create_fundamentals_analyst(llm)
            result = node(_make_state())

        envelope = json.loads(result["fundamentals_report"])
        details = envelope["details"]

        assert "context" in details
        assert "annual" in details
        assert "insider_sentiment" in details
        assert "forecast" in details
        assert "value" in details
        assert "growth" in details

    def test_signal_derives_correctly_when_agree(self):
        """When value and growth signals agree, top-level signal should match."""
        llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '{"value": {"signal": "BUY", "confidence": "HIGH", "data_confidence": "HIGH", "key_ratios": ["pe"]}, "growth": {"signal": "BUY", "confidence": "HIGH", "data_confidence": "HIGH", "key_ratios": ["gross_margin"]}}'
        mock_summary = MagicMock()
        mock_summary.content = "Strong growth and value"

        mock_llm = MagicMock(side_effect=[mock_response, mock_summary])
        llm.get_llm.return_value = mock_llm

        with patch(
            "tradingagents.agents.analysts.fundamentals_analyst.compute",
            return_value=_make_sample_ratios(),
        ):
            node = create_fundamentals_analyst(llm)
            result = node(_make_state())

        envelope = json.loads(result["fundamentals_report"])
        assert envelope["signal"] == "BUY"

    def test_signal_derives_hold_when_disagree(self):
        """When value and growth signals disagree, top-level signal should be HOLD."""
        llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '{"value": {"signal": "BUY", "confidence": "HIGH", "data_confidence": "HIGH", "key_ratios": ["pe"]}, "growth": {"signal": "SELL", "confidence": "MEDIUM", "data_confidence": "HIGH", "key_ratios": ["gross_margin"]}}'
        mock_summary = MagicMock()
        mock_summary.content = "Mixed signals"

        mock_llm = MagicMock(side_effect=[mock_response, mock_summary])
        llm.get_llm.return_value = mock_llm

        with patch(
            "tradingagents.agents.analysts.fundamentals_analyst.compute",
            return_value=_make_sample_ratios(),
        ):
            node = create_fundamentals_analyst(llm)
            result = node(_make_state())

        envelope = json.loads(result["fundamentals_report"])
        assert envelope["signal"] == "HOLD"

    def test_confidence_takes_higher_when_agree(self):
        """When signals agree, take higher confidence."""
        llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '{"value": {"signal": "BUY", "confidence": "HIGH", "data_confidence": "HIGH", "key_ratios": []}, "growth": {"signal": "BUY", "confidence": "MEDIUM", "data_confidence": "HIGH", "key_ratios": []}}'
        mock_summary = MagicMock()
        mock_summary.content = "BUY"

        mock_llm = MagicMock(side_effect=[mock_response, mock_summary])
        llm.get_llm.return_value = mock_llm

        with patch(
            "tradingagents.agents.analysts.fundamentals_analyst.compute",
            return_value=_make_sample_ratios(),
        ):
            node = create_fundamentals_analyst(llm)
            result = node(_make_state())

        envelope = json.loads(result["fundamentals_report"])
        assert envelope["confidence"] == "HIGH"

    def test_confidence_takes_lower_when_disagree(self):
        """When signals disagree, take lower confidence."""
        llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '{"value": {"signal": "BUY", "confidence": "HIGH", "data_confidence": "HIGH", "key_ratios": []}, "growth": {"signal": "SELL", "confidence": "LOW", "data_confidence": "MEDIUM", "key_ratios": []}}'
        mock_summary = MagicMock()
        mock_summary.content = "Conflicting analysis"

        mock_llm = MagicMock(side_effect=[mock_response, mock_summary])
        llm.get_llm.return_value = mock_llm

        with patch(
            "tradingagents.agents.analysts.fundamentals_analyst.compute",
            return_value=_make_sample_ratios(),
        ):
            node = create_fundamentals_analyst(llm)
            result = node(_make_state())

        envelope = json.loads(result["fundamentals_report"])
        assert envelope["confidence"] == "LOW"

    def test_missing_fundamental_data_degradation(self):
        """Should degrade gracefully when fundamental data unavailable."""
        llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '{"value": {"signal": null, "confidence": null, "data_confidence": "LOW", "key_ratios": []}, "growth": {"signal": null, "confidence": null, "data_confidence": "LOW", "key_ratios": []}}'
        mock_summary = MagicMock()
        mock_summary.content = "Insufficient data"

        mock_llm = MagicMock(side_effect=[mock_response, mock_summary])
        llm.get_llm.return_value = mock_llm

        with patch(
            "tradingagents.agents.analysts.fundamentals_analyst.compute",
            side_effect=Exception("Data fetch failed"),
        ):
            node = create_fundamentals_analyst(llm)
            result = node(_make_state())

        envelope = json.loads(result["fundamentals_report"])
        assert envelope["signal"] is None
        assert envelope["confidence"] is None
        assert "details" in envelope
        assert "context" in envelope["details"]

    def test_llm_parses_value_growth_json(self):
        """LLM should return JSON with value/growth evaluations."""
        llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '{"value": {"signal": "BUY", "confidence": "HIGH", "data_confidence": "HIGH", "key_ratios": ["pe", "roe"]}, "growth": {"signal": "BUY", "confidence": "HIGH", "data_confidence": "HIGH", "key_ratios": ["gross_margin", "fcf_margin"]}}'
        mock_summary = MagicMock()
        mock_summary.content = "Strong fundamentals"

        mock_llm = MagicMock(side_effect=[mock_response, mock_summary])
        llm.get_llm.return_value = mock_llm

        with patch(
            "tradingagents.agents.analysts.fundamentals_analyst.compute",
            return_value=_make_sample_ratios(),
        ):
            node = create_fundamentals_analyst(llm)
            result = node(_make_state())

        envelope = json.loads(result["fundamentals_report"])
        assert envelope["details"]["value"]["signal"] == "BUY"
        assert envelope["details"]["value"]["key_ratios"] == ["pe", "roe"]
        assert envelope["details"]["growth"]["signal"] == "BUY"
        assert envelope["details"]["growth"]["key_ratios"] == ["gross_margin", "fcf_margin"]

    def test_messages_cleared_after_processing(self):
        """Node should return empty messages list after processing."""
        llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '{"value": {"signal": "HOLD", "confidence": "MEDIUM", "data_confidence": "MEDIUM", "key_ratios": []}, "growth": {"signal": "HOLD", "confidence": "MEDIUM", "data_confidence": "MEDIUM", "key_ratios": []}}'
        mock_summary = MagicMock()
        mock_summary.content = "Fairly valued"

        mock_llm = MagicMock(side_effect=[mock_response, mock_summary])
        llm.get_llm.return_value = mock_llm

        with patch(
            "tradingagents.agents.analysts.fundamentals_analyst.compute",
            return_value=_make_sample_ratios(),
        ):
            node = create_fundamentals_analyst(llm)
            result = node(_make_state())

        # Messages should be empty per market analyst pattern
        assert result["messages"] == []
