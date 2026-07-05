"""Tests for the market analyst node (issue #30): deterministic indicator
computation with JSON envelope output.
"""

import json
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from tradingagents.agents.analysts.market_analyst import create_market_analyst


def _make_state(**overrides):
    state = {
        "trade_date": "2026-05-13",
        "company_of_interest": "AAPL",
        "messages": [],
    }
    state.update(overrides)
    return state


def _make_sample_ohlcv():
    """Create sample OHLCV data for testing."""
    base_date = datetime(2024, 1, 1)
    records = []
    price = 100.0
    for i in range(100):
        date = base_date + timedelta(days=i)
        price += 0.1 + (0.05 if i % 3 == 0 else -0.02)
        records.append({
            "Date": date.strftime("%Y-%m-%d"),
            "Open": price - 0.5,
            "High": price + 0.75,
            "Low": price - 0.75,
            "Close": price,
            "Volume": 1000000 + (i * 1000),
        })
    return pd.DataFrame(records)


@pytest.mark.unit
class TestMarketAnalystJsonEnvelope:
    """Tests for JSON envelope output per issue #30."""

    def test_market_report_is_json_string(self):
        """market_report should contain JSON envelope, not markdown."""
        llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Test summary"
        mock_llm = MagicMock(return_value=mock_response)
        llm.get_llm.return_value = mock_llm

        with patch(
            "tradingagents.agents.analysts.market_analyst.load_ohlcv",
            return_value=_make_sample_ohlcv(),
        ):
            node = create_market_analyst(llm)
            result = node(_make_state())

        assert "market_report" in result
        report_str = result["market_report"]

        # Should be valid JSON
        envelope = json.loads(report_str)
        assert isinstance(envelope, dict)

    def test_envelope_has_required_fields(self):
        """JSON envelope must have skill, ticker, date, signal, confidence, summary, details."""
        llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Test summary"
        mock_llm = MagicMock(return_value=mock_response)
        llm.get_llm.return_value = mock_llm

        with patch(
            "tradingagents.agents.analysts.market_analyst.load_ohlcv",
            return_value=_make_sample_ohlcv(),
        ):
            node = create_market_analyst(llm)
            result = node(_make_state())

        envelope = json.loads(result["market_report"])
        assert envelope["skill"] == "market-analyst"
        assert envelope["ticker"] == "AAPL"
        assert envelope["date"] == "2026-05-13"
        assert "signal" in envelope
        assert "confidence" in envelope
        assert "summary" in envelope
        assert "details" in envelope

    def test_envelope_details_has_indicators_list(self):
        """details should contain indicators list with all 7 indicators."""
        llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Test summary"
        mock_llm = MagicMock(return_value=mock_response)
        llm.get_llm.return_value = mock_llm

        with patch(
            "tradingagents.agents.analysts.market_analyst.load_ohlcv",
            return_value=_make_sample_ohlcv(),
        ):
            node = create_market_analyst(llm)
            result = node(_make_state())

        envelope = json.loads(result["market_report"])
        indicators = envelope["details"]["indicators"]
        assert len(indicators) == 7

        # Check indicator structure
        ind_names = {ind["indicator"] for ind in indicators}
        expected = {"sma_50", "macdh", "rsi", "boll_ub", "boll_lb", "atr", "vwma"}
        assert ind_names == expected

    def test_signal_confidence_deterministic(self):
        """Signal and confidence should be deterministic (same for same data)."""
        ohlcv = _make_sample_ohlcv()

        llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Test summary"
        mock_llm = MagicMock(return_value=mock_response)
        llm.get_llm.return_value = mock_llm

        def load_ohlcv_patched(*args, **kwargs):
            return ohlcv.copy()

        with patch(
            "tradingagents.agents.analysts.market_analyst.load_ohlcv",
            side_effect=load_ohlcv_patched,
        ):
            node = create_market_analyst(llm)
            result1 = node(_make_state())
            result2 = node(_make_state())

        envelope1 = json.loads(result1["market_report"])
        envelope2 = json.loads(result2["market_report"])

        assert envelope1["signal"] == envelope2["signal"]
        assert envelope1["confidence"] == envelope2["confidence"]
        assert envelope1["details"] == envelope2["details"]

    def test_missing_ohlcv_degradation(self):
        """Should degrade gracefully when OHLCV unavailable."""
        llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "No data available"
        mock_llm = MagicMock(return_value=mock_response)
        llm.get_llm.return_value = mock_llm

        with patch(
            "tradingagents.agents.analysts.market_analyst.load_ohlcv",
            return_value=pd.DataFrame(),  # Empty DataFrame
        ):
            node = create_market_analyst(llm)
            result = node(_make_state())

        envelope = json.loads(result["market_report"])
        assert envelope["signal"] is None
        assert envelope["confidence"] is None
        assert envelope["details"]["close"] is None
        # When empty DataFrame, compute_indicators still runs and returns proper structure
        assert isinstance(envelope["details"]["convergence"]["missing"], list)

    def test_llm_writes_summary_only(self):
        """LLM should write summary; signal/confidence from computation."""
        llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Custom summary from LLM"

        # Make the mock callable so it works with the pipe operator
        mock_llm = MagicMock(return_value=mock_response)
        llm.get_llm.return_value = mock_llm

        with patch(
            "tradingagents.agents.analysts.market_analyst.load_ohlcv",
            return_value=_make_sample_ohlcv(),
        ):
            node = create_market_analyst(llm)
            result = node(_make_state())

        envelope = json.loads(result["market_report"])
        # Summary should be LLM's output
        assert envelope["summary"] == "Custom summary from LLM"
        # Signal and confidence should come from computation (not None)
        assert envelope["signal"] in ("BUY", "HOLD", "SELL")
        assert envelope["confidence"] in ("HIGH", "MEDIUM", "LOW")

    def test_convergence_and_trade_setup_present(self):
        """details should include convergence and trade_setup."""
        llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Test summary"
        mock_llm = MagicMock(return_value=mock_response)
        llm.get_llm.return_value = mock_llm

        with patch(
            "tradingagents.agents.analysts.market_analyst.load_ohlcv",
            return_value=_make_sample_ohlcv(),
        ):
            node = create_market_analyst(llm)
            result = node(_make_state())

        envelope = json.loads(result["market_report"])
        details = envelope["details"]

        assert "convergence" in details
        assert "confirms" in details["convergence"]
        assert "conflicts" in details["convergence"]
        assert "missing" in details["convergence"]

        assert "trade_setup" in details
        if details["trade_setup"]:
            assert "bias" in details["trade_setup"]
            assert "entry_trigger" in details["trade_setup"]
