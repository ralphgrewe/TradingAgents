"""
Unit tests for fundamentals_computation.py

Tests the deterministic ratio computation module with fixed financial data fixtures.
"""

import json

import pytest

from tradingagents.agents.analysts.fundamentals_computation import (
    build_json_envelope,
    build_key_drivers,
    confidence_to_score,
    normalize_statements,
    pick,
    r2,
    ratio,
)


@pytest.fixture
def sample_annual_financials():
    """Sample financial data in yfinance format (columnar dict of dicts)."""
    return {
        "Total Revenue": {
            "2024-01-01": 100000000,
            "2023-01-01": 90000000,
            "2022-01-01": 80000000,
        },
        "Gross Profit": {
            "2024-01-01": 40000000,
            "2023-01-01": 36000000,
            "2022-01-01": 32000000,
        },
        "Operating Income": {
            "2024-01-01": 20000000,
            "2023-01-01": 18000000,
            "2022-01-01": 16000000,
        },
        "Net Income": {
            "2024-01-01": 15000000,
            "2023-01-01": 13500000,
            "2022-01-01": 12000000,
        },
        "Total Assets": {
            "2024-01-01": 200000000,
            "2023-01-01": 190000000,
            "2022-01-01": 180000000,
        },
        "Total Stockholder Equity": {
            "2024-01-01": 100000000,
            "2023-01-01": 95000000,
            "2022-01-01": 90000000,
        },
        "Total Debt": {
            "2024-01-01": 30000000,
            "2023-01-01": 28500000,
            "2022-01-01": 27000000,
        },
    }


@pytest.mark.unit
class TestHelperFunctions:
    """Test standalone helper functions."""

    def test_r2_rounds_to_4_decimals(self):
        """r2() should round to 4 decimal places."""
        assert r2(100.123456) == 100.1235
        assert r2(100.1) == 100.1
        assert r2(0.0) == 0.0

    def test_r2_returns_none_for_none(self):
        """r2() should return None for None input."""
        assert r2(None) is None

    def test_ratio_computes_correctly(self):
        """ratio() should compute a/b."""
        assert ratio(100, 50) == 2.0
        assert ratio(1, 3) == 0.3333

    def test_ratio_returns_none_on_invalid(self):
        """ratio() should return None for invalid input."""
        assert ratio(None, 50) is None
        assert ratio(100, None) is None
        assert ratio(100, 0) is None

    def test_pick_returns_first_non_none(self):
        """pick() should return first non-None value."""
        d = {"a": None, "b": 100, "c": 200}
        assert pick(d, "a", "b", "c") == 100
        assert pick(d, "a", "c") == 200
        assert pick(d, "nonexistent", "b") == 100

    def test_pick_returns_none_when_all_missing(self):
        """pick() should return None when all keys missing."""
        d = {"a": None}
        assert pick(d, "x", "y") is None

    def test_normalize_statements_columnar_dict(self):
        """normalize_statements should handle columnar dict format."""
        data = {
            "Revenue": {"2024-01-15": 1000, "2023-01-15": 900},
            "Net Income": {"2024-01-15": 100, "2023-01-15": 90},
        }
        result = normalize_statements(data)
        assert "2024-01-15" in result
        assert result["2024-01-15"]["Revenue"] == 1000
        assert result["2024-01-15"]["Net Income"] == 100

    def test_normalize_statements_row_oriented_list(self):
        """normalize_statements should handle row-oriented list format."""
        data = [
            {"Date": "2024-01-15", "Revenue": 1000, "Net Income": 100},
            {"Date": "2023-01-15", "Revenue": 900, "Net Income": 90},
        ]
        result = normalize_statements(data)
        assert "2024-01-15" in result
        assert result["2024-01-15"]["Revenue"] == 1000


@pytest.mark.unit
class TestConfidenceScoring:
    """Test confidence scoring functions."""

    def test_confidence_to_score_high(self):
        """HIGH confidence maps to 1.0."""
        assert confidence_to_score("HIGH") == 1.0

    def test_confidence_to_score_medium(self):
        """MEDIUM confidence maps to 0.6."""
        assert confidence_to_score("MEDIUM") == 0.6

    def test_confidence_to_score_low(self):
        """LOW confidence maps to 0.3."""
        assert confidence_to_score("LOW") == 0.3

    def test_confidence_to_score_unknown_defaults_to_low(self):
        """Unknown confidence defaults to 0.3."""
        assert confidence_to_score("UNKNOWN") == 0.3
        assert confidence_to_score(None) == 0.3


@pytest.mark.unit
class TestBuildKeyDrivers:
    """Test key_drivers construction for memory wiring."""

    def test_build_key_drivers_structure(self):
        """build_key_drivers should extract value/growth/insider signals."""
        details = {
            "value": {
                "signal": "BUY",
                "confidence": "HIGH",
                "key_ratios": ["pe", "roe"],
            },
            "growth": {
                "signal": "HOLD",
                "confidence": "MEDIUM",
                "key_ratios": ["gross_margin"],
            },
            "insider_sentiment": "BULLISH",
        }
        result = build_key_drivers(details)
        assert result["value"]["signal"] == "BUY"
        assert result["value"]["confidence"] == "HIGH"
        assert result["growth"]["signal"] == "HOLD"
        assert result["insider_sentiment"] == "BULLISH"

    def test_build_key_drivers_handles_missing_fields(self):
        """build_key_drivers should handle missing fields gracefully."""
        details = {"insider_sentiment": "NEUTRAL"}
        result = build_key_drivers(details)
        assert result["value"]["signal"] is None
        assert result["growth"]["signal"] is None
        assert result["insider_sentiment"] == "NEUTRAL"


@pytest.mark.unit
class TestBuildJsonEnvelope:
    """Test JSON envelope construction."""

    def test_envelope_has_required_fields(self):
        """JSON envelope must have all required fields."""
        envelope_str = build_json_envelope(
            signal="BUY",
            confidence="HIGH",
            summary="Strong value opportunity",
            details={"context": {}},
            ticker="AAPL",
            date="2026-07-05",
        )
        envelope = json.loads(envelope_str)

        assert envelope["skill"] == "fundamental-analyst"
        assert envelope["ticker"] == "AAPL"
        assert envelope["date"] == "2026-07-05"
        assert envelope["signal"] == "BUY"
        assert envelope["confidence"] == "HIGH"
        assert envelope["summary"] == "Strong value opportunity"
        assert "details" in envelope

    def test_envelope_is_valid_json(self):
        """Envelope should be valid JSON string."""
        envelope_str = build_json_envelope(
            signal="HOLD",
            confidence="MEDIUM",
            summary="Mixed signals",
            details={},
            ticker="MSFT",
            date="2026-07-05",
        )
        # Should not raise
        envelope = json.loads(envelope_str)
        assert isinstance(envelope, dict)

    def test_envelope_with_none_values(self):
        """Envelope should handle None signal/confidence."""
        envelope_str = build_json_envelope(
            signal=None,
            confidence=None,
            summary="Unable to compute",
            details={},
            ticker="UNKNOWN",
            date="2026-07-05",
        )
        envelope = json.loads(envelope_str)
        assert envelope["signal"] is None
        assert envelope["confidence"] is None
