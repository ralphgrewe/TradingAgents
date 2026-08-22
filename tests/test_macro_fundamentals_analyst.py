"""Tests for the macro fundamentals analyst node (issue #132): the LLM reads
the deterministic macro indicator pack (#131) and emits a JSON envelope,
following the news_analyst.py / news_computation.py pattern.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from tradingagents.agents.analysts.macro_fundamentals_analyst import (
    create_macro_fundamentals_analyst,
)


def _make_state(**overrides):
    state = {
        "trade_date": "2026-07-06",
        "company_of_interest": "AAPL",
        "messages": [],
    }
    state.update(overrides)
    return state


def _make_sample_pack():
    """A minimal-but-realistic macro indicator pack, matching macro_pack.py's shape."""
    return {
        "curr_date": "2026-07-06",
        "indicators": {
            "fed_funds_rate": {
                "alias": "fed_funds_rate",
                "series_id": "FEDFUNDS",
                "available": True,
                "title": "Federal Funds Effective Rate",
                "units": "Percent",
                "frequency": "Monthly",
                "latest_value": 4.33,
                "latest_date": "2026-06-01",
                "prior_value": 4.58,
                "prior_date": "2026-05-01",
                "delta": -0.25,
                "direction": "down",
                "z_score": -1.2,
                "z_score_note": None,
                "observations_used": 24,
            },
            "cpi": {
                "alias": "cpi",
                "series_id": "CPIAUCSL",
                "available": True,
                "title": "Consumer Price Index",
                "units": "Index",
                "frequency": "Monthly",
                "latest_value": 320.5,
                "latest_date": "2026-06-01",
                "prior_value": 319.0,
                "prior_date": "2026-05-01",
                "delta": 1.5,
                "direction": "up",
                "z_score": 0.4,
                "z_score_note": None,
                "observations_used": 24,
            },
            "cot": {"alias": "cot", "available": False, "reason": "no vendor integrated"},
        },
    }


def _make_sample_llm_output(**overrides):
    """Sample macro fundamentals analyst LLM output matching the expected JSON schema."""
    output = {
        "macro_regime": "RISK_OFF",
        "regime_rationale": "Fed cutting but inflation still sticky above target.",
        "drivers": [
            {"indicator": "fed_funds_rate", "reading": "Down 25bp, z-score -1.2 (easing)"},
            {"indicator": "cpi", "reading": "Up 1.5, z-score 0.4 (mild reacceleration)"},
        ],
        "conservative": {"rating": "HOLD", "confidence": 0.55},
        "risky": {"rating": "BUY", "confidence": 0.7},
    }
    output.update(overrides)
    return output


def _run_node(llm_content, pack=None, state_overrides=None):
    """Build the node with a mocked llm + mocked route_to_vendor and invoke it."""
    llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = llm_content
    llm.return_value = mock_response

    pack_value = pack if pack is not None else _make_sample_pack()

    with patch(
        "tradingagents.agents.analysts.macro_fundamentals_analyst.route_to_vendor",
        return_value=pack_value,
    ) as mock_route:
        node = create_macro_fundamentals_analyst(llm)
        result = node(_make_state(**(state_overrides or {})))
    return result, mock_route


@pytest.mark.unit
class TestMacroFundamentalsAnalystJsonEnvelope:
    def test_macro_report_is_json_string(self):
        result, _ = _run_node(json.dumps(_make_sample_llm_output()))
        assert "macro_report" in result
        envelope = json.loads(result["macro_report"])
        assert isinstance(envelope, dict)

    def test_envelope_has_required_fields(self):
        result, _ = _run_node(json.dumps(_make_sample_llm_output()))
        envelope = json.loads(result["macro_report"])
        assert envelope["skill"] == "macro-fundamentals-analyst"
        assert envelope["ticker"] == "AAPL"
        assert envelope["date"] == "2026-07-06"
        assert "signal" in envelope
        assert "confidence" in envelope
        assert "summary" in envelope
        assert "details" in envelope

    def test_details_carries_regime_and_drivers_not_full_pack(self):
        """details should carry the compact regime read, not the full indicator pack."""
        result, _ = _run_node(json.dumps(_make_sample_llm_output()))
        envelope = json.loads(result["macro_report"])
        details = envelope["details"]

        assert details["macro_regime"] == "RISK_OFF"
        assert "regime_rationale" in details
        assert len(details["drivers"]) == 2
        # The full pack's per-series numeric fields must not leak into details.
        assert "indicators" not in details

    def test_pack_attached_as_sibling_key_when_available(self):
        """The full pack is attached at the envelope's top level (for the report
        file / audit trail), not nested inside details."""
        pack = _make_sample_pack()
        result, _ = _run_node(json.dumps(_make_sample_llm_output()), pack=pack)
        envelope = json.loads(result["macro_report"])
        assert envelope["pack"] == pack

    def test_pack_omitted_when_whole_pack_fetch_fails(self):
        """route_to_vendor returning a string sentinel (whole-pack failure) means
        no 'pack' key and the LLM prompt sees no indicator data."""
        result, mock_route = _run_node(
            json.dumps(_make_sample_llm_output()),
            pack="DATA_UNAVAILABLE: optional macro_data could not be retrieved (boom)",
        )
        envelope = json.loads(result["macro_report"])
        assert "pack" not in envelope
        mock_route.assert_called_once_with("get_macro_pack", "2026-07-06")

    def test_route_to_vendor_called_with_trade_date(self):
        _, mock_route = _run_node(json.dumps(_make_sample_llm_output()))
        mock_route.assert_called_once_with("get_macro_pack", "2026-07-06")

    def test_signal_derived_from_conservative_rating(self):
        output = _make_sample_llm_output()
        output["conservative"]["rating"] = "SELL"
        output["risky"]["rating"] = "BUY"
        result, _ = _run_node(json.dumps(output))
        envelope = json.loads(result["macro_report"])
        assert envelope["signal"] == "SELL"

    def test_confidence_high_when_mean_above_0_7(self):
        output = _make_sample_llm_output()
        output["conservative"]["confidence"] = 0.8
        output["risky"]["confidence"] = 0.8
        result, _ = _run_node(json.dumps(output))
        envelope = json.loads(result["macro_report"])
        assert envelope["confidence"] == "HIGH"

    def test_confidence_medium_when_mean_between_0_4_and_0_7(self):
        output = _make_sample_llm_output()
        output["conservative"]["confidence"] = 0.5
        output["risky"]["confidence"] = 0.5
        result, _ = _run_node(json.dumps(output))
        envelope = json.loads(result["macro_report"])
        assert envelope["confidence"] == "MEDIUM"

    def test_confidence_low_when_mean_below_0_4(self):
        output = _make_sample_llm_output()
        output["conservative"]["confidence"] = 0.2
        output["risky"]["confidence"] = 0.3
        result, _ = _run_node(json.dumps(output))
        envelope = json.loads(result["macro_report"])
        assert envelope["confidence"] == "LOW"

    def test_graceful_degradation_on_llm_json_parse_failure(self):
        """Should degrade gracefully (neutral default) on unparseable LLM output."""
        result, _ = _run_node("This is not valid JSON at all")
        envelope = json.loads(result["macro_report"])
        assert envelope["signal"] == "HOLD"
        assert envelope["confidence"] == "MEDIUM"
        assert envelope["details"]["macro_regime"] == "NEUTRAL"
        assert envelope["details"]["drivers"] == []

    def test_graceful_degradation_on_schema_violation(self):
        """Should degrade gracefully when JSON parses but fails pydantic validation."""
        bad_output = {"macro_regime": "BULLISH_AF", "drivers": []}  # invalid enum value
        result, _ = _run_node(json.dumps(bad_output))
        envelope = json.loads(result["macro_report"])
        assert envelope["signal"] == "HOLD"
        assert envelope["details"]["macro_regime"] == "NEUTRAL"

    def test_messages_cleared_after_processing(self):
        result, _ = _run_node(json.dumps(_make_sample_llm_output()))
        assert result["messages"] == []

    def test_summary_reflects_regime(self):
        result, _ = _run_node(json.dumps(_make_sample_llm_output()))
        envelope = json.loads(result["macro_report"])
        assert "RISK_OFF" in envelope["summary"]

    def test_past_context_included_in_prompt(self):
        """macro_past_context on state should reach the LLM prompt."""
        captured = {}

        llm = MagicMock()

        def fake_call(formatted_prompt):
            # Capture the rendered prompt text to check past-context injection.
            captured["text"] = str(formatted_prompt)
            response = MagicMock()
            response.content = json.dumps(_make_sample_llm_output())
            return response

        llm.side_effect = fake_call

        with patch(
            "tradingagents.agents.analysts.macro_fundamentals_analyst.route_to_vendor",
            return_value=_make_sample_pack(),
        ):
            node = create_macro_fundamentals_analyst(llm)
            node(_make_state(macro_past_context="Prior macro call: RISK_ON, was wrong."))

        assert "Prior macro call: RISK_ON, was wrong." in captured["text"]

    def test_indicators_json_included_in_prompt(self):
        """The pack's indicator data should reach the LLM prompt (as pre-computed
        values, per LEARNINGS.md — the LLM must not be asked to derive them)."""
        captured = {}

        llm = MagicMock()

        def fake_call(formatted_prompt):
            captured["text"] = str(formatted_prompt)
            response = MagicMock()
            response.content = json.dumps(_make_sample_llm_output())
            return response

        llm.side_effect = fake_call

        with patch(
            "tradingagents.agents.analysts.macro_fundamentals_analyst.route_to_vendor",
            return_value=_make_sample_pack(),
        ):
            node = create_macro_fundamentals_analyst(llm)
            node(_make_state())

        assert "fed_funds_rate" in captured["text"]
        assert "z_score" in captured["text"]
