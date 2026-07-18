"""Tests for issue #33: non-LLM consumers of the JSON-envelope analyst reports.

`market_report`/`fundamentals_report`/`news_report`/`sentiment_report` are all
JSON envelope strings since #30-#32 and #71; every other text field (debate
history, trader plan, final decision) intentionally remains prose.
`tradingagents/reporting.py`'s `format_report_preview`/`format_report_markdown`
detect which shape a field is and render it accordingly; this module checks
those helpers directly plus their use in `write_report_tree`, the CLI's
`MessageBuffer`/`display_complete_report`, and `run_trading_agents.py`'s
`display_summary`.
"""

import json

import pytest

from tradingagents.reporting import (
    format_report_markdown,
    format_report_preview,
    write_report_tree,
)

pytestmark = pytest.mark.unit


def _envelope(summary="Bullish trend", signal="BUY", confidence="HIGH", **details):
    return json.dumps(
        {
            "skill": "market-analyst",
            "ticker": "NVDA",
            "date": "2026-07-06",
            "signal": signal,
            "confidence": confidence,
            "summary": summary,
            "details": details or {"close": 100.0},
        },
        indent=2,
    )


class TestFormatReportPreview:
    def test_envelope_uses_signal_confidence_summary(self):
        preview = format_report_preview(_envelope(summary="Strong uptrend confirmed"))
        assert preview == "[BUY/HIGH] Strong uptrend confirmed"

    def test_prose_passes_through_unchanged(self):
        assert format_report_preview("Sentiment is upbeat.") == "Sentiment is upbeat."

    def test_truncates_long_text(self):
        long_summary = "x" * 200
        preview = format_report_preview(_envelope(summary=long_summary), max_len=20)
        assert preview.endswith("...")
        assert len(preview) == 23  # 20 chars + "..."

    def test_empty_and_none_are_safe(self):
        assert format_report_preview("") == ""
        assert format_report_preview(None) == ""

    def test_non_envelope_json_passes_through(self):
        """A JSON array/scalar (not an envelope dict) is not mistaken for one."""
        assert format_report_preview("[1, 2, 3]") == "[1, 2, 3]"


class TestFormatReportMarkdown:
    def test_envelope_renders_signal_header_and_fenced_json(self):
        raw = _envelope(summary="Strong uptrend confirmed")
        rendered = format_report_markdown(raw)
        assert "**Signal:** BUY (confidence: HIGH)" in rendered
        assert "Strong uptrend confirmed" in rendered
        assert "```json" in rendered
        assert raw in rendered  # original envelope preserved verbatim inside the fence

    def test_prose_passes_through_unchanged(self):
        prose = "### Sentiment\nUpbeat social chatter."
        assert format_report_markdown(prose) == prose

    def test_non_dict_json_passes_through(self):
        assert format_report_markdown("[1, 2, 3]") == "[1, 2, 3]"


class TestWriteReportTreeWithEnvelopes:
    def _final_state(self):
        return {
            "market_report": _envelope(summary="Technical picture is bullish"),
            "sentiment_report": _envelope(summary="Sentiment is upbeat on social media.", signal="BUY", confidence="MEDIUM"),
            "news_report": _envelope(summary="Headlines skew positive", signal="HOLD"),
            "fundamentals_report": _envelope(summary="Fairly valued", signal="HOLD", confidence="MEDIUM"),
        }

    def test_per_analyst_files_keep_raw_json(self, tmp_path):
        """Individual .json files stay machine-parseable (raw envelope, no fencing)."""
        save_path = tmp_path / "report"
        write_report_tree(self._final_state(), "NVDA", save_path)

        market_text = (save_path / "1_analysts" / "market.json").read_text()
        assert json.loads(market_text)["signal"] == "BUY"

        sentiment_text = (save_path / "1_analysts" / "sentiment.json").read_text()
        assert json.loads(sentiment_text)["signal"] == "BUY"

    def test_complete_report_renders_envelopes_readably(self, tmp_path):
        """The consolidated report fences JSON envelopes, including sentiment (#71)."""
        save_path = tmp_path / "report"
        report_file = write_report_tree(self._final_state(), "NVDA", save_path)
        complete = report_file.read_text()

        assert "```json" in complete
        assert "**Signal:** BUY" in complete
        assert "Sentiment is upbeat on social media." in complete

    def test_prose_only_state_unaffected(self, tmp_path):
        """Non-JSON reports (pre-existing behavior) render exactly as before."""
        save_path = tmp_path / "report"
        state = {"market_report": "Market looks bullish."}
        report_file = write_report_tree(state, "MSFT", save_path)
        assert "Market looks bullish." in report_file.read_text()


class TestCliMessageBufferWithEnvelopes:
    def test_final_report_renders_json_analyst_sections(self):
        from cli.main import MessageBuffer

        buf = MessageBuffer()
        buf.init_for_analysis(["market", "social", "news", "fundamentals"])
        buf.update_report_section("market_report", _envelope(summary="Technical picture is bullish"))
        buf.update_report_section(
            "sentiment_report",
            _envelope(summary="Sentiment is upbeat on social media.", signal="BUY", confidence="MEDIUM"),
        )

        assert "```json" in buf.final_report
        assert "**Signal:** BUY" in buf.final_report
        assert "Sentiment is upbeat on social media." in buf.final_report

    def test_current_report_renders_latest_json_section(self):
        from cli.main import MessageBuffer

        buf = MessageBuffer()
        buf.init_for_analysis(["market"])
        buf.update_report_section("market_report", _envelope(summary="Technical picture is bullish"))

        assert "```json" in buf.current_report
        assert "**Signal:** BUY" in buf.current_report


class TestRunTradingAgentsDisplaySummary:
    def test_display_summary_shows_envelope_preview_not_raw_json(self, capsys):
        from run_trading_agents import display_summary

        final_state = {
            "market_report": _envelope(summary="Technical picture is bullish"),
            "sentiment_report": _envelope(
                summary="Sentiment is upbeat on social media.", signal="BUY", confidence="MEDIUM"
            ),
        }
        display_summary(final_state, "NVDA")
        out = capsys.readouterr().out

        assert "[BUY/HIGH] Technical picture is bullish" in out
        assert "[BUY/MEDIUM] Sentiment is upbeat on social media." in out
        # The raw envelope's opening brace should not leak into the one-liner.
        assert '{\n  "skill"' not in out
