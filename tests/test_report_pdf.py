"""Tests for the standalone reportlab-based PDF renderer (issue #164).

Covers the acceptance criteria from #164: a full section list renders a
multi-page PDF with all headings present, JSON-envelope-shaped entries render
their signal/confidence header and content readably, and an empty sections
list still produces a valid (minimal) PDF rather than raising. The rendered
PDF is read back with ``pypdf`` so assertions run against extracted text
rather than requiring visual inspection.
"""

import json

import pytest
from pypdf import PdfReader

from tradingagents.report_pdf import (
    ReportSection,
    ReportSubsection,
    render_complete_report_pdf,
)

pytestmark = pytest.mark.unit


def _extract_text(pdf_path) -> str:
    reader = PdfReader(str(pdf_path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _full_sections() -> list[ReportSection]:
    envelope = json.dumps(
        {
            "skill": "market-analyst",
            "ticker": "AAPL",
            "date": "2024-01-15",
            "signal": "BUY",
            "confidence": 0.82,
            "summary": "Momentum & breadth <both> look constructive.",
            "details": {"rsi": 61.2},
        }
    )
    return [
        ReportSection(
            "I. Analyst Team Reports",
            [
                ReportSubsection("Market Analyst", envelope),
                ReportSubsection("Fundamentals Analyst", "Solid balance sheet."),
            ],
        ),
        ReportSection(
            "II. Research Team Decision",
            [
                ReportSubsection("Bull Researcher", "Bull case text."),
                ReportSubsection("Bear Researcher", "Bear case text."),
                ReportSubsection("Research Manager", "Manager verdict."),
            ],
        ),
        ReportSection(
            "III. Trading Team Plan",
            [ReportSubsection("Trader", "Buy 10 shares.")],
        ),
        ReportSection(
            "IV. Risk Management Team Decision",
            [
                ReportSubsection("Aggressive Analyst", "Aggressive take."),
                ReportSubsection("Conservative Analyst", "Conservative take."),
                ReportSubsection("Neutral Analyst", "Neutral take."),
            ],
        ),
        ReportSection(
            "V. Portfolio Manager Decision",
            [ReportSubsection("Portfolio Manager", "Final decision: BUY.")],
        ),
    ]


class TestRenderCompleteReportPdf:
    def test_full_section_list_renders_multi_page_pdf_with_all_headings(self, tmp_path):
        out_path = tmp_path / "complete_report.pdf"
        result = render_complete_report_pdf("AAPL", _full_sections(), out_path)

        assert result == out_path
        assert out_path.exists()
        assert out_path.stat().st_size > 0

        reader = PdfReader(str(out_path))
        assert len(reader.pages) > 1

        text = _extract_text(out_path)
        # Section headings, in order.
        section_titles = [
            "I. Analyst Team Reports",
            "II. Research Team Decision",
            "III. Trading Team Plan",
            "IV. Risk Management Team Decision",
            "V. Portfolio Manager Decision",
        ]
        positions = [text.index(title) for title in section_titles]
        assert positions == sorted(positions)

        # Agent-name sub-headings are present.
        for agent_name in [
            "Market Analyst",
            "Fundamentals Analyst",
            "Bull Researcher",
            "Bear Researcher",
            "Research Manager",
            "Trader",
            "Aggressive Analyst",
            "Conservative Analyst",
            "Neutral Analyst",
            "Portfolio Manager",
        ]:
            assert agent_name in text

        # Prose body text made it in.
        assert "Solid balance sheet." in text
        assert "Buy 10 shares." in text
        assert "Final decision: BUY." in text

    def test_json_envelope_renders_signal_confidence_header_and_content(self, tmp_path):
        out_path = tmp_path / "envelope.pdf"
        envelope = json.dumps(
            {
                "skill": "market-analyst",
                "ticker": "AAPL",
                "date": "2024-01-15",
                "signal": "BUY",
                "confidence": 0.82,
                "summary": "Momentum & breadth <both> look constructive.",
                "details": {"rsi": 61.2},
            }
        )
        sections = [
            ReportSection(
                "I. Analyst Team Reports",
                [ReportSubsection("Market Analyst", envelope)],
            )
        ]

        render_complete_report_pdf("AAPL", sections, out_path)
        text = _extract_text(out_path)

        assert "Signal: BUY" in text
        assert "confidence: 0.82" in text
        # Escaped-and-unescaped-back-out prose summary (via Paragraph escaping).
        assert "Momentum" in text
        assert "breadth" in text
        # Raw JSON content came through (via Preformatted), including the
        # angle brackets and ampersand that would otherwise break XML markup.
        assert '"signal": "BUY"' in text
        assert '"rsi": 61.2' in text

    def test_empty_sections_list_produces_valid_pdf(self, tmp_path):
        out_path = tmp_path / "empty.pdf"
        result = render_complete_report_pdf("AAPL", [], out_path)

        assert result == out_path
        assert out_path.exists()
        assert out_path.stat().st_size > 0

        reader = PdfReader(str(out_path))
        assert len(reader.pages) >= 1
        text = _extract_text(out_path)
        assert "AAPL" in text

    def test_prose_with_markup_characters_is_escaped_safely(self, tmp_path):
        out_path = tmp_path / "prose.pdf"
        sections = [
            ReportSection(
                "III. Trading Team Plan",
                [ReportSubsection("Trader", "Buy if price < 150 & momentum > 0.")],
            )
        ]

        # Should not raise even though the content contains characters that
        # are meaningful in reportlab's Paragraph XML-like markup.
        render_complete_report_pdf("AAPL", sections, out_path)
        text = _extract_text(out_path)
        assert "Buy if price" in text
        assert "momentum" in text

    def test_section_with_no_subsections_does_not_raise(self, tmp_path):
        out_path = tmp_path / "no_subsections.pdf"
        sections = [ReportSection("I. Analyst Team Reports", [])]

        render_complete_report_pdf("AAPL", sections, out_path)
        text = _extract_text(out_path)
        assert "I. Analyst Team Reports" in text

    def test_creates_parent_directories(self, tmp_path):
        out_path = tmp_path / "nested" / "dir" / "report.pdf"
        result = render_complete_report_pdf("AAPL", [], out_path)
        assert result.exists()
