"""Standalone reportlab-based PDF renderer for the complete report (issue #164).

Part of #150 (converting the complete report to PDF). This module knows nothing
about ``AgentState``, LangGraph, or the on-disk report tree written by
``tradingagents/reporting.py::write_report_tree`` — it takes an already-built,
ordered list of :class:`ReportSection` objects (mirroring today's
``## I.``-``## V.`` markdown headings) and writes a single well-formatted PDF
with ``reportlab``'s Platypus layer. No LLM call happens anywhere in this path.

Wiring this into ``write_report_tree`` (building the ``ReportSection`` list
from ``AgentState`` fields) and deciding the on-disk filename are out of scope
for this module — that's the caller's job (see the #150 follow-up sub-issue).

JSON-envelope content (the analyst JSON fields — ``skill``/``ticker``/``date``/
``signal``/``confidence``/``summary``/``details``, see skills/SCHEMA.md) is
detected with the same heuristic ``tradingagents/reporting.py`` uses
(``_try_parse_envelope``: strips the text, checks it starts with ``{`` and
parses as a JSON object) and rendered the same way conceptually — a short
signal/confidence header followed by the fenced/raw content — mirroring
``format_report_markdown``'s convention instead of inventing a new one. The
raw JSON text is rendered with ``Preformatted`` (which does not interpret
markup) rather than escaped and passed to ``Paragraph``, since a JSON blob is
already tricky to escape correctly. Plain prose subsections are also safe
against reportlab's XML-like ``Paragraph`` markup: ``&``, ``<``, ``>`` are
escaped before the text reaches ``Paragraph``.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape as _xml_escape

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
)

from tradingagents.reporting import _try_parse_envelope

__all__ = ["ReportSection", "ReportSubsection", "render_complete_report_pdf"]


@dataclass
class ReportSubsection:
    """One agent's contribution within a top-level report section.

    ``content`` is the raw field value exactly as it would be written to the
    per-ticker report tree — either prose or a JSON-envelope string. This
    mirrors the ``(agent_name, text)`` tuples ``write_report_tree`` already
    builds internally (e.g. ``analyst_parts``, ``research_parts``), so the
    #150 follow-up can construct this structure directly from those lists.
    """

    agent_name: str
    content: str


@dataclass
class ReportSection:
    """A top-level report section (mirrors the ``## I.``-``## V.`` headings)."""

    title: str
    subsections: list[ReportSubsection] = field(default_factory=list)


def _escape(text: str) -> str:
    """Escape ``&``, ``<``, ``>`` so ``text`` is safe inside a reportlab ``Paragraph``."""
    return _xml_escape(text or "")


def _build_styles() -> dict[str, ParagraphStyle]:
    """Build the paragraph/preformatted styles used throughout the document."""
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=base["Title"],
            fontSize=20,
            spaceAfter=4,
        ),
        "meta": ParagraphStyle(
            "ReportMeta",
            parent=base["Normal"],
            fontSize=9,
            textColor="#555555",
            spaceAfter=2,
        ),
        "section": ParagraphStyle(
            "SectionHeading",
            parent=base["Heading1"],
            fontSize=16,
            spaceBefore=12,
            spaceAfter=8,
            textColor="#1a1a1a",
        ),
        "agent": ParagraphStyle(
            "AgentHeading",
            parent=base["Heading2"],
            fontSize=12,
            spaceBefore=10,
            spaceAfter=4,
            textColor="#2952a3",
        ),
        "envelope_header": ParagraphStyle(
            "EnvelopeHeader",
            parent=base["Normal"],
            fontSize=10,
            leading=13,
            spaceAfter=4,
            textColor="#333333",
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["Normal"],
            fontSize=10,
            leading=14,
            spaceAfter=6,
        ),
        "code": ParagraphStyle(
            "Code",
            parent=base["Code"],
            fontName="Courier",
            fontSize=8,
            leading=10,
            spaceAfter=8,
        ),
        "empty": ParagraphStyle(
            "Empty",
            parent=base["Italic"],
            fontSize=10,
        ),
    }


def _prose_flowables(text: str, style: ParagraphStyle) -> list:
    """Render plain prose text as one or more escaped ``Paragraph`` flowables."""
    text = (text or "").strip()
    if not text:
        return []
    flowables = []
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        escaped = _escape(para).replace("\n", "<br/>")
        flowables.append(Paragraph(escaped, style))
    return flowables


def _envelope_flowables(envelope: dict, raw_content: str, styles: dict) -> list:
    """Render a JSON-envelope subsection: signal/confidence header + raw JSON.

    Mirrors ``tradingagents/reporting.py::format_report_markdown``'s
    convention (a one-line signal/confidence header followed by the envelope
    content), but uses ``Preformatted`` instead of a markdown fenced code
    block since this is a PDF, not markdown.
    """
    flowables = []
    signal = envelope.get("signal")
    confidence = envelope.get("confidence")
    if signal:
        header = f"Signal: {_escape(str(signal))} (confidence: {_escape(str(confidence))})"
        flowables.append(Paragraph(header, styles["envelope_header"]))
    summary = envelope.get("summary")
    if summary:
        flowables.extend(_prose_flowables(str(summary), styles["body"]))
    flowables.append(Preformatted(raw_content, styles["code"]))
    return flowables


def render_complete_report_pdf(
    ticker: str,
    sections: list[ReportSection],
    out_path: Path,
) -> Path:
    """Render ``sections`` to a PDF at ``out_path`` using reportlab and return the path.

    ``sections`` is an ordered list of :class:`ReportSection`, each carrying an
    ordered list of :class:`ReportSubsection` (agent name + content). Section
    headings and agent-name sub-headings get visually distinct styles; JSON
    envelope content is rendered readably (signal/confidence header + raw
    JSON) via :func:`_envelope_flowables`. An empty ``sections`` list still
    produces a valid (minimal) PDF rather than raising.

    No LLM call happens in this path.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    styles = _build_styles()
    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=LETTER,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        title=f"Trading Analysis Report: {ticker}",
    )

    story: list = [
        Paragraph(_escape(f"Trading Analysis Report: {ticker}"), styles["title"]),
        Paragraph(
            _escape(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"),
            styles["meta"],
        ),
        Spacer(1, 18),
    ]

    if not sections:
        story.append(Paragraph("No report sections available.", styles["empty"]))

    for i, section in enumerate(sections):
        if i > 0:
            story.append(PageBreak())
        story.append(Paragraph(_escape(section.title), styles["section"]))

        if not section.subsections:
            story.append(Paragraph("(no content)", styles["empty"]))
            continue

        for sub in section.subsections:
            story.append(Paragraph(_escape(sub.agent_name), styles["agent"]))
            envelope = _try_parse_envelope(sub.content)
            if envelope is not None:
                story.extend(_envelope_flowables(envelope, sub.content, styles))
            else:
                prose = _prose_flowables(sub.content, styles["body"])
                story.extend(prose or [Paragraph("(empty)", styles["empty"])])
            story.append(Spacer(1, 10))

    doc.build(story)
    return out_path
