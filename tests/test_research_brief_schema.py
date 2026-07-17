"""Tests for ResearchBrief schema and renderer (issue #84).

Tests cover:
1. Schema list-bounds validation: empty args list rejected, 6+ args rejected
2. Render → parse_rating() round-trip for each PortfolioRating value
3. Renderer produces markdown with web research metadata
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from tradingagents.agents.schemas import (
    BriefConfidence,
    PortfolioRating,
    ResearchArgument,
    ResearchBrief,
    render_research_brief,
)
from tradingagents.agents.utils.rating import parse_rating

pytestmark = pytest.mark.unit


class TestResearchArgumentSchema:
    """Tests for ResearchArgument schema."""

    def test_valid_argument_with_analyst_source(self):
        """Test creating a ResearchArgument with an analyst source."""
        arg = ResearchArgument(
            statement="The company has strong earnings growth.",
            source="fundamentals",
        )
        assert arg.statement == "The company has strong earnings growth."
        assert arg.source == "fundamentals"

    def test_valid_argument_with_web_source(self):
        """Test creating a ResearchArgument with a web evidence source."""
        arg = ResearchArgument(
            statement="Market sentiment is positive.",
            source="web:3",
        )
        assert arg.source == "web:3"

    def test_all_valid_sources(self):
        """Test that all specified sources are accepted."""
        sources = ["market", "sentiment", "news", "fundamentals", "web:0", "web:5"]
        for source in sources:
            arg = ResearchArgument(statement="Test statement.", source=source)
            assert arg.source == source


class TestResearchBriefSchema:
    """Tests for ResearchBrief schema."""

    def test_valid_brief_minimal(self):
        """Test creating a minimal valid ResearchBrief."""
        brief = ResearchBrief(
            bull_arguments=[
                ResearchArgument(statement="Bullish point.", source="market"),
            ],
            bear_arguments=[
                ResearchArgument(statement="Bearish point.", source="news"),
            ],
            lean=PortfolioRating.BUY,
            confidence=BriefConfidence.HIGH,
        )
        assert len(brief.bull_arguments) == 1
        assert len(brief.bear_arguments) == 1
        assert brief.new_information is None

    def test_valid_brief_with_new_information(self):
        """Test ResearchBrief with new_information field set."""
        brief = ResearchBrief(
            bull_arguments=[
                ResearchArgument(statement="Strong fundamentals.", source="fundamentals"),
            ],
            bear_arguments=[
                ResearchArgument(statement="Market headwinds.", source="sentiment"),
            ],
            lean=PortfolioRating.HOLD,
            confidence=BriefConfidence.MEDIUM,
            new_information="Web research revealed a upcoming product launch.",
        )
        assert brief.new_information is not None

    def test_bull_arguments_min_length_rejected(self):
        """Test that empty bull_arguments list is rejected."""
        with pytest.raises(ValidationError):
            ResearchBrief(
                bull_arguments=[],
                bear_arguments=[
                    ResearchArgument(statement="Bearish point.", source="news"),
                ],
                lean=PortfolioRating.BUY,
                confidence=BriefConfidence.HIGH,
            )

    def test_bear_arguments_min_length_rejected(self):
        """Test that empty bear_arguments list is rejected."""
        with pytest.raises(ValidationError):
            ResearchBrief(
                bull_arguments=[
                    ResearchArgument(statement="Bullish point.", source="market"),
                ],
                bear_arguments=[],
                lean=PortfolioRating.BUY,
                confidence=BriefConfidence.HIGH,
            )

    def test_bull_arguments_max_length_exceeded(self):
        """Test that 6+ bull_arguments are rejected."""
        with pytest.raises(ValidationError):
            ResearchBrief(
                bull_arguments=[
                    ResearchArgument(statement=f"Bull point {i}.", source="market")
                    for i in range(6)
                ],
                bear_arguments=[
                    ResearchArgument(statement="Bearish point.", source="news"),
                ],
                lean=PortfolioRating.BUY,
                confidence=BriefConfidence.HIGH,
            )

    def test_bear_arguments_max_length_exceeded(self):
        """Test that 6+ bear_arguments are rejected."""
        with pytest.raises(ValidationError):
            ResearchBrief(
                bull_arguments=[
                    ResearchArgument(statement="Bullish point.", source="market"),
                ],
                bear_arguments=[
                    ResearchArgument(statement=f"Bear point {i}.", source="news")
                    for i in range(6)
                ],
                lean=PortfolioRating.BUY,
                confidence=BriefConfidence.HIGH,
            )

    def test_max_arguments_accepted(self):
        """Test that exactly 5 arguments on each side are accepted."""
        brief = ResearchBrief(
            bull_arguments=[
                ResearchArgument(statement=f"Bull point {i}.", source="market")
                for i in range(5)
            ],
            bear_arguments=[
                ResearchArgument(statement=f"Bear point {i}.", source="news")
                for i in range(5)
            ],
            lean=PortfolioRating.HOLD,
            confidence=BriefConfidence.MEDIUM,
        )
        assert len(brief.bull_arguments) == 5
        assert len(brief.bear_arguments) == 5


class TestResearchBriefRenderer:
    """Tests for render_research_brief() function."""

    def test_render_basic_structure(self):
        """Test that the renderer produces all required sections."""
        brief = ResearchBrief(
            bull_arguments=[
                ResearchArgument(statement="Bull point 1.", source="market"),
                ResearchArgument(statement="Bull point 2.", source="fundamentals"),
            ],
            bear_arguments=[
                ResearchArgument(statement="Bear point 1.", source="news"),
            ],
            lean=PortfolioRating.BUY,
            confidence=BriefConfidence.HIGH,
        )
        rendered = render_research_brief(brief, "disabled (config)")
        assert "**Recommendation**: Buy" in rendered
        assert "**Bull Arguments**:" in rendered
        assert "**Bear Arguments**:" in rendered
        assert "**Confidence**: High" in rendered
        assert "Web research: disabled (config)" in rendered

    def test_render_with_new_information(self):
        """Test that new_information is rendered when present."""
        brief = ResearchBrief(
            bull_arguments=[
                ResearchArgument(statement="Bull point.", source="market"),
            ],
            bear_arguments=[
                ResearchArgument(statement="Bear point.", source="news"),
            ],
            lean=PortfolioRating.BUY,
            confidence=BriefConfidence.HIGH,
            new_information="Web search found a new partnership.",
        )
        rendered = render_research_brief(brief, "enabled", 5)
        assert "**New Information**: Web search found a new partnership." in rendered

    def test_render_without_new_information(self):
        """Test that new_information section is omitted when None."""
        brief = ResearchBrief(
            bull_arguments=[
                ResearchArgument(statement="Bull point.", source="market"),
            ],
            bear_arguments=[
                ResearchArgument(statement="Bear point.", source="news"),
            ],
            lean=PortfolioRating.BUY,
            confidence=BriefConfidence.HIGH,
            new_information=None,
        )
        rendered = render_research_brief(brief, "disabled (historical date)")
        assert "**New Information**:" not in rendered

    def test_render_web_research_enabled(self):
        """Test web research metadata when enabled."""
        brief = ResearchBrief(
            bull_arguments=[
                ResearchArgument(statement="Bull point.", source="market"),
            ],
            bear_arguments=[
                ResearchArgument(statement="Bear point.", source="news"),
            ],
            lean=PortfolioRating.HOLD,
            confidence=BriefConfidence.MEDIUM,
        )
        rendered = render_research_brief(brief, "enabled", 12)
        assert "Web research: enabled (12 results)" in rendered

    def test_render_web_research_disabled_variants(self):
        """Test web research metadata for different disabled reasons."""
        brief = ResearchBrief(
            bull_arguments=[
                ResearchArgument(statement="Bull point.", source="market"),
            ],
            bear_arguments=[
                ResearchArgument(statement="Bear point.", source="news"),
            ],
            lean=PortfolioRating.HOLD,
            confidence=BriefConfidence.MEDIUM,
        )

        for status in [
            "disabled (historical date)",
            "disabled (no API key)",
            "disabled (config)",
        ]:
            rendered = render_research_brief(brief, status)
            assert f"Web research: {status}" in rendered

    def test_render_argument_sources_preserved(self):
        """Test that argument sources are rendered as tags."""
        brief = ResearchBrief(
            bull_arguments=[
                ResearchArgument(statement="Strong cash flow.", source="fundamentals"),
                ResearchArgument(statement="Positive sentiment.", source="web:2"),
            ],
            bear_arguments=[
                ResearchArgument(statement="Market saturation.", source="market"),
            ],
            lean=PortfolioRating.OVERWEIGHT,
            confidence=BriefConfidence.HIGH,
        )
        rendered = render_research_brief(brief, "enabled", 3)
        assert "[fundamentals]" in rendered
        assert "[web:2]" in rendered
        assert "[market]" in rendered

    @pytest.mark.parametrize(
        "rating",
        [
            PortfolioRating.BUY,
            PortfolioRating.OVERWEIGHT,
            PortfolioRating.HOLD,
            PortfolioRating.UNDERWEIGHT,
            PortfolioRating.SELL,
        ],
    )
    def test_render_to_parse_rating_roundtrip(self, rating):
        """Test that rendered brief can be parsed back to the original rating.

        This ensures compatibility with downstream parse_rating() usage,
        which looks for the label-first "**Recommendation**: X" format.
        """
        brief = ResearchBrief(
            bull_arguments=[
                ResearchArgument(statement="Bull argument.", source="market"),
            ],
            bear_arguments=[
                ResearchArgument(statement="Bear argument.", source="news"),
            ],
            lean=rating,
            confidence=BriefConfidence.HIGH,
        )
        rendered = render_research_brief(brief, "disabled (config)")
        parsed = parse_rating(rendered)
        assert parsed == rating.value
