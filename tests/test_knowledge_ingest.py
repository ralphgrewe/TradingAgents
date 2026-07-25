"""Tests for tradingagents.knowledge.ingest -- the PDF -> LLM-wiki article
ingestion pipeline (#102).

No real LLM/network calls: LLM responses are canned (either a plain object
with a ``.content`` attribute, or via the ``mock_llm_client`` fixture from
``tests/conftest.py`` for the factory-wiring path). PDF extraction is
monkeypatched where the orchestration layer is under test, since exercising
real ``pypdf`` parsing isn't the "drafting/assembly logic" this issue asks to
unit test.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tradingagents.knowledge import ingest
from tradingagents.knowledge.wiki_schema import (
    REQUIRED_FRONTMATTER_KEYS,
    REQUIRED_SECTIONS,
    validate_article,
    validate_article_file,
)

# A well-formed article body the "LLM" drafts, deliberately using a WRONG
# source.file and a non-kebab-case id -- ingest.py must force-correct both
# rather than trust the model's output (see #102 caveats).
_DRAFTED_ARTICLE = """---
id: Example Signal!!
title: Example Signal
tags: [example, test]
signals: [example_signal]
asset_classes: [equity]
horizon: [swing]
source: {authors: "A. Author", title: "Example Paper", year: 2000, file: paper/WRONG_PATH.pdf}
---
## Summary

An example summary grounded in the source text.

## Signal — what it is

Precise definition of the example signal.

## How to compute

signal = a / b, computed from line items X and Y.

## Empirical evidence

The paper found a statistically significant effect over 1990-2010.

## When to apply / regime

Equities, position horizon, broad liquid markets.

## Caveats

Look-ahead bias and data requirements apply.
"""


def _fake_response(content: str):
    return SimpleNamespace(content=content)


# ---------------------------------------------------------------------------
# slugify
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSlugify:
    def test_lowercases_and_hyphenates(self):
        assert ingest.slugify("Piotroski F-Score") == "piotroski-f-score"

    def test_collapses_repeated_separators(self):
        assert ingest.slugify("A!!  B__C") == "a-b-c"

    def test_strips_leading_trailing_separators(self):
        assert ingest.slugify("--Example--") == "example"

    def test_empty_input_falls_back_to_article(self):
        assert ingest.slugify("!!!") == "article"


# ---------------------------------------------------------------------------
# build_draft_prompt
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildDraftPrompt:
    def test_reproduces_required_sections_verbatim(self):
        prompt = ingest.build_draft_prompt("source text", "Paper.pdf", "paper/Paper.pdf")
        for section in REQUIRED_SECTIONS:
            assert section in prompt, f"prompt missing verbatim heading: {section!r}"

    def test_includes_source_relpath_and_filename(self):
        prompt = ingest.build_draft_prompt("source text", "Paper.pdf", "paper/Paper.pdf")
        assert "paper/Paper.pdf" in prompt
        assert "Paper.pdf" in prompt

    def test_includes_source_text(self):
        prompt = ingest.build_draft_prompt("UNIQUE_SOURCE_MARKER", "Paper.pdf", "paper/Paper.pdf")
        assert "UNIQUE_SOURCE_MARKER" in prompt


# ---------------------------------------------------------------------------
# draft_article / postprocess_draft (the core drafting+assembly logic)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDraftArticle:
    def test_drafted_article_passes_schema_validation(self):
        llm = SimpleNamespace(invoke=lambda prompt: _fake_response(_DRAFTED_ARTICLE))
        article_id, text = ingest.draft_article(llm, "source text", "Example.pdf", "paper/Example.pdf")
        result = validate_article(text)
        assert result.ok, result.errors
        assert article_id == "example-signal"

    def test_forces_correct_source_file_over_llm_output(self):
        """The model drafted 'paper/WRONG_PATH.pdf'; the real source path must win."""
        llm = SimpleNamespace(invoke=lambda prompt: _fake_response(_DRAFTED_ARTICLE))
        _, text = ingest.draft_article(llm, "source text", "Example.pdf", "paper/Example.pdf")
        frontmatter, _ = ingest._split_frontmatter(text)
        assert frontmatter["source"]["file"] == "paper/Example.pdf"

    def test_slugifies_non_kebab_case_id_from_llm(self):
        llm = SimpleNamespace(invoke=lambda prompt: _fake_response(_DRAFTED_ARTICLE))
        article_id, text = ingest.draft_article(llm, "source text", "Example.pdf", "paper/Example.pdf")
        assert article_id == "example-signal"
        frontmatter, _ = ingest._split_frontmatter(text)
        assert frontmatter["id"] == "example-signal"

    def test_forced_id_overrides_llm_proposed_id(self):
        llm = SimpleNamespace(invoke=lambda prompt: _fake_response(_DRAFTED_ARTICLE))
        article_id, text = ingest.draft_article(
            llm, "source text", "Example.pdf", "paper/Example.pdf", forced_id="totally-different-id",
        )
        assert article_id == "totally-different-id"
        frontmatter, _ = ingest._split_frontmatter(text)
        assert frontmatter["id"] == "totally-different-id"

    def test_strips_wrapping_code_fence(self):
        fenced = f"```markdown\n{_DRAFTED_ARTICLE}\n```"
        llm = SimpleNamespace(invoke=lambda prompt: _fake_response(fenced))
        _, text = ingest.draft_article(llm, "source text", "Example.pdf", "paper/Example.pdf")
        assert validate_article(text).ok
        assert not text.startswith("```")

    def test_response_without_content_attribute_falls_back_to_str(self):
        llm = SimpleNamespace(invoke=lambda prompt: _DRAFTED_ARTICLE)
        article_id, text = ingest.draft_article(llm, "source text", "Example.pdf", "paper/Example.pdf")
        assert article_id == "example-signal"
        assert validate_article(text).ok

    def test_missing_frontmatter_raises(self):
        llm = SimpleNamespace(invoke=lambda prompt: _fake_response("## Summary\nNo frontmatter here.\n"))
        with pytest.raises(ValueError):
            ingest.draft_article(llm, "source text", "Example.pdf", "paper/Example.pdf")

    def test_drafted_article_missing_a_section_fails_validation(self):
        """The pipeline's own validation step (not draft_article itself) is what
        catches this -- draft_article just assembles what the model produced."""
        broken = _DRAFTED_ARTICLE.replace("## Caveats\n\nLook-ahead bias and data requirements apply.\n", "")
        llm = SimpleNamespace(invoke=lambda prompt: _fake_response(broken))
        _, text = ingest.draft_article(llm, "source text", "Example.pdf", "paper/Example.pdf")
        result = validate_article(text)
        assert not result.ok
        assert any("Caveats" in err for err in result.errors)


# ---------------------------------------------------------------------------
# find_existing_article_for_source
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFindExistingArticleForSource:
    def test_finds_article_whose_source_file_matches(self, tmp_path):
        wiki_dir = tmp_path / "wiki"
        wiki_dir.mkdir()
        article_text = _DRAFTED_ARTICLE.replace("file: paper/WRONG_PATH.pdf", "file: paper/Real.pdf")
        (wiki_dir / "example-signal.md").write_text(article_text, encoding="utf-8")

        found = ingest.find_existing_article_for_source(wiki_dir, "paper/Real.pdf")
        assert found == wiki_dir / "example-signal.md"

    def test_returns_none_when_no_match(self, tmp_path):
        wiki_dir = tmp_path / "wiki"
        wiki_dir.mkdir()
        (wiki_dir / "example-signal.md").write_text(_DRAFTED_ARTICLE, encoding="utf-8")

        assert ingest.find_existing_article_for_source(wiki_dir, "paper/Other.pdf") is None

    def test_ignores_template_and_readme(self, tmp_path):
        wiki_dir = tmp_path / "wiki"
        wiki_dir.mkdir()
        (wiki_dir / "_TEMPLATE.md").write_text("not real frontmatter", encoding="utf-8")
        (wiki_dir / "README.md").write_text("# readme", encoding="utf-8")

        assert ingest.find_existing_article_for_source(wiki_dir, "paper/Anything.pdf") is None

    def test_missing_wiki_dir_returns_none(self, tmp_path):
        assert ingest.find_existing_article_for_source(tmp_path / "does-not-exist", "paper/X.pdf") is None

    def test_malformed_article_is_skipped_not_raised(self, tmp_path):
        wiki_dir = tmp_path / "wiki"
        wiki_dir.mkdir()
        (wiki_dir / "broken.md").write_text("no frontmatter fence at all", encoding="utf-8")

        assert ingest.find_existing_article_for_source(wiki_dir, "paper/X.pdf") is None


# ---------------------------------------------------------------------------
# ingest_pdf (orchestration: skip / create / force / invalid)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestIngestPdf:
    def test_creates_article_and_writes_valid_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ingest, "extract_pdf_text", lambda pdf_path, **kw: "extracted source text")
        wiki_dir = tmp_path / "wiki"
        pdf_path = tmp_path / "paper" / "Example.pdf"
        pdf_path.parent.mkdir(parents=True)
        pdf_path.write_bytes(b"%PDF-fake")

        llm = SimpleNamespace(invoke=lambda prompt: _fake_response(_DRAFTED_ARTICLE))
        outcome = ingest.ingest_pdf(pdf_path, llm, wiki_dir, "paper")

        assert outcome.status == "created"
        assert outcome.article_id == "example-signal"
        assert outcome.article_path == wiki_dir / "example-signal.md"
        assert outcome.article_path.exists()
        result = validate_article_file(outcome.article_path)
        assert result.ok, result.errors

    def test_skips_when_existing_article_covers_source(self, tmp_path, monkeypatch):
        wiki_dir = tmp_path / "wiki"
        wiki_dir.mkdir()
        pdf_path = tmp_path / "paper" / "Example.pdf"
        pdf_path.parent.mkdir(parents=True)
        pdf_path.write_bytes(b"%PDF-fake")

        existing_text = _DRAFTED_ARTICLE.replace("file: paper/WRONG_PATH.pdf", "file: paper/Example.pdf")
        (wiki_dir / "already-covered.md").write_text(existing_text, encoding="utf-8")

        def _boom(*args, **kwargs):
            raise AssertionError("extract_pdf_text must not be called when skipping")

        monkeypatch.setattr(ingest, "extract_pdf_text", _boom)

        def _llm_boom(prompt):
            raise AssertionError("LLM must not be called when skipping")

        llm = SimpleNamespace(invoke=_llm_boom)
        outcome = ingest.ingest_pdf(pdf_path, llm, wiki_dir, "paper")

        assert outcome.status == "skipped"
        assert outcome.article_path == wiki_dir / "already-covered.md"
        assert "already-covered.md" in outcome.reason

    def test_force_regenerates_in_place_keeping_same_id(self, tmp_path, monkeypatch):
        wiki_dir = tmp_path / "wiki"
        wiki_dir.mkdir()
        pdf_path = tmp_path / "paper" / "Example.pdf"
        pdf_path.parent.mkdir(parents=True)
        pdf_path.write_bytes(b"%PDF-fake")

        existing_text = _DRAFTED_ARTICLE.replace("file: paper/WRONG_PATH.pdf", "file: paper/Example.pdf")
        existing_path = wiki_dir / "already-covered.md"
        existing_path.write_text(existing_text, encoding="utf-8")

        monkeypatch.setattr(ingest, "extract_pdf_text", lambda pdf_path, **kw: "new extracted text")
        llm = SimpleNamespace(invoke=lambda prompt: _fake_response(_DRAFTED_ARTICLE))
        outcome = ingest.ingest_pdf(pdf_path, llm, wiki_dir, "paper", force=True)

        assert outcome.status == "created"
        assert outcome.article_path == existing_path
        assert outcome.article_id == "already-covered"  # kept, not the LLM's proposed id
        result = validate_article_file(existing_path)
        assert result.ok, result.errors

    def test_reports_invalid_without_writing_when_draft_fails_schema(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ingest, "extract_pdf_text", lambda pdf_path, **kw: "extracted source text")
        wiki_dir = tmp_path / "wiki"
        pdf_path = tmp_path / "paper" / "Example.pdf"
        pdf_path.parent.mkdir(parents=True)
        pdf_path.write_bytes(b"%PDF-fake")

        broken = _DRAFTED_ARTICLE.replace("## Caveats\n\nLook-ahead bias and data requirements apply.\n", "")
        llm = SimpleNamespace(invoke=lambda prompt: _fake_response(broken))
        outcome = ingest.ingest_pdf(pdf_path, llm, wiki_dir, "paper")

        assert outcome.status == "invalid"
        assert any("Caveats" in err for err in outcome.errors)
        assert not wiki_dir.exists() or not any(wiki_dir.glob("*.md"))

    def test_id_collision_with_unrelated_article_is_skipped(self, tmp_path, monkeypatch):
        wiki_dir = tmp_path / "wiki"
        wiki_dir.mkdir()
        # An unrelated article that happens to already occupy the id the LLM
        # will propose for our new PDF.
        unrelated = _DRAFTED_ARTICLE.replace("file: paper/WRONG_PATH.pdf", "file: paper/SomeOtherPaper.pdf")
        (wiki_dir / "example-signal.md").write_text(unrelated, encoding="utf-8")

        monkeypatch.setattr(ingest, "extract_pdf_text", lambda pdf_path, **kw: "extracted source text")
        pdf_path = tmp_path / "paper" / "New.pdf"
        pdf_path.parent.mkdir(parents=True)
        pdf_path.write_bytes(b"%PDF-fake")

        llm = SimpleNamespace(invoke=lambda prompt: _fake_response(_DRAFTED_ARTICLE))
        outcome = ingest.ingest_pdf(pdf_path, llm, wiki_dir, "paper")

        assert outcome.status == "skipped"
        assert outcome.reason == "id collision with existing article"

    def test_skips_pdf_with_no_extractable_text(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ingest, "extract_pdf_text", lambda pdf_path, **kw: "   ")
        wiki_dir = tmp_path / "wiki"
        pdf_path = tmp_path / "paper" / "Empty.pdf"
        pdf_path.parent.mkdir(parents=True)
        pdf_path.write_bytes(b"%PDF-fake")

        def _llm_boom(prompt):
            raise AssertionError("LLM must not be called for a PDF with no extractable text")

        outcome = ingest.ingest_pdf(pdf_path, SimpleNamespace(invoke=_llm_boom), wiki_dir, "paper")
        assert outcome.status == "skipped"
        assert outcome.reason == "no extractable text"


# ---------------------------------------------------------------------------
# ingest_directory / run_ingest
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestIngestDirectoryAndRunIngest:
    def test_ingest_directory_processes_all_pdfs_sorted(self, tmp_path, monkeypatch):
        ingest_dir = tmp_path / "paper"
        ingest_dir.mkdir()
        (ingest_dir / "b.pdf").write_bytes(b"%PDF-fake")
        (ingest_dir / "a.pdf").write_bytes(b"%PDF-fake")
        (ingest_dir / "not_a_pdf.txt").write_text("ignore me")
        wiki_dir = tmp_path / "wiki"

        monkeypatch.setattr(ingest, "extract_pdf_text", lambda pdf_path, **kw: "extracted source text")
        llm = SimpleNamespace(invoke=lambda prompt: _fake_response(_DRAFTED_ARTICLE))

        # Second draft would collide on id with the first; force a distinct id
        # per call so both are "created" for this test's purposes.
        calls = {"n": 0}

        def _invoke(prompt):
            calls["n"] += 1
            text = _DRAFTED_ARTICLE.replace("id: Example Signal!!", f"id: Example Signal {calls['n']}")
            return _fake_response(text)

        llm = SimpleNamespace(invoke=_invoke)

        outcomes = ingest.ingest_directory(ingest_dir, wiki_dir, llm)

        assert [o.pdf_path.name for o in outcomes] == ["a.pdf", "b.pdf"]
        assert all(o.status == "created" for o in outcomes)

    def test_run_ingest_defaults_to_config_dirs(self, tmp_path, monkeypatch):
        ingest_dir = tmp_path / "paper"
        ingest_dir.mkdir()
        (ingest_dir / "only.pdf").write_bytes(b"%PDF-fake")
        wiki_dir = tmp_path / "wiki"

        monkeypatch.setattr(ingest, "extract_pdf_text", lambda pdf_path, **kw: "extracted source text")
        llm = SimpleNamespace(invoke=lambda prompt: _fake_response(_DRAFTED_ARTICLE))

        config = {"knowledge_ingest_dir": str(ingest_dir), "knowledge_base_dir": str(wiki_dir)}
        outcomes = ingest.run_ingest(llm, config=config)

        assert len(outcomes) == 1
        assert outcomes[0].status == "created"
        assert outcomes[0].article_path.parent == wiki_dir

    def test_run_ingest_missing_ingest_dir_returns_empty(self, tmp_path):
        config = {
            "knowledge_ingest_dir": str(tmp_path / "does-not-exist"),
            "knowledge_base_dir": str(tmp_path / "wiki"),
        }
        outcomes = ingest.run_ingest(SimpleNamespace(invoke=lambda p: _fake_response("")), config=config)
        assert outcomes == []


# ---------------------------------------------------------------------------
# build_quick_think_llm (factory wiring; uses the mock_llm_client fixture
# pattern from tests/conftest.py)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildQuickThinkLlm:
    def test_uses_quick_think_model_and_returns_get_llm(self, mock_llm_client):
        config = {"llm_provider": "openai", "quick_think_llm": "gpt-test", "backend_url": None}
        llm = ingest.build_quick_think_llm(config)

        assert llm is mock_llm_client.get_llm.return_value
        mock_llm_client.get_llm.assert_called_once()

    def test_defaults_to_get_config_when_no_config_passed(self, mock_llm_client):
        llm = ingest.build_quick_think_llm()
        assert llm is mock_llm_client.get_llm.return_value


# ---------------------------------------------------------------------------
# Sanity: required frontmatter keys are all present in the fixture draft
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_fixture_drafted_article_has_all_required_frontmatter_keys():
    frontmatter, _ = ingest._split_frontmatter(_DRAFTED_ARTICLE)
    for key in REQUIRED_FRONTMATTER_KEYS:
        assert key in frontmatter
