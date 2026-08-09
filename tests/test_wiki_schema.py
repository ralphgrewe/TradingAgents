"""Tests for tradingagents.knowledge.wiki_schema — LLM-wiki article schema validation."""

from pathlib import Path

import pytest

from tradingagents.knowledge.wiki_schema import (
    REQUIRED_FRONTMATTER_KEYS,
    REQUIRED_SECTIONS,
    ValidationResult,
    validate_article,
    validate_article_file,
)

VALID_ARTICLE = """---
id: example-signal
title: Example Signal
tags: [example, test]
signals: [example_signal]
asset_classes: [equity]
horizon: [swing]
source: {authors: "A. Author", title: "Example Paper", year: 2000, file: paper/Example.pdf}
---
## Summary

An example summary.

## Signal — what it is

Definition of the signal.

## How to compute

The formula.

## Empirical evidence

What the paper found.

## When to apply / regime

When it works.

## Caveats

What to watch out for.
"""


def _drop_frontmatter_key(text: str, key: str) -> str:
    lines = [line for line in text.splitlines() if not line.startswith(f"{key}:")]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Valid article
# ---------------------------------------------------------------------------


def test_valid_article_passes():
    result = validate_article(VALID_ARTICLE)
    assert result.ok is True
    assert result.errors == []
    assert bool(result) is True


def test_real_example_article_passes():
    """The committed knowledge/wiki/piotroski-f-score.md example must validate."""
    repo_root = Path(__file__).resolve().parent.parent
    article_path = repo_root / "knowledge" / "wiki" / "piotroski-f-score.md"
    result = validate_article_file(article_path)
    assert result.ok, result.errors


def test_template_article_is_not_required_to_pass():
    """_TEMPLATE.md has placeholder frontmatter/body and is not itself a valid article."""
    repo_root = Path(__file__).resolve().parent.parent
    template_path = repo_root / "knowledge" / "wiki" / "_TEMPLATE.md"
    # The template exists and has the right sections/keys (it's the schema's own
    # worked skeleton) -- this just documents that we don't assert anything
    # stronger about it here; piotroski-f-score.md is the real fixture.
    assert template_path.exists()


# ---------------------------------------------------------------------------
# Missing frontmatter block entirely
# ---------------------------------------------------------------------------


def test_missing_frontmatter_block():
    text = VALID_ARTICLE.split("---\n", 2)[-1]  # strip the frontmatter block
    result = validate_article(text)
    assert result.ok is False
    assert any("frontmatter" in err for err in result.errors)


def test_invalid_yaml_frontmatter():
    text = "---\nid: [unclosed\n---\n## Summary\n"
    result = validate_article(text)
    assert result.ok is False
    assert any("not valid YAML" in err for err in result.errors)


def test_frontmatter_not_a_mapping():
    text = "---\n- just\n- a\n- list\n---\n## Summary\n"
    result = validate_article(text)
    assert result.ok is False
    assert any("mapping" in err for err in result.errors)


# ---------------------------------------------------------------------------
# Missing required frontmatter keys (parametrized over all seven)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", REQUIRED_FRONTMATTER_KEYS)
def test_missing_required_frontmatter_key(key):
    text = _drop_frontmatter_key(VALID_ARTICLE, key)
    result = validate_article(text)
    assert result.ok is False
    assert any(key in err for err in result.errors)


def test_id_not_kebab_case():
    text = VALID_ARTICLE.replace("id: example-signal", "id: Example_Signal")
    result = validate_article(text)
    assert result.ok is False
    assert any("kebab-case" in err for err in result.errors)


def test_list_field_not_a_list():
    text = VALID_ARTICLE.replace("tags: [example, test]", "tags: example")
    result = validate_article(text)
    assert result.ok is False
    assert any("'tags' must be a list" in err for err in result.errors)


def test_source_not_a_mapping():
    text = VALID_ARTICLE.replace(
        'source: {authors: "A. Author", title: "Example Paper", year: 2000, file: paper/Example.pdf}',
        "source: paper/Example.pdf",
    )
    result = validate_article(text)
    assert result.ok is False
    assert any("'source' must be a mapping" in err for err in result.errors)


def test_source_missing_nested_key():
    text = VALID_ARTICLE.replace(
        'source: {authors: "A. Author", title: "Example Paper", year: 2000, file: paper/Example.pdf}',
        'source: {authors: "A. Author", title: "Example Paper", year: 2000}',
    )
    result = validate_article(text)
    assert result.ok is False
    assert any("source' key: 'file'" in err for err in result.errors)


# ---------------------------------------------------------------------------
# Missing required body sections (parametrized over all six)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("section", REQUIRED_SECTIONS)
def test_missing_required_section(section):
    text = VALID_ARTICLE.replace(f"{section}\n", "")
    result = validate_article(text)
    assert result.ok is False
    assert any(section in err for err in result.errors)


def test_multiple_missing_sections_all_reported():
    text = VALID_ARTICLE.replace("## Caveats\n", "").replace("## Summary\n", "")
    result = validate_article(text)
    assert result.ok is False
    assert sum("missing body section" in err for err in result.errors) == 2


# ---------------------------------------------------------------------------
# validate_article_file
# ---------------------------------------------------------------------------


def test_validate_article_file_reads_and_validates(tmp_path):
    article_path = tmp_path / "example-signal.md"
    article_path.write_text(VALID_ARTICLE, encoding="utf-8")
    result = validate_article_file(article_path)
    assert result.ok is True


def test_validate_article_file_accepts_str_path(tmp_path):
    article_path = tmp_path / "example-signal.md"
    article_path.write_text(VALID_ARTICLE, encoding="utf-8")
    result = validate_article_file(str(article_path))
    assert result.ok is True


def test_validation_result_bool():
    assert bool(ValidationResult(ok=True)) is True
    assert bool(ValidationResult(ok=False, errors=["x"])) is False
