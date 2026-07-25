"""Pure-function schema validation for ``knowledge/wiki/*.md`` articles.

Every LLM-wiki article is a markdown file: a YAML frontmatter block followed by
six required ``## ``-headed body sections. The schema itself is defined and
documented in ``docs/design/llm-wiki.md`` and ``knowledge/wiki/README.md``;
this module is only the machine-checkable version of that contract.

``validate_article`` is intentionally a pure function of the file's text ->
:class:`ValidationResult`, with no file I/O and no network access, so it can be
reused unmodified by:

- the (future) PDF->article ingestion pipeline (issue #102), to reject a
  drafted article before it is committed, and
- the (future) BM25 retrieval dataflow (issue #103), to skip or flag malformed
  articles at index-build time,

without either of those depending on the other, or on how the text was
produced (hand-written, LLM-drafted, read from disk, ...).

``validate_article_file`` is a thin convenience wrapper that does the one bit
of I/O (reading the file) and delegates everything else to
``validate_article``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# Required YAML frontmatter keys (see docs/design/llm-wiki.md "Article schema").
REQUIRED_FRONTMATTER_KEYS: tuple[str, ...] = (
    "id",
    "title",
    "tags",
    "signals",
    "asset_classes",
    "horizon",
    "source",
)

# Frontmatter keys whose values must be YAML lists.
REQUIRED_LIST_KEYS: tuple[str, ...] = ("tags", "signals", "asset_classes", "horizon")

# Required keys within the nested `source` mapping.
REQUIRED_SOURCE_KEYS: tuple[str, ...] = ("authors", "title", "year", "file")

# Required body sections, matched by exact heading text. Order is house style
# (see _TEMPLATE.md), not a validated constraint -- only presence is checked.
REQUIRED_SECTIONS: tuple[str, ...] = (
    "## Summary",
    "## Signal — what it is",
    "## How to compute",
    "## Empirical evidence",
    "## When to apply / regime",
    "## Caveats",
)

_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)
_KEBAB_CASE_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


@dataclass
class ValidationResult:
    """Outcome of validating one article's text against the wiki schema.

    ``bool(result)`` is equivalent to ``result.ok``, so callers can write
    ``if not validate_article(text): ...`` without reaching into ``.ok``.
    """

    ok: bool
    errors: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.ok


def validate_article(text: str) -> ValidationResult:
    """Validate one article's raw markdown text against the LLM-wiki schema.

    Checks (accumulating all failures rather than stopping at the first):

    - A YAML frontmatter block is present and parses to a mapping.
    - All of ``REQUIRED_FRONTMATTER_KEYS`` are present.
    - ``id`` is a string in kebab-case.
    - Each of ``REQUIRED_LIST_KEYS`` is a YAML list.
    - ``source`` is a mapping containing all of ``REQUIRED_SOURCE_KEYS``.
    - All of ``REQUIRED_SECTIONS`` appear as a heading line in the body.

    Args:
        text: the full raw text of an article file.

    Returns:
        A :class:`ValidationResult` with ``ok=True`` and no errors if the
        article conforms to the schema, otherwise ``ok=False`` and one
        human-readable message per problem found.
    """
    errors: list[str] = []

    match = _FRONTMATTER_RE.match(text)
    if not match:
        return ValidationResult(False, ["missing YAML frontmatter block (--- ... ---)"])

    try:
        frontmatter = yaml.safe_load(match.group(1))
    except yaml.YAMLError as exc:
        return ValidationResult(False, [f"frontmatter is not valid YAML: {exc}"])

    if frontmatter is None:
        frontmatter = {}
    if not isinstance(frontmatter, dict):
        return ValidationResult(False, ["frontmatter must be a YAML mapping"])

    for key in REQUIRED_FRONTMATTER_KEYS:
        if key not in frontmatter:
            errors.append(f"missing frontmatter key: {key!r}")

    if "id" in frontmatter:
        article_id = frontmatter["id"]
        if not isinstance(article_id, str) or not _KEBAB_CASE_RE.match(article_id):
            errors.append(f"frontmatter 'id' must be kebab-case, got {article_id!r}")

    for list_key in REQUIRED_LIST_KEYS:
        if list_key in frontmatter and not isinstance(frontmatter[list_key], list):
            errors.append(f"frontmatter {list_key!r} must be a list")

    if "source" in frontmatter:
        source = frontmatter["source"]
        if not isinstance(source, dict):
            errors.append("frontmatter 'source' must be a mapping")
        else:
            for source_key in REQUIRED_SOURCE_KEYS:
                if source_key not in source:
                    errors.append(f"missing frontmatter 'source' key: {source_key!r}")

    body = text[match.end() :]
    for section in REQUIRED_SECTIONS:
        if not re.search(rf"^{re.escape(section)}\s*$", body, re.MULTILINE):
            errors.append(f"missing body section: {section!r}")

    return ValidationResult(ok=not errors, errors=errors)


def validate_article_file(path: str | Path) -> ValidationResult:
    """Read ``path`` and validate its contents via :func:`validate_article`."""
    text = Path(path).read_text(encoding="utf-8")
    return validate_article(text)
