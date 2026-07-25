"""PDF -> LLM-wiki article ingestion pipeline (issue #102).

Turns user-provided PDFs under ``knowledge_ingest_dir`` (config key, default
``"paper"``) into schema-conformant articles under ``knowledge_base_dir``
(config key, default ``"knowledge/wiki"``). The article schema itself —
required frontmatter keys, required body sections, exact heading text
(including the em dash in "## Signal — what it is") — is defined and
validated by ``tradingagents.knowledge.wiki_schema`` (issue #101); this
module only drafts content that conforms to it and writes it to disk.

Pipeline, per PDF:

1. Skip if an existing article's ``source.file`` frontmatter already points
   at this PDF (idempotent, human-in-the-loop review — see module docstring
   note below), unless ``force=True``.
2. Extract text via ``pypdf`` (:func:`extract_pdf_text`).
3. Ask the quick-thinking LLM to draft a full article matching the schema
   (:func:`draft_article`), reproducing the required section headings
   verbatim (copied straight from ``wiki_schema.REQUIRED_SECTIONS`` so the
   em dash is byte-for-byte correct).
4. Force-correct the frontmatter ``id`` (kebab-case slug) and ``source.file``
   (always the real, repo-relative path to the PDF actually being read —
   never trust the LLM's own guess here, per the #101 design review note
   that ``source.file`` is not path-checked by the validator).
5. Validate the result with ``wiki_schema.validate_article`` before writing
   anything to disk.

Human-in-the-loop: this module never overwrites an existing article unless
the caller explicitly passes ``force=True`` — generated articles are meant
to be reviewed (and possibly hand-edited) before being committed, and a
silent re-run must not clobber that review work.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from tradingagents.dataflows.config import get_config
from tradingagents.knowledge.wiki_schema import REQUIRED_SECTIONS, validate_article

logger = logging.getLogger(__name__)

# Filenames under the wiki directory that are never treated as articles.
_NON_ARTICLE_FILENAMES = {"_TEMPLATE.md", "README.md"}

# Cap on extracted source text fed into the drafting prompt (~25k tokens at a
# ~4 chars/token rule of thumb). The seed corpus (#100) ranges from ~54k to
# ~178k extracted chars; this keeps prompts bounded for the quick-think model
# while still covering most or all of the shorter papers in full. Papers
# longer than this are truncated (front matter of an academic paper --
# abstract, model, headline results -- is generally front-loaded), which is
# an acceptable tradeoff given generated articles are human-reviewed before
# commit, not auto-trusted.
_MAX_SOURCE_CHARS = 100_000

_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)
_FENCE_RE = re.compile(r"\A```(?:markdown|md|yaml)?\s*\n(.*?)\n```\s*\Z", re.DOTALL)
_SLUG_INVALID_RE = re.compile(r"[^a-z0-9]+")


@dataclass
class IngestOutcome:
    """Result of attempting to ingest one source PDF."""

    pdf_path: Path
    status: str  # "created" | "skipped" | "invalid"
    article_path: Path | None = None
    article_id: str | None = None
    reason: str | None = None
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# LLM client wiring
# ---------------------------------------------------------------------------


def build_quick_think_llm(config: dict | None = None) -> Any:
    """Build the quick-thinking LLM used to draft articles.

    Goes through the ``tradingagents.llm_clients.factory`` module attribute
    (not a top-level ``from tradingagents.llm_clients import
    create_llm_client``) so that ``tests/conftest.py``'s ``mock_llm_client``
    fixture -- which patches
    ``tradingagents.llm_clients.factory.create_llm_client`` -- takes effect
    when this function is called from a test, without needing a second patch
    target.
    """
    from tradingagents.llm_clients import factory as llm_factory

    cfg = config if config is not None else get_config()
    client = llm_factory.create_llm_client(
        provider=cfg.get("llm_provider"),
        model=cfg.get("quick_think_llm"),
        base_url=cfg.get("backend_url"),
    )
    return client.get_llm()


# ---------------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------------


def extract_pdf_text(pdf_path: Path, max_chars: int | None = _MAX_SOURCE_CHARS) -> str:
    """Extract plain text from a PDF via ``pypdf``.

    Per-page extraction failures are logged and skipped rather than aborting
    the whole document -- a handful of unreadable pages (e.g. a scanned
    figure) shouldn't block drafting an article from the rest of the paper.
    """
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    pages: list[str] = []
    for i, page in enumerate(reader.pages):
        try:
            pages.append(page.extract_text() or "")
        except Exception as exc:  # pragma: no cover - defensive, pypdf-internal
            logger.warning("Failed to extract text from %s page %d: %s", pdf_path.name, i, exc)

    text = "\n".join(pages).strip()
    if max_chars and len(text) > max_chars:
        logger.info(
            "Truncating extracted text for %s from %d to %d chars",
            pdf_path.name, len(text), max_chars,
        )
        text = text[:max_chars]
    return text


# ---------------------------------------------------------------------------
# Drafting / assembly (pure-ish: no file I/O beyond the LLM call itself)
# ---------------------------------------------------------------------------


def slugify(text: str) -> str:
    """Normalize arbitrary text into a kebab-case slug (lowercase, [a-z0-9-])."""
    slug = _SLUG_INVALID_RE.sub("-", text.strip().lower()).strip("-")
    slug = re.sub(r"-{2,}", "-", slug)
    return slug or "article"


def build_draft_prompt(source_text: str, pdf_filename: str, source_relpath: str) -> str:
    """Build the article-drafting prompt for the quick-thinking LLM.

    Reproduces the required section headings verbatim from
    ``wiki_schema.REQUIRED_SECTIONS`` (not hand-typed) so the em dash in
    "## Signal — what it is" is guaranteed byte-identical to what the
    validator's regex matches (see #101 design review / #102 caveats).
    """
    sections_block = "\n".join(REQUIRED_SECTIONS)
    return f"""You are drafting one article for the TradingAgents LLM-wiki, a curated \
knowledge base of trading-strategy articles consulted by autonomous trading agents \
(portfolio manager, swing trader).

Source document: {pdf_filename}

Extracted source text (from the PDF; may include OCR/layout noise -- use your \
judgement to reconstruct the paper's actual content and ignore artifacts like \
running headers, page numbers, or broken line wraps):
---
{source_text}
---

Draft ONE markdown article summarizing this paper's trading-relevant strategy or \
signal, following this EXACT schema. Output ONLY the raw markdown for the article \
-- no commentary before or after, no code fences.

Required format (YAML frontmatter, then body sections):

---
id: kebab-case-unique-id
title: Human-Readable Title
tags: [tag-one, tag-two]
signals: [signal_one, signal_two]
asset_classes: [equity]
horizon: [swing, position]
source: {{authors: "Author One, Author Two", title: "Paper Title", year: 2000, file: {source_relpath}}}
---
{sections_block}

Rules:
- Reproduce the body section headings shown above EXACTLY, character for character
  -- including the em dash character (—, not a hyphen) in "## Signal — what it is".
- `id`: kebab-case (lowercase letters, digits, hyphens only), short and descriptive
  of the strategy/signal itself (e.g. "piotroski-f-score"), not a restatement of the
  filename.
- `tags`, `signals`, `asset_classes`, `horizon` must be YAML lists (square-bracket or
  block form).
- `source.authors`, `source.title`, `source.year` must reflect the actual paper (read
  them from the source text). Set `source.file` to exactly `{source_relpath}` --
  do not change it.
- Every body section must contain real, substantive content grounded in the source
  text -- no placeholders, no "TBD", no boilerplate.
- "## How to compute" must give a precise, implementable formula or procedure (cf.
  this repo's convention of precomputing numeric signals in Python rather than
  asking an LLM to do arithmetic), not just a prose description.
- "## Empirical evidence" must cite the paper's actual sample, period, and effect
  size where the source text provides them.
"""


def _strip_code_fence(text: str) -> str:
    """Strip a single wrapping ```...``` fence, if the model added one despite
    being asked not to."""
    text = text.strip()
    match = _FENCE_RE.match(text)
    return match.group(1).strip() if match else text


def _split_frontmatter(text: str) -> tuple[dict, str]:
    """Split ``text`` into (frontmatter dict, body). Reparses independently via
    ``yaml.safe_load`` rather than reusing any part of ``wiki_schema``'s
    internals, which are intentionally private (see #102 caveats)."""
    match = _FRONTMATTER_RE.match(text)
    if not match:
        raise ValueError("drafted article text has no YAML frontmatter block (--- ... ---)")
    frontmatter = yaml.safe_load(match.group(1))
    if not isinstance(frontmatter, dict):
        frontmatter = {}
    body = text[match.end():]
    return frontmatter, body


def _render_article(frontmatter: dict, body: str) -> str:
    fm_yaml = yaml.safe_dump(frontmatter, sort_keys=False, allow_unicode=True).strip()
    return f"---\n{fm_yaml}\n---\n{body.strip(chr(10))}\n"


def postprocess_draft(
    raw_text: str,
    source_relpath: str,
    fallback_id: str,
    forced_id: str | None = None,
) -> str:
    """Force-correct the parts of the drafted frontmatter this module owns.

    - ``id``: normalized to a kebab-case slug. If ``forced_id`` is given
      (regenerating an article that already exists under a known id), that id
      wins outright; otherwise the LLM's proposed id (falling back to its
      title, then to a slug of the source filename) is slugified.
    - ``source.file``: always overwritten to ``source_relpath`` -- the actual
      repo-relative path of the PDF being processed -- never trusted from the
      LLM's output, since a drafted value here isn't otherwise checked against
      a real path (see #102 caveats).
    """
    frontmatter, body = _split_frontmatter(raw_text)

    if forced_id:
        article_id = slugify(forced_id)
    else:
        candidate = frontmatter.get("id") or frontmatter.get("title") or fallback_id
        article_id = slugify(str(candidate)) or fallback_id
    frontmatter["id"] = article_id

    source = frontmatter.get("source")
    if not isinstance(source, dict):
        source = {}
    source["file"] = source_relpath
    frontmatter["source"] = source

    return _render_article(frontmatter, body)


def draft_article(
    llm: Any,
    source_text: str,
    pdf_filename: str,
    source_relpath: str,
    forced_id: str | None = None,
) -> tuple[str, str]:
    """Draft one article from extracted PDF text via ``llm``.

    Returns ``(article_id, article_markdown_text)``. Does not validate or
    write anything -- callers (:func:`ingest_pdf`) own that.
    """
    prompt = build_draft_prompt(source_text, pdf_filename, source_relpath)
    response = llm.invoke(prompt)
    content = response.content if hasattr(response, "content") else str(response)
    raw = _strip_code_fence(content)

    fallback_id = slugify(Path(pdf_filename).stem)
    article_text = postprocess_draft(raw, source_relpath, fallback_id, forced_id=forced_id)
    frontmatter, _ = _split_frontmatter(article_text)
    return frontmatter["id"], article_text


# ---------------------------------------------------------------------------
# Orchestration (file I/O, skip/force semantics)
# ---------------------------------------------------------------------------


def find_existing_article_for_source(wiki_dir: Path, source_relpath: str) -> Path | None:
    """Return the path of the existing article (if any) whose ``source.file``
    frontmatter already points at ``source_relpath``.

    This -- not a filename convention -- is the idempotency check: a source
    PDF is "already covered" if *some* article in the wiki cites it as its
    source, regardless of what id/filename a human (or a prior ingestion run)
    gave that article. This is what lets a hand-written article like
    ``knowledge/wiki/piotroski-f-score.md`` (whose id has nothing to do with
    its source filename) correctly suppress re-drafting the same PDF.
    """
    if not wiki_dir.exists():
        return None
    for md_path in sorted(wiki_dir.glob("*.md")):
        if md_path.name in _NON_ARTICLE_FILENAMES:
            continue
        try:
            frontmatter, _ = _split_frontmatter(md_path.read_text(encoding="utf-8"))
        except (ValueError, yaml.YAMLError):
            continue
        source = frontmatter.get("source")
        if isinstance(source, dict) and source.get("file") == source_relpath:
            return md_path
    return None


def ingest_pdf(
    pdf_path: Path,
    llm: Any,
    wiki_dir: Path,
    ingest_dir_name: str,
    force: bool = False,
) -> IngestOutcome:
    """Ingest one PDF: skip/draft/validate/write. See module docstring for the
    full pipeline description."""
    source_relpath = f"{ingest_dir_name.rstrip('/')}/{pdf_path.name}"
    existing = find_existing_article_for_source(wiki_dir, source_relpath)

    if existing is not None and not force:
        logger.info("Skipping %s: already covered by %s", pdf_path.name, existing.name)
        return IngestOutcome(
            pdf_path, "skipped", article_path=existing,
            reason=f"already covered by {existing.name}",
        )

    source_text = extract_pdf_text(pdf_path)
    if not source_text.strip():
        logger.warning("No extractable text in %s; skipping", pdf_path.name)
        return IngestOutcome(pdf_path, "skipped", reason="no extractable text")

    forced_id = existing.stem if existing is not None else None
    article_id, article_text = draft_article(
        llm, source_text, pdf_path.name, source_relpath, forced_id=forced_id,
    )

    target_path = existing if existing is not None else wiki_dir / f"{article_id}.md"
    if existing is None and target_path.exists() and not force:
        logger.warning(
            "Drafted id %r for %s collides with an existing unrelated article %s; skipping",
            article_id, pdf_path.name, target_path.name,
        )
        return IngestOutcome(
            pdf_path, "skipped", article_path=target_path, article_id=article_id,
            reason="id collision with existing article",
        )

    result = validate_article(article_text)
    if not result.ok:
        logger.error(
            "Drafted article for %s failed schema validation: %s", pdf_path.name, result.errors,
        )
        return IngestOutcome(pdf_path, "invalid", article_id=article_id, errors=result.errors)

    wiki_dir.mkdir(parents=True, exist_ok=True)
    target_path.write_text(article_text, encoding="utf-8")
    logger.info(
        "%s %s -> %s", "Regenerated" if existing is not None else "Created",
        pdf_path.name, target_path.name,
    )
    return IngestOutcome(pdf_path, "created", article_path=target_path, article_id=article_id)


def ingest_directory(
    ingest_dir: Path,
    wiki_dir: Path,
    llm: Any,
    force: bool = False,
) -> list[IngestOutcome]:
    """Ingest every ``*.pdf`` directly under ``ingest_dir``, in sorted order."""
    ingest_dir_name = ingest_dir.name
    outcomes = []
    for pdf_path in sorted(ingest_dir.glob("*.pdf")):
        outcomes.append(ingest_pdf(pdf_path, llm, wiki_dir, ingest_dir_name, force=force))
    return outcomes


def run_ingest(
    llm: Any,
    ingest_dir: Path | str | None = None,
    wiki_dir: Path | str | None = None,
    force: bool = False,
    config: dict | None = None,
) -> list[IngestOutcome]:
    """Top-level entry point: resolve directories from config (unless
    overridden) and ingest every PDF found."""
    cfg = config if config is not None else get_config()
    ingest_dir = Path(ingest_dir) if ingest_dir is not None else Path(cfg.get("knowledge_ingest_dir", "paper"))
    wiki_dir = Path(wiki_dir) if wiki_dir is not None else Path(cfg.get("knowledge_base_dir", "knowledge/wiki"))

    if not ingest_dir.exists():
        logger.warning("Ingest directory %s does not exist; nothing to do", ingest_dir)
        return []

    return ingest_directory(ingest_dir, wiki_dir, llm, force=force)
