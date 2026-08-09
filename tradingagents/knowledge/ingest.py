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
   note below), unless ``force=True``. The comparison is made on *normalized*
   paths (:func:`normalize_source_path`) so a hand-edited article whose
   ``source.file`` differs only cosmetically (``./paper/X.pdf``, a trailing
   slash, a different case, an absolute path inside the repo) still counts
   as covering the same source instead of silently spawning a duplicate.
2. Extract text via ``pypdf`` (:func:`extract_pdf_text`).
3. Ask the quick-thinking LLM to draft a full article matching the schema
   (:func:`draft_article`), reproducing the required section headings
   verbatim (copied straight from ``wiki_schema.REQUIRED_SECTIONS`` so the
   em dash is byte-for-byte correct).
4. Force-correct the frontmatter ``id`` (kebab-case slug) and ``source.file``
   (always the real, repo-relative path to the PDF actually being read —
   never trust the LLM's own guess here, per the #101 design review note
   that ``source.file`` is not path-checked by the validator). The
   repo-relative path is derived by :func:`compute_source_relpath` from the
   PDF's own location, so it stays correct for an ingest directory at any
   depth (``paper/``, ``data/papers/``, ...).
5. Validate the result with ``wiki_schema.validate_article`` before writing
   anything to disk.

Every step above runs inside :func:`ingest_pdf`, which never raises: a
corrupt PDF, a failed LLM call or a malformed drafted frontmatter is
reported as an ``"error"`` :class:`IngestOutcome` for that one file, so a
batch run keeps the outcomes it already collected and moves on to the next
PDF instead of aborting.

Frontmatter parsing is *not* re-implemented here: ``wiki_schema.parse_article``
(extracted in #103 for exactly this purpose) is the single place that knows
how an article's ``---`` fence + YAML block is split.

Human-in-the-loop: this module never overwrites an existing article unless
the caller explicitly passes ``force=True`` — generated articles are meant
to be reviewed (and possibly hand-edited) before being committed, and a
silent re-run must not clobber that review work.
"""

from __future__ import annotations

import contextlib
import logging
import os
import posixpath
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from tradingagents.dataflows.config import get_config
from tradingagents.knowledge.wiki_schema import (
    REQUIRED_SECTIONS,
    parse_article,
    validate_article,
)

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

_FENCE_RE = re.compile(r"\A```(?:markdown|md|yaml)?\s*\n(.*?)\n```\s*\Z", re.DOTALL)
_SLUG_INVALID_RE = re.compile(r"[^a-z0-9]+")

# Repo root, used to turn a PDF's location into the repo-relative path written
# to ``source.file`` (and to normalize absolute paths found in existing
# articles back to that same frame of reference). Derived from this file's
# location -- ``tradingagents/knowledge/ingest.py`` -- rather than from the
# working directory, so it is correct however the CLI is invoked.
_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class IngestOutcome:
    """Result of attempting to ingest one source PDF.

    ``status`` is one of:

    - ``"created"``  -- an article was drafted, validated and written.
    - ``"skipped"``  -- nothing to do (already covered, id collision, no text).
    - ``"invalid"``  -- a draft was produced but failed schema validation.
    - ``"error"``    -- processing this PDF raised (corrupt PDF, LLM failure,
      malformed drafted frontmatter). The message is in both ``reason`` and
      ``errors`` so it shows up in the CLI's per-file report.
    """

    pdf_path: Path
    status: str  # "created" | "skipped" | "invalid" | "error"
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
    frontmatter, body = parse_article(raw_text)

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
    frontmatter, _ = parse_article(article_text)
    return frontmatter["id"], article_text


# ---------------------------------------------------------------------------
# Source-path resolution / normalization
# ---------------------------------------------------------------------------


def compute_source_relpath(
    pdf_path: Path | str,
    ingest_dir: Path | str,
    repo_root: Path | str | None = None,
) -> str:
    """Return the path written to an article's ``source.file`` for ``pdf_path``.

    The contract (documented in ``knowledge/wiki/README.md`` and #101's design
    review) is a repo-root-relative POSIX path, e.g. ``paper/Foo.pdf``. It is
    derived from the PDF's *own* location rather than from the ingest
    directory's basename, so an ingest directory of any depth works:
    ``--ingest-dir data/papers`` yields ``data/papers/Foo.pdf``, not the
    ``papers/Foo.pdf`` a basename-only join would produce.

    Fallbacks, in order, for PDFs that are not under the repo root at all
    (an out-of-tree ``--ingest-dir``, or a test's ``tmp_path``):

    1. If the ingest directory was given as a *relative* path, reuse it as
       given (it is by construction relative to the caller's working
       directory) joined with the PDF's filename.
    2. Otherwise fall back to the PDF's absolute path -- unambiguous, and
       still matched by :func:`normalize_source_path` on a later run.
    """
    pdf_path = Path(pdf_path)
    ingest_dir = Path(ingest_dir)
    root = Path(repo_root).resolve() if repo_root is not None else _REPO_ROOT

    # ``resolve()`` (not ``abspath``) on both sides: ``_REPO_ROOT`` is itself
    # resolved, so a symlinked checkout or temp dir must be resolved too for
    # ``relative_to`` to see the two as related.
    absolute = pdf_path.resolve()
    try:
        return absolute.relative_to(root).as_posix()
    except ValueError:
        pass

    if not ingest_dir.is_absolute():
        joined = os.path.normpath(str(ingest_dir / pdf_path.name)).replace(os.sep, "/")
        return PurePosixPath(joined).as_posix()
    return absolute.as_posix()


def normalize_source_path(value: Any, repo_root: Path | str | None = None) -> str:
    """Normalize a ``source.file`` value for equality comparison.

    Two ``source.file`` strings denote the same source PDF if they normalize
    to the same string here. Normalization collapses the differences that a
    hand-edit or a differently-configured run can introduce without changing
    which file is meant: backslash separators, ``./`` prefixes and ``..``
    segments, duplicate/trailing slashes, an absolute path inside the repo
    (rewritten to its repo-relative form), and letter case.

    Case folding is deliberate leniency: the two failure modes are not
    symmetric. A false *match* only means an existing article is left alone
    for a human to look at, whereas a false *miss* silently writes a second
    article covering a PDF the wiki already documents -- the exact
    duplicate-article failure the idempotency check exists to prevent.

    Returns ``""`` for anything that isn't a usable path (``None``, a
    non-string, an empty string); callers must treat ``""`` as "no match".
    """
    if not isinstance(value, str):
        return ""
    text = value.strip().replace("\\", "/")
    if not text:
        return ""

    normalized = posixpath.normpath(text)
    if normalized.startswith("/"):
        root = Path(repo_root).resolve() if repo_root is not None else _REPO_ROOT
        # Out-of-repo absolute paths stay absolute; there is no repo-relative
        # form for them, and the absolute form still compares consistently.
        with contextlib.suppress(ValueError):
            normalized = PurePosixPath(normalized).relative_to(root.as_posix()).as_posix()
    normalized = normalized.rstrip("/")
    return normalized.casefold()


# ---------------------------------------------------------------------------
# Orchestration (file I/O, skip/force semantics)
# ---------------------------------------------------------------------------


def find_existing_article_for_source(
    wiki_dir: Path,
    source_relpath: str,
    repo_root: Path | str | None = None,
) -> Path | None:
    """Return the path of the existing article (if any) whose ``source.file``
    frontmatter already points at ``source_relpath``.

    This -- not a filename convention -- is the idempotency check: a source
    PDF is "already covered" if *some* article in the wiki cites it as its
    source, regardless of what id/filename a human (or a prior ingestion run)
    gave that article. This is what lets a hand-written article like
    ``knowledge/wiki/piotroski-f-score.md`` (whose id has nothing to do with
    its source filename) correctly suppress re-drafting the same PDF.

    Both sides of the comparison go through :func:`normalize_source_path`, so
    a cosmetically different but equivalent path (``./paper/X.pdf``, an
    in-repo absolute path, different case) still matches instead of producing
    a duplicate article. Frontmatter is parsed with
    ``wiki_schema.parse_article``; articles that fail to parse are skipped,
    not raised on -- a malformed neighbour must not block ingestion.
    """
    target = normalize_source_path(source_relpath, repo_root)
    if not target or not wiki_dir.exists():
        return None
    for md_path in sorted(wiki_dir.glob("*.md")):
        if md_path.name in _NON_ARTICLE_FILENAMES:
            continue
        try:
            frontmatter, _ = parse_article(md_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, yaml.YAMLError):
            continue
        source = frontmatter.get("source")
        if not isinstance(source, dict):
            continue
        if normalize_source_path(source.get("file"), repo_root) == target:
            return md_path
    return None


def ingest_pdf(
    pdf_path: Path,
    llm: Any,
    wiki_dir: Path,
    ingest_dir: Path | str,
    force: bool = False,
    repo_root: Path | str | None = None,
) -> IngestOutcome:
    """Ingest one PDF: skip/draft/validate/write. See module docstring for the
    full pipeline description.

    Never raises. Any failure while processing this one PDF -- an unreadable
    file, a failed LLM call, malformed drafted YAML -- is caught and returned
    as an ``"error"`` :class:`IngestOutcome`, so a batch caller keeps the
    outcomes it has already collected and can continue with the next file.
    """
    try:
        return _ingest_pdf_unguarded(pdf_path, llm, wiki_dir, ingest_dir, force, repo_root)
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        logger.error("Failed to ingest %s: %s", pdf_path.name, message, exc_info=True)
        return IngestOutcome(pdf_path, "error", reason=message, errors=[message])


def _ingest_pdf_unguarded(
    pdf_path: Path,
    llm: Any,
    wiki_dir: Path,
    ingest_dir: Path | str,
    force: bool = False,
    repo_root: Path | str | None = None,
) -> IngestOutcome:
    """The actual per-PDF pipeline. May raise; :func:`ingest_pdf` is the
    exception-isolating public entry point."""
    source_relpath = compute_source_relpath(pdf_path, ingest_dir, repo_root)
    existing = find_existing_article_for_source(wiki_dir, source_relpath, repo_root)

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
    repo_root: Path | str | None = None,
) -> list[IngestOutcome]:
    """Ingest every ``*.pdf`` directly under ``ingest_dir``, in sorted order.

    One failing PDF does not abort the batch: :func:`ingest_pdf` converts any
    exception into an ``"error"`` outcome for that file, and the loop carries
    on -- outcomes collected for earlier files are never discarded.
    """
    outcomes = []
    for pdf_path in sorted(ingest_dir.glob("*.pdf")):
        outcomes.append(
            ingest_pdf(pdf_path, llm, wiki_dir, ingest_dir, force=force, repo_root=repo_root),
        )
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
