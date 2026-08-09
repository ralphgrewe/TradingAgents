"""BM25 keyword retrieval over ``knowledge/wiki/*.md`` articles.

Provides ``search_wiki``, the "bm25" vendor implementation for the
``data_vendors["knowledge_base"]`` category (see ``dataflows/interface.py``
and CLAUDE.md "Data vendors"). Deterministic, offline, keyword-only: no
vector store, no embedding provider. Routing through the standard vendor
seam means a future embedding-based backend is a config change (a new
"knowledge_base" vendor entry), not a rewrite of callers.

Design notes (see ``docs/design/llm-wiki.md`` "Consumption approach"):

- Articles are parsed with :func:`tradingagents.knowledge.wiki_schema.
  validate_article` / ``parse_article`` -- the same schema module the
  ingestion pipeline (#102) validates drafts against -- so a malformed
  article is skipped here with a warning instead of poisoning the index or
  raising.
- Retrieval is whole-article, not section-aware: the BM25 document for each
  article is its title + tags + full body text. The result dict's
  ``section`` key is reserved for a future section-level chunking pass (see
  #103's "Entry points" note); until that lands it is always ``None``.
- No third-party BM25 dependency is pinned: the wiki corpus is small (single
  digits to low tens of articles for the foreseeable future), so a plain
  O(n) pure-Python BM25 Okapi implementation keeps ranking fully
  reproducible without adding an install-time dependency for a handful of
  documents.
"""

from __future__ import annotations

import logging
import math
import re
from pathlib import Path

from tradingagents.knowledge.wiki_schema import parse_article, validate_article

from .config import get_config

logger = logging.getLogger(__name__)

# BM25 Okapi hyperparameters (standard defaults; see Robertson & Zaragoza 2009).
_BM25_K1 = 1.5
_BM25_B = 0.75

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase, alnum-run tokenizer. Deterministic, locale-independent."""
    return _TOKEN_RE.findall(text.lower())


class _Bm25Index:
    """Minimal, dependency-free BM25 (Okapi) scorer over a fixed corpus.

    Pure Python; fine for the wiki's expected size (tens of articles). Kept
    private to this module -- the public surface is :func:`search_wiki`.
    """

    def __init__(self, tokenized_docs: list[list[str]], k1: float = _BM25_K1, b: float = _BM25_B):
        self.k1 = k1
        self.b = b
        self.doc_len = [len(doc) for doc in tokenized_docs]
        self.avg_doc_len = (sum(self.doc_len) / len(self.doc_len)) if tokenized_docs else 0.0

        self.doc_term_freqs: list[dict[str, int]] = []
        doc_freq: dict[str, int] = {}
        for doc in tokenized_docs:
            freqs: dict[str, int] = {}
            for term in doc:
                freqs[term] = freqs.get(term, 0) + 1
            self.doc_term_freqs.append(freqs)
            for term in freqs:
                doc_freq[term] = doc_freq.get(term, 0) + 1

        n_docs = len(tokenized_docs)
        self.idf: dict[str, float] = {
            term: math.log((n_docs - freq + 0.5) / (freq + 0.5) + 1) for term, freq in doc_freq.items()
        }

    def scores(self, query_tokens: list[str]) -> list[float]:
        """Return one BM25 score per document, in corpus order."""
        results = []
        for i, freqs in enumerate(self.doc_term_freqs):
            doc_len = self.doc_len[i]
            score = 0.0
            for term in query_tokens:
                f = freqs.get(term)
                if not f:
                    continue
                idf = self.idf.get(term, 0.0)
                denom = f + self.k1 * (1 - self.b + self.b * doc_len / (self.avg_doc_len or 1.0))
                score += idf * (f * (self.k1 + 1)) / denom
            results.append(score)
        return results


def _load_articles(wiki_dir: Path) -> list[dict]:
    """Load, validate, and parse every article under ``wiki_dir``.

    Returns one dict per valid article with keys: id, title, tags, source,
    doc_tokens (tokenized title+tags+body, used to build the BM25 index).
    Malformed articles are skipped with a logged warning. A missing or empty
    directory yields an empty list -- never an exception.
    """
    articles: list[dict] = []

    if not wiki_dir.is_dir():
        logger.warning(
            "Wiki knowledge base directory %s does not exist; search_wiki will return no results.",
            wiki_dir,
        )
        return articles

    for path in sorted(wiki_dir.glob("*.md")):
        if path.stem.startswith("_"):
            continue  # scaffolding, e.g. _TEMPLATE.md -- not a real article

        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("Skipping unreadable wiki article %s: %s", path, exc)
            continue

        result = validate_article(text)
        if not result.ok:
            logger.warning(
                "Skipping malformed wiki article %s: %s", path, "; ".join(result.errors)
            )
            continue

        try:
            frontmatter, body = parse_article(text)
        except ValueError as exc:
            # Shouldn't happen -- validate_article already confirmed the
            # frontmatter parses -- but guard so a schema/parser drift can
            # never turn into an unhandled exception at index-build time.
            logger.warning("Skipping wiki article %s: %s", path, exc)
            continue

        article_id = frontmatter.get("id", path.stem)
        title = frontmatter.get("title", "")
        tags = frontmatter.get("tags", []) or []
        source = frontmatter.get("source", {}) or {}

        doc_text = " ".join([title, " ".join(str(tag) for tag in tags), body])
        articles.append({
            "id": article_id,
            "title": title,
            "tags": tags,
            "source": source,
            "doc_tokens": _tokenize(doc_text),
        })

    return articles


def search_wiki(query: str, k: int = 3) -> list[dict]:
    """Search the LLM-wiki knowledge base for the ``k`` most relevant articles.

    Loads and parses every article under ``config["knowledge_base_dir"]``
    (default ``"knowledge/wiki"``, resolved relative to the current working
    directory when not absolute), builds a BM25 index over each article's
    title + tags + body, and returns the top-``k`` matches for ``query``.

    Args:
        query: free-text search query.
        k: maximum number of results to return (default 3).

    Returns:
        A list of up to ``k`` dicts with keys ``id``, ``title``, ``tags``,
        ``section``, ``source``, ``score``, sorted deterministically by
        score descending, then ``id`` ascending to break ties. ``section``
        is always ``None`` today (retrieval is whole-article, not
        section-aware; see module docstring). Returns ``[]`` -- never
        raises -- when the corpus is empty/missing, the knowledge base is
        disabled, or the query has no token overlap with any article.
    """
    config = get_config()
    if not config.get("knowledge_base_enabled", True):
        return []

    wiki_dir = Path(config.get("knowledge_base_dir", "knowledge/wiki"))
    articles = _load_articles(wiki_dir)
    if not articles:
        return []

    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    index = _Bm25Index([article["doc_tokens"] for article in articles])
    scores = index.scores(query_tokens)

    # Exclude articles with zero token overlap with the query rather than
    # filtering on "score > 0" -- BM25's idf term can go negative for very
    # common terms, so a genuinely non-matching article could otherwise be
    # kept (score == 0, filtered) while an overlapping-but-common-term
    # article could be wrongly dropped (score < 0). Overlap is the correct
    # "did this even match" signal; the score only decides ranking among
    # matches.
    results = [
        {
            "id": article["id"],
            "title": article["title"],
            "tags": article["tags"],
            "section": None,
            "source": article["source"],
            "score": score,
        }
        for article, score in zip(articles, scores, strict=True)
        if set(query_tokens) & set(article["doc_tokens"])
    ]

    # Deterministic order: score descending, then id ascending to break ties
    # (matches build_evidence_pack's score-desc/id-asc convention in
    # tavily_search.py, adapted to this corpus's natural key).
    results.sort(key=lambda r: (-r["score"], r["id"]))

    return results[:k]
