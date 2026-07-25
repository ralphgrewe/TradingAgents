# `knowledge/wiki/`

This is the **LLM-wiki**: a curated, human-reviewed knowledge base of trading-strategy
articles, built from the papers under `paper/` (see the ingest folder convention below).
Decision agents (portfolio manager, swing trader, and later others — see
`docs/design/llm-wiki.md` "Extensibility") consult it via a retrieval tool instead of
re-deriving strategy knowledge from model priors on every run.

Full design rationale — why in-repo markdown over an external system, why BM25 over
vector retrieval, why a tool call over prompt injection or an MCP server — lives in
[`docs/design/llm-wiki.md`](../../docs/design/llm-wiki.md). This README only covers the
practical "how do I add/read an article" layout.

## Layout

- One markdown file per article: `knowledge/wiki/<id>.md`, where `<id>` matches the
  article's own `id` frontmatter field exactly (kebab-case, e.g. `piotroski-f-score.md`
  for `id: piotroski-f-score`).
- `_TEMPLATE.md` — copy this to start a new article, whether hand-written or drafted by
  the future ingestion pipeline (#102) for human review.
- Every article's `source.file` frontmatter field points at the source document under
  `paper/` (the default ingest folder — config key `knowledge_ingest_dir`, default
  `"paper"`) that the article was derived from.

## Article schema

Every article is a YAML frontmatter block followed by six required `## `-headed body
sections, in this order:

```markdown
---
id: kebab-case-unique-id
title: Human-Readable Title
tags: [tag-one, tag-two]
signals: [signal_one, signal_two]
asset_classes: [equity]
horizon: [swing, position]
source: {authors: "Author Name", title: "Paper Title", year: 2000, file: paper/Source.pdf}
---
## Summary
## Signal — what it is
## How to compute
## Empirical evidence
## When to apply / regime
## Caveats
```

| Frontmatter key | Type | Meaning |
|---|---|---|
| `id` | string, kebab-case, unique across the wiki | Must match the filename stem. |
| `title` | string | Human-readable article title. |
| `tags` | list of strings | Free-text topic tags used by BM25 retrieval (#103). |
| `signals` | list of strings | Named computable signals the article defines. |
| `asset_classes` | list of strings | e.g. `equity`, `crypto`, `fx`. |
| `horizon` | list of strings | e.g. `swing`, `position`, `intraday`. |
| `source` | mapping `{authors, title, year, file}` | `file` is a `paper/`-relative path to the source PDF. |

Body sections, one paragraph-or-more each:

1. `## Summary` — the strategy's thesis in a few sentences.
2. `## Signal — what it is` — precise definition of the `signals` named above.
3. `## How to compute` — the formula/procedure in enough detail to implement.
4. `## Empirical evidence` — what the source paper found (sample, period, effect size).
5. `## When to apply / regime` — asset classes/horizons/regimes it's expected to work in.
6. `## Caveats` — known failure modes, data requirements, decay/crowding concerns.

See [`piotroski-f-score.md`](piotroski-f-score.md) for a complete worked example.

## Validating an article

`tradingagents/knowledge/wiki_schema.py` provides a pure-function validator,
`validate_article(text)` / `validate_article_file(path)`, that checks the required
frontmatter keys and body sections listed above are present. Run it against a new or
edited article before committing:

```bash
./venv/bin/python -c "
from tradingagents.knowledge.wiki_schema import validate_article_file
result = validate_article_file('knowledge/wiki/your-new-article.md')
print('OK' if result.ok else result.errors)
"
```

This same helper is what the future ingestion pipeline (#102) and retrieval dataflow
(#103) reuse to reject malformed articles, so keeping an article schema-valid here
means it stays usable everywhere else the wiki is consumed.
