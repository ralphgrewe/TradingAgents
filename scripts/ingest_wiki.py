"""CLI entry point for the LLM-wiki PDF -> article ingestion pipeline (#102).

Reads PDFs from ``knowledge_ingest_dir`` (config key, default ``paper/``),
drafts a schema-conformant article per PDF via the quick-thinking LLM
(``tradingagents.knowledge.ingest``), validates each draft against
``tradingagents.knowledge.wiki_schema``, and writes it to
``knowledge_base_dir`` (config key, default ``knowledge/wiki/``).

Idempotent / human-in-the-loop: an existing article whose ``source.file``
frontmatter already points at a given PDF (compared on normalized paths) is
left untouched (skipped) unless ``--force``/``--overwrite`` is passed.
Generated articles are meant to be reviewed (and edited if needed) before
being committed.

Batch-resilient: a PDF that fails to process (corrupt file, LLM/parse
failure) is reported as an ``error`` in the final tally and the run
continues with the remaining PDFs; the exit status is non-zero if any file
came back ``invalid`` or ``error``.

Usage:
    ./venv/bin/python scripts/ingest_wiki.py
    ./venv/bin/python scripts/ingest_wiki.py --provider mistral --model mistral-small
    ./venv/bin/python scripts/ingest_wiki.py --ingest-dir paper --wiki-dir knowledge/wiki --force
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from tradingagents.dataflows.config import get_config
from tradingagents.knowledge.ingest import build_quick_think_llm, run_ingest


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--ingest-dir", type=Path, default=None,
        help="Folder of source PDFs (default: config knowledge_ingest_dir, normally 'paper')",
    )
    parser.add_argument(
        "--wiki-dir", type=Path, default=None,
        help="Output folder for articles (default: config knowledge_base_dir, normally 'knowledge/wiki')",
    )
    parser.add_argument("--provider", default=None, help="LLM provider override (default: config llm_provider)")
    parser.add_argument("--model", default=None, help="Quick-think model override (default: config quick_think_llm)")
    parser.add_argument(
        "--force", "--overwrite", dest="force", action="store_true",
        help="Regenerate articles that already cover a source PDF (default: skip them)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    config = get_config()
    if args.provider:
        config["llm_provider"] = args.provider
    if args.model:
        config["quick_think_llm"] = args.model

    llm = build_quick_think_llm(config)
    outcomes = run_ingest(
        llm,
        ingest_dir=args.ingest_dir,
        wiki_dir=args.wiki_dir,
        force=args.force,
        config=config,
    )

    created = [o for o in outcomes if o.status == "created"]
    skipped = [o for o in outcomes if o.status == "skipped"]
    invalid = [o for o in outcomes if o.status == "invalid"]
    errored = [o for o in outcomes if o.status == "error"]

    print(
        f"\n{len(created)} created, {len(skipped)} skipped, {len(invalid)} invalid, "
        f"{len(errored)} error (of {len(outcomes)} PDFs)"
    )
    for outcome in created:
        print(f"  created  {outcome.pdf_path.name} -> {outcome.article_path}")
    for outcome in skipped:
        print(f"  skipped  {outcome.pdf_path.name} ({outcome.reason})")
    for outcome in invalid:
        print(f"  invalid  {outcome.pdf_path.name}: {'; '.join(outcome.errors)}")
    for outcome in errored:
        print(f"  error    {outcome.pdf_path.name}: {outcome.reason}")

    return 1 if (invalid or errored) else 0


if __name__ == "__main__":
    sys.exit(main())
