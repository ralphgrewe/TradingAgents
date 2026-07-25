"""LLM-wiki strategy knowledge base support code.

Home for logic shared between the (future) PDF->article ingestion pipeline and
the (future) BM25 retrieval dataflow, but that belongs to neither on its own —
starting with the article schema validator in ``wiki_schema.py``. See
``docs/design/llm-wiki.md`` for the full design and ``knowledge/wiki/`` for the
article corpus this package validates.
"""
