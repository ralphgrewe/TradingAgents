"""Tests for the BM25 LLM-wiki retrieval dataflow (issue #103).

Uses a small, self-written fixture corpus under ``tmp_path`` rather than the
live ``knowledge/wiki/`` directory -- issue #102 (PDF ingestion, running in
parallel) actively adds articles there, so depending on its contents would
make these tests non-deterministic. No network access.
"""

import unittest

import pytest

from tradingagents.dataflows import interface
from tradingagents.dataflows.config import set_config
from tradingagents.dataflows.wiki_search import search_wiki

_MOMENTUM_ARTICLE = """\
---
id: momentum-factor
title: Momentum Factor
tags: [momentum, technical, trend-following]
signals: [momentum_12_1]
asset_classes: [equity]
horizon: [swing]
source: {authors: "Jegadeesh, Titman", title: "Returns to Buying Winners and Selling Losers", year: 1993, file: paper/jegadeesh_titman_1993.pdf}
---
## Summary

Momentum strategies buy recent winners and sell recent losers, exploiting
short- to medium-term trend continuation in stock returns.

## Signal — what it is

`momentum_12_1` is the cumulative return over the trailing 12 months,
skipping the most recent month, used to rank stocks by trend strength.

## How to compute

```
momentum_12_1 = price[t-1] / price[t-12] - 1
```

## Empirical evidence

Jegadeesh and Titman (1993) find winners continue to outperform losers over
3-12 month holding periods.

## When to apply / regime

Works best in trending, low-volatility regimes; momentum crashes occur
during sharp reversals.

## Caveats

Momentum crash risk and high turnover/transaction costs.
"""

_MEAN_REVERSION_ARTICLE = """\
---
id: mean-reversion-rsi
title: Mean Reversion via RSI
tags: [mean-reversion, oscillator, value]
signals: [rsi_14]
asset_classes: [equity]
horizon: [intraday]
source: {authors: "Wilder", title: "New Concepts in Technical Trading Systems", year: 1978, file: paper/wilder_1978.pdf}
---
## Summary

Mean reversion strategies bet that oversold or overbought assets revert
toward a recent average price.

## Signal — what it is

`rsi_14` is the 14-period Relative Strength Index, used to flag oversold
(<30) or overbought (>70) conditions for reversion trades.

## How to compute

```
rsi_14 = 100 - 100 / (1 + avg_gain_14 / avg_loss_14)
```

## Empirical evidence

Short-horizon reversal effects are well documented following sharp
oversold/overbought moves.

## When to apply / regime

Works best in range-bound, mean-reverting regimes; fails badly in strong
trends (the reversion never comes).

## Caveats

Prone to whipsaws in trending markets; not a substitute for trend filters.
"""

# Missing several required frontmatter keys (signals, asset_classes, horizon,
# source) and all body sections -- deliberately malformed for the
# skip-and-warn test.
_MALFORMED_ARTICLE = """\
---
id: broken-article
title: Broken Article
tags: [nonsense-only-tag]
---
This article has no required body sections and incomplete frontmatter.
"""


def _write_corpus(wiki_dir, include_malformed: bool = False):
    wiki_dir.mkdir(parents=True, exist_ok=True)
    (wiki_dir / "momentum-factor.md").write_text(_MOMENTUM_ARTICLE, encoding="utf-8")
    (wiki_dir / "mean-reversion-rsi.md").write_text(_MEAN_REVERSION_ARTICLE, encoding="utf-8")
    if include_malformed:
        (wiki_dir / "broken-article.md").write_text(_MALFORMED_ARTICLE, encoding="utf-8")


@pytest.mark.unit
class WikiSearchTests(unittest.TestCase):
    def test_obvious_best_match_returns_first(self):
        wiki_dir = self._tmp_wiki_dir()
        _write_corpus(wiki_dir)
        set_config({"knowledge_base_dir": str(wiki_dir)})

        results = search_wiki("momentum trend following winners losers", k=3)

        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "momentum-factor")
        self.assertEqual(results[0]["title"], "Momentum Factor")
        self.assertEqual(results[0]["tags"], ["momentum", "technical", "trend-following"])
        self.assertIsNone(results[0]["section"])
        self.assertIn("authors", results[0]["source"])
        # Sorted score desc.
        scores = [r["score"] for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_other_query_matches_other_article_first(self):
        wiki_dir = self._tmp_wiki_dir()
        _write_corpus(wiki_dir)
        set_config({"knowledge_base_dir": str(wiki_dir)})

        results = search_wiki("oversold overbought RSI mean reversion", k=3)

        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "mean-reversion-rsi")

    def test_result_dict_shape(self):
        wiki_dir = self._tmp_wiki_dir()
        _write_corpus(wiki_dir)
        set_config({"knowledge_base_dir": str(wiki_dir)})

        results = search_wiki("momentum", k=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(
            set(results[0].keys()), {"id", "title", "tags", "section", "source", "score"}
        )

    def test_k_limits_result_count(self):
        wiki_dir = self._tmp_wiki_dir()
        _write_corpus(wiki_dir)
        set_config({"knowledge_base_dir": str(wiki_dir)})

        # Query overlapping both articles (e.g. "regime" appears in both bodies).
        results = search_wiki("regime strategy", k=1)
        self.assertLessEqual(len(results), 1)

    def test_empty_corpus_returns_empty_list(self):
        wiki_dir = self._tmp_wiki_dir()
        wiki_dir.mkdir(parents=True, exist_ok=True)
        set_config({"knowledge_base_dir": str(wiki_dir)})

        self.assertEqual(search_wiki("anything", k=3), [])

    def test_missing_directory_returns_empty_list(self):
        wiki_dir = self._tmp_wiki_dir() / "does-not-exist"
        set_config({"knowledge_base_dir": str(wiki_dir)})

        self.assertEqual(search_wiki("anything", k=3), [])

    def test_no_matching_tokens_returns_empty_list(self):
        wiki_dir = self._tmp_wiki_dir()
        _write_corpus(wiki_dir)
        set_config({"knowledge_base_dir": str(wiki_dir)})

        self.assertEqual(search_wiki("zzzznonexistentqueryterm", k=3), [])

    def test_malformed_article_is_skipped_not_raised(self):
        wiki_dir = self._tmp_wiki_dir()
        _write_corpus(wiki_dir, include_malformed=True)
        set_config({"knowledge_base_dir": str(wiki_dir)})

        with self.assertLogs("tradingagents.dataflows.wiki_search", level="WARNING") as cm:
            # "nonsense-only-tag" only appears in the malformed article's tags;
            # if it were indexed, it would be the sole hit.
            results = search_wiki("nonsense-only-tag", k=3)

        self.assertEqual(results, [])
        joined = "\n".join(cm.output)
        self.assertIn("broken-article", joined)

        # Valid articles are still indexed and searchable alongside the skip.
        momentum_results = search_wiki("momentum trend", k=3)
        self.assertTrue(any(r["id"] == "momentum-factor" for r in momentum_results))

    def test_disabled_returns_empty_list_without_reading_disk(self):
        wiki_dir = self._tmp_wiki_dir()
        _write_corpus(wiki_dir)
        set_config({"knowledge_base_dir": str(wiki_dir), "knowledge_base_enabled": False})

        self.assertEqual(search_wiki("momentum", k=3), [])

    def test_tie_break_is_deterministic_by_id(self):
        wiki_dir = self._tmp_wiki_dir()
        wiki_dir.mkdir(parents=True, exist_ok=True)
        # Two articles with identical bodies (so identical BM25 scores for any
        # query token they share) except id/title -- exercises the score-desc,
        # id-asc tie-break.
        for article_id in ("zzz-twin", "aaa-twin"):
            text = _MOMENTUM_ARTICLE.replace("momentum-factor", article_id).replace(
                "Momentum Factor", article_id.upper()
            )
            (wiki_dir / f"{article_id}.md").write_text(text, encoding="utf-8")
        set_config({"knowledge_base_dir": str(wiki_dir)})

        results = search_wiki("momentum trend", k=2)

        self.assertEqual(len(results), 2)
        self.assertAlmostEqual(results[0]["score"], results[1]["score"])
        self.assertEqual([r["id"] for r in results], ["aaa-twin", "zzz-twin"])

    def test_route_to_vendor_wiring(self):
        wiki_dir = self._tmp_wiki_dir()
        _write_corpus(wiki_dir)
        set_config({
            "knowledge_base_dir": str(wiki_dir),
            "data_vendors": {"knowledge_base": "bm25"},
        })

        results = interface.route_to_vendor("search_wiki", "momentum trend following", 3)

        self.assertIsInstance(results, list)
        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "momentum-factor")

    def _tmp_wiki_dir(self):
        import tempfile
        from pathlib import Path

        tmp = tempfile.mkdtemp(prefix="wiki-search-test-")
        self.addCleanup(lambda: __import__("shutil").rmtree(tmp, ignore_errors=True))
        return Path(tmp) / "wiki"


if __name__ == "__main__":
    unittest.main()
