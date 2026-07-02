"""Memory-core glue for the news analyst skill (issue #10).

Mechanical repetition of the memory wiring pattern validated for the quant
skill (issue #8, commit 5173818): `skills/news/SKILL.md` calls the `memory_*`
MCP tools directly, but the payload it sends to `memory_store_decision` is
built here as pure functions of the already-written envelope, not re-derived
by the model.

Unlike `skills/quant/compute_indicators.py`, this skill has no deterministic
compute step to expose: the news analyst's `signal`/`confidence` come from
the model's own reading of the fetched articles (SKILL.md Steps 1-4), not
from a script — see the "no new design" note in issue #10 and the
"structural discipline ... as closely as the news skill's own architecture
allows" language, which is why there is no `compute(records, ticker)`
equivalent here. These two helpers only reshape the envelope's `details`
after the model has already fixed `signal`/`confidence`/the `conservative`/
`risky` ratings, the same way `build_key_drivers` in the quant script
reshapes `details` without touching them.
"""

from __future__ import annotations

# Same HIGH/MEDIUM/LOW -> numeric convention as
# skills/quant/compute_indicators.py's confidence_to_score and
# skills/trader/score_trader.py's conf_weight — reused here rather than
# inventing a second mapping.
CONFIDENCE_SCORE = {"HIGH": 1.0, "MEDIUM": 0.6, "LOW": 0.3}


def confidence_to_score(confidence):
    """Map the envelope's HIGH/MEDIUM/LOW confidence to the numeric score
    `memory_store_decision`'s `confidence` argument expects. Unknown/missing
    values fall back to 0.3 (LOW), matching conf_weight's default."""
    return CONFIDENCE_SCORE.get(str(confidence).upper(), 0.3)


def build_key_drivers(details):
    """Build the `key_drivers` payload for `memory_store_decision` from the
    news envelope's `details` (issue #10): the four top-headline lists
    (`top_positive_fundamentals`/`top_negative_fundamentals`/
    `top_positive_sentiment`/`top_negative_sentiment`) and the
    `conservative` vs `risky` rating split, verbatim. No new derivation —
    every value here already exists in `details`."""
    return {
        "top_positive_fundamentals": details.get("top_positive_fundamentals", []),
        "top_negative_fundamentals": details.get("top_negative_fundamentals", []),
        "top_positive_sentiment": details.get("top_positive_sentiment", []),
        "top_negative_sentiment": details.get("top_negative_sentiment", []),
        "conservative": details.get("conservative"),
        "risky": details.get("risky"),
    }
