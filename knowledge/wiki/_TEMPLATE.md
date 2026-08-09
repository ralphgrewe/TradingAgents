---
id: kebab-case-unique-id
title: Human-Readable Article Title
tags: [tag-one, tag-two]
signals: [signal_one, signal_two]
asset_classes: [equity]
horizon: [swing, position]
source: {authors: "Author One, Author Two", title: "Source Paper Title", year: 2000, file: paper/SourceFile.pdf}
---
## Summary

One or two paragraphs: what does this strategy/signal claim, and why (the economic or
behavioral rationale)?

## Signal — what it is

Precise definition of each signal named in the `signals` frontmatter list above.

## How to compute

The formula/procedure in enough detail that it could be implemented in code (cf.
`tradingagents/agents/analysts/fundamentals_computation.py`'s "precompute in Python,
don't ask the LLM to do arithmetic" convention). Use inputs an agent could plausibly
source from this repo's existing data vendors where possible.

## Empirical evidence

What the source paper(s) found: sample, time period, effect size, statistical
significance. Note any known replications or failures to replicate.

## When to apply / regime

Which asset classes, holding horizons, and market regimes this signal is expected to
work (or specifically not work) in.

## Caveats

Known failure modes, data requirements, look-ahead-bias risks, crowding/decay
concerns, and anything a consuming agent should weigh before acting on this signal.
