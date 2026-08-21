"""Deterministic macro fundamentals indicator pack.

Builds a single, LLM-ready snapshot of the macro indicator universe for a
given ``curr_date`` in one call — the deterministic pre-computation layer
``LEARNINGS.md`` calls for: hand the agent Python-computed values instead of
asking it to do cross-series arithmetic itself (issue #131).

Indicator universe (decided on issue #131, in place of the #127 research
ticket which was not run this round):

- Rates/curve, inflation, growth, labor, money/markets, sentiment — all via
  the existing ``fred`` vendor (``fred.MACRO_SERIES``).
- Gold — FRED's LBMA gold series was discontinued in 2015, so spot gold is
  sourced from the ``yfinance`` vendor instead, symbol ``GC=F`` (COMEX
  front-month future), via ``y_finance.get_price_history_points``.
- CoT (Commitments of Traders) — **not implemented this batch**; no CFTC
  vendor is integrated. The pack always includes an explicit
  ``{"available": False, "reason": "no vendor integrated"}`` entry so the gap
  is visible rather than silently omitted.

Derived features (computed in Python, per series):

- ``latest_value`` / ``latest_date``: the most recent observation on or
  before ``curr_date``.
- ``prior_value`` / ``prior_date`` / ``delta``: the previous observation and
  ``latest - prior``.
- ``direction``: ``"up"``/``"down"``/``"flat"`` from the sign of ``delta``
  (flat within a small epsilon, to absorb floating-point noise rather than
  genuine movement).
- ``z_score``: over the trailing ``Z_SCORE_WINDOW`` (24) observations in the
  series' own native frequency (24 monthly obs ~= 2y, 24 daily obs ~= 1
  month) — sample mean/stdev of that window. z-scores are read per-series,
  never compared cross-series, so mixing frequencies here is fine. Below
  ``Z_SCORE_MIN_OBSERVATIONS`` (8) available observations, ``z_score`` is
  ``None`` with a ``"insufficient history"`` note rather than computed on a
  tiny, unstable sample.

Frequency alignment / point-in-time: the pack is a **per-series snapshot**,
not a merged single-frequency table, so no cross-series resampling happens —
each series is fetched and evaluated independently as-of ``curr_date``. Point
-in-time safety reuses the existing FRED ``observation_end=curr_date`` /
yfinance inclusive-end window, which already excludes observations dated
after ``curr_date``. Release lag falls out of this for free: a low-frequency
series' ``latest_date`` will naturally trail ``curr_date`` by however long
that series' real-world publication lag is — no separate lag table is
needed. **No ALFRED/``realtime_*`` integration this batch**: FRED's standard
endpoint returns the latest-revised vintage for historical dates, i.e. mild
revision leakage for backtests against past dates. Revisit if backtest
fidelity on macro-sensitive strategies becomes a problem.

Partial failure: each series is fetched independently with its own
try/except. A failed or unavailable series is marked
``{"available": False, "error": ...}`` in the pack while every other series
still populates — one dead series (or a missing ``FRED_API_KEY``, surfaced
via ``FredNotConfiguredError``'s message) never fails the whole pack.

Deterministic: given the same ``curr_date`` and identical upstream vendor
data, ``get_macro_pack`` returns a value-identical dict — no LLM call is made
anywhere in this module, and no wall-clock read beyond ``curr_date`` itself.
"""
from __future__ import annotations

import statistics
from datetime import datetime

from .fred import (
    FredNotConfiguredError,
    _fetch_observations,
    _fetch_series_meta,
    _resolve_series_id,
)
from .y_finance import get_price_history_points

# The indicator universe decided on issue #131. Aliases map onto
# fred.MACRO_SERIES; anything not resolvable there is a bug in this list, not
# a per-call failure, so it is intentionally not wrapped defensively.
FRED_INDICATOR_ALIASES = [
    # Rates / curve
    "fed_funds_rate",
    "2y_treasury",
    "10y_treasury",
    "yield_curve",
    # Inflation
    "cpi",
    "core_cpi",
    "inflation_expectations",
    # Growth
    "real_gdp",
    "industrial_production",
    # Labor
    "unemployment_rate",
    "nonfarm_payrolls",
    "initial_claims",
    # Money / markets
    "dollar_index",
    "vix",
    # Sentiment / housing
    "consumer_sentiment",
]

# Spot gold has no clean FRED series (LBMA discontinued 2015); sourced from
# yfinance instead, since it is already the core_stock_apis default vendor.
GOLD_SYMBOL = "GC=F"

# Trailing window (in the series' own native frequency) for the z-score.
Z_SCORE_WINDOW = 24
# Below this many observations, a z-score is too unstable to be useful.
Z_SCORE_MIN_OBSERVATIONS = 8

# Generous lookback so even the lowest-frequency series in the universe
# (real GDP, quarterly: 24 obs ~= 6 years) has enough history for a full
# Z_SCORE_WINDOW, with buffer for gaps/holidays in higher-frequency series.
LOOKBACK_DAYS = 15 * 365


def _direction(delta: float, scale: float) -> str:
    """Classify ``delta``'s sign as up/down/flat, absorbing float noise.

    ``scale`` (typically the latest value) sizes the epsilon so "flat" means
    genuinely unchanged rather than a real move on a large-magnitude series
    being misread as noise.
    """
    epsilon = max(abs(scale), 1.0) * 1e-9
    if delta > epsilon:
        return "up"
    if delta < -epsilon:
        return "down"
    return "flat"


def _z_score(values: list[float]) -> tuple[float | None, str | None]:
    """Sample z-score of the last value over its trailing Z_SCORE_WINDOW.

    Returns ``(z_score, note)``; ``z_score`` is ``None`` (with an explanatory
    ``note``) when there is insufficient history or the window has zero
    variance, rather than raising or returning a misleading value.
    """
    window = values[-Z_SCORE_WINDOW:]
    if len(window) < Z_SCORE_MIN_OBSERVATIONS:
        return None, "insufficient history"
    mean = statistics.fmean(window)
    stdev = statistics.stdev(window)
    if stdev == 0:
        return None, "zero variance in window"
    return (window[-1] - mean) / stdev, None


def _derived_features(points: list[tuple[str, float]]) -> dict:
    """Compute the per-series derived features from ascending (date, value) points.

    ``points`` must be non-empty and sorted ascending by date.
    """
    latest_date, latest_value = points[-1]
    features = {
        "latest_value": latest_value,
        "latest_date": latest_date,
        "prior_value": None,
        "prior_date": None,
        "delta": None,
        "direction": None,
    }

    if len(points) >= 2:
        prior_date, prior_value = points[-2]
        delta = latest_value - prior_value
        features["prior_value"] = prior_value
        features["prior_date"] = prior_date
        features["delta"] = delta
        features["direction"] = _direction(delta, latest_value)

    z_score, z_score_note = _z_score([v for _, v in points])
    features["z_score"] = z_score
    features["z_score_note"] = z_score_note
    features["observations_used"] = min(len(points), Z_SCORE_WINDOW)

    return features


def _to_float_points(raw_points: list[tuple[str, str]]) -> list[tuple[str, float]]:
    """Coerce FRED's string-valued observations to floats, dropping any that aren't numeric."""
    result = []
    for date, value in raw_points:
        try:
            result.append((date, float(value)))
        except (TypeError, ValueError):
            continue
    return result


def _fred_series_entry(alias: str, curr_date: str) -> dict:
    """Fetch and compute derived features for one FRED-backed series.

    Never raises: any failure (unknown alias, missing FRED_API_KEY, network
    error, no observations) is caught and returned as
    ``{"available": False, "error": ...}`` so one dead series doesn't abort
    the rest of the pack.
    """
    try:
        series_id = _resolve_series_id(alias)
    except ValueError as e:
        return {"alias": alias, "available": False, "error": str(e)}

    try:
        meta = _fetch_series_meta(series_id)
        raw_points = _fetch_observations(series_id, curr_date, LOOKBACK_DAYS)
    except FredNotConfiguredError as e:
        # Surface the existing "FRED_API_KEY not set" message verbatim rather
        # than letting it propagate as a stack trace.
        return {"alias": alias, "series_id": series_id, "available": False, "error": str(e)}
    except Exception as e:
        return {"alias": alias, "series_id": series_id, "available": False, "error": str(e)}

    points = _to_float_points(raw_points)
    if not points:
        return {
            "alias": alias,
            "series_id": series_id,
            "available": False,
            "error": f"no observations for {series_id} on or before {curr_date}",
        }

    entry = {
        "alias": alias,
        "series_id": series_id,
        "available": True,
        "title": meta.get("title", series_id),
        "units": meta.get("units_short") or meta.get("units", ""),
        "frequency": meta.get("frequency", ""),
    }
    entry.update(_derived_features(points))
    return entry


def _gold_entry(curr_date: str) -> dict:
    """Fetch and compute derived features for spot gold (GC=F, via yfinance).

    Never raises: mirrors ``_fred_series_entry``'s partial-failure contract.
    """
    try:
        raw_points = get_price_history_points(GOLD_SYMBOL, curr_date, LOOKBACK_DAYS)
    except Exception as e:
        return {"alias": "gold", "symbol": GOLD_SYMBOL, "available": False, "error": str(e)}

    points = [(d, float(v)) for d, v in raw_points]
    if not points:
        return {
            "alias": "gold",
            "symbol": GOLD_SYMBOL,
            "available": False,
            "error": f"no price observations for {GOLD_SYMBOL} on or before {curr_date}",
        }

    entry = {
        "alias": "gold",
        "symbol": GOLD_SYMBOL,
        "available": True,
        "title": "Gold (COMEX front-month future)",
        "units": "USD/oz",
        "frequency": "Daily",
    }
    entry.update(_derived_features(points))
    return entry


def _cot_entry() -> dict:
    """CoT (Commitments of Traders): explicitly marked unavailable this batch.

    No CFTC vendor is integrated. Included as a visible gap, not omitted
    silently, so future work has a clean slot (see the #131 decision comment).
    """
    return {"alias": "cot", "available": False, "reason": "no vendor integrated"}


def get_macro_pack(curr_date: str) -> dict:
    """Build the full deterministic macro indicator pack for ``curr_date``.

    One call fetches every series in the indicator universe (FRED-backed
    macro series + gold via yfinance), computes derived features (latest/
    prior value, delta, direction, trailing z-score) in Python, and returns a
    structured dict ready to hand an LLM — no LLM call happens here. A CoT
    slot is always present, marked unavailable (see module docstring).

    Args:
        curr_date: The as-of date, ``yyyy-mm-dd``. No observation dated after
            this date is ever included (point-in-time safe).

    Returns:
        ``{"curr_date": curr_date, "indicators": {alias: entry, ...}}`` where
        each ``entry`` is either an "available" series with its derived
        features, or ``{"available": False, "error"/"reason": ...}``.
    """
    try:
        datetime.strptime(curr_date, "%Y-%m-%d")
    except (TypeError, ValueError) as e:
        raise ValueError(f"curr_date must be in yyyy-mm-dd format, got {curr_date!r}") from e

    indicators = {
        alias: _fred_series_entry(alias, curr_date) for alias in FRED_INDICATOR_ALIASES
    }
    indicators["gold"] = _gold_entry(curr_date)
    indicators["cot"] = _cot_entry()

    return {"curr_date": curr_date, "indicators": indicators}
