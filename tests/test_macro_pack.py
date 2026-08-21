"""Tests for the deterministic macro indicator pack (tradingagents/dataflows/macro_pack.py).

All FRED and yfinance access is mocked, so these run without a network
connection and without a real FRED_API_KEY (issue #131).
"""
import unittest
from unittest import mock

import pytest

from tradingagents.dataflows import fred, macro_pack
from tradingagents.dataflows.symbol_utils import NoMarketDataError

_DEFAULT_META = {
    "seriess": [
        {
            "title": "Test Series",
            "units_short": "Percent",
            "frequency": "Monthly",
            "seasonal_adjustment_short": "SA",
        }
    ]
}


def _make_observations(n, start_value=2.0, increment=0.1, start_date="2023-01-01", step_days=30):
    """Build n ascending FRED-style observations with a steady increment."""
    from datetime import datetime, timedelta

    d = datetime.strptime(start_date, "%Y-%m-%d")
    obs = []
    value = start_value
    for _ in range(n):
        obs.append({"date": d.strftime("%Y-%m-%d"), "value": f"{value:.4f}"})
        d += timedelta(days=step_days)
        value += increment
    return {"observations": obs}


_DEFAULT_OBS = _make_observations(30)


def _stub_request(overrides=None, default_meta=None, default_obs=None):
    """Build a fred._request replacement dispatching on series_id.

    overrides: dict[series_id -> {"raise": exc}] or {"meta": ..., "obs": ...}]
    """
    overrides = overrides or {}
    default_meta = default_meta or _DEFAULT_META
    default_obs = default_obs or _DEFAULT_OBS

    def _impl(path, params):
        series_id = params.get("series_id")
        override = overrides.get(series_id)
        if override and "raise" in override:
            raise override["raise"]
        meta = override["meta"] if override and "meta" in override else default_meta
        obs = override["obs"] if override and "obs" in override else default_obs
        if path == "series":
            return meta
        if path == "series/observations":
            return obs
        raise AssertionError(f"unexpected FRED path: {path}")

    return _impl


def _gold_points(n=10, start_value=1900.0, increment=5.0, start_date="2026-07-01"):
    from datetime import datetime, timedelta

    d = datetime.strptime(start_date, "%Y-%m-%d")
    points = []
    value = start_value
    for _ in range(n):
        points.append((d.strftime("%Y-%m-%d"), value))
        d += timedelta(days=1)
        value += increment
    return points


@pytest.mark.unit
class DirectionTests(unittest.TestCase):
    def test_positive_delta_is_up(self):
        self.assertEqual(macro_pack._direction(1.5, 100.0), "up")

    def test_negative_delta_is_down(self):
        self.assertEqual(macro_pack._direction(-0.7, 100.0), "down")

    def test_zero_delta_is_flat(self):
        self.assertEqual(macro_pack._direction(0.0, 100.0), "flat")

    def test_tiny_float_noise_is_flat(self):
        # Floating point noise well inside epsilon must not read as a move.
        self.assertEqual(macro_pack._direction(1e-12, 100.0), "flat")


@pytest.mark.unit
class ZScoreTests(unittest.TestCase):
    def test_insufficient_history_below_min_observations(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]  # 5 < Z_SCORE_MIN_OBSERVATIONS (8)
        z, note = macro_pack._z_score(values)
        self.assertIsNone(z)
        self.assertEqual(note, "insufficient history")

    def test_exactly_min_observations_computes_a_score(self):
        values = [float(i) for i in range(8)]  # exactly 8
        z, note = macro_pack._z_score(values)
        self.assertIsNotNone(z)
        self.assertIsNone(note)

    def test_window_caps_at_24_most_recent(self):
        # 30 values -> only the trailing 24 feed the z-score.
        values = [float(i) for i in range(30)]
        z, note = macro_pack._z_score(values)
        window = values[-24:]
        import statistics
        expected = (window[-1] - statistics.fmean(window)) / statistics.stdev(window)
        self.assertAlmostEqual(z, expected)

    def test_zero_variance_window_is_null_not_a_crash(self):
        values = [5.0] * 10
        z, note = macro_pack._z_score(values)
        self.assertIsNone(z)
        self.assertEqual(note, "zero variance in window")


@pytest.mark.unit
class DerivedFeaturesTests(unittest.TestCase):
    def test_single_point_has_no_prior_or_direction(self):
        features = macro_pack._derived_features([("2026-01-01", 4.0)])
        self.assertEqual(features["latest_value"], 4.0)
        self.assertIsNone(features["prior_value"])
        self.assertIsNone(features["delta"])
        self.assertIsNone(features["direction"])

    def test_two_points_compute_delta_and_direction(self):
        points = [("2026-01-01", 4.0), ("2026-02-01", 4.5)]
        features = macro_pack._derived_features(points)
        self.assertEqual(features["prior_value"], 4.0)
        self.assertEqual(features["latest_value"], 4.5)
        self.assertAlmostEqual(features["delta"], 0.5)
        self.assertEqual(features["direction"], "up")


@pytest.mark.unit
class FredSeriesEntryTests(unittest.TestCase):
    def test_happy_path_computes_all_features(self):
        with mock.patch.object(fred, "_request", side_effect=_stub_request()):
            entry = macro_pack._fred_series_entry("cpi", "2026-06-01")
        self.assertTrue(entry["available"])
        self.assertEqual(entry["series_id"], "CPIAUCSL")
        self.assertIn("latest_value", entry)
        self.assertIn("z_score", entry)
        self.assertEqual(entry["direction"], "up")  # steadily increasing series

    def test_unknown_alias_is_marked_unavailable(self):
        # A descriptive phrase (not a known alias / valid series ID) fails
        # resolution before any request is made.
        entry = macro_pack._fred_series_entry("not a real indicator!!", "2026-06-01")
        self.assertFalse(entry["available"])
        self.assertIn("error", entry)

    def test_missing_fred_api_key_is_a_clear_message_not_a_crash(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            entry = macro_pack._fred_series_entry("cpi", "2026-06-01")
        self.assertFalse(entry["available"])
        self.assertIn("FRED_API_KEY", entry["error"])

    def test_vendor_failure_is_marked_unavailable_with_series_id(self):
        overrides = {"CPIAUCSL": {"raise": ValueError("FRED request failed: boom")}}
        with mock.patch.object(fred, "_request", side_effect=_stub_request(overrides)):
            entry = macro_pack._fred_series_entry("cpi", "2026-06-01")
        self.assertFalse(entry["available"])
        self.assertEqual(entry["series_id"], "CPIAUCSL")
        self.assertIn("boom", entry["error"])

    def test_insufficient_history_still_available_but_null_z_score(self):
        overrides = {"CPIAUCSL": {"obs": _make_observations(5)}}
        with mock.patch.object(fred, "_request", side_effect=_stub_request(overrides)):
            entry = macro_pack._fred_series_entry("cpi", "2026-06-01")
        self.assertTrue(entry["available"])
        self.assertIsNone(entry["z_score"])
        self.assertEqual(entry["z_score_note"], "insufficient history")


@pytest.mark.unit
class GoldEntryTests(unittest.TestCase):
    def test_happy_path(self):
        with mock.patch.object(macro_pack, "get_price_history_points", return_value=_gold_points()):
            entry = macro_pack._gold_entry("2026-07-10")
        self.assertTrue(entry["available"])
        self.assertEqual(entry["symbol"], "GC=F")
        self.assertEqual(entry["direction"], "up")

    def test_vendor_failure_is_marked_unavailable(self):
        with mock.patch.object(
            macro_pack, "get_price_history_points",
            side_effect=NoMarketDataError("GC=F", "GC=F", "no rows"),
        ):
            entry = macro_pack._gold_entry("2026-07-10")
        self.assertFalse(entry["available"])
        self.assertIn("error", entry)


@pytest.mark.unit
class GetMacroPackTests(unittest.TestCase):
    def test_happy_path_includes_full_universe(self):
        with mock.patch.object(fred, "_request", side_effect=_stub_request()), \
                mock.patch.object(macro_pack, "get_price_history_points", return_value=_gold_points()):
            pack = macro_pack.get_macro_pack("2026-06-01")

        self.assertEqual(pack["curr_date"], "2026-06-01")
        indicators = pack["indicators"]
        for alias in macro_pack.FRED_INDICATOR_ALIASES:
            self.assertIn(alias, indicators)
            self.assertTrue(indicators[alias]["available"], msg=alias)
        self.assertIn("gold", indicators)
        self.assertTrue(indicators["gold"]["available"])
        self.assertEqual(
            indicators["cot"], {"alias": "cot", "available": False, "reason": "no vendor integrated"}
        )

    def test_deterministic_across_repeated_calls(self):
        with mock.patch.object(fred, "_request", side_effect=_stub_request()), \
                mock.patch.object(macro_pack, "get_price_history_points", return_value=_gold_points()):
            pack1 = macro_pack.get_macro_pack("2026-06-01")
            pack2 = macro_pack.get_macro_pack("2026-06-01")

        self.assertEqual(pack1, pack2)

    def test_one_dead_series_does_not_fail_the_pack(self):
        # yield_curve (T10Y2Y) fails; every other series and gold/cot still
        # populate — a single dead series must not abort the whole pack.
        overrides = {"T10Y2Y": {"raise": ValueError("connection reset")}}
        with mock.patch.object(fred, "_request", side_effect=_stub_request(overrides)), \
                mock.patch.object(macro_pack, "get_price_history_points", return_value=_gold_points()):
            pack = macro_pack.get_macro_pack("2026-06-01")

        indicators = pack["indicators"]
        self.assertFalse(indicators["yield_curve"]["available"])
        self.assertIn("connection reset", indicators["yield_curve"]["error"])
        # Every other FRED alias is unaffected.
        for alias in macro_pack.FRED_INDICATOR_ALIASES:
            if alias == "yield_curve":
                continue
            self.assertTrue(indicators[alias]["available"], msg=alias)
        self.assertTrue(indicators["gold"]["available"])

    def test_missing_fred_api_key_degrades_fred_series_only(self):
        # Gold (yfinance) and cot need no FRED key, so only the FRED-backed
        # aliases go unavailable when FRED_API_KEY is absent.
        with mock.patch.dict("os.environ", {}, clear=True), \
                mock.patch.object(macro_pack, "get_price_history_points", return_value=_gold_points()):
            pack = macro_pack.get_macro_pack("2026-06-01")

        indicators = pack["indicators"]
        for alias in macro_pack.FRED_INDICATOR_ALIASES:
            self.assertFalse(indicators[alias]["available"], msg=alias)
            self.assertIn("FRED_API_KEY", indicators[alias]["error"])
        self.assertTrue(indicators["gold"]["available"])

    def test_frequency_alignment_is_per_series_independent(self):
        # A monthly series and a daily series get independently fetched and
        # evaluated as-of curr_date; the pack is a per-series snapshot, not a
        # merged/resampled single-frequency table, so each keeps its own
        # native cadence and observation count.
        monthly_obs = _make_observations(24, start_date="2024-06-01", step_days=30)
        daily_obs = _make_observations(40, start_date="2026-04-01", step_days=1)
        overrides = {
            "CPIAUCSL": {"obs": monthly_obs},   # cpi: monthly
            "VIXCLS": {"obs": daily_obs},        # vix: daily
        }
        with mock.patch.object(fred, "_request", side_effect=_stub_request(overrides)), \
                mock.patch.object(macro_pack, "get_price_history_points", return_value=_gold_points()):
            pack = macro_pack.get_macro_pack("2026-06-01")

        indicators = pack["indicators"]
        cpi_entry = indicators["cpi"]
        vix_entry = indicators["vix"]

        # Each series' own last observation date is preserved, not aligned to
        # a shared date.
        self.assertEqual(cpi_entry["latest_date"], monthly_obs["observations"][-1]["date"])
        self.assertEqual(vix_entry["latest_date"], daily_obs["observations"][-1]["date"])
        self.assertNotEqual(cpi_entry["latest_date"], vix_entry["latest_date"])
        # Both have enough history (24 and 40 obs) for a real z-score, capped
        # at the trailing 24 observations independently.
        self.assertEqual(cpi_entry["observations_used"], 24)
        self.assertEqual(vix_entry["observations_used"], 24)
        self.assertIsNotNone(cpi_entry["z_score"])
        self.assertIsNotNone(vix_entry["z_score"])

    def test_invalid_curr_date_raises(self):
        with self.assertRaises(ValueError):
            macro_pack.get_macro_pack("not-a-date")


@pytest.mark.unit
class GetMacroPackToolTests(unittest.TestCase):
    """The @tool wrapper in tradingagents/agents/utils/macro_data_tools.py."""

    def test_dict_result_is_json_serialized(self):
        from tradingagents.agents.utils import macro_data_tools

        fake_pack = {"curr_date": "2026-06-01", "indicators": {"cpi": {"available": True}}}
        with mock.patch.object(macro_data_tools, "route_to_vendor", return_value=fake_pack):
            out = macro_data_tools.get_macro_pack.func("2026-06-01")

        import json
        self.assertEqual(json.loads(out), fake_pack)

    def test_string_sentinel_passes_through_unchanged(self):
        from tradingagents.agents.utils import macro_data_tools

        sentinel = "DATA_UNAVAILABLE: optional macro_data could not be retrieved"
        with mock.patch.object(macro_data_tools, "route_to_vendor", return_value=sentinel):
            out = macro_data_tools.get_macro_pack.func("2026-06-01")

        self.assertEqual(out, sentinel)


if __name__ == "__main__":
    unittest.main()
