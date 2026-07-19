"""Tests for get_earnings_calendar vendor function."""

import unittest
from datetime import datetime, timedelta
from unittest import mock

import pandas as pd
import pytz
import pytest

from tradingagents.dataflows import y_finance


@pytest.mark.unit
class TestGetEarningsCalendarYFinance(unittest.TestCase):
    def test_known_future_earnings_date(self):
        """Test that a known future earnings date is returned correctly."""
        # Mock earnings data: one future and one past date
        future_date = datetime.now(pytz.UTC) + timedelta(days=10)
        past_date = datetime.now(pytz.UTC) - timedelta(days=30)

        # Create a DataFrame with datetime index (with timezone, as yfinance returns)
        earnings_index = pd.DatetimeIndex(
            [future_date, past_date],
            tz="UTC"
        )
        earnings_df = pd.DataFrame(
            {
                "EPS Estimate": [1.5, 1.2],
                "Reported EPS": [float('nan'), 1.18],
                "Surprise(%)": [float('nan'), -1.67],
            },
            index=earnings_index
        )

        curr_date = datetime.now().strftime("%Y-%m-%d")

        with mock.patch.object(y_finance.yf, "Ticker") as mock_ticker_class:
            mock_ticker = mock.Mock()
            mock_ticker_class.return_value = mock_ticker
            mock_ticker.get_earnings_dates.return_value = earnings_df

            result = y_finance.get_earnings_calendar("AAPL", curr_date)

        self.assertIn("# Earnings Calendar for AAPL", result)
        self.assertIn("Next Earnings Date:", result)
        self.assertIn("Days Until Next Earnings:", result)
        self.assertIn("Most Recent Past Earnings Date:", result)
        # Verify that days until is reasonable (between 1 and 20)
        lines = result.split("\n")
        for line in lines:
            if "Days Until Next Earnings:" in line:
                days_str = line.split(":")[-1].strip()
                days = int(days_str)
                self.assertGreaterEqual(days, 1)
                self.assertLess(days, 20)

    def test_no_earnings_data_available(self):
        """Test that no earnings data returns an 'UNKNOWN' sentinel."""
        curr_date = "2026-07-19"

        with mock.patch.object(y_finance.yf, "Ticker") as mock_ticker_class:
            mock_ticker = mock.Mock()
            mock_ticker_class.return_value = mock_ticker
            mock_ticker.get_earnings_dates.return_value = pd.DataFrame()

            result = y_finance.get_earnings_calendar("UNKNOWNTICKER", curr_date)

        self.assertIn("UNKNOWN", result)
        self.assertIn("No earnings calendar data available", result)

    def test_non_equity_symbol_futures(self):
        """Test that futures symbols return 'NO_EARNINGS_CALENDAR_AVAILABLE'."""
        curr_date = "2026-07-19"

        # GC=F is gold futures
        result = y_finance.get_earnings_calendar("XAUUSD", curr_date)

        self.assertIn("NO_EARNINGS_CALENDAR_AVAILABLE", result)
        self.assertIn("non-equity", result)
        self.assertIn("commodity", result)

    def test_non_equity_symbol_forex(self):
        """Test that forex symbols return 'NO_EARNINGS_CALENDAR_AVAILABLE'."""
        curr_date = "2026-07-19"

        # EURUSD=X is a forex pair
        result = y_finance.get_earnings_calendar("EURUSD", curr_date)

        self.assertIn("NO_EARNINGS_CALENDAR_AVAILABLE", result)
        self.assertIn("non-equity", result)
        self.assertIn("forex", result)

    def test_non_equity_symbol_crypto(self):
        """Test that crypto symbols return 'NO_EARNINGS_CALENDAR_AVAILABLE'."""
        curr_date = "2026-07-19"

        # BTC-USD is crypto
        result = y_finance.get_earnings_calendar("BTCUSD", curr_date)

        self.assertIn("NO_EARNINGS_CALENDAR_AVAILABLE", result)
        self.assertIn("non-equity", result)
        self.assertIn("crypto", result)

    def test_invalid_date_format(self):
        """Test that an invalid date format returns 'UNKNOWN' sentinel."""
        with mock.patch.object(y_finance.yf, "Ticker"):
            result = y_finance.get_earnings_calendar("AAPL", "invalid-date")

        self.assertIn("UNKNOWN", result)
        self.assertIn("Invalid current date format", result)

    def test_vendor_error_graceful_degradation(self):
        """Test that vendor errors degrade to 'UNKNOWN' sentinel."""
        curr_date = "2026-07-19"

        with mock.patch.object(y_finance.yf, "Ticker") as mock_ticker_class:
            mock_ticker = mock.Mock()
            mock_ticker_class.return_value = mock_ticker
            mock_ticker.get_earnings_dates.side_effect = Exception("Network error")

            result = y_finance.get_earnings_calendar("AAPL", curr_date)

        self.assertIn("UNKNOWN", result)
        self.assertIn("Error retrieving earnings calendar", result)
        self.assertIn("Network error", result)

    def test_most_recent_past_earnings(self):
        """Test that the most recent past earnings date is included."""
        # Create multiple past and future dates
        now = datetime.now(pytz.UTC)
        dates_to_include = [
            now + timedelta(days=30),  # Future
            now + timedelta(days=60),  # Future
            now - timedelta(days=5),   # Recent past
            now - timedelta(days=35),  # Older past
        ]

        earnings_index = pd.DatetimeIndex(dates_to_include, tz="UTC")
        earnings_df = pd.DataFrame(
            {
                "EPS Estimate": [1.5, 1.4, 1.2, 1.1],
                "Reported EPS": [float('nan'), float('nan'), 1.18, 1.08],
                "Surprise(%)": [float('nan'), float('nan'), -1.67, -1.82],
            },
            index=earnings_index
        )

        curr_date = now.strftime("%Y-%m-%d")

        with mock.patch.object(y_finance.yf, "Ticker") as mock_ticker_class:
            mock_ticker = mock.Mock()
            mock_ticker_class.return_value = mock_ticker
            mock_ticker.get_earnings_dates.return_value = earnings_df

            result = y_finance.get_earnings_calendar("AAPL", curr_date)

        self.assertIn("Most Recent Past Earnings Date:", result)
        # The most recent past should be 5 days ago
        lines = result.split("\n")
        for line in lines:
            if "Most Recent Past Earnings Date:" in line:
                # Parse the date from the line
                date_str = line.split(":")[-1].strip()
                past_date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
                current_date_obj = datetime.strptime(curr_date, "%Y-%m-%d").date()
                days_diff = (current_date_obj - past_date_obj).days
                self.assertGreaterEqual(days_diff, 5)
                self.assertLess(days_diff, 10)

    def test_no_future_earnings_only_past(self):
        """Test behavior when only past earnings are available."""
        # Create only past dates
        now = datetime.now(pytz.UTC)
        dates_to_include = [
            now - timedelta(days=5),
            now - timedelta(days=35),
        ]

        earnings_index = pd.DatetimeIndex(dates_to_include, tz="UTC")
        earnings_df = pd.DataFrame(
            {
                "EPS Estimate": [1.2, 1.1],
                "Reported EPS": [1.18, 1.08],
                "Surprise(%)": [-1.67, -1.82],
            },
            index=earnings_index
        )

        curr_date = now.strftime("%Y-%m-%d")

        with mock.patch.object(y_finance.yf, "Ticker") as mock_ticker_class:
            mock_ticker = mock.Mock()
            mock_ticker_class.return_value = mock_ticker
            mock_ticker.get_earnings_dates.return_value = earnings_df

            result = y_finance.get_earnings_calendar("AAPL", curr_date)

        self.assertIn("Next Earnings Date: Not scheduled", result)
        self.assertIn("Most Recent Past Earnings Date:", result)


if __name__ == "__main__":
    unittest.main()
