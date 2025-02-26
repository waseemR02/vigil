"""Tests for time utility functions."""

import pytest
import datetime
import pytz
from vigil.utils.time_utils import (parse_date, normalize_date, parse_relative_time,
                                   get_date_difference, format_timedelta)


class TestTimeUtils:
    """Test cases for time utilities."""

    def test_parse_date(self, sample_dates):
        """Test date parsing from various formats."""
        # Test standard ISO format
        dt = parse_date("2023-07-15T14:30:00Z")
        assert dt is not None
        assert dt.year == 2023
        assert dt.month == 7
        assert dt.day == 15
        assert dt.hour == 14
        assert dt.minute == 30
        assert dt.tzinfo is not None
        
        # Test date-only format
        dt = parse_date("2023-07-15")
        assert dt is not None
        assert dt.year == 2023
        assert dt.month == 7
        assert dt.day == 15
        
        # Test US format
        dt = parse_date("07/15/2023")
        assert dt is not None
        assert dt.year == 2023
        assert dt.month == 7
        assert dt.day == 15
        
        # Test named month format
        dt = parse_date("July 15, 2023")
        assert dt is not None
        assert dt.year == 2023
        assert dt.month == 7
        assert dt.day == 15
        
        # Test invalid format
        dt = parse_date("invalid date")
        assert dt is None
        
        # Test empty input
        dt = parse_date("")
        assert dt is None

    def test_normalize_date(self):
        """Test date normalization."""
        # Create a datetime with timezone for testing
        dt = datetime.datetime(2023, 7, 15, 14, 30, 0, tzinfo=pytz.UTC)
        
        # Test default format (ISO)
        normalized = normalize_date(dt)
        assert normalized == "2023-07-15T14:30:00+0000"
        
        # Test custom format
        normalized = normalize_date(dt, format_str="%Y-%m-%d %H:%M:%S")
        assert normalized == "2023-07-15 14:30:00"
        
        # Test timezone conversion
        normalized = normalize_date(dt, target_timezone="America/New_York", 
                                   format_str="%Y-%m-%d %H:%M:%S %Z")
        assert "EDT" in normalized or "EST" in normalized
        assert "10:30:00" in normalized or "09:30:00" in normalized  # Depends on DST
        
        # Test None input
        normalized = normalize_date(None)
        assert normalized == ""

    def test_parse_relative_time(self, reference_date):
        """Test parsing of relative time expressions."""
        # Test minutes ago
        dt = parse_relative_time("5 minutes ago", reference_date)
        assert dt is not None
        assert (reference_date - dt).total_seconds() == 5 * 60
        
        # Test hours ago
        dt = parse_relative_time("2 hours ago", reference_date)
        assert dt is not None
        assert (reference_date - dt).total_seconds() == 2 * 3600
        
        # Test days ago
        dt = parse_relative_time("3 days ago", reference_date)
        assert dt is not None
        assert (reference_date - dt).total_seconds() == 3 * 86400
        
        # Test weeks ago
        dt = parse_relative_time("1 week ago", reference_date)
        assert dt is not None
        assert (reference_date - dt).total_seconds() == 7 * 86400
        
        # Test months ago (approximate)
        dt = parse_relative_time("2 months ago", reference_date)
        assert dt is not None
        assert (reference_date - dt).days == 60  # Approximate
        
        # Test years ago (approximate)
        dt = parse_relative_time("1 year ago", reference_date)
        assert dt is not None
        assert (reference_date - dt).days == 365  # Approximate
        
        # Test yesterday
        dt = parse_relative_time("yesterday", reference_date)
        assert dt is not None
        assert (reference_date - dt).days == 1
        
        # Test last week
        dt = parse_relative_time("last week", reference_date)
        assert dt is not None
        assert (reference_date - dt).days == 7
        
        # Test invalid format
        dt = parse_relative_time("not a relative time", reference_date)
        assert dt is None
        
        # Test empty input
        dt = parse_relative_time("", reference_date)
        assert dt is None
        
        # Test with None reference date (should use current time)
        dt = parse_relative_time("5 minutes ago")
        assert dt is not None
        # Hard to test exact value since it uses current time

    def test_get_date_difference(self):
        """Test date difference calculation."""
        # Test with datetime objects
        date1 = datetime.datetime(2023, 7, 15, 12, 0, 0, tzinfo=pytz.UTC)
        date2 = datetime.datetime(2023, 7, 20, 12, 0, 0, tzinfo=pytz.UTC)
        diff = get_date_difference(date1, date2)
        assert diff is not None
        assert diff.days == 5
        
        # Test with string dates
        diff = get_date_difference("2023-07-15T12:00:00Z", "2023-07-20T12:00:00Z")
        assert diff is not None
        assert diff.days == 5
        
        # Test with mixed types
        diff = get_date_difference(date1, "2023-07-20T12:00:00Z")
        assert diff is not None
        assert diff.days == 5
        
        # Test with invalid string
        diff = get_date_difference("invalid date", date2)
        assert diff is None
        
        # Test with one empty string
        diff = get_date_difference("", date2)
        assert diff is None

    def test_format_timedelta(self):
        """Test timedelta formatting."""
        # Create timedeltas for testing
        one_day = datetime.timedelta(days=1)
        complex_time = datetime.timedelta(days=2, hours=5, minutes=30, seconds=15)
        small_time = datetime.timedelta(minutes=10, seconds=30)
        
        # Test with days only
        formatted = format_timedelta(one_day)
        assert formatted == "1 day"
        
        # Test with multiple components - long format
        formatted = format_timedelta(complex_time)
        assert formatted == "2 days 5 hours 30 minutes"
        
        # Test with multiple components - short format
        formatted = format_timedelta(complex_time, short_format=True)
        assert formatted == "2d 5h 30m"
        
        # Test with small time
        formatted = format_timedelta(small_time)
        assert formatted == "10 minutes 30 seconds"
        formatted = format_timedelta(small_time, short_format=True)
        assert formatted == "10m 30s"
        
        # Test with zero
        formatted = format_timedelta(datetime.timedelta())
        assert formatted == "0 seconds"
        formatted = format_timedelta(datetime.timedelta(), short_format=True)
        assert formatted == "0s"
        
        # Test with None
        formatted = format_timedelta(None)
        assert formatted == ""
