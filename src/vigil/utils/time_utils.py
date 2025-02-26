"""Date and time utilities for the Vigil monitoring system."""

import re
import datetime
from typing import Optional, Union
from dateutil import parser
import pytz


def parse_date(date_str: str, default_timezone: str = 'UTC') -> Optional[datetime.datetime]:
    """
    Parse a date string in various formats to a datetime object.
    
    Args:
        date_str: The date string to parse
        default_timezone: Timezone to use if none specified in the date string
        
    Returns:
        Datetime object or None if parsing fails
    """
    if not date_str:
        return None
    
    try:
        # Use dateutil parser which handles many formats
        dt = parser.parse(date_str)
        
        # If no timezone info was in the string, add the default timezone
        if dt.tzinfo is None:
            tz = pytz.timezone(default_timezone)
            dt = tz.localize(dt)
            
        return dt
    except (ValueError, parser.ParserError):
        return None


def normalize_date(dt: datetime.datetime, target_timezone: str = 'UTC', 
                  format_str: str = '%Y-%m-%dT%H:%M:%S%z') -> str:
    """
    Normalize a datetime to a standard string format.
    
    Args:
        dt: Datetime object to normalize
        target_timezone: Target timezone for the result
        format_str: Output format string
        
    Returns:
        Normalized date string
    """
    if not dt:
        return ""
    
    # Convert to the target timezone
    if dt.tzinfo is None:
        dt = pytz.utc.localize(dt)
        
    target_tz = pytz.timezone(target_timezone)
    normalized_dt = dt.astimezone(target_tz)
    
    # Format the datetime
    return normalized_dt.strftime(format_str)


def parse_relative_time(relative_time: str, reference_date: Optional[datetime.datetime] = None) -> Optional[datetime.datetime]:
    """
    Parse a relative time expression like "2 days ago" to an absolute datetime.
    
    Args:
        relative_time: Relative time string
        reference_date: Reference date for the relative time (default: current time)
        
    Returns:
        Datetime object or None if parsing fails
    """
    if not relative_time:
        return None
    
    # Use current time if no reference date provided
    if reference_date is None:
        reference_date = datetime.datetime.now(datetime.timezone.utc)
    elif reference_date.tzinfo is None:
        reference_date = pytz.utc.localize(reference_date)
    
    # Common relative time patterns
    patterns = [
        # "X minutes ago"
        (r'(\d+)\s*minute[s]?\s*ago', lambda x: reference_date - datetime.timedelta(minutes=int(x))),
        # "X hours ago"
        (r'(\d+)\s*hour[s]?\s*ago', lambda x: reference_date - datetime.timedelta(hours=int(x))),
        # "X days ago"
        (r'(\d+)\s*day[s]?\s*ago', lambda x: reference_date - datetime.timedelta(days=int(x))),
        # "X weeks ago"
        (r'(\d+)\s*week[s]?\s*ago', lambda x: reference_date - datetime.timedelta(weeks=int(x))),
        # "X months ago" (approximate)
        (r'(\d+)\s*month[s]?\s*ago', lambda x: reference_date - datetime.timedelta(days=int(x)*30)),
        # "X years ago" (approximate)
        (r'(\d+)\s*year[s]?\s*ago', lambda x: reference_date - datetime.timedelta(days=int(x)*365)),
        # "yesterday"
        (r'yesterday', lambda x: reference_date - datetime.timedelta(days=1)),
        # "last week"
        (r'last\s*week', lambda x: reference_date - datetime.timedelta(weeks=1)),
        # "last month"
        (r'last\s*month', lambda x: reference_date - datetime.timedelta(days=30)),
    ]
    
    # Try each pattern
    for pattern, time_func in patterns:
        match = re.search(pattern, relative_time, re.IGNORECASE)
        if match:
            try:
                if len(match.groups()) > 0:
                    return time_func(match.group(1))
                else:
                    return time_func(None)
            except ValueError:
                continue
    
    # No pattern matched
    return None


def get_date_difference(date1: Union[str, datetime.datetime], 
                       date2: Union[str, datetime.datetime]) -> Optional[datetime.timedelta]:
    """
    Calculate the difference between two dates.
    
    Args:
        date1: First date (string or datetime)
        date2: Second date (string or datetime)
        
    Returns:
        Timedelta object representing the difference or None if parsing fails
    """
    # Parse dates if they are strings
    if isinstance(date1, str):
        date1 = parse_date(date1)
        if date1 is None:
            return None
            
    if isinstance(date2, str):
        date2 = parse_date(date2)
        if date2 is None:
            return None
    
    # Calculate difference
    return abs(date1 - date2)


def format_timedelta(delta: datetime.timedelta, 
                    short_format: bool = False) -> str:
    """
    Format a timedelta to a human-readable string.
    
    Args:
        delta: Timedelta object to format
        short_format: Whether to use short format (e.g., "2d 5h") or long format (e.g., "2 days 5 hours")
        
    Returns:
        Formatted time difference string
    """
    if delta is None:
        return ""
    
    # Extract components
    days = delta.days
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    parts = []
    if days:
        if short_format:
            parts.append(f"{days}d")
        else:
            parts.append(f"{days} day{'s' if days != 1 else ''}")
    
    if hours:
        if short_format:
            parts.append(f"{hours}h")
        else:
            parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    
    if minutes:
        if short_format:
            parts.append(f"{minutes}m")
        else:
            parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    
    if seconds and not (days or hours):
        if short_format:
            parts.append(f"{seconds}s")
        else:
            parts.append(f"{seconds} second{'s' if seconds != 1 else ''}")
    
    if not parts:
        if short_format:
            return "0s"
        else:
            return "0 seconds"
    
    return " ".join(parts)
