"""Test configuration and fixtures for Vigil."""

import os
import sys
import pytest
from datetime import datetime, timezone

@pytest.fixture
def sample_html():
    """Return sample HTML content for testing."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Sample Page</title>
        <meta charset="utf-8">
        <style>
            body { font-family: Arial; }
        </style>
        <script>
            function test() { console.log('test'); }
        </script>
    </head>
    <body>
        <h1>Main Heading</h1>
        <p>This is a paragraph with <strong>bold text</strong> and <a href="https://example.com">a link</a>.</p>
        <p>Second paragraph with some additional text.</p>
        <ul>
            <li>List item 1</li>
            <li>List item 2</li>
        </ul>
        <!-- Comment -->
    </body>
    </html>
    """


@pytest.fixture
def sample_urls():
    """Return a list of sample URLs for testing."""
    return [
        "http://example.com",
        "https://www.example.com/",
        "https://example.com/path/to/page?q=query&sort=desc",
        "https://subdomain.example.co.uk/path?a=1&b=2#fragment",
        "http://example.com:80/path/",
        "https://user:pass@example.com/path",
        "http://invalid",
        ""
    ]


@pytest.fixture
def sample_dates():
    """Return a list of sample dates in various formats for testing."""
    return [
        "2023-07-15",
        "07/15/2023",
        "July 15, 2023",
        "15-Jul-2023",
        "2023-07-15T14:30:00Z",
        "2023-07-15T14:30:00+02:00",
        "Tue, 15 Jul 2023 14:30:00 GMT",
        "yesterday",
        "2 days ago",
        "last week",
        "invalid date"
    ]


@pytest.fixture
def reference_date():
    """Return a fixed reference date for testing relative dates."""
    return datetime(2023, 7, 20, 12, 0, 0, tzinfo=timezone.utc)
