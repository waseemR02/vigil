"""Tests for URL utility functions."""

import pytest
from vigil.utils.url_utils import (normalize_url, extract_domain, is_valid_url,
                                  get_url_params, update_url_params, join_url_path)


class TestUrlUtils:
    """Test cases for URL utilities."""

    def test_normalize_url(self):
        """Test URL normalization."""
        # Test scheme and host normalization
        assert normalize_url("HTTP://EXAMPLE.COM") == "http://example.com"
        
        # Test trailing slash removal
        assert normalize_url("http://example.com/path/") == "http://example.com/path"
        
        # Test port removal
        assert normalize_url("http://example.com:80/") == "http://example.com"
        assert normalize_url("https://example.com:443/") == "https://example.com"
        
        # Test www removal
        assert normalize_url("http://www.example.com") == "http://example.com"
        
        # Test query parameter sorting
        assert normalize_url("http://example.com?b=2&a=1") == "http://example.com?a=1&b=2"
        
        # Test empty URL
        assert normalize_url("") == ""
        
        # Test complete normalization
        assert normalize_url("HTTP://WWW.EXAMPLE.COM:80/path/?b=2&a=1") == "http://example.com/path?a=1&b=2"

    def test_extract_domain(self):
        """Test domain extraction."""
        assert extract_domain("http://www.example.com/path") == "example.com"
        assert extract_domain("https://sub.example.co.uk") == "example.co.uk"
        assert extract_domain("http://example.com:8080") == "example.com"
        assert extract_domain("") == ""
        assert extract_domain("http://localhost") == "localhost"
        assert extract_domain("http://sub.domain.example.com") == "example.com"

    def test_is_valid_url(self):
        """Test URL validation."""
        assert is_valid_url("http://example.com")
        assert is_valid_url("https://example.com/path")
        assert not is_valid_url("http://")
        assert not is_valid_url("example.com")
        assert not is_valid_url("")
        assert not is_valid_url("just text")

    def test_get_url_params(self):
        """Test URL parameter extraction."""
        assert get_url_params("http://example.com?a=1&b=2") == {"a": "1", "b": "2"}
        assert get_url_params("http://example.com") == {}
        assert get_url_params("") == {}
        assert get_url_params("http://example.com?param=value&empty=") == {"param": "value", "empty": ""}

    def test_update_url_params(self):
        """Test URL parameter updates."""
        # Test adding parameters
        assert update_url_params(
            "http://example.com", 
            {"a": "1", "b": "2"}
        ) == "http://example.com?a=1&b=2"
        
        # Test updating parameters
        assert update_url_params(
            "http://example.com?a=1&b=2", 
            {"a": "3", "c": "4"}
        ) == "http://example.com?a=3&b=2&c=4"
        
        # Test removing parameters
        assert update_url_params(
            "http://example.com?a=1&b=2&c=3", 
            {}, 
            ["b"]
        ) == "http://example.com?a=1&c=3"
        
        # Test empty URL
        assert update_url_params("", {"a": "1"}) == ""

    def test_join_url_path(self):
        """Test URL path joining."""
        assert join_url_path("http://example.com", "path") == "http://example.com/path"
        assert join_url_path("http://example.com/", "path") == "http://example.com/path"
        assert join_url_path("http://example.com", "/path") == "http://example.com/path"
        assert join_url_path("http://example.com/", "/path") == "http://example.com/path"
        assert join_url_path("", "path") == "path"
        assert join_url_path("http://example.com", "") == "http://example.com"
