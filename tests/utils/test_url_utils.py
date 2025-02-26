"""Unit tests for URL utilities."""

import pytest
from vigil.utils.url_utils import (
    normalize_url, extract_domain, is_valid_url,
    get_url_params, update_url_params, join_url_path
)


class TestUrlUtils:
    """Test cases for URL utilities."""
    
    def test_normalize_url(self):
        """Test URL normalization."""
        # Test scheme and host normalization
        assert normalize_url("HTTP://EXAMPLE.COM") == "http://example.com/"
        
        # Test trailing slash removal for paths (but not for root path)
        assert normalize_url("http://example.com/path/") == "http://example.com/path"
        assert normalize_url("http://example.com/") == "http://example.com/"
        
        # Test empty path becomes root path
        assert normalize_url("http://example.com") == "http://example.com/"
        
        # Test www prefix removal
        assert normalize_url("http://www.example.com") == "http://example.com/"
        
        # Test default port removal
        assert normalize_url("http://example.com:80") == "http://example.com/"
        assert normalize_url("https://example.com:443") == "https://example.com/"
        
        # Test non-default port preservation
        assert normalize_url("http://example.com:8080") == "http://example.com:8080/"
        
        # Test query parameter sorting
        assert normalize_url("http://example.com/?b=2&a=1") == "http://example.com/?a=1&b=2"
        
        # Test empty URL
        assert normalize_url("") == ""
    
    def test_extract_domain(self):
        """Test domain extraction."""
        assert extract_domain("https://www.example.com/path") == "example.com"
        assert extract_domain("https://blog.example.co.uk") == "example.co.uk"
        assert extract_domain("https://test.github.com") == "github.com"
        assert extract_domain("") == ""
    
    def test_is_valid_url(self):
        """Test URL validation."""
        assert is_valid_url("https://example.com")
        assert is_valid_url("http://localhost:8080")
        assert not is_valid_url("")
        assert not is_valid_url("example.com")  # Missing scheme
        assert not is_valid_url("https://")  # Missing netloc
    
    def test_get_url_params(self):
        """Test URL parameter extraction."""
        url = "https://example.com/search?q=test&page=2&filter="
        params = get_url_params(url)
        assert params == {"q": "test", "page": "2", "filter": ""}
        assert get_url_params("") == {}
    
    def test_update_url_params(self):
        """Test URL parameter updates."""
        url = "https://example.com/search?q=test&page=1"
        
        # Add and update parameters
        updated = update_url_params(url, {"page": "2", "sort": "date"})
        assert "q=test" in updated
        assert "page=2" in updated
        assert "sort=date" in updated
        
        # Remove parameters
        updated = update_url_params(url, {"sort": "date"}, ["q"])
        assert "q=test" not in updated
        assert "page=1" in updated
        assert "sort=date" in updated
        
        # Empty URL
        assert update_url_params("", {}) == ""
    
    def test_join_url_path(self):
        """Test URL path joining."""
        assert join_url_path("https://example.com", "path") == "https://example.com/path"
        assert join_url_path("https://example.com/", "/path") == "https://example.com/path"
        assert join_url_path("https://example.com", "") == "https://example.com"
        assert join_url_path("", "path") == "path"
