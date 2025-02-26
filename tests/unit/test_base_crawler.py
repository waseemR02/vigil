"""Unit tests for the BaseCrawler class using pytest."""

import os
import time
import pytest
from unittest import mock
import requests
from requests.exceptions import ConnectionError, Timeout, HTTPError, RequestException

from vigil.data_collection.base import BaseCrawler


class MockResponse:
    """Mock response object for testing."""
    
    def __init__(self, status_code=200, content=b"", text="", headers=None, url="https://example.com"):
        self.status_code = status_code
        self.content = content
        self.text = text
        self.headers = headers or {"Content-Type": "text/html"}
        self.url = url
        self._content_consumed = True
    
    def raise_for_status(self):
        """Raise an exception if status code indicates an error."""
        if 400 <= self.status_code < 600:
            raise HTTPError(f"HTTP Error: {self.status_code}")
    
    def iter_content(self, chunk_size=1):
        """Mock iterator for content streaming."""
        remaining = len(self.content)
        pos = 0
        while remaining > 0:
            size = min(chunk_size, remaining)
            yield self.content[pos:pos+size]
            pos += size
            remaining -= size
    
    def json(self):
        """Return JSON data."""
        import json
        try:
            return json.loads(self.text)
        except Exception:
            raise ValueError("Invalid JSON")


@pytest.fixture
def crawler():
    """Create a BaseCrawler instance for testing."""
    with mock.patch("urllib.robotparser.RobotFileParser") as mock_robot_parser:
        # Configure mock robot parser to always allow access
        mock_parser_instance = mock.Mock()
        mock_robot_parser.return_value = mock_parser_instance
        mock_parser_instance.can_fetch.return_value = True
        
        crawler = BaseCrawler(
            respect_robots_txt=True,
            delay=0.01,  # Small delay for testing
            min_delay=0.01,
            max_delay=0.01
        )
        
        yield crawler
        
        # Clean up
        crawler.close()


def test_init_default_parameters():
    """Test initialization with default parameters."""
    crawler = BaseCrawler()
    assert crawler.delay == 1.0
    assert crawler.min_delay == 0.5
    assert crawler.max_delay == 3.0
    assert crawler.timeout == 10
    assert crawler.max_retries == 3
    assert crawler.verify_ssl is True
    assert crawler.respect_robots_txt is True
    assert crawler.user_agent in BaseCrawler.DEFAULT_USER_AGENTS
    assert isinstance(crawler.session, requests.Session)
    assert "User-Agent" in crawler.headers


def test_init_custom_parameters():
    """Test initialization with custom parameters."""
    custom_ua = "Custom User Agent"
    custom_headers = {"X-Test": "Value"}
    custom_proxies = {"http": "http://proxy.example.com"}
    
    crawler = BaseCrawler(
        delay=2.0,
        min_delay=1.0,
        max_delay=5.0,
        timeout=20,
        max_retries=5,
        verify_ssl=False,
        respect_robots_txt=False,
        user_agent=custom_ua,
        headers=custom_headers,
        proxies=custom_proxies
    )
    
    assert crawler.delay == 2.0
    assert crawler.min_delay == 1.0
    assert crawler.max_delay == 5.0
    assert crawler.timeout == 20
    assert crawler.max_retries == 5
    assert crawler.verify_ssl is False
    assert crawler.respect_robots_txt is False
    assert crawler.user_agent == custom_ua
    assert crawler.headers["User-Agent"] == custom_ua
    assert crawler.headers["X-Test"] == "Value"
    assert crawler.proxies == custom_proxies


def test_create_session():
    """Test session creation with retry configuration."""
    crawler = BaseCrawler(max_retries=3)
    session = crawler.session
    
    # Check that session has adapters for both HTTP and HTTPS
    assert "http://" in session.adapters
    assert "https://" in session.adapters
    
    # Examine the retry configuration
    http_adapter = session.adapters["http://"]
    retry_config = http_adapter.max_retries
    
    assert retry_config.total == 3
    assert retry_config.backoff_factor == 1.0
    assert 429 in retry_config.status_forcelist
    assert 500 in retry_config.status_forcelist


def test_respect_robots_enabled(crawler):
    """Test respecting robots.txt when enabled."""
    # Get the mocked robot parser from the fixture
    robot_parser = crawler.robots_cache.get("https://example.com", None)
    
    # Set up the mock to disallow a specific URL
    with mock.patch.object(crawler, "_respect_robots") as mock_respect:
        mock_respect.return_value = False
        
        # Try to request the disallowed URL
        with mock.patch("requests.Session.request") as mock_request:
            response = crawler.get("https://example.com/disallowed")
            
            # Check that the request was not made
            mock_request.assert_not_called()
            assert response is None


def test_respect_robots_disabled():
    """Test that robots.txt is ignored when disabled."""
    crawler = BaseCrawler(respect_robots_txt=False)
    
    # Even with a URL that would be disallowed, request should proceed
    with mock.patch("requests.Session.request") as mock_request:
        mock_request.return_value = MockResponse(status_code=200)
        response = crawler.get("https://example.com/disallowed")
        
        # Check that the request was made despite robots.txt
        mock_request.assert_called_once()
        assert response is not None


@mock.patch("time.sleep")  # Mock sleep to speed up tests
def test_enforce_delay(mock_sleep, crawler):
    """Test that delay is enforced between requests."""
    # Make first request (should not delay)
    with mock.patch("requests.Session.request") as mock_request:
        mock_request.return_value = MockResponse()
        crawler.get("https://example.com/page1")
        mock_sleep.assert_not_called()
    
    # Make second request immediately (should enforce delay)
    with mock.patch("requests.Session.request") as mock_request:
        mock_request.return_value = MockResponse()
        crawler.get("https://example.com/page2")
        mock_sleep.assert_called_once()


def test_change_user_agent(crawler):
    """Test changing the user agent."""
    original_ua = crawler.user_agent
    
    # Change to specific user agent
    new_ua = "New User Agent String"
    crawler.change_user_agent(new_ua)
    assert crawler.user_agent == new_ua
    assert crawler.headers["User-Agent"] == new_ua
    
    # Change to random user agent but explicitly ensure it's different
    # We need to modify the DEFAULT_USER_AGENTS list for the test to ensure we get a different one
    with mock.patch.object(BaseCrawler, 'DEFAULT_USER_AGENTS', 
                          ["Different UA 1", "Different UA 2", "Different UA 3"]):
        crawler.change_user_agent()
        assert crawler.user_agent in BaseCrawler.DEFAULT_USER_AGENTS
        assert crawler.user_agent != new_ua  # Compare with the previous value instead of original
        assert crawler.headers["User-Agent"] == crawler.user_agent


@pytest.mark.parametrize("method", ["GET", "POST", "PUT", "DELETE"])
def test_request_method(method, crawler):
    """Test making requests with different HTTP methods."""
    url = "https://example.com/api"
    
    with mock.patch("requests.Session.request") as mock_request:
        mock_request.return_value = MockResponse(status_code=200)
        response = crawler.request(url=url, method=method)
        
        mock_request.assert_called_once()
        kwargs = mock_request.call_args[1]
        assert kwargs["method"] == method
        assert kwargs["url"] == url


@pytest.mark.parametrize("status_code,should_succeed", [
    (200, True),
    (201, True),
    (301, True),  # Redirects should succeed by default
    (400, False),
    (404, False),
    (500, False),
    (503, False)
])
def test_request_status_codes(status_code, should_succeed, crawler):
    """Test handling of different HTTP status codes."""
    with mock.patch("requests.Session.request") as mock_request:
        mock_response = MockResponse(status_code=status_code)
        mock_request.return_value = mock_response
        
        response = crawler.get("https://example.com/page")
        
        if should_succeed:
            assert response is not None
            assert response.status_code == status_code
        else:
            assert response is None


@pytest.mark.parametrize("exception_class", [
    ConnectionError, 
    Timeout, 
    HTTPError, 
    RequestException, 
    Exception
])
def test_request_exceptions(exception_class, crawler):
    """Test handling of various request exceptions."""
    with mock.patch("requests.Session.request") as mock_request:
        mock_request.side_effect = exception_class("Test error")
        
        response = crawler.get("https://example.com/page")
        
        assert response is None


def test_get_request(crawler):
    """Test the GET request helper method."""
    url = "https://example.com/page"
    params = {"param1": "value1"}
    custom_headers = {"X-Custom": "Value"}
    
    with mock.patch("requests.Session.request") as mock_request:
        mock_request.return_value = MockResponse(status_code=200)
        
        response = crawler.get(
            url=url,
            params=params,
            custom_headers=custom_headers,
            timeout=5,
            allow_redirects=False
        )
        
        mock_request.assert_called_once()
        kwargs = mock_request.call_args[1]
        assert kwargs["method"] == "GET"
        assert kwargs["url"] == url
        assert kwargs["params"] == params
        assert "X-Custom" in kwargs["headers"]
        assert kwargs["timeout"] == 5
        assert kwargs["allow_redirects"] is False


def test_post_request(crawler):
    """Test the POST request helper method."""
    url = "https://example.com/api"
    data = {"field1": "value1"}
    json_data = {"json_field": "json_value"}
    
    with mock.patch("requests.Session.request") as mock_request:
        mock_request.return_value = MockResponse(status_code=201)
        
        response = crawler.post(
            url=url,
            data=data,
            json=json_data
        )
        
        mock_request.assert_called_once()
        kwargs = mock_request.call_args[1]
        assert kwargs["method"] == "POST"
        assert kwargs["url"] == url
        assert kwargs["data"] == data
        assert kwargs["json"] == json_data


def test_fetch_url_success(crawler):
    """Test successful URL fetching."""
    url = "https://example.com/page"
    content = "<html><body>Test content</body></html>"
    
    with mock.patch("requests.Session.request") as mock_request:
        mock_request.return_value = MockResponse(
            status_code=200,
            text=content,
            content=content.encode('utf-8')
        )
        
        result_content, result_response = crawler.fetch_url(url)
        
        assert result_content == content
        assert result_response.status_code == 200


def test_fetch_url_failure(crawler):
    """Test URL fetching failure."""
    url = "https://example.com/nonexistent"
    
    with mock.patch("requests.Session.request") as mock_request:
        mock_request.return_value = MockResponse(status_code=404)
        
        result_content, result_response = crawler.fetch_url(url)
        
        assert result_content is None
        assert result_response is None


def test_fetch_json_success(crawler):
    """Test successful JSON fetching."""
    url = "https://example.com/api/data"
    json_data = '{"key": "value", "number": 42}'
    
    with mock.patch("requests.Session.request") as mock_request:
        mock_request.return_value = MockResponse(
            status_code=200,
            text=json_data,
            content=json_data.encode('utf-8'),
            headers={"Content-Type": "application/json"}
        )
        
        result_json, result_response = crawler.fetch_json(url)
        
        assert result_json == {"key": "value", "number": 42}
        assert result_response.status_code == 200
        
        # Verify that Accept header was set for JSON
        kwargs = mock_request.call_args[1]
        assert kwargs["headers"]["Accept"] == "application/json"


def test_fetch_json_invalid(crawler):
    """Test handling of invalid JSON responses."""
    url = "https://example.com/api/invalid"
    invalid_json = "{not valid json}"
    
    with mock.patch("requests.Session.request") as mock_request:
        mock_request.return_value = MockResponse(
            status_code=200,
            text=invalid_json,
            content=invalid_json.encode('utf-8'),
            headers={"Content-Type": "application/json"}
        )
        
        result_json, result_response = crawler.fetch_json(url)
        
        assert result_json is None
        assert result_response is not None  # Response should still be returned


def test_download_file_success(crawler, tmp_path):
    """Test successful file download."""
    url = "https://example.com/file.txt"
    content = b"File content for testing"
    output_path = tmp_path / "downloaded_file.txt"
    
    with mock.patch("requests.Session.request") as mock_request:
        mock_request.return_value = MockResponse(
            status_code=200,
            content=content
        )
        
        success = crawler.download_file(url, str(output_path))
        
        assert success is True
        assert output_path.exists()
        assert output_path.read_bytes() == content


def test_download_file_request_failure(crawler, tmp_path):
    """Test file download when the request fails."""
    url = "https://example.com/file.txt"
    output_path = tmp_path / "should_not_exist.txt"
    
    with mock.patch("requests.Session.request") as mock_request:
        mock_request.return_value = MockResponse(status_code=404)
        
        success = crawler.download_file(url, str(output_path))
        
        assert success is False
        assert not output_path.exists()


def test_download_file_write_error(crawler, tmp_path):
    """Test file download with write errors."""
    url = "https://example.com/file.txt"
    content = b"File content"
    
    # Create a directory with the same name as the file we want to create
    # This will cause a write error
    output_dir = tmp_path / "output_file.txt"
    output_dir.mkdir()
    
    with mock.patch("requests.Session.request") as mock_request:
        mock_request.return_value = MockResponse(
            status_code=200,
            content=content
        )
        
        success = crawler.download_file(url, str(output_dir))
        
        assert success is False


def test_normalize_url(crawler):
    """Test URL normalization."""
    # Test absolute URL with trailing slash
    abs_url = "HTTPS://Example.COM/Path/"
    normalized = crawler.normalize_url(abs_url)
    assert normalized == "https://example.com/Path"
    
    # Test absolute URL without trailing slash
    abs_url = "HTTPS://Example.COM/Path"
    normalized = crawler.normalize_url(abs_url)
    assert normalized == "https://example.com/Path"
    
    # Test root path
    root_url = "https://example.com/"
    normalized = crawler.normalize_url(root_url)
    assert normalized == "https://example.com/"
    
    # Test relative URL with base
    rel_url = "/relative/path"
    base_url = "https://example.com"
    normalized = crawler.normalize_url(rel_url, base_url)
    assert normalized == "https://example.com/relative/path"


def test_clear_robots_cache(crawler):
    """Test clearing the robots.txt cache."""
    # First request should create a cache entry
    with mock.patch("requests.Session.request") as mock_request:
        mock_request.return_value = MockResponse()
        crawler.robots_cache["https://example.com"] = mock.Mock()
        
        crawler.clear_robots_cache()
        
        assert len(crawler.robots_cache) == 0


def test_close(crawler):
    """Test proper cleanup of resources."""
    with mock.patch.object(crawler.session, "close") as mock_close:
        crawler.close()
        mock_close.assert_called_once()
