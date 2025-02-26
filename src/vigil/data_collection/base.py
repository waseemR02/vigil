"""
Base crawler implementation for data collection in the Vigil monitoring system.
"""

import logging
import random
import time
from typing import Dict, List, Optional, Tuple, Union, Any
import urllib.robotparser
from urllib.parse import urlparse, urljoin

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from vigil.utils.url_utils import normalize_url


class BaseCrawler:
    """
    Base crawler class with built-in politeness mechanisms, retry logic,
    robots.txt handling, and proper error management.
    """

    # Default user agents to rotate through
    DEFAULT_USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
    ]

    def __init__(
        self,
        delay: float = 1.0,
        min_delay: float = 0.5,
        max_delay: float = 3.0,
        timeout: int = 10,
        max_retries: int = 3,
        verify_ssl: bool = True,
        respect_robots_txt: bool = True,
        user_agent: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        proxies: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize the base crawler with configurable parameters.

        Args:
            delay: Base delay between requests in seconds (default: 1.0)
            min_delay: Minimum delay when using random delays (default: 0.5)
            max_delay: Maximum delay when using random delays (default: 3.0)
            timeout: Request timeout in seconds (default: 10)
            max_retries: Maximum number of retries for failed requests (default: 3)
            verify_ssl: Whether to verify SSL certificates (default: True)
            respect_robots_txt: Whether to respect robots.txt rules (default: True)
            user_agent: User agent string, or None to use a random one from defaults
            headers: Additional headers to include in requests
            proxies: Proxy configuration for requests
        """
        self.delay = delay
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.timeout = timeout
        self.max_retries = max_retries
        self.verify_ssl = verify_ssl
        self.respect_robots_txt = respect_robots_txt
        
        # Set up user agent
        if user_agent is None:
            self.user_agent = random.choice(self.DEFAULT_USER_AGENTS)
        else:
            self.user_agent = user_agent

        # Set up session
        self.session = self._create_session()
        
        # Set default headers
        self.headers = {
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }
        
        # Update with user-provided headers
        if headers:
            self.headers.update(headers)
        
        # Set proxies
        self.proxies = proxies
        
        # Initialize last request time
        self.last_request_time = 0
        
        # Initialize robots.txt cache - {domain: parser}
        self.robots_cache = {}
        
        # Logger
        self.logger = logging.getLogger("vigil.crawler")

    def _create_session(self) -> requests.Session:
        """
        Create a requests session with retry and backoff configuration.
        
        Returns:
            Configured requests session
        """
        session = requests.Session()
        
        # Configure retry strategy with exponential backoff
        retry_strategy = Retry(
            total=self.max_retries,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "HEAD", "OPTIONS"],
            backoff_factor=1.0,  # 1, 2, 4... seconds between retries
        )
        
        # Apply retry strategy to both HTTP and HTTPS connections
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session

    def _respect_robots(self, url: str) -> bool:
        """
        Check if the URL can be accessed according to robots.txt rules.
        
        Args:
            url: URL to check
            
        Returns:
            True if allowed, False otherwise
        """
        if not self.respect_robots_txt:
            return True
        
        parsed_url = urlparse(url)
        domain = f"{parsed_url.scheme}://{parsed_url.netloc}"
        
        # Check if we already have a parser for this domain
        if domain not in self.robots_cache:
            robots_url = f"{domain}/robots.txt"
            parser = urllib.robotparser.RobotFileParser()
            parser.set_url(robots_url)
            
            try:
                parser.read()
                self.robots_cache[domain] = parser
            except Exception as e:
                self.logger.warning(f"Failed to read robots.txt at {robots_url}: {e}")
                # Allow access if robots.txt cannot be read
                return True
        
        # Check if the URL is allowed
        if self.robots_cache[domain].can_fetch(self.user_agent, url):
            return True
        else:
            self.logger.info(f"URL {url} is disallowed by robots.txt")
            return False

    def _enforce_delay(self) -> None:
        """
        Enforce a delay between requests to be polite.
        """
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        
        if elapsed < self.delay:
            # Use random delay between min and max values
            actual_delay = random.uniform(self.min_delay, self.max_delay)
            
            # Don't delay more than necessary
            actual_delay = min(actual_delay, self.delay - elapsed)
            
            if actual_delay > 0:
                self.logger.debug(f"Sleeping for {actual_delay:.2f} seconds")
                time.sleep(actual_delay)

    def change_user_agent(self, user_agent: Optional[str] = None) -> None:
        """
        Change the user agent for the crawler.
        
        Args:
            user_agent: New user agent, or None to use a random one from defaults
        """
        if user_agent is None:
            user_agent = random.choice(self.DEFAULT_USER_AGENTS)
        
        self.user_agent = user_agent
        self.headers["User-Agent"] = user_agent
        self.logger.debug(f"Changed user agent to: {user_agent}")

    def request(
        self,
        url: str,
        method: str = "GET",
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        custom_headers: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
        allow_redirects: bool = True,
        stream: bool = False,
    ) -> Optional[requests.Response]:
        """
        Make an HTTP request with proper handling.
        
        Args:
            url: URL to request
            method: HTTP method (default: GET)
            params: URL parameters
            data: Form data
            json: JSON data
            custom_headers: Additional headers for this specific request
            timeout: Request timeout override
            allow_redirects: Whether to follow redirects
            stream: Whether to stream the response
            
        Returns:
            Response object or None if request fails
        """
        # Check if we're allowed to access this URL
        if not self._respect_robots(url):
            return None
        
        # Enforce delay between requests
        self._enforce_delay()
        
        # Prepare headers
        headers = self.headers.copy()
        if custom_headers:
            headers.update(custom_headers)
        
        # Use instance timeout if not overridden
        if timeout is None:
            timeout = self.timeout
        
        # Make the request
        try:
            self.logger.debug(f"Making {method} request to: {url}")
            self.last_request_time = time.time()
            
            response = self.session.request(
                method=method,
                url=url,
                params=params,
                data=data,
                json=json,
                headers=headers,
                timeout=timeout,
                verify=self.verify_ssl,
                proxies=self.proxies,
                allow_redirects=allow_redirects,
                stream=stream,
            )
            
            # Log response info
            self.logger.debug(f"Response: {response.status_code}, {len(response.content)} bytes")
            
            # Raise exception for client and server errors
            response.raise_for_status()
            
            return response
            
        except requests.exceptions.HTTPError as e:
            self.logger.error(f"HTTP error for {url}: {e}")
        except requests.exceptions.ConnectionError as e:
            self.logger.error(f"Connection error for {url}: {e}")
        except requests.exceptions.Timeout as e:
            self.logger.error(f"Timeout error for {url}: {e}")
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request error for {url}: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error for {url}: {e}")
        
        return None

    def get(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        custom_headers: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
        allow_redirects: bool = True,
        stream: bool = False,  # Added stream parameter
    ) -> Optional[requests.Response]:
        """
        Make a GET request.
        
        Args:
            url: URL to request
            params: URL parameters
            custom_headers: Additional headers for this specific request
            timeout: Request timeout override
            allow_redirects: Whether to follow redirects
            stream: Whether to stream the response
            
        Returns:
            Response object or None if request fails
        """
        return self.request(
            url=url,
            method="GET",
            params=params,
            custom_headers=custom_headers,
            timeout=timeout,
            allow_redirects=allow_redirects,
            stream=stream,  # Pass the stream parameter
        )

    def post(
        self,
        url: str,
        data: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        custom_headers: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
        allow_redirects: bool = True,
    ) -> Optional[requests.Response]:
        """
        Make a POST request.
        
        Args:
            url: URL to request
            data: Form data
            json: JSON data
            params: URL parameters
            custom_headers: Additional headers for this specific request
            timeout: Request timeout override
            allow_redirects: Whether to follow redirects
            
        Returns:
            Response object or None if request fails
        """
        return self.request(
            url=url,
            method="POST",
            data=data,
            json=json,
            params=params,
            custom_headers=custom_headers,
            timeout=timeout,
            allow_redirects=allow_redirects,
        )

    def fetch_url(
        self, url: str, params: Optional[Dict[str, Any]] = None
    ) -> Tuple[Optional[str], Optional[requests.Response]]:
        """
        Fetch content from URL.
        
        Args:
            url: URL to fetch
            params: URL parameters
            
        Returns:
            Tuple of (content as string, response object) or (None, None) if request fails
        """
        response = self.get(url, params=params)
        if response is not None and response.status_code == 200:
            return response.text, response
        return None, None

    def fetch_json(
        self, url: str, params: Optional[Dict[str, Any]] = None
    ) -> Tuple[Optional[Dict[str, Any]], Optional[requests.Response]]:
        """
        Fetch JSON from URL.
        
        Args:
            url: URL to fetch
            params: URL parameters
            
        Returns:
            Tuple of (JSON data, response object) or (None, None) if request fails
        """
        response = self.get(
            url, 
            params=params, 
            custom_headers={"Accept": "application/json"}
        )
        
        if response is not None and response.status_code == 200:
            try:
                return response.json(), response
            except ValueError as e:
                self.logger.error(f"JSON parsing error for {url}: {e}")
                # Return None for JSON data but still return the response
                return None, response
        
        return None, None

    def download_file(
        self, url: str, output_path: str, chunk_size: int = 8192
    ) -> bool:
        """
        Download a file from URL.
        
        Args:
            url: URL to download from
            output_path: Path to save the file
            chunk_size: Size of chunks to download
            
        Returns:
            True if download succeeded, False otherwise
        """
        response = self.get(url, stream=True)
        if response is None:
            return False
            
        try:
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:  # Filter out keep-alive chunks
                        f.write(chunk)
            return True
        except IOError as e:
            self.logger.error(f"File write error for {output_path}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error while downloading {url} to {output_path}: {e}")
            return False

    def normalize_url(self, url: str, base_url: Optional[str] = None) -> str:
        """
        Normalize URL and resolve relative URLs.
        
        Args:
            url: URL to normalize
            base_url: Base URL for resolving relative URLs
            
        Returns:
            Normalized URL
        """
        if base_url and not url.startswith(('http://', 'https://')):
            url = urljoin(base_url, url)
            
        return normalize_url(url)

    def clear_robots_cache(self) -> None:
        """Clear the robots.txt cache."""
        self.robots_cache.clear()
        self.logger.debug("Cleared robots.txt cache")

    def close(self) -> None:
        """Close the session and release resources."""
        if self.session:
            self.session.close()
            self.logger.debug("Closed session")
