"""
Web crawler module for fetching cybersecurity news and information.
"""

import logging
import re
import time
from collections import deque
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from vigil.data_collection.content_extractor import ContentExtractor

# Configure module logger
logger = logging.getLogger('vigil.data_collection.crawler')

class Crawler:
    """A web crawler for fetching cybersecurity news and articles."""
    
    def __init__(self, user_agent=None, timeout=30, delay=1.0, max_retries=3):
        """
        Initialize the crawler with configuration parameters.
        
        Args:
            user_agent (str): User agent string to use in requests
            timeout (int): Request timeout in seconds
            delay (float): Delay between requests to the same domain (in seconds)
            max_retries (int): Maximum number of retry attempts for failed requests
        """
        self.user_agent = user_agent or 'VIGIL Cybersecurity News Crawler/0.1'
        self.timeout = timeout
        self.delay = delay
        self.max_retries = max_retries
        self.domain_last_access = {}
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': self.user_agent})
        self.content_extractor = ContentExtractor()
    
    def fetch_url(self, url):
        """
        Fetch content from a URL with error handling and retry logic.
        
        Args:
            url (str): URL to fetch
            
        Returns:
            tuple: (content, status_code) or (None, error_code) if failed
        """
        # Extract domain for rate limiting
        domain = urlparse(url).netloc
        
        # Apply rate limiting
        current_time = time.time()
        if domain in self.domain_last_access:
            time_since_last_request = current_time - self.domain_last_access[domain]
            if time_since_last_request < self.delay:
                sleep_time = self.delay - time_since_last_request
                logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s for {domain}")
                time.sleep(sleep_time)
        
        # Update last access time
        self.domain_last_access[domain] = time.time()
        
        retries = 0
        while retries <= self.max_retries:
            try:
                logger.debug(f"Fetching URL: {url}")
                response = self.session.get(url, timeout=self.timeout)
                
                if response.status_code == 200:
                    return response.text, response.status_code
                elif response.status_code == 404:
                    logger.warning(f"URL not found (404): {url}")
                    return None, 404
                elif 400 <= response.status_code < 500:
                    logger.warning(f"Client error: {response.status_code} for URL: {url}")
                    return None, response.status_code
                else:
                    logger.warning(f"Server error: {response.status_code} for URL: {url}")
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout while fetching URL: {url}")
            except requests.exceptions.ConnectionError:
                logger.warning(f"Connection error while fetching URL: {url}")
            except requests.exceptions.RequestException as e:
                logger.warning(f"Error fetching URL {url}: {str(e)}")
            
            # Exponential backoff
            retries += 1
            if retries <= self.max_retries:
                wait_time = 2 ** retries  # Exponential backoff: 2, 4, 8 seconds
                logger.info(f"Retrying {url} in {wait_time} seconds (attempt {retries}/{self.max_retries})")
                time.sleep(wait_time)
        
        return None, 0
    
    def parse_html(self, html_content):
        """
        Parse HTML content using BeautifulSoup.
        
        Args:
            html_content (str): HTML content to parse
            
        Returns:
            BeautifulSoup: Parsed HTML
        """
        if not html_content:
            return None
        
        try:
            return BeautifulSoup(html_content, 'html.parser')
        except Exception as e:
            logger.error(f"Error parsing HTML: {str(e)}")
            return None
    
    def extract_links(self, soup, base_url):
        """
        Extract all links from a BeautifulSoup parsed page.
        
        Args:
            soup (BeautifulSoup): Parsed HTML
            base_url (str): Base URL for resolving relative links
            
        Returns:
            list: List of absolute URLs found in the page
        """
        if not soup:
            return []
        
        links = []
        for anchor in soup.find_all('a', href=True):
            href = anchor['href']
            absolute_url = urljoin(base_url, href)
            
            # Filter out javascript, mailto, etc.
            if absolute_url.startswith(('http://', 'https://')):
                links.append(absolute_url)
        
        return links
    
    def should_follow_url(self, url, allowed_domains=None, url_patterns=None):
        """
        Check if a URL should be followed based on domain and patterns.
        
        Args:
            url (str): URL to check
            allowed_domains (list): List of allowed domains to follow
            url_patterns (list): List of regex patterns for allowed URLs
            
        Returns:
            bool: True if URL should be followed, False otherwise
        """
        parsed = urlparse(url)
        domain = parsed.netloc
        
        # Check if domain is allowed
        if allowed_domains and domain not in allowed_domains:
            return False
        
        # Check if URL matches any pattern
        if url_patterns:
            return any(re.search(pattern, url) for pattern in url_patterns)
        
        return True
    
    def crawl(self, start_urls, max_depth=2, allowed_domains=None, url_patterns=None, max_urls=100):
        """
        Crawl web pages starting from given URLs using BFS approach.
        
        Args:
            start_urls (list): List of URLs to start crawling from
            max_depth (int): Maximum depth to crawl
            allowed_domains (list): List of domains to restrict crawling to
            url_patterns (list): Regex patterns of URLs to follow
            max_urls (int): Maximum number of URLs to crawl
            
        Returns:
            dict: Dictionary of crawled URLs with their content and metadata
        """
        queue = deque([(url, 0) for url in start_urls])  # (url, depth)
        visited = set()
        results = {}
        
        logger.info(f"Starting crawl with {len(start_urls)} seed URLs, max depth {max_depth}")
        
        while queue and len(results) < max_urls:
            url, depth = queue.popleft()
            
            # Skip if already visited or max depth reached
            if url in visited or depth > max_depth:
                continue
            
            visited.add(url)
            logger.info(f"Crawling [{len(results)+1}/{max_urls}] {url} (depth {depth}/{max_depth})")
            
            # Fetch and process the URL
            content, status = self.fetch_url(url)
            
            if content and status == 200:
                soup = self.parse_html(content)
                
                # Extract structured content
                extracted_content = self.content_extractor.extract_content(content, url)
                
                # Store the result
                results[url] = {
                    'url': url,
                    'depth': depth,
                    'content': content,
                    'soup': soup,
                    'status': status,
                    'timestamp': time.time(),
                    'extracted_content': extracted_content
                }
                
                # If we're not at max depth, extract and queue links
                if depth < max_depth:
                    links = self.extract_links(soup, url)
                    logger.debug(f"Found {len(links)} links on {url}")
                    
                    for link in links:
                        if (link not in visited and 
                            self.should_follow_url(link, allowed_domains, url_patterns)):
                            queue.append((link, depth + 1))
            else:
                # Store failed attempts too
                results[url] = {
                    'url': url,
                    'depth': depth,
                    'content': None,
                    'soup': None,
                    'status': status,
                    'timestamp': time.time(),
                    'extracted_content': None
                }
        
        logger.info(f"Crawling complete. Processed {len(results)} URLs.")
        return results
    
    # The extract_article_content method is now superseded by the ContentExtractor class
    # We can either delete it or keep it for backward compatibility
