"""
URL frontier implementation for managing crawling workloads.
"""

import heapq
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, Callable
from urllib.parse import urlparse

from vigil.utils.url_utils import normalize_url, extract_domain


@dataclass(order=True)
class URLEntry:
    """
    An entry in the URL frontier with priority information.
    """
    priority: float
    url: str = field(compare=False)
    metadata: Dict[str, Any] = field(default_factory=dict, compare=False)
    timestamp: float = field(default_factory=time.time, compare=False)

    def __str__(self) -> str:
        return f"URLEntry(priority={self.priority}, url='{self.url}')"


class DomainBucket:
    """
    A bucket for URLs from a single domain with rate limiting capabilities.
    """
    def __init__(self, domain: str, delay: float = 1.0):
        self.domain = domain
        self.delay = delay  # Minimum delay between requests in seconds
        self.last_access_time = 0.0
        self.urls = deque()  # URLs for this domain
    
    def add_url(self, entry: URLEntry) -> None:
        """Add a URL entry to this domain bucket."""
        self.urls.append(entry)
    
    def is_empty(self) -> bool:
        """Check if the bucket is empty."""
        return len(self.urls) == 0
    
    def ready_time(self) -> float:
        """Get the time when this domain is ready for the next request."""
        return self.last_access_time + self.delay
    
    def is_ready(self) -> bool:
        """Check if enough time has passed to crawl this domain again."""
        return time.time() >= self.ready_time()
    
    def get_url(self) -> Optional[URLEntry]:
        """Get the next URL if available and update the last access time."""
        if self.is_empty():
            return None
        
        self.last_access_time = time.time()
        return self.urls.popleft()
    

class URLFrontier:
    """
    URL frontier for managing a prioritized queue of URLs to crawl.
    
    Features:
    - Maintains a queue of URLs to crawl
    - Tracks visited URLs to avoid duplicates
    - Supports prioritization of URLs
    - Handles domain-based rate limiting
    """
    
    def __init__(
        self,
        default_delay: float = 1.0,
        max_urls_per_domain: int = 1000,
        max_priority_queue_size: int = 10000,
        priority_function: Optional[Callable[[str, Dict[str, Any]], float]] = None
    ):
        """
        Initialize the URL frontier.
        
        Args:
            default_delay: Default delay between requests to the same domain in seconds
            max_urls_per_domain: Maximum number of URLs to keep per domain
            max_priority_queue_size: Maximum size of the priority queue
            priority_function: Function to calculate URL priority (url, metadata) -> float
        """
        # Logger
        self.logger = logging.getLogger("vigil.frontier")
        
        # Configuration
        self.default_delay = default_delay
        self.max_urls_per_domain = max_urls_per_domain
        self.max_priority_queue_size = max_priority_queue_size
        
        # Priority function - defaults to FIFO if None
        self.priority_function = priority_function or (lambda url, metadata: 0.0)
        
        # Storage structures
        self.visited_urls: Set[str] = set()  # URLs we've seen before
        self.priority_queue: List[URLEntry] = []  # Priority queue of URLs to visit
        self.domain_buckets: Dict[str, DomainBucket] = {}  # Domain-specific queues
        self.domain_delays: Dict[str, float] = defaultdict(lambda: self.default_delay)  # Delay per domain
        
        # Statistics
        self.stats = {
            "added_urls": 0,
            "duplicate_urls": 0,
            "fetched_urls": 0,
            "empty_gets": 0
        }
    
    def _get_domain(self, url: str) -> str:
        """Extract domain from URL."""
        parsed = urlparse(url)
        return parsed.netloc.lower()
    
    def _calculate_priority(self, url: str, metadata: Dict[str, Any] = None) -> float:
        """Calculate priority for a URL using the configured priority function."""
        metadata = metadata or {}
        try:
            return self.priority_function(url, metadata)
        except Exception as e:
            self.logger.error(f"Error calculating priority for {url}: {str(e)}")
            return 0.0  # Default priority
    
    def add_domain_delay(self, domain: str, delay: float) -> None:
        """
        Set a custom delay for a specific domain.
        
        Args:
            domain: Domain name
            delay: Delay in seconds between requests
        """
        self.domain_delays[domain.lower()] = max(0.1, delay)  # Ensure minimum delay
        
        # Update existing bucket if it exists
        if domain in self.domain_buckets:
            self.domain_buckets[domain].delay = self.domain_delays[domain]
    
    def add_url(
        self, url: str, metadata: Dict[str, Any] = None, priority: Optional[float] = None
    ) -> bool:
        """
        Add a URL to the frontier.
        
        Args:
            url: URL to add
            metadata: Additional metadata about the URL
            priority: Explicit priority (overrides priority function)
            
        Returns:
            True if the URL was added, False if it was a duplicate
        """
        # Normalize URL
        normalized_url = normalize_url(url)
        
        # Check if we've seen this URL before
        if normalized_url in self.visited_urls:
            self.stats["duplicate_urls"] += 1
            return False
        
        # Mark as visited
        self.visited_urls.add(normalized_url)
        
        # Extract domain
        domain = self._get_domain(normalized_url)
        
        # Calculate or use provided priority
        if priority is None:
            priority = self._calculate_priority(normalized_url, metadata)
        
        # Create URL entry
        entry = URLEntry(
            priority=priority,
            url=normalized_url,
            metadata=metadata or {},
            timestamp=time.time()
        )
        
        # Add to domain bucket
        if domain not in self.domain_buckets:
            self.domain_buckets[domain] = DomainBucket(
                domain=domain,
                delay=self.domain_delays[domain]
            )
        
        # Check if we have too many URLs for this domain
        bucket = self.domain_buckets[domain]
        if len(bucket.urls) >= self.max_urls_per_domain:
            self.logger.debug(f"Domain {domain} has reached max URLs limit. Skipping {url}")
            return False
        
        # Add to domain bucket
        bucket.add_url(entry)
        
        # Also add to priority queue for global prioritization
        if len(self.priority_queue) < self.max_priority_queue_size:
            heapq.heappush(self.priority_queue, entry)
        else:
            # If queue is full, only add if better than worst entry
            worst_entry = self.priority_queue[0]  # Heap root is the minimum
            if entry.priority > worst_entry.priority:
                heapq.heapreplace(self.priority_queue, entry)
        
        self.stats["added_urls"] += 1
        return True
    
    def add_urls(self, urls: List[str], metadata: Dict[str, Any] = None) -> int:
        """
        Add multiple URLs to the frontier.
        
        Args:
            urls: List of URLs to add
            metadata: Shared metadata for all URLs
            
        Returns:
            Number of URLs successfully added
        """
        added_count = 0
        for url in urls:
            if self.add_url(url, metadata):
                added_count += 1
        
        return added_count
    
    def get_next_url(self) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Get the next URL to crawl based on priority and domain politeness.
        
        Returns:
            Tuple of (URL, metadata) or None if no URLs are available
        """
        # Return None if everything is empty
        if all(bucket.is_empty() for bucket in self.domain_buckets.values()):
            self.stats["empty_gets"] += 1
            return None
            
        # Rebuild the priority queue to ensure URLs are in proper order
        self._rebuild_priority_queue()
        
        # Look for ready domains first
        ready_domains = {domain: bucket for domain, bucket in self.domain_buckets.items() 
                        if not bucket.is_empty() and bucket.is_ready()}
        
        if ready_domains:
            # Choose the domain with the highest priority URL
            best_domain = None
            best_entry = None
            best_priority = float('-inf')
            
            for domain, bucket in ready_domains.items():
                # Find the highest priority URL in this domain
                for entry in bucket.urls:
                    if entry.priority > best_priority:
                        best_priority = entry.priority
                        best_entry = entry
                        best_domain = domain
            
            if best_domain and best_entry:
                # Remove the entry from the domain bucket
                bucket = self.domain_buckets[best_domain]
                bucket.urls.remove(best_entry)
                bucket.last_access_time = time.time()
                self.stats["fetched_urls"] += 1
                return best_entry.url, best_entry.metadata
        
        # If no ready domains with URLs, pick the next one that will be ready
        non_empty_domains = [(d, b) for d, b in self.domain_buckets.items() if not b.is_empty()]
        
        if non_empty_domains:
            # Sort by ready time to find the domain that will be ready soonest
            domain, bucket = min(non_empty_domains, key=lambda x: x[1].ready_time())
            
            # Wait for it to be ready
            wait_time = max(0, bucket.ready_time() - time.time())
            if wait_time > 0:
                time.sleep(wait_time)
                
            # Get URL with highest priority from this domain
            best_entry = max(bucket.urls, key=lambda entry: entry.priority)
            bucket.urls.remove(best_entry)
            bucket.last_access_time = time.time()
            self.stats["fetched_urls"] += 1
            return best_entry.url, best_entry.metadata
        
        self.stats["empty_gets"] += 1
        return None
    
    def _rebuild_priority_queue(self) -> None:
        """
        Rebuild the priority queue from all domain buckets.
        This ensures that URLs are retrieved in proper priority order.
        """
        # Start with an empty list and collect all entries
        all_entries = []
        
        # Collect from all domain buckets
        for domain, bucket in self.domain_buckets.items():
            all_entries.extend(list(bucket.urls))
        
        # If no entries, nothing to do
        if not all_entries:
            self.priority_queue = []
            return
            
        # Sort by priority in descending order (higher priority first)
        all_entries.sort(key=lambda entry: entry.priority, reverse=True)
        
        # Limit to max size and set as the new priority queue
        self.priority_queue = all_entries[:self.max_priority_queue_size]
    
    def get_domain_count(self) -> int:
        """Get the number of domains in the frontier."""
        return len(self.domain_buckets)
    
    def get_url_count(self) -> int:
        """Get the total number of URLs in the frontier."""
        return sum(len(bucket.urls) for bucket in self.domain_buckets.values())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the frontier."""
        stats = self.stats.copy()
        stats["visited_urls"] = len(self.visited_urls)
        stats["queued_urls"] = self.get_url_count()
        stats["domains"] = self.get_domain_count()
        return stats
    
    def is_empty(self) -> bool:
        """Check if the frontier is empty."""
        return all(bucket.is_empty() for bucket in self.domain_buckets.values())
    
    def clear(self) -> None:
        """Clear the frontier."""
        self.visited_urls.clear()
        self.priority_queue.clear()
        self.domain_buckets.clear()
        # Reset statistics
        self.stats = {
            "added_urls": 0,
            "duplicate_urls": 0, 
            "fetched_urls": 0,
            "empty_gets": 0
        }
        self.logger.info("URL frontier cleared")
    
    def clear_domain(self, domain: str) -> int:
        """
        Clear URLs for a specific domain.
        
        Args:
            domain: Domain to clear
            
        Returns:
            Number of URLs removed
        """
        domain = domain.lower()
        if domain in self.domain_buckets:
            removed_count = len(self.domain_buckets[domain].urls)
            self.domain_buckets[domain].urls.clear()
            
            # Remove entries from priority queue
            self.priority_queue = [
                entry for entry in self.priority_queue 
                if self._get_domain(entry.url) != domain
            ]
            heapq.heapify(self.priority_queue)
            
            return removed_count
        return 0


class BreadthFirstFrontier(URLFrontier):
    """
    Breadth-first search implementation of URL frontier.
    All URLs have the same priority based on when they were added.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize a breadth-first frontier.
        
        Args:
            **kwargs: Arguments to pass to URLFrontier constructor
        """
        # Pass a FIFO priority function (earlier timestamp = higher priority)
        def fifo_priority(url, metadata):
            # For BFS, URLs discovered earlier should have higher priority
            return 1000.0 - metadata.get('timestamp', time.time())
        
        super().__init__(priority_function=fifo_priority, **kwargs)


class DepthFirstFrontier(URLFrontier):
    """
    Depth-first search implementation of URL frontier.
    Prioritizes newly discovered URLs.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize a depth-first frontier.
        
        Args:
            **kwargs: Arguments to pass to URLFrontier constructor
        """
        # Pass a LIFO priority function (later timestamp = higher priority)
        def lifo_priority(url, metadata):
            # For DFS, URLs discovered later should have higher priority
            return metadata.get('timestamp', time.time())
        
        super().__init__(priority_function=lifo_priority, **kwargs)


class PriorityFrontier(URLFrontier):
    """
    Priority-based frontier with configurable score function.
    """
    
    def __init__(
        self,
        score_function: Callable[[str, Dict[str, Any]], float],
        **kwargs
    ):
        """
        Initialize a priority frontier.
        
        Args:
            score_function: Function that calculates a score (url, metadata) -> float
                           Higher scores will be crawled first
            **kwargs: Arguments to pass to URLFrontier constructor
        """
        super().__init__(priority_function=score_function, **kwargs)
