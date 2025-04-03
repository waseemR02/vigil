"""
Pipeline module for the Vigil cybersecurity platform.
Connects crawler, ML analyzer, and database components.
"""
import logging
import time
import os
from datetime import datetime
from collections import deque
from urllib.parse import urlparse, urlunparse, parse_qs, urlencode
from typing import Dict, Tuple, List, Set, Optional

from bs4 import BeautifulSoup
import requests

from vigil.config import load_config
from vigil.data_collection.crawler import Crawler
from vigil.data_collection.content_extractor import ContentExtractor
from vigil.model.training import ContentPredictor
from vigil.database.connection import init_db
from vigil.database import queries
from vigil.database import url_tracking  # Import the new module

# Set up logging
logger = logging.getLogger("vigil.core.pipeline")

class Pipeline:
    """Pipeline class to manage the flow: crawl → analyze → store."""
    
    def __init__(self, config_path=None, model_path=None, vectorizer_path=None):
        """
        Initialize pipeline components.
        
        Args:
            config_path: Path to configuration file
            model_path: Path to ML model file (overrides config setting)
            vectorizer_path: Path to vectorizer file (overrides config setting)
        """
        # Load configuration
        self.config = self._load_config()
        
        # Initialize database
        try:
            init_db()
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            logger.warning("Pipeline will continue but database operations may fail")
        
        # Initialize crawler
        crawler_config = self.config.get('crawler', {})
        self.crawler = Crawler(
            user_agent=crawler_config.get('user_agent'),
            timeout=crawler_config.get('timeout', 30),
            delay=crawler_config.get('delay', 1.0),
            max_retries=crawler_config.get('max_retries', 3)
        )
        
        # Initialize content extractor
        self.content_extractor = ContentExtractor()
        
        # Initialize ML predictor
        # If model_path is provided as an argument, use that; otherwise check config
        if not model_path:
            model_path = self.config.get('model', {}).get('path')
            
        # If still no model path, use default location
        if not model_path:
            model_path = os.path.join(
                self.config.get('storage', {}).get('file_store_path', "data"), 
                "models", 
                "latest_model.pkl"
            )
        
        # Get vectorizer path from args or config
        if not vectorizer_path:
            vectorizer_path = self.config.get('model', {}).get('vectorizer_path')
        
        # Initialize the predictor with both model and vectorizer
        self.predictor = ContentPredictor()
        
        # Check if model exists
        model_exists = os.path.exists(model_path) if model_path else False
        if model_exists:
            logger.info(f"Loading model from {model_path}")
            # If vectorizer path is provided, use it when loading the model
            if vectorizer_path and os.path.exists(vectorizer_path):
                logger.info(f"Loading vectorizer from {vectorizer_path}")
                self.predictor.load_model(model_path, vectorizer_path=vectorizer_path)
            else:
                self.predictor.load_model(model_path)
        else:
            logger.warning(f"Model not found at {model_path}. Classification will be unavailable.")
        
        # Initialize tracking sets and queues
        self.visited_urls = set()
        self.queued_urls = set()  # Keep track of URLs already in the queue
        self.url_queue = []       # The actual queue
        
        # Create URL tracking table if it doesn't exist
        url_tracking.init_url_tracking_table()
        
        logger.info("Pipeline initialized")
    
    def _load_config(self):
        """Load configuration from YAML file."""
        
        try:
            return load_config()
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            return {}
    
    def _normalize_url(self, url: str) -> str:
        """
        Normalize URL by removing fragments and non-essential query parameters.
        
        This prevents processing duplicate content from URLs like:
        - https://example.com/page#fragment
        - https://example.com/page?replytocom=123#respond
        
        Args:
            url: The URL to normalize
            
        Returns:
            Normalized URL without fragments and with only essential query params
        """
        try:
            # Parse the URL
            parsed = urlparse(url)
            
            # Get query parameters
            query_params = parse_qs(parsed.query)
            
            # Filter out non-essential query parameters that don't change main content
            # For example, remove 'replytocom' which is common in CMS comment systems
            non_essential_params = {'replytocom', 'ref', 'utm_source', 'utm_medium', 
                                    'utm_campaign', 'utm_term', 'utm_content'}
            
            filtered_params = {k: v for k, v in query_params.items() 
                              if k.lower() not in non_essential_params}
            
            # Rebuild query string
            query_string = urlencode(filtered_params, doseq=True) if filtered_params else ''
            
            # Rebuild URL without fragment
            normalized = urlunparse((
                parsed.scheme,
                parsed.netloc,
                parsed.path,
                parsed.params,
                query_string,
                ''  # No fragment
            ))
            
            return normalized
        except Exception as e:
            logger.warning(f"Error normalizing URL {url}: {str(e)}")
            return url
    
    def _is_url_processed(self, url: str) -> bool:
        """Check if a URL has been processed before."""
        # Normalize the URL before checking
        normalized_url = self._normalize_url(url)
        return url_tracking.is_url_processed(normalized_url)
    
    def _mark_url_processed(self, url: str):
        """Mark a URL as processed in the database."""
        # Normalize the URL before marking as processed
        normalized_url = self._normalize_url(url)
        url_tracking.mark_url_processed(normalized_url)
    
    def _get_processed_urls(self, domain: str = None, limit: int = None) -> List[str]:
        """Get list of processed URLs, optionally filtered by domain."""
        return url_tracking.get_processed_urls(domain, limit)
    
    def process_url(self, url: str) -> Tuple[bool, Dict]:
        """
        Process a single URL through the pipeline:
        1. Crawl and extract content
        2. Check if content is relevant (ML classification)
        3. Store relevant content in the database
        4. Return processing status and metadata
        """
        # Normalize the URL before processing
        url = self._normalize_url(url)
        
        start_time = time.time()
        result = {
            "url": url,
            "timestamp": datetime.now().isoformat(),
            "relevant": False,
            "new_urls": 0
        }
        
        try:
            logger.info(f"Processing URL: {url}")
            self.visited_urls.add(url)
            
            # Mark URL as processed in persistent storage
            self._mark_url_processed(url)
            
            # 1. Fetch content
            content, status = self.crawler.fetch_url(url)
            if not content or status != 200:
                logger.warning(f"Failed to fetch content from {url} (status {status})")
                return False, result
            
            # 2. Extract structured content
            extracted = self.content_extractor.extract_content(content, url)
            if not extracted["content"]:
                logger.warning(f"Failed to extract content from {url}")
                return False, result
            
            # 3. Check relevance with ML model
            if self.predictor.model_trainer and self.predictor.model_trainer.model:
                try:
                    is_relevant, confidence = self.predictor.predict_with_score(extracted["content"])
                    result["relevant"] = is_relevant
                    result["confidence"] = confidence
                    
                    if not is_relevant:
                        logger.info(f"Content from {url} classified as not relevant (confidence: {confidence:.2f})")
                        return True, result
                except Exception as e:
                    logger.error(f"Error classifying content: {str(e)}")
                    # Continue anyway, assume it's relevant if classification fails
                    result["relevant"] = True
                    result["confidence"] = 0.5
            else:
                # No model available, assume relevant
                logger.warning("No model available for classification, assuming content is relevant")
                result["relevant"] = True
                result["confidence"] = 1.0
            
            # 4. Store relevant content in the database
            try:
                # Get or create source
                domain = urlparse(url).netloc
                source_id = self._get_or_create_source(domain, url)
                
                # Create incident record
                incident_id = queries.create_incident(
                    title=extracted["title"],
                    description=extracted["content"],
                    source_id=source_id,
                    source_reference=url,
                    tags=["auto-extracted"]  # Basic tagging
                )
                
                result["stored"] = True
                result["incident_id"] = incident_id
                logger.info(f"Stored incident {incident_id} from {url}")
                
            except Exception as e:
                logger.error(f"Error storing content in database: {str(e)}")
                result["stored"] = False
                result["error"] = str(e)
            
            # 5. Extract new URLs for future processing
            soup = BeautifulSoup(content, 'html.parser')
            new_urls = self._extract_urls(soup, url)
            result["new_urls"] = len(new_urls)
            result["new_urls_list"] = new_urls  # Store the actual URLs
            
            # Calculate processing time
            result["processing_time"] = time.time() - start_time
            return True, result
            
        except Exception as e:
            logger.error(f"Error processing {url}: {str(e)}")
            result["error"] = str(e)
            return False, result
    
    def _extract_urls(self, soup, base_url):
        """Extract and filter URLs from the content."""
        if not soup:
            return []
        
        links = []
        allowed_domains = self.config.get('crawler', {}).get('allowed_domains', [])
        
        for anchor in soup.find_all('a', href=True):
            href = anchor['href']
            full_url = requests.compat.urljoin(base_url, href)
            
            # Normalize the URL
            normalized_url = self._normalize_url(full_url)
            
            # Parse URL to clean it
            parsed = urlparse(normalized_url)
            if not parsed.scheme or not parsed.netloc:
                continue
                
            # Filter by domain if allowed_domains is specified
            if allowed_domains and parsed.netloc not in allowed_domains:
                continue
                
            # Skip already visited or queued URLs in this session
            if normalized_url in self.visited_urls or normalized_url in self.queued_urls:
                continue
                
            # Skip URLs that were processed in previous sessions
            if self._is_url_processed(normalized_url):
                continue
                
            links.append(normalized_url)
        
        return links
    
    def _prioritize_urls(self, urls, seed_domain):
        """Prioritize URLs from the same domain as the seed URL."""
        same_domain = []
        other_domains = []
        
        for url in urls:
            parsed = urlparse(url)
            if parsed.netloc == seed_domain:
                same_domain.append(url)
            else:
                other_domains.append(url)
        
        # Return same domain URLs first, then others
        return same_domain + other_domains
    
    def _add_to_queue(self, urls, seed_domain, max_queue_size=1000):
        """Add URLs to the queue with deduplication and size management."""
        # Skip if the queue is already at max size
        if len(self.url_queue) >= max_queue_size:
            logger.warning(f"Queue size limit reached ({max_queue_size}). Skipping {len(urls)} new URLs.")
            return 0
            
        # Prioritize URLs from the same domain
        prioritized_urls = self._prioritize_urls(urls, seed_domain)
        
        # Add URLs to the queue if not already visited, queued, or processed in previous runs
        added_count = 0
        for url in prioritized_urls:
            # Make sure we're using the normalized URL
            normalized_url = self._normalize_url(url)
            
            if (normalized_url not in self.visited_urls and 
                normalized_url not in self.queued_urls and 
                not self._is_url_processed(normalized_url)):
                
                # Only add if we're still under the limit
                if len(self.url_queue) < max_queue_size:
                    self.url_queue.append(normalized_url)
                    self.queued_urls.add(normalized_url)
                    added_count += 1
                else:
                    break
        
        return added_count
    
    def _get_or_create_source(self, name, url=None):
        """Get or create a source in the database."""
        sources = queries.list_sources()
        for source in sources:
            if source.name == name:
                return source.id
        
        # Create new source if not found
        return queries.create_source(name, url)
    
    def _find_alternative_urls(self, seed_url: str, max_attempts: int = 5) -> List[str]:
        """
        Find alternative URLs to crawl if the seed URL has already been processed.
        
        Args:
            seed_url: The original seed URL
            max_attempts: Maximum number of URLs to try to fetch
            
        Returns:
            List of alternative URLs that haven't been processed yet
        """
        logger.info(f"Finding alternative start URLs for {seed_url}")
        
        # Extract domain from seed URL
        seed_domain = urlparse(seed_url).netloc
        
        # Try to fetch the seed URL to get links without processing it
        try:
            content, status = self.crawler.fetch_url(seed_url)
            if content and status == 200:
                soup = BeautifulSoup(content, 'html.parser')
                all_links = []
                
                # Extract all links from the page
                for anchor in soup.find_all('a', href=True):
                    href = anchor['href']
                    full_url = requests.compat.urljoin(seed_url, href)
                    
                    # Parse URL to clean it
                    parsed = urlparse(full_url)
                    if not parsed.scheme or not parsed.netloc:
                        continue
                    
                    # Filter to same domain
                    if parsed.netloc != seed_domain:
                        continue
                        
                    all_links.append(full_url)
                
                # Filter out already processed URLs
                new_urls = []
                for link in all_links:
                    if not self._is_url_processed(link):
                        new_urls.append(link)
                        if len(new_urls) >= max_attempts:
                            break
                
                if new_urls:
                    logger.info(f"Found {len(new_urls)} alternative URLs to process")
                    return new_urls
                else:
                    logger.warning(f"All links from {seed_url} have already been processed")
            else:
                logger.warning(f"Could not fetch {seed_url} to find alternative URLs")
        
        except Exception as e:
            logger.error(f"Error finding alternative URLs: {str(e)}")
        
        return []
    
    def _run_bfs_crawl(self, seed_url: str, max_urls: int = 50, max_depth: int = 3, max_queue_size: int = 5000) -> Dict:
        """
        Run a breadth-first crawler starting from the seed URL.
        Uses URL normalization and tracking to avoid duplicates.
        
        Args:
            seed_url: The starting URL
            max_urls: Maximum number of URLs to process
            max_depth: Maximum crawl depth
            max_queue_size: Maximum size of the BFS queue to prevent memory issues
            
        Returns:
            Dictionary with crawl statistics
        """
        seed_domain = urlparse(seed_url).netloc
        queue = deque([(seed_url, 0)])  # (url, depth)
        visited = set()
        processed_count = 0
        relevant_count = 0
        failed_count = 0
        skipped_count = 0
        
        logger.info(f"Starting BFS crawl from {seed_url} with max depth {max_depth}")
        start_time = time.time()
        
        crawl_stats = {
            "seed_url": seed_url,
            "start_time": datetime.now().isoformat(),
            "processed": 0,
            "relevant": 0,
            "failed": 0
        }
        
        while queue and processed_count < max_urls:
            current_url, depth = queue.popleft()
            
            # Normalize URL
            current_url = self._normalize_url(current_url)
            
            # Skip if already visited in this session or previously processed
            if current_url in visited or self._is_url_processed(current_url):
                continue
                
            # Mark as visited for this session
            visited.add(current_url)
            
            # Process the URL
            success, result = self.process_url(current_url)
            processed_count += 1
            
            if success:
                # If relevant, collect new URLs
                if result.get("relevant", False):
                    relevant_count += 1
                    
                    # If we're not at max depth and the content was successfully processed,
                    # extract and queue new URLs
                    if depth < max_depth and "new_urls_list" in result:
                        new_urls = result["new_urls_list"]
                        
                        # Filter and prioritize URLs from the same domain
                        prioritized_urls = self._prioritize_urls(new_urls, seed_domain)
                        
                        # Check if adding more URLs would exceed the queue size limit
                        remaining_capacity = max_queue_size - len(queue)
                        
                        if remaining_capacity <= 0:
                            logger.warning(f"Queue size limit ({max_queue_size}) reached. Skipping {len(prioritized_urls)} new URLs.")
                            skipped_count += len(prioritized_urls)
                        else:
                            # Add URLs to the queue up to the remaining capacity
                            urls_to_add = prioritized_urls[:remaining_capacity]
                            skipped_count += len(prioritized_urls) - len(urls_to_add)
                            
                            # Add URLs to the queue
                            for url in urls_to_add:
                                # Normalize URL
                                normalized_url = self._normalize_url(url)
                                
                                # Skip already visited or processed URLs
                                if (normalized_url not in visited and 
                                    not self._is_url_processed(normalized_url)):
                                    queue.append((normalized_url, depth + 1))
            else:
                failed_count += 1
                
            logger.info(f"Processed {processed_count}/{max_urls} URLs. " 
                      f"Queue size: {len(queue)}, "
                      f"Relevant: {relevant_count}, Failed: {failed_count}")
        
        # Collect unprocessed URLs for future runs
        unprocessed_urls = []
        for url, _ in queue:
            normalized_url = self._normalize_url(url)
            if normalized_url not in visited:
                unprocessed_urls.append(normalized_url)
                
        # Save unprocessed URLs for future runs
        if unprocessed_urls:
            url_tracking.save_url_queue(seed_domain, unprocessed_urls)
            logger.info(f"Saved {len(unprocessed_urls)} URLs for future runs")
            
        duration = time.time() - start_time
        
        crawl_stats.update({
            "processed": processed_count,
            "relevant": relevant_count,
            "failed": failed_count,
            "skipped": skipped_count,  # Add count of skipped URLs due to queue limit
            "duration_seconds": duration,
            "end_time": datetime.now().isoformat(),
            "saved_urls": len(unprocessed_urls)
        })
        
        return crawl_stats
    
    def run_workflow(self, seed_url: str, max_urls: int = 50) -> Dict:
        """
        Run a complete workflow starting from a seed URL.
        Process URLs using breadth-first search up to max_urls.
        """
        workflow_start = time.time()
        
        # Get seed domain for prioritization
        seed_domain = urlparse(seed_url).netloc
        
        # Reset tracking
        self.visited_urls = set()
        self.queued_urls = set()
        self.url_queue = []
        
        # Check if the seed URL has already been processed
        seed_already_processed = self._is_url_processed(seed_url)
        
        # Where to start the crawl from
        start_urls = []
        
        if seed_already_processed:
            logger.info(f"Seed URL {seed_url} has already been processed")
            
            # Try to get saved URLs from previous runs first
            saved_urls = url_tracking.get_url_queue(seed_domain)
            
            if saved_urls:
                # Filter out any URLs that have been processed since they were saved
                unprocessed_urls = [url for url in saved_urls if not self._is_url_processed(url)]
                
                if unprocessed_urls:
                    logger.info(f"Using {len(unprocessed_urls)} URLs from saved queue")
                    start_urls = unprocessed_urls
                else:
                    logger.warning("All saved URLs have been processed since they were saved")
            
            # If we have no URLs yet, try to find alternative URLs
            if not start_urls:
                logger.info("Looking for alternative URLs to process")
                alternative_urls = self._find_alternative_urls(seed_url)
                
                if alternative_urls:
                    logger.info(f"Found {len(alternative_urls)} alternative URLs to process")
                    start_urls = alternative_urls
                else:
                    logger.warning("No alternative URLs found, ending workflow run")
        else:
            # Seed URL not processed before, use it as start
            start_urls = [seed_url]
        
        # Prepare workflow stats
        workflow_stats = {
            "seed_url": seed_url,
            "start_time": datetime.now().isoformat(),
            "urls_processed": 0,
            "relevant_found": 0,
            "failed": 0
        }
        
        # If we have no URLs to process, return early
        if not start_urls:
            logger.warning("No URLs to process")
            workflow_stats["urls_processed"] = 0
            workflow_stats["duration_seconds"] = time.time() - workflow_start
            workflow_stats["end_time"] = datetime.now().isoformat()
            workflow_stats["status"] = "no_urls"
            return workflow_stats
        
        # Run the BFS crawl
        max_depth = self.config.get('crawler', {}).get('max_depth', 3)
        max_queue_size = self.config.get('crawler', {}).get('max_queue_size', 5000)
        crawl_result = self._run_bfs_crawl(start_urls[0], max_urls, max_depth, max_queue_size)
        
        # Update workflow stats with crawl results
        workflow_stats.update({
            "urls_processed": crawl_result["processed"],
            "relevant_found": crawl_result["relevant"],
            "failed": crawl_result["failed"],
            "end_time": crawl_result["end_time"],
            "duration_seconds": crawl_result["duration_seconds"],
            "saved_urls": crawl_result.get("saved_urls", 0)
        })
        
        logger.info(f"Workflow completed in {workflow_stats['duration_seconds']:.2f} seconds")
        logger.info(f"Processed: {workflow_stats['urls_processed']}, "
                   f"Relevant: {workflow_stats['relevant_found']}, "
                   f"Failed: {workflow_stats['failed']}")
        
        return workflow_stats

def main():
    """Entry point for command line execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run the Vigil pipeline")
    parser.add_argument("url", help="Seed URL to start the crawling process")
    parser.add_argument("--max", type=int, default=50, help="Maximum number of URLs to process")
    parser.add_argument("--config", help="Path to configuration file", default="config.yml")
    parser.add_argument("--model", help="Path to ML model file (overrides config)", default=None)
    parser.add_argument("--vectorizer", help="Path to vectorizer file (overrides config)", default=None)
    
    args = parser.parse_args()
    
    pipeline = Pipeline(config_path=args.config, model_path=args.model, vectorizer_path=args.vectorizer)
    results = pipeline.run_workflow(args.url, args.max)
    
    print("\n=== Workflow Results ===")
    print(f"Seed URL: {results['seed_url']}")
    print(f"Duration: {results['duration_seconds']:.2f} seconds")
    print(f"URLs Processed: {results['urls_processed']}")
    print(f"Relevant Articles: {results['relevant_found']}")
    print(f"Failed: {results['failed']}")

if __name__ == "__main__":
    main()
