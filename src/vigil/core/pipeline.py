"""
Pipeline module for the Vigil cybersecurity platform.
Connects crawler, ML analyzer, and database components.
"""
import logging
import time
import yaml
import os
from datetime import datetime
from urllib.parse import urlparse
from typing import Dict, Tuple

from bs4 import BeautifulSoup
import requests

from vigil.data_collection.crawler import Crawler
from vigil.data_collection.content_extractor import ContentExtractor
from vigil.model.training import ContentPredictor
from vigil.database.connection import init_db
from vigil.database import queries

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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
        self.config = self._load_config(config_path)
        
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
        
        logger.info("Pipeline initialized")
    
    def _load_config(self, config_path=None):
        """Load configuration from YAML file."""
        if not config_path:
            config_path = "config.yml"
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            logger.info("Using default configuration")
            return {
                'crawler': {
                    'user_agent': "VIGIL Cybersecurity News Crawler/0.1",
                    'timeout': 30,
                    'delay': 1.5,
                    'max_retries': 3,
                    'max_depth': 2,
                    'max_urls': 50
                },
                'storage': {
                    'file_store_path': "data",
                    'db_path': "data/vigil_data.db"
                },
                'model': {
                    'path': None,  # Default to None, will be handled in __init__
                    'vectorizer_path': None  # Default to None, will be handled in __init__
                }
            }
    
    def process_url(self, url: str) -> Tuple[bool, Dict]:
        """
        Process a single URL through the pipeline:
        1. Crawl and extract content
        2. Check if content is relevant (ML classification)
        3. Store relevant content in the database
        4. Return processing status and metadata
        """
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
            
            # Parse URL to clean it
            parsed = urlparse(full_url)
            if not parsed.scheme or not parsed.netloc:
                continue
                
            # Filter by domain if allowed_domains is specified
            if allowed_domains and parsed.netloc not in allowed_domains:
                continue
                
            # Skip already visited or queued URLs
            if full_url in self.visited_urls or full_url in self.queued_urls:
                continue
                
            links.append(full_url)
        
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
        
        # Add URLs to the queue if not already visited or queued
        added_count = 0
        for url in prioritized_urls:
            if url not in self.visited_urls and url not in self.queued_urls:
                # Only add if we're still under the limit
                if len(self.url_queue) < max_queue_size:
                    self.url_queue.append(url)
                    self.queued_urls.add(url)
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
    
    def run_workflow(self, seed_url: str, max_urls: int = 50) -> Dict:
        """
        Run a complete workflow starting from a seed URL.
        Process URLs in a breadth-first manner up to max_urls.
        """
        workflow_start = time.time()
        
        # Reset tracking
        self.visited_urls = set()
        self.queued_urls = set()
        self.url_queue = [seed_url]
        self.queued_urls.add(seed_url)
        
        # Get seed domain for prioritization
        seed_domain = urlparse(seed_url).netloc
        
        # Statistics
        processed_count = 0
        relevant_count = 0
        failed_count = 0
        
        workflow_stats = {
            "seed_url": seed_url,
            "start_time": datetime.now().isoformat(),
            "urls_processed": 0,
            "relevant_found": 0,
            "failed": 0,
            "queue_peak": 1
        }
        
        logger.info(f"Starting workflow with seed URL: {seed_url}")
        
        while self.url_queue and processed_count < max_urls:
            current_url = self.url_queue.pop(0)
            self.queued_urls.remove(current_url)  # Remove from queued set
            
            if current_url in self.visited_urls:
                continue
                
            success, result = self.process_url(current_url)
            processed_count += 1
            
            if success:
                if result.get("relevant", False):
                    relevant_count += 1
                    
                    # Add new URLs to the queue with prioritization and deduplication
                    if "new_urls_list" in result:
                        # Limit queue size to a reasonable number to prevent explosion
                        max_queue_size = self.config.get('crawler', {}).get('max_queue_size', 1000)
                        added = self._add_to_queue(result["new_urls_list"], seed_domain, max_queue_size)
                        if added > 0:
                            logger.debug(f"Added {added} new URLs to queue (total queue size: {len(self.url_queue)})")
                        
                        # Update the peak queue size for reporting
                        workflow_stats["queue_peak"] = max(workflow_stats["queue_peak"], len(self.url_queue))
            else:
                failed_count += 1
            
            logger.info(f"Processed {processed_count}/{max_urls} URLs. " 
                      f"Queue size: {len(self.url_queue)}, "
                      f"Relevant: {relevant_count}, Failed: {failed_count}")
        
        workflow_end = time.time()
        duration = workflow_end - workflow_start
        
        workflow_stats.update({
            "urls_processed": processed_count,
            "relevant_found": relevant_count,
            "failed": failed_count,
            "end_time": datetime.now().isoformat(),
            "duration_seconds": duration,
            "urls_visited": len(self.visited_urls),
            "queue_size_final": len(self.url_queue)
        })
        
        logger.info(f"Workflow completed in {duration:.2f} seconds")
        logger.info(f"Processed: {processed_count}, Relevant: {relevant_count}, Failed: {failed_count}")
        
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
