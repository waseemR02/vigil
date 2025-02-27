"""
Main entry point for the VIGIL application, demonstrating the crawler functionality.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

from vigil.config import load_config
from vigil.data_collection.crawler import Crawler
from vigil.logging import setup_logging


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="VIGIL Cybersecurity News Crawler")
    
    parser.add_argument("--config", "-c", 
                        help="Path to config file (default: config.yml)")
    
    parser.add_argument("--output", "-o", 
                        help="Output directory for crawled data")
    
    parser.add_argument("--max-urls", "-m", type=int,
                        help="Maximum number of URLs to crawl")
    
    parser.add_argument("--depth", "-d", type=int,
                        help="Maximum crawl depth")
    
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose output")
    
    parser.add_argument("--url", "-u", action="append",
                        help="Add a start URL (can be used multiple times)")
    
    return parser.parse_args()


def save_crawled_data(results, output_dir):
    """Save crawled data to output directory."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate a timestamp for the results directory
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    results_dir = os.path.join(output_dir, f"crawl-{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Save a summary JSON file
    summary = []
    for url, data in results.items():
        # Don't include the soup and full content in summary
        summary_item = {
            'url': data['url'],
            'depth': data['depth'],
            'status': data['status'],
            'timestamp': data['timestamp'],
            'has_content': data['content'] is not None
        }
        
        # Add title from extracted content if available
        if data.get('extracted_content') and data['extracted_content'].get('title'):
            summary_item['title'] = data['extracted_content']['title']
        elif data['soup']:
            title_tag = data['soup'].find('title')
            if title_tag:
                summary_item['title'] = title_tag.get_text().strip()
                
        summary.append(summary_item)
    
    # Write the summary JSON
    with open(os.path.join(results_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save raw HTML content
    content_dir = os.path.join(results_dir, 'html')
    os.makedirs(content_dir, exist_ok=True)
    
    # Save extracted content as JSON
    extracted_dir = os.path.join(results_dir, 'extracted')
    os.makedirs(extracted_dir, exist_ok=True)
    
    for url, data in results.items():
        # Create a safe filename from URL
        filename = ''.join(c if c.isalnum() else '_' for c in url)
        filename = filename[-150:] if len(filename) > 150 else filename  # Limit length
        
        # Save raw HTML if available
        if data['content']:
            html_filename = f"{filename}.html"
            with open(os.path.join(content_dir, html_filename), 'w', encoding='utf-8') as f:
                f.write(data['content'])
        
        # Save extracted content if available
        if data.get('extracted_content'):
            json_filename = f"{filename}.json"
            with open(os.path.join(extracted_dir, json_filename), 'w', encoding='utf-8') as f:
                # Remove BeautifulSoup object before serializing to JSON
                extracted_content = data['extracted_content']
                json.dump(extracted_content, f, indent=2)
    
    return results_dir


def main():
    """Main entry point."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Load configuration
    config = load_config()
    if args.config:
        try:
            config.load_from_file(args.config)
        except Exception as e:
            print(f"Error loading config file: {e}")
            sys.exit(1)
    
    # Set up logging
    log_config = config.config.get('logging', {})
    if args.verbose:
        log_config['console_level'] = 'DEBUG'
    
    logger = setup_logging(log_config)
    logger.info("VIGIL Cybersecurity News Crawler starting...")
    
    # Get crawler configuration
    crawler_config = config.config.get('crawler', {})
    
    # Override with command line arguments
    if args.max_urls:
        crawler_config['max_urls'] = args.max_urls
    
    if args.depth:
        crawler_config['max_depth'] = args.depth
    
    if args.output:
        crawler_config['output_dir'] = args.output
    
    # Default values if not in config
    max_urls = crawler_config.get('max_urls', 50)
    max_depth = crawler_config.get('max_depth', 2)
    output_dir = crawler_config.get('output_dir', 'crawled_data')
    
    # Get start URLs, prioritizing command line arguments
    start_urls = args.url if args.url else crawler_config.get('start_urls', [])
    if not start_urls:
        logger.error("No start URLs specified!")
        sys.exit(1)
    
    # Initialize crawler
    logger.info(f"Initializing crawler with user agent: {crawler_config.get('user_agent')}")
    crawler = Crawler(
        user_agent=crawler_config.get('user_agent'),
        timeout=crawler_config.get('timeout', 30),
        delay=crawler_config.get('delay', 1.0),
        max_retries=crawler_config.get('max_retries', 3)
    )
    
    # Start crawling
    logger.info(f"Starting crawl with {len(start_urls)} seed URLs, max depth {max_depth}, max URLs {max_urls}")
    logger.info(f"Seed URLs: {', '.join(start_urls)}")
    
    start_time = time.time()
    results = crawler.crawl(
        start_urls=start_urls,
        max_depth=max_depth,
        allowed_domains=crawler_config.get('allowed_domains'),
        url_patterns=crawler_config.get('url_patterns'),
        max_urls=max_urls
    )
    end_time = time.time()
    
    # Save results
    results_dir = save_crawled_data(results, output_dir)
    
    # Print summary
    success_count = sum(1 for data in results.values() if data['status'] == 200)
    extracted_count = sum(1 for data in results.values() if data.get('extracted_content') and data['extracted_content'].get('content'))
    
    logger.info("Crawl completed!")
    logger.info(f"Time taken: {end_time - start_time:.2f} seconds")
    logger.info(f"URLs processed: {len(results)}")
    logger.info(f"Successful fetches: {success_count}")
    logger.info(f"Failed fetches: {len(results) - success_count}")
    logger.info(f"Articles with extracted content: {extracted_count}")
    logger.info(f"Results saved to: {results_dir}")


if __name__ == "__main__":
    main()
