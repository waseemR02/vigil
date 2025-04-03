"""
Main entry point for the VIGIL application, demonstrating the crawler functionality.
"""

import argparse
import json
import logging
import os
import sys
import time
import datetime
from pathlib import Path

from vigil.config import load_config
from vigil.data_collection.crawler import Crawler
from vigil.logging import setup_logging
from vigil.storage import init_storage


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
    
    parser.add_argument("--no-db", action="store_true",
                        help="Don't store results in database")
    
    parser.add_argument("--export", "-e", choices=["json", "csv", "none"],
                        help="Export results after crawling")
    
    return parser.parse_args()


def save_crawled_data(results, output_dir, storage=None, storage_config=None):
    """Save crawled data to output directory and database."""
    # Get logger
    logger = logging.getLogger('vigil.crawl')
    
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
    
    # Track articles to add to database
    articles_to_store = []
    
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
            
            # Add to list for database storage
            articles_to_store.append(extracted_content)
    
    # If we have database storage and auto_save is enabled, save there too
    stored_count = 0
    if storage and storage.get('db_store') and articles_to_store:
        # Check if auto_save is enabled in config
        auto_save = True  # Default to True
        if storage_config and 'auto_save' in storage_config:
            auto_save = storage_config.get('auto_save')
        
        if auto_save:
            db_store = storage['db_store']
            stored_count = db_store.add_articles_batch(articles_to_store)
            logger.info(f"Stored {stored_count} articles in database")
        else:
            logger.info("Database storage disabled (auto_save=False)")
    
    # Handle auto export if configured
    if storage and storage.get('db_store') and storage_config:
        auto_export = storage_config.get('auto_export', False)
        export_format = storage_config.get('export_format', 'json')
        
        if auto_export and export_format:
            export_file = os.path.join(results_dir, f"export.{export_format}")
            db_store = storage['db_store']
            
            if export_format == 'json':
                count = db_store.export_to_json(export_file)
                logger.info(f"Auto-exported {count} articles to {export_file}")
            elif export_format == 'csv':
                count = db_store.export_to_csv(export_file)
                logger.info(f"Auto-exported {count} articles to {export_file}")
    
    return results_dir, stored_count


def log_crawl_session(storage, start_time, end_time, urls_processed, success_count, config):
    """Log the crawl session to the database."""
    if storage and storage.get('db_store'):
        try:
            session_id = storage['db_store'].log_crawl_session(
                start_time=start_time,
                end_time=end_time,
                urls_processed=urls_processed,
                success_count=success_count,
                config=config
            )
            return session_id
        except Exception as e:
            logger = logging.getLogger('vigil.crawl')
            logger.error(f"Failed to log crawl session: {str(e)}")
    
    return None


def export_results(storage, export_format, output_dir, filters=None):
    """Export crawled data to a file."""
    logger = logging.getLogger('vigil.crawl')
    
    if not storage or not storage.get('db_store'):
        logger.warning("No database storage available for export")
        return 0
    
    if not export_format or export_format == 'none':
        return 0
        
    # Create export filename with timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    export_file = os.path.join(output_dir, f"export-{timestamp}.{export_format}")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(export_file), exist_ok=True)
    
    # Export based on format
    db_store = storage['db_store']
    count = 0
    
    if export_format == 'json':
        count = db_store.export_to_json(export_file, filters)
        logger.info(f"Exported {count} articles to {export_file}")
    elif export_format == 'csv':
        count = db_store.export_to_csv(export_file, filters)
        logger.info(f"Exported {count} articles to {export_file}")
    else:
        logger.warning(f"Unsupported export format: {export_format}")
    
    return count


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
    
    # Initialize storage if not disabled via command line
    storage = None
    storage_config = config.config.get('storage', {})
    if not args.no_db:
        storage = init_storage(config.config)
        logger.info(f"Storage initialized: file_store={storage['file_store'].base_dir}, db_store={storage['db_store'].db_path}")
    else:
        logger.info("Database storage disabled via command line")
    
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
    
    # Record start time for logging
    start_time_iso = datetime.datetime.now().isoformat()
    start_time = time.time()
    
    # Do the crawl
    results = crawler.crawl(
        start_urls=start_urls,
        max_depth=max_depth,
        allowed_domains=crawler_config.get('allowed_domains'),
        url_patterns=crawler_config.get('url_patterns'),
        max_urls=max_urls
    )
    
    # Record end time
    end_time = time.time()
    end_time_iso = datetime.datetime.now().isoformat()
    
    # Save results
    results_dir, stored_count = save_crawled_data(
        results, 
        output_dir, 
        storage=storage,
        storage_config=storage_config
    )
    
    # Log the crawl session
    if storage:
        success_count = sum(1 for data in results.values() if data['status'] == 200)
        session_id = log_crawl_session(
            storage,
            start_time=start_time_iso,
            end_time=end_time_iso,
            urls_processed=len(results),
            success_count=success_count,
            config=crawler_config
        )
        if session_id:
            logger.info(f"Crawl session logged with ID: {session_id}")
    
    # Handle explicit export request (overrides auto-export)
    if args.export and args.export != 'none' and storage:
        export_results(storage, args.export, output_dir)
    
    # Print summary
    success_count = sum(1 for data in results.values() if data['status'] == 200)
    extracted_count = sum(1 for data in results.values() if data.get('extracted_content') and data['extracted_content'].get('content'))
    
    logger.info("Crawl completed!")
    logger.info(f"Time taken: {end_time - start_time:.2f} seconds")
    logger.info(f"URLs processed: {len(results)}")
    logger.info(f"Successful fetches: {success_count}")
    logger.info(f"Failed fetches: {len(results) - success_count}")
    logger.info(f"Articles with extracted content: {extracted_count}")
    logger.info(f"Articles stored in database: {stored_count}")
    logger.info(f"Results saved to: {results_dir}")

