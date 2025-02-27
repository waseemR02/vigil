"""
Command-line tool for preparing machine learning datasets.

This script loads labeled articles, preprocesses them, 
extracts features, and saves the prepared dataset.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Add parent directory to sys.path if running as script
if __name__ == "__main__" and __package__ is None:
    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, parent_dir)

from vigil.config import load_config
from vigil.logging import setup_logging
from vigil.model import prepare_dataset, TextPreprocessor, DataLoader, FeatureExtractor, DatasetPreparer


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="VIGIL ML Dataset Preparation Tool")
    
    parser.add_argument("--config", "-c", 
                        help="Path to config file (default: config.yml)")
    
    parser.add_argument("--db", "-d",
                        help="Path to SQLite database")
    
    parser.add_argument("--input", "-i",
                        help="Path to labeled JSON file (used if --db not provided)")
    
    parser.add_argument("--output-dir", "-o", default="dataset",
                        help="Directory to save the prepared dataset")
    
    parser.add_argument("--test-size", "-t", type=float, default=0.2,
                        help="Proportion of data to use for testing (default: 0.2)")
    
    parser.add_argument("--no-tfidf", action="store_true",
                        help="Use CountVectorizer instead of TF-IDF")
    
    parser.add_argument("--max-features", "-m", type=int, default=10000,
                        help="Maximum number of features to extract (default: 10000)")
    
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose output")
    
    return parser.parse_args()


def main():
    """Main entry point."""
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
    logger.info("VIGIL ML Dataset Preparation Tool starting...")
    
    # Get database path
    db_path = args.db
    if not db_path and 'storage' in config.config:
        db_path = config.config['storage'].get('db_path')
    
    # Get input file path
    file_path = args.input
    
    # Ensure we have at least one data source
    if not db_path and not file_path:
        logger.error("No data source provided. Use --db or --input")
        sys.exit(1)
    
    # Generate dataset directory name
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(args.output_dir, f"dataset-{timestamp}")
    
    logger.info(f"Preparing dataset from {'database' if db_path else 'file'}")
    logger.info(f"Output directory: {output_dir}")
    
    # Prepare the dataset
    start_time = time.time()
    success = prepare_dataset(
        db_path=db_path,
        file_path=file_path,
        output_dir=output_dir,
        test_size=args.test_size,
        use_tfidf=not args.no_tfidf,
        max_features=args.max_features
    )
    end_time = time.time()
    
    if success:
        logger.info(f"Dataset preparation completed in {end_time - start_time:.2f} seconds")
        logger.info(f"Dataset saved to: {output_dir}")
    else:
        logger.error("Dataset preparation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
