"""
Script to run Vigil workflows.
Provides a command-line interface to the workflow system.
"""
import argparse
import json
import os
import time
import sys
import logging

from vigil.core.pipeline import Pipeline
from vigil.logging import setup_logging
from vigil.config import load_config

def run_workflow(url, max_urls=50, output_file=None, config_path=None, model_path=None, vectorizer_path=None, verbose=False):
    """Run a complete workflow and optionally save results to a file."""
    # Load configuration
    config = load_config()
    if config_path:
        try:
            config.load_from_file(config_path)
        except Exception as e:
            print(f"Error loading config file: {e}")
            sys.exit(1)
    
    # Set up logging
    log_config = config.config.get('logging', {})
    if verbose:
        log_config['console_level'] = 'DEBUG'

    # Initialize logging system and get the logger
    logger = setup_logging(log_config)
    
    try:
        # Ensure database directory exists
        from vigil.database.connection import DB_PATH
        
        db_dir = os.path.dirname(DB_PATH)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
            logger.info(f"Created database directory: {db_dir}")
        
        # Initialize pipeline
        pipeline = Pipeline(config_path=config_path, model_path=model_path, vectorizer_path=vectorizer_path)
        
        # Run workflow
        logger.info(f"Starting workflow with seed URL: {url}")
        start_time = time.time()
        results = pipeline.run_workflow(url, max_urls)
        duration = time.time() - start_time
        
        # Add additional info
        results["duration_formatted"] = f"{duration:.2f} seconds"
        
        # Print summary
        print("\n=== Workflow Results ===")
        print(f"Seed URL: {url}")
        print(f"Duration: {duration:.2f} seconds")
        print(f"URLs Processed: {results['urls_processed']}")
        print(f"Relevant Articles: {results['relevant_found']}")
        print(f"Failed: {results['failed']}")
        
        # Save to file if requested
        if output_file:
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"Results saved to {output_file}")
        
        return results
    except Exception as e:
        logger.error(f"Error initializing workflow: {str(e)}")
        raise

def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Run Vigil workflows")
    parser.add_argument("url", help="Seed URL to crawl")
    parser.add_argument("--max", type=int, default=50, help="Maximum URLs to process")
    parser.add_argument("--output", help="Output file for results (JSON)")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--model", help="Path to ML model file (overrides config)")
    parser.add_argument("--vectorizer", help="Path to vectorizer file (overrides config)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    try:
        run_workflow(args.url, args.max, args.output, args.config, args.model, args.vectorizer, args.verbose)
    except KeyboardInterrupt:
        print("\nWorkflow interrupted by user.")
    except Exception as e:
        # Initialize minimal logging if not already done
        minimal_logger = logging.getLogger('vigil.scripts.workflow')
        minimal_logger.error(f"Error running workflow: {str(e)}")
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
