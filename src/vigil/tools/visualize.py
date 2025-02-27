"""
Command-line tool for generating visualizations and reports.

This script creates visualizations of model performance, dataset
statistics, and generates HTML reports with the results.
"""

import argparse
import logging
import os
import sys
import webbrowser
from pathlib import Path

# Add parent directory to sys.path if running as script
if __name__ == "__main__" and __package__ is None:
    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, parent_dir)

from vigil.config import load_config
from vigil.logging import setup_logging
from vigil.model import generate_model_report


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="VIGIL Visualization Tool")
    
    parser.add_argument("--config", "-c", 
                        help="Path to config file (default: config.yml)")
    
    parser.add_argument("--model", "-m", required=True,
                        help="Path to the trained model (.pkl file)")
    
    parser.add_argument("--dataset", "-d", required=True,
                        help="Path to the dataset directory")
    
    parser.add_argument("--output-dir", "-o", default="reports",
                        help="Directory to save the report and visualizations")
    
    parser.add_argument("--open", action="store_true",
                        help="Open the report in a web browser after generation")
    
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
    logger.info("VIGIL Visualization Tool starting...")
    
    # Check if paths exist
    if not os.path.exists(args.model):
        logger.error(f"Model file not found: {args.model}")
        sys.exit(1)
    
    if not os.path.exists(args.dataset) or not os.path.isdir(args.dataset):
        logger.error(f"Dataset directory not found: {args.dataset}")
        sys.exit(1)
    
    # Generate the report
    logger.info(f"Generating report for model: {args.model}")
    logger.info(f"Using dataset: {args.dataset}")
    logger.info(f"Output directory: {args.output_dir}")
    
    report_path = generate_model_report(
        model_path=args.model,
        dataset_path=args.dataset,
        output_dir=args.output_dir
    )
    
    if not report_path:
        logger.error("Failed to generate report")
        sys.exit(1)
    
    logger.info(f"Report generated successfully: {report_path}")
    
    # Open in browser if requested
    if args.open:
        logger.info("Opening report in web browser...")
        try:
            webbrowser.open(f"file://{os.path.abspath(report_path)}")
        except Exception as e:
            logger.error(f"Failed to open browser: {e}")


if __name__ == "__main__":
    main()
