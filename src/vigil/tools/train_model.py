"""
Command-line tool for training machine learning models.

This script trains classifiers on prepared datasets to predict
whether articles are relevant to cybersecurity incidents.
"""

import argparse
import os
import sys
import json

# Add parent directory to sys.path if running as script
if __name__ == "__main__" and __package__ is None:
    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, parent_dir)

from vigil.config import load_config
from vigil.logging import setup_logging
from vigil.model import train_model


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="VIGIL Model Training Tool")
    
    parser.add_argument("--config", "-c", 
                        help="Path to config file (default: config.yml)")
    
    parser.add_argument("--dataset", "-d", required=True,
                        help="Path to prepared dataset directory")
    
    parser.add_argument("--output-dir", "-o",
                        help="Directory to save the trained model")
    
    parser.add_argument("--model-type", "-m", default="logistic_regression",
                        choices=["logistic_regression", "linear_svc"],
                        help="Type of model to train")
    
    parser.add_argument("--params", "-p",
                        help="JSON string or file path containing model parameters")
    
    parser.add_argument("--test", "-t", action="store_true",
                        help="Test the trained model with sample predictions")
    
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose output")
                        
    return parser.parse_args()


def load_model_params(params_arg):
    """Load model parameters from JSON string or file."""
    if not params_arg:
        return {}
        
    # Check if it's a file path
    if os.path.exists(params_arg):
        try:
            with open(params_arg, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in file {params_arg}")
            sys.exit(1)
    
    # Try parsing as a JSON string
    try:
        return json.loads(params_arg)
    except json.JSONDecodeError:
        print(f"Error: Provided string is not valid JSON: {params_arg}")
        sys.exit(1)


def test_model_predictions(predictor):
    """Test the trained model with some sample predictions."""
    from rich.console import Console
    from rich.table import Table
    
    console = Console()
    
    console.print("\n[bold]Testing model with sample texts:[/bold]")
    
    # Example texts for testing predictions
    test_texts = [
        "We are excited to announce our new website launch.",
        "Critical security vulnerability found in popular software allows remote code execution.",
        "The company reported strong financial results for the quarter.",
        "Ransomware attack cripples hospital network, patient data compromised.",
        "Weather forecast predicts sunny conditions for the weekend.",
        "Zero-day vulnerability in browser extension allows attackers to steal credentials."
    ]
    
    results_table = Table(title="Prediction Results")
    results_table.add_column("Text", style="cyan", no_wrap=False)
    results_table.add_column("Prediction", style="bold")
    results_table.add_column("Confidence", justify="right")
    
    for text in test_texts:
        # Get prediction with score
        is_relevant, confidence = predictor.predict_with_score(text)
        
        prediction_str = "[green]Relevant[/green]" if is_relevant else "[red]Not relevant[/red]"
        confidence_str = f"{confidence:.2f}"
        
        # Truncate text if too long for display
        display_text = text[:80] + "..." if len(text) > 80 else text
        
        results_table.add_row(display_text, prediction_str, confidence_str)
    
    console.print(results_table)


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
    logger.info("VIGIL Model Training Tool starting...")
    
    # Validate dataset path
    dataset_path = args.dataset
    if not os.path.exists(dataset_path) or not os.path.isdir(dataset_path):
        logger.error(f"Dataset path not found or not a directory: {dataset_path}")
        sys.exit(1)
    
    # Set output directory
    output_dir = args.output_dir
    if not output_dir:
        output_dir = os.path.join(dataset_path, "models")
    
    # Load model parameters
    model_params = load_model_params(args.params)
    logger.info(f"Using model parameters: {model_params}")
    
    # Train the model
    logger.info(f"Training {args.model_type} model on dataset: {dataset_path}")
    
    result = train_model(
        dataset_path=dataset_path,
        model_type=args.model_type,
        model_params=model_params,
        output_dir=output_dir
    )
    
    if not result:
        logger.error("Model training failed")
        sys.exit(1)
    
    # Show metrics
    metrics = result.get('metrics', {})
    if metrics:
        logger.info("Model performance:")
        logger.info(f"  Accuracy:  {metrics.get('accuracy', 0):.4f}")
        logger.info(f"  Precision: {metrics.get('precision', 0):.4f}")
        logger.info(f"  Recall:    {metrics.get('recall', 0):.4f}")
        logger.info(f"  F1 Score:  {metrics.get('f1_score', 0):.4f}")
    
    # Show saved files
    saved_files = result.get('saved_files', {})
    if saved_files:
        logger.info("Files saved:")
        for file_type, file_path in saved_files.items():
            logger.info(f"  {file_type}: {file_path}")
    
    # Test predictions if requested
    if args.test and 'predictor' in result:
        test_model_predictions(result['predictor'])


if __name__ == "__main__":
    main()
