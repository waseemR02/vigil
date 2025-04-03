"""
Command-line tool for classifying cybersecurity content.

This script uses a trained model to classify articles or text
as relevant or not relevant to cybersecurity incidents.
"""

import argparse
import json
import os
import sys

from rich.console import Console
from rich.panel import Panel

# Add parent directory to sys.path if running as script
if __name__ == "__main__" and __package__ is None:
    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, parent_dir)

from vigil.config import load_config
from vigil.logging import setup_logging
from vigil.model import ContentPredictor
from vigil.storage import init_storage


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="VIGIL Content Classification Tool")
    
    parser.add_argument("--config", "-c", 
                        help="Path to config file (default: config.yml)")
    
    parser.add_argument("--model", "-m", required=True,
                        help="Path to trained model (.pkl file)")
    
    parser.add_argument("--vectorizer", "-v",
                        help="Path to vectorizer (.pkl file), if not using the default")
    
    parser.add_argument("--db", "-d",
                        help="Path to SQLite database for loading articles")
    
    parser.add_argument("--article-id", "-a", type=int,
                        help="ID of article to classify (requires --db)")
    
    parser.add_argument("--input-file", "-i",
                        help="Path to text file to classify")
    
    parser.add_argument("--text", "-t",
                        help="Text string to classify")
    
    parser.add_argument("--batch", "-b",
                        help="Path to file with one text per line for batch classification")
    
    parser.add_argument("--output", "-o",
                        help="Path to output file for batch results (JSON format)")
    
    parser.add_argument("--interactive", action="store_true",
                        help="Run in interactive mode")
    
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")
                        
    return parser.parse_args()


def find_vectorizer(model_path):
    """Find the associated vectorizer for a model."""
    # Try common locations for the vectorizer
    model_dir = os.path.dirname(model_path)
    
    # Check if there's a models directory within a dataset directory
    if os.path.basename(model_dir) == 'models':
        dataset_dir = os.path.dirname(model_dir)
        vectorizer_path = os.path.join(dataset_dir, 'vectorizer.pkl')
        if os.path.exists(vectorizer_path):
            return vectorizer_path
    
    # Check for vectorizer in the same directory as the model
    vectorizer_path = os.path.join(model_dir, 'vectorizer.pkl')
    if os.path.exists(vectorizer_path):
        return vectorizer_path
    
    # Check if there's a model name that we can use to find the metadata
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    metadata_path = os.path.join(model_dir, f"{model_name}_metadata.json")
    
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                # Check if metadata has vectorizer path
                if 'vectorizer_path' in metadata:
                    vectorizer_path = metadata['vectorizer_path']
                    if os.path.exists(vectorizer_path):
                        return vectorizer_path
        except:
            pass
    
    # Look for any vectorizer.pkl in the directory
    for filename in os.listdir(model_dir):
        if 'vectorizer' in filename.lower() and filename.endswith('.pkl'):
            return os.path.join(model_dir, filename)
    
    return None


def classify_from_file(predictor, file_path, output_file=None):
    """Classify texts from a file, one text per line."""
    console = Console()
    
    try:
        # Read texts from file
        with open(file_path, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        if not texts:
            console.print("[yellow]No texts found in the file[/yellow]")
            return
        
        # Classify each text
        results = []
        for text in texts:
            is_relevant, confidence = predictor.predict_with_score(text)
            results.append({
                'text': text,
                'is_relevant': bool(is_relevant),
                'confidence': float(confidence)
            })
        
        # Show summary
        relevant_count = sum(1 for r in results if r['is_relevant'])
        console.print(f"\nClassified {len(results)} texts: "
                    f"[green]{relevant_count} relevant[/green], "
                    f"[red]{len(results) - relevant_count} not relevant[/red]")
        
        # Save to output file if specified
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            console.print(f"[green]Results saved to {output_file}[/green]")
            
    except Exception as e:
        console.print(f"[red]Error classifying from file: {str(e)}[/red]")


def classify_from_db(predictor, db_path, article_id):
    """Classify an article from the database."""
    console = Console()
    
    try:
        # Initialize storage
        storage = init_storage({"storage": {"db_path": db_path}})
        db_store = storage.get('db_store')
        
        if not db_store:
            console.print("[red]Failed to initialize database storage[/red]")
            return
        
        # Retrieve article
        article = db_store.get_article(article_id)
        
        if not article:
            console.print(f"[red]Article with ID {article_id} not found[/red]")
            return
        
        # Classify the article
        evaluation = predictor.evaluate_text(article['content'])
        
        # Display the article and classification
        console.print(Panel(f"[bold cyan]Article ID:[/bold cyan] {article_id}", 
                           title="Article Information", border_style="blue"))
        
        # Display article title
        console.print(Panel(article['title'], title="Title", border_style="green"))
        
        # Display classification result
        relevance_str = "[green]RELEVANT[/green]" if evaluation['relevant'] else "[red]NOT RELEVANT[/red]"
        console.print(Panel(
            f"[bold]Classification:[/bold] {relevance_str}\n"
            f"[bold]Confidence:[/bold] {evaluation['confidence']:.2f}",
            title="Classification Result", border_style="yellow"
        ))
        
        # Display the first part of content
        content_preview = article['content'][:500] + "..." if len(article['content']) > 500 else article['content']
        console.print(Panel(content_preview, title="Content Preview", border_style="blue"))
        
    except Exception as e:
        console.print(f"[red]Error classifying from database: {str(e)}[/red]")


def classify_text(predictor, text):
    """Classify a single text string."""
    console = Console()
    
    try:
        # Classify the text
        evaluation = predictor.evaluate_text(text)
        
        # Display classification result
        relevance_str = "[green]RELEVANT[/green]" if evaluation['relevant'] else "[red]NOT RELEVANT[/red]"
        console.print(Panel(
            f"[bold]Classification:[/bold] {relevance_str}\n"
            f"[bold]Confidence:[/bold] {evaluation['confidence']:.2f}",
            title="Classification Result", border_style="yellow"
        ))
        
        # Display the text
        text_preview = text[:200] + "..." if len(text) > 200 else text
        console.print(Panel(text_preview, title="Text", border_style="blue"))
        
        return evaluation
        
    except Exception as e:
        console.print(f"[red]Error classifying text: {str(e)}[/red]")
        return None


def run_interactive_mode(predictor):
    """Run an interactive classification session."""
    console = Console()
    
    console.print(Panel.fit(
        "[bold]VIGIL Interactive Classification Tool[/bold]\n\n"
        "Enter text to classify it as relevant or not relevant to cybersecurity incidents.\n"
        "Type 'exit' or 'quit' to end the session.",
        title="Interactive Mode",
        border_style="green"
    ))
    
    while True:
        console.print("\n[bold cyan]Enter text to classify (or 'exit' to quit):[/bold cyan]")
        text = input("> ")
        
        if text.lower() in ('exit', 'quit', 'q'):
            console.print("[yellow]Exiting interactive mode[/yellow]")
            break
        
        if not text.strip():
            continue
        
        # Classify and show result
        classify_text(predictor, text)


def batch_classify_from_db(predictor, db_path, output_file=None, limit=100):
    """Classify all articles in the database and optionally save results."""
    console = Console()
    
    try:
        # Initialize storage
        storage = init_storage({"storage": {"db_path": db_path}})
        db_store = storage.get('db_store')
        
        if not db_store:
            console.print("[red]Failed to initialize database storage[/red]")
            return
        
        # Get all articles
        articles = db_store.search_articles(limit=limit)
        
        if not articles:
            console.print("[yellow]No articles found in the database[/yellow]")
            return
        
        console.print(f"[cyan]Classifying {len(articles)} articles from database...[/cyan]")
        
        # Classify each article
        results = []
        for i, article in enumerate(articles):
            if i % 10 == 0:  # Show progress every 10 articles
                console.print(f"Processing article {i+1}/{len(articles)}")
                
            # Skip articles without content
            if not article.get('content'):
                continue
                
            is_relevant, confidence = predictor.predict_with_score(article['content'])
            
            results.append({
                'id': article.get('id'),
                'url': article.get('url'),
                'title': article.get('title'),
                'is_relevant': bool(is_relevant),
                'confidence': float(confidence)
            })
        
        # Show summary
        relevant_count = sum(1 for r in results if r['is_relevant'])
        console.print(f"\nClassified {len(results)} articles: "
                    f"[green]{relevant_count} relevant[/green], "
                    f"[red]{len(results) - relevant_count} not relevant[/red]")
        
        # Save to output file if specified
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            console.print(f"[green]Results saved to {output_file}[/green]")
            
        return results
            
    except Exception as e:
        console.print(f"[red]Error during batch classification: {str(e)}[/red]")
        return []


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
    logger.info("VIGIL Content Classification Tool starting...")
    
    # Initialize console
    console = Console()
    
    # Determine vectorizer path if not provided
    vectorizer_path = args.vectorizer
    if not vectorizer_path:
        # Try to find a vectorizer that matches the model
        vectorizer_path = find_vectorizer(args.model)
        if vectorizer_path:
            logger.info(f"Found matching vectorizer: {vectorizer_path}")
        else:
            # Check if there's a vectorizer in the dataset directory
            dataset_dir = os.path.dirname(os.path.dirname(args.model))
            if os.path.exists(os.path.join(dataset_dir, 'vectorizer.pkl')):
                vectorizer_path = os.path.join(dataset_dir, 'vectorizer.pkl')
                logger.info(f"Using dataset vectorizer: {vectorizer_path}")
            else:
                logger.warning("No vectorizer specified and no matching vectorizer found")
                console.print("[yellow]Warning: No vectorizer specified. Classification may fail.[/yellow]")
    
    # Initialize predictor
    try:
        predictor = ContentPredictor(model_path=args.model, vectorizer_path=vectorizer_path)
        logger.info(f"Model loaded from {args.model}")
        
        # Check if we have a feature extractor
        if not predictor.feature_extractor:
            logger.error("No feature extractor loaded. Cannot continue.")
            console.print("[red]Error: Model doesn't have an associated feature extractor.[/red]")
            console.print("[yellow]Please specify a vectorizer using the --vectorizer option.[/yellow]")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        console.print(f"[red]Error loading model: {str(e)}[/red]")
        sys.exit(1)
    
    # Handle various modes of operation
    if args.interactive:
        run_interactive_mode(predictor)
    elif args.article_id is not None:
        if not args.db:
            logger.error("--db option required with --article-id")
            console.print("[red]Error: --db option required with --article-id[/red]")
            sys.exit(1)
        classify_from_db(predictor, args.db, args.article_id)
    elif args.batch:
        classify_from_file(predictor, args.batch, args.output)
    elif args.input_file:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            text = f.read()
        classify_text(predictor, text)
    elif args.text:
        classify_text(predictor, args.text)
    else:
        logger.error("No input provided. Use --text, --input-file, --batch, --article-id, or --interactive")
        console.print("[red]Error: No input provided. Use --text, --input-file, --batch, --article-id, or --interactive[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
