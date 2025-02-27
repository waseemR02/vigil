"""
Command-line tool for manually labeling crawled articles.
"""

import argparse
import json
import logging
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table

# Add parent directory to sys.path if running as script
if __name__ == "__main__" and __package__ is None:
    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, parent_dir)

from vigil.config import load_config
from vigil.storage import init_storage
from vigil.logging import setup_logging


class LabelingTool:
    """Tool for manually labeling articles."""
    
    def __init__(self, storage_config: Dict = None, output_dir: str = "dataset"):
        """
        Initialize the labeling tool.
        
        Args:
            storage_config: Configuration for the storage backend
            output_dir: Directory to save labeled data
        """
        self.logger = logging.getLogger('vigil.tools.labeler')
        self.console = Console()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize storage
        if storage_config is None:
            storage_config = {}
        self.storage = init_storage({"storage": storage_config})
        
        # State data for labeling session
        self.session_data = {
            "total_labeled": 0,
            "labeled_relevant": 0,
            "labeled_not_relevant": 0,
            "skipped": 0,
            "start_time": datetime.now().isoformat(),
            "last_article_id": None,
            "labeled_ids": set(),
        }
        
        # Pre-fetched articles to label
        self.articles_to_label = []
        
        # Current batch of articles being processed
        self.current_batch = []
        
    def load_progress(self, session_file: str) -> bool:
        """
        Load progress from a previous labeling session.
        
        Args:
            session_file: Path to the session file
            
        Returns:
            True if loaded successfully, False otherwise
        """
        session_path = Path(session_file)
        if not session_path.exists():
            self.logger.warning(f"Session file not found: {session_file}")
            return False
        
        try:
            with open(session_path, 'rb') as f:
                self.session_data = pickle.load(f)
            
            self.console.print(f"[green]Loaded session with {self.session_data['total_labeled']} labeled articles[/green]")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading session: {str(e)}")
            self.console.print(f"[red]Error loading session: {str(e)}[/red]")
            return False
    
    def save_progress(self, session_file: str) -> bool:
        """
        Save the current progress to a session file.
        
        Args:
            session_file: Path to save the session file
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Update the last save time
            self.session_data["last_saved"] = datetime.now().isoformat()
            
            with open(session_file, 'wb') as f:
                pickle.dump(self.session_data, f)
            
            self.console.print(f"[green]Progress saved to {session_file}[/green]")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving session: {str(e)}")
            self.console.print(f"[red]Error saving session: {str(e)}[/red]")
            return False
    
    def fetch_unlabeled_articles(self, batch_size: int = 20) -> List[Dict[str, Any]]:
        """
        Fetch a batch of unlabeled articles from the database.
        
        Args:
            batch_size: Number of articles to fetch
            
        Returns:
            List of articles for labeling
        """
        if not self.storage or not self.storage.get('db_store'):
            self.console.print("[red]No database storage available[/red]")
            return []
        
        db_store = self.storage['db_store']
        
        try:
            # Get articles that don't have labels or have empty labels
            conn = db_store._get_connection()
            cursor = conn.cursor()
            
            # Query for articles without labels or with empty label object
            cursor.execute("""
                SELECT * FROM articles 
                WHERE labels IS NULL 
                   OR labels = '{}' 
                   OR labels = ''
                ORDER BY date_extracted DESC
                LIMIT ?
            """, (batch_size,))
            
            articles = []
            for row in cursor.fetchall():
                article = dict(row)
                
                # Skip if we've already labeled this in the current session
                if article['id'] in self.session_data['labeled_ids']:
                    continue
                    
                articles.append(article)
            
            conn.close()
            return articles
            
        except Exception as e:
            self.logger.error(f"Error fetching unlabeled articles: {str(e)}")
            self.console.print(f"[red]Error fetching unlabeled articles: {str(e)}[/red]")
            return []
    
    def display_article(self, article: Dict[str, Any]) -> None:
        """
        Display an article for labeling.
        
        Args:
            article: Article data to display
        """
        # Clear the screen for better visibility
        self.console.clear()
        
        # Create a header with article info
        header = Table.grid(expand=True)
        header.add_column("Key", style="bold cyan", no_wrap=True)
        header.add_column("Value", style="white")
        
        # Add metadata rows
        header.add_row("ID", str(article['id']))
        header.add_row("URL", article['url'])
        header.add_row("Domain", article['domain'])
        header.add_row("Date Extracted", article['date_extracted'])
        if article['publish_date']:
            header.add_row("Publish Date", article['publish_date'])
        
        # Display the header
        self.console.print(Panel(header, title=f"Article Information", border_style="blue"))
        
        # Display the title
        self.console.print(Panel(article['title'], title="Title", border_style="green", expand=False))
        
        # Display the content (first 1000 chars, then ask if user wants to see more)
        content = article['content']
        if len(content) > 1000:
            self.console.print(Panel(content[:1000] + "...", title="Content Preview", border_style="yellow"))
            if Confirm.ask("Show full content?"):
                self.console.print(Panel(content, title="Full Content", border_style="yellow"))
        else:
            self.console.print(Panel(content, title="Content", border_style="yellow"))
    
    def get_label(self) -> Tuple[int, bool]:
        """
        Get user input for labeling an article.
        
        Returns:
            Tuple of (label, continue_labeling) where:
                label: 1 for relevant, 0 for not relevant, -1 for skip
                continue_labeling: False if the user wants to exit
        """
        self.console.print("\n[bold]Labeling Options:[/bold]")
        self.console.print("[1] Relevant to cybersecurity incidents")
        self.console.print("[0] Not relevant to cybersecurity incidents")
        self.console.print("[s] Skip this article")
        self.console.print("[q] Quit labeling session")
        
        while True:
            choice = Prompt.ask(
                "Enter your choice", 
                choices=["1", "0", "s", "q"], 
                default="s"
            )
            
            if choice == "1":
                return 1, True
            elif choice == "0":
                return 0, True
            elif choice == "s":
                return -1, True
            elif choice == "q":
                return -1, False
    
    def save_label(self, article_id: int, label: int) -> bool:
        """
        Save a label for an article.
        
        Args:
            article_id: ID of the article
            label: 1 for relevant, 0 for not relevant
            
        Returns:
            True if saved successfully, False otherwise
        """
        if not self.storage or not self.storage.get('db_store'):
            self.logger.error("No database storage available")
            return False
        
        db_store = self.storage['db_store']
        
        try:
            # Create label object
            labels = {
                "is_relevant": bool(label),
                "labeled_by": "manual",
                "labeled_date": datetime.now().isoformat()
            }
            
            # Update the article
            success = db_store.update_article(
                article_id,
                {"labels": labels}
            )
            
            if success:
                # Update session stats
                self.session_data['total_labeled'] += 1
                if label == 1:
                    self.session_data['labeled_relevant'] += 1
                else:
                    self.session_data['labeled_not_relevant'] += 1
                
                # Record the ID as labeled
                self.session_data['labeled_ids'].add(article_id)
                self.session_data['last_article_id'] = article_id
                
                return True
            else:
                self.logger.warning(f"Failed to update article {article_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error saving label: {str(e)}")
            return False
    
    def export_labeled_data(self, format_type: str, output_file: str = None) -> bool:
        """
        Export labeled data to file.
        
        Args:
            format_type: 'csv' or 'json'
            output_file: Path for the output file (optional)
            
        Returns:
            True if exported successfully, False otherwise
        """
        if not self.storage or not self.storage.get('db_store'):
            self.console.print("[red]No database storage available for export[/red]")
            return False
        
        db_store = self.storage['db_store']
        
        # Generate a default filename if none provided
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            output_file = self.output_dir / f"labeled_articles_{timestamp}.{format_type}"
        
        # Make sure the output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        try:
            # Fetch only labeled articles
            conn = db_store._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM articles 
                WHERE labels IS NOT NULL 
                   AND labels != '{}'
                   AND labels != ''
                ORDER BY id
            """)
            
            articles = []
            for row in cursor.fetchall():
                article = dict(row)
                
                # Parse labels
                try:
                    if article['labels']:
                        article['labels'] = json.loads(article['labels'])
                    else:
                        article['labels'] = {}
                except:
                    article['labels'] = {}
                
                # Add a simple is_relevant flag for easier processing
                if article['labels'] and article['labels'].get('is_relevant') is not None:
                    article['is_relevant'] = int(article['labels'].get('is_relevant'))
                else:
                    article['is_relevant'] = -1  # Unlabeled
                
                articles.append(article)
            
            conn.close()
            
            # Export based on format
            if format_type.lower() == 'json':
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(articles, f, indent=2)
                
                self.console.print(f"[green]Exported {len(articles)} labeled articles to {output_file}[/green]")
                return True
                
            elif format_type.lower() == 'csv':
                import csv
                
                if not articles:
                    self.console.print("[yellow]No labeled articles found[/yellow]")
                    return False
                
                with open(output_file, 'w', newline='', encoding='utf-8') as f:
                    # Determine fields - include everything except content to keep CSV size reasonable
                    fields = [field for field in articles[0].keys() if field != 'content']
                    
                    writer = csv.DictWriter(f, fieldnames=fields)
                    writer.writeheader()
                    
                    for article in articles:
                        # Convert labels back to string for CSV
                        if isinstance(article.get('labels'), dict):
                            article['labels'] = json.dumps(article['labels'])
                        
                        # Remove content to keep CSV size reasonable
                        row = {field: article[field] for field in fields}
                        writer.writerow(row)
                
                self.console.print(f"[green]Exported {len(articles)} labeled articles to {output_file}[/green]")
                return True
            
            else:
                self.console.print(f"[red]Unsupported format: {format_type}[/red]")
                return False
                
        except Exception as e:
            self.logger.error(f"Error exporting labeled data: {str(e)}")
            self.console.print(f"[red]Error exporting labeled data: {str(e)}[/red]")
            return False
    
    def display_stats(self) -> None:
        """Display statistics about the current labeling session."""
        # Calculate elapsed time
        start_time = datetime.fromisoformat(self.session_data["start_time"])
        elapsed = datetime.now() - start_time
        
        # Create a table for stats
        table = Table(title="Labeling Session Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Articles Labeled", str(self.session_data["total_labeled"]))
        table.add_row("Relevant Articles", str(self.session_data["labeled_relevant"]))
        table.add_row("Non-Relevant Articles", str(self.session_data["labeled_not_relevant"]))
        table.add_row("Skipped Articles", str(self.session_data["skipped"]))
        table.add_row("Session Duration", str(elapsed).split('.')[0])  # Remove microseconds
        
        # Display relevance percentage if we have labeled articles
        if self.session_data["total_labeled"] > 0:
            relevance_pct = (self.session_data["labeled_relevant"] / self.session_data["total_labeled"]) * 100
            table.add_row("Relevance Percentage", f"{relevance_pct:.1f}%")
        
        self.console.print(table)
    
    def run_labeling_session(self, session_file: str = None, batch_size: int = 20, 
                          auto_save_frequency: int = 5) -> None:
        """
        Run an interactive labeling session.
        
        Args:
            session_file: Path to save/load session progress
            batch_size: Number of articles to fetch at once
            auto_save_frequency: How often to auto-save progress (in articles)
        """
        # Set default session file if none provided
        if session_file is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            session_file = self.output_dir / f"labeling_session_{timestamp}.pkl"
        
        # Load existing progress if file exists
        if os.path.exists(session_file):
            self.load_progress(session_file)
        
        # Welcome message
        self.console.clear()
        self.console.print(Panel.fit(
            "[bold]Welcome to the VIGIL Article Labeling Tool[/bold]\n\n"
            "This tool will help you label articles as:\n"
            "• [green]Relevant[/green] to cybersecurity incidents (1)\n"
            "• [red]Not relevant[/red] to cybersecurity incidents (0)\n\n"
            "Your progress will be saved automatically.",
            title="Article Labeling Tool",
            border_style="blue"
        ))
        
        self.console.print("\nPress Enter to start labeling...", end="")
        input()
        
        # Main labeling loop
        continue_labeling = True
        articles_since_save = 0
        
        while continue_labeling:
            # Fetch more articles if our queue is empty
            if not self.articles_to_label:
                self.console.print("[yellow]Fetching more articles to label...[/yellow]")
                self.articles_to_label = self.fetch_unlabeled_articles(batch_size)
                
                if not self.articles_to_label:
                    self.console.print("[yellow]No more unlabeled articles found in the database.[/yellow]")
                    break
            
            # Get the next article
            article = self.articles_to_label.pop(0)
            
            # Display the article
            self.display_article(article)
            
            # Get label from user
            label, continue_labeling = self.get_label()
            
            if not continue_labeling:
                self.console.print("[yellow]Exiting labeling session...[/yellow]")
                break
            
            # Save the label if not skipped
            if label >= 0:
                if self.save_label(article['id'], label):
                    self.console.print(f"[green]Article {article['id']} labeled as {'relevant' if label == 1 else 'not relevant'}[/green]")
                else:
                    self.console.print(f"[red]Failed to save label for article {article['id']}[/red]")
            else:
                self.session_data['skipped'] += 1
                self.console.print(f"[yellow]Article {article['id']} skipped[/yellow]")
            
            # Auto-save progress
            articles_since_save += 1
            if articles_since_save >= auto_save_frequency:
                self.save_progress(session_file)
                articles_since_save = 0
        
        # Final save and display stats
        self.save_progress(session_file)
        self.display_stats()
        
        # Ask if user wants to export labeled data
        if self.session_data["total_labeled"] > 0 and Confirm.ask("Export labeled data?"):
            format_type = Prompt.ask(
                "Export format", 
                choices=["json", "csv"], 
                default="json"
            )
            self.export_labeled_data(format_type)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="VIGIL Article Labeling Tool")
    
    parser.add_argument("--config", "-c", 
                        help="Path to config file (default: config.yml)")
    
    parser.add_argument("--session", "-s", 
                        help="Path to session file (to resume labeling)")
    
    parser.add_argument("--output-dir", "-o", default="dataset",
                        help="Directory to save labeled data")
    
    parser.add_argument("--batch-size", "-b", type=int, default=20,
                        help="Number of articles to fetch at once")
    
    parser.add_argument("--auto-save", "-a", type=int, default=5,
                        help="Auto-save frequency (in number of articles)")
    
    parser.add_argument("--export", "-e", choices=["json", "csv"], 
                        help="Export labeled data without starting interactive session")
    
    parser.add_argument("--export-file", "-f",
                        help="Path for exported data (used with --export)")
    
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose output")
                        
    args = parser.parse_args()
    
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
    logger.info("VIGIL Article Labeling Tool starting...")
    
    # Initialize the labeling tool
    labeler = LabelingTool(
        storage_config=config.config.get('storage', {}),
        output_dir=args.output_dir
    )
    
    # If export flag is set, just export and exit
    if args.export:
        success = labeler.export_labeled_data(args.export, args.export_file)
        sys.exit(0 if success else 1)
    
    # Otherwise, start interactive session
    try:
        labeler.run_labeling_session(
            session_file=args.session,
            batch_size=args.batch_size,
            auto_save_frequency=args.auto_save
        )
    except KeyboardInterrupt:
        print("\nLabeling session interrupted. Saving progress...")
        if args.session:
            labeler.save_progress(args.session)
        sys.exit(0)


if __name__ == "__main__":
    main()
