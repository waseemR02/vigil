"""
Data storage module for saving and retrieving crawled and processed content.
"""

import csv
import datetime
import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

# Configure module logger
logger = logging.getLogger('vigil.storage.data_store')

class FileStore:
    """File-based storage for extracted content."""
    
    def __init__(self, base_dir: str = "data"):
        """
        Initialize the file store with a base directory.
        
        Args:
            base_dir: Base directory for file storage
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different data types
        self.content_dir = self.base_dir / "content"
        self.content_dir.mkdir(exist_ok=True)
        
        self.metadata_dir = self.base_dir / "metadata"
        self.metadata_dir.mkdir(exist_ok=True)
    
    def save_content(self, content_id: str, data: Dict[str, Any], overwrite: bool = False) -> str:
        """
        Save content data to a JSON file.
        
        Args:
            content_id: Unique identifier for the content
            data: Dictionary of content data
            overwrite: Whether to overwrite existing files
            
        Returns:
            Path to saved file
        """
        # Sanitize the ID for use as a filename
        safe_id = self._sanitize_filename(content_id)
        
        # Create the file path
        file_path = self.content_dir / f"{safe_id}.json"
        
        # Check if file exists and overwrite is not allowed
        if file_path.exists() and not overwrite:
            logger.warning(f"File already exists and overwrite=False: {file_path}")
            return str(file_path)
        
        # Ensure data has timestamp
        if "timestamp" not in data:
            data["timestamp"] = datetime.datetime.now().isoformat()
        
        # Save the data
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.debug(f"Saved content to {file_path}")
            return str(file_path)
        except Exception as e:
            logger.error(f"Error saving content to {file_path}: {str(e)}")
            raise
    
    def load_content(self, content_id: str) -> Optional[Dict[str, Any]]:
        """
        Load content data from a JSON file.
        
        Args:
            content_id: Unique identifier for the content
            
        Returns:
            Dictionary of content data or None if not found
        """
        # Sanitize the ID for use as a filename
        safe_id = self._sanitize_filename(content_id)
        
        # Create the file path
        file_path = self.content_dir / f"{safe_id}.json"
        
        # Check if file exists
        if not file_path.exists():
            logger.warning(f"Content file not found: {file_path}")
            return None
        
        # Load the data
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading content from {file_path}: {str(e)}")
            return None
    
    def save_batch(self, data_items: List[Dict[str, Any]], id_field: str = "url") -> int:
        """
        Save multiple content items in batch.
        
        Args:
            data_items: List of data dictionaries
            id_field: Field to use as identifier
            
        Returns:
            Number of successfully saved items
        """
        saved_count = 0
        for item in data_items:
            if id_field not in item:
                logger.warning(f"Item missing ID field '{id_field}', skipping")
                continue
            
            try:
                self.save_content(item[id_field], item)
                saved_count += 1
            except Exception as e:
                logger.error(f"Error saving item {item.get(id_field, 'unknown')}: {str(e)}")
        
        return saved_count
    
    def list_contents(self) -> List[str]:
        """
        List all available content IDs.
        
        Returns:
            List of content IDs
        """
        return [
            f.stem for f in self.content_dir.glob("*.json")
        ]
    
    def _sanitize_filename(self, filename: str) -> str:
        """
        Sanitize a string for use as a filename.
        
        Args:
            filename: Input string
            
        Returns:
            Sanitized filename
        """
        # Replace invalid characters with underscore
        safe_name = ''.join(c if c.isalnum() or c in '-._' else '_' for c in filename)
        
        # Limit length
        if len(safe_name) > 100:
            # Keep the start and end, but truncate the middle
            safe_name = safe_name[:80] + "___" + safe_name[-17:]
            
        return safe_name


class SQLiteStore:
    """SQLite-based storage for extracted content."""
    
    # SQL for creating tables
    CREATE_TABLES_SQL = """
    CREATE TABLE IF NOT EXISTS articles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        url TEXT UNIQUE NOT NULL,
        title TEXT,
        content TEXT,
        domain TEXT,
        publish_date TEXT,
        date_extracted TEXT,
        date_stored TEXT,
        source TEXT,
        status INTEGER DEFAULT 0,
        labels TEXT
    );
    
    CREATE TABLE IF NOT EXISTS crawl_sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        start_time TEXT,
        end_time TEXT,
        urls_processed INTEGER,
        success_count INTEGER,
        config TEXT
    );
    
    CREATE INDEX IF NOT EXISTS idx_articles_url ON articles(url);
    CREATE INDEX IF NOT EXISTS idx_articles_domain ON articles(domain);
    CREATE INDEX IF NOT EXISTS idx_articles_date_extracted ON articles(date_extracted);
    """
    
    def __init__(self, db_path: str = "vigil_data.db"):
        """
        Initialize the SQLite database.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self._create_tables()
    
    def _create_tables(self):
        """Create the database tables if they don't exist."""
        try:
            with self._get_connection() as conn:
                conn.executescript(self.CREATE_TABLES_SQL)
            logger.debug("Database tables created or verified")
        except sqlite3.Error as e:
            logger.error(f"Error creating database tables: {str(e)}")
            raise
    
    def _get_connection(self):
        """Get a database connection with row factory."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # This enables column access by name
            return conn
        except sqlite3.Error as e:
            logger.error(f"Error connecting to database: {str(e)}")
            raise
    
    def add_article(self, article_data: Dict[str, Any]) -> int:
        """
        Add an article to the database.
        
        Args:
            article_data: Dictionary with article data
            
        Returns:
            ID of the inserted article or -1 if failed
        """
        # Ensure required fields are present
        required_fields = ['url', 'title', 'content']
        for field in required_fields:
            if field not in article_data:
                logger.warning(f"Article missing required field: {field}")
                return -1
        
        # Prepare metadata
        metadata = article_data.get('metadata', {})
        
        # Extract fields
        url = article_data['url']
        title = article_data['title']
        content = article_data['content']
        domain = metadata.get('domain', '')
        publish_date = metadata.get('publish_date', '')
        date_extracted = article_data.get('date_extracted', '')
        date_stored = datetime.datetime.now().isoformat()
        source = article_data.get('source', 'crawler')
        
        # Convert any labels to JSON string
        labels = json.dumps(article_data.get('labels', {}))
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                INSERT OR REPLACE INTO articles 
                (url, title, content, domain, publish_date, date_extracted, date_stored, source, labels)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (url, title, content, domain, publish_date, date_extracted, 
                     date_stored, source, labels))
                
                # Get the ID of the inserted row
                if cursor.lastrowid:
                    return cursor.lastrowid
                else:
                    # If replaced an existing row, get its ID
                    cursor.execute("SELECT id FROM articles WHERE url = ?", (url,))
                    result = cursor.fetchone()
                    return result[0] if result else -1
                    
        except sqlite3.Error as e:
            logger.error(f"Error adding article to database: {str(e)}")
            return -1
    
    def add_articles_batch(self, articles: List[Dict[str, Any]]) -> int:
        """
        Add multiple articles in a batch operation.
        
        Args:
            articles: List of article dictionaries
            
        Returns:
            Number of successfully added articles
        """
        added_count = 0
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                for article in articles:
                    try:
                        # Extract and prepare data
                        metadata = article.get('metadata', {})
                        
                        url = article['url']
                        title = article.get('title', '')
                        content = article.get('content', '')
                        domain = metadata.get('domain', '')
                        publish_date = metadata.get('publish_date', '')
                        date_extracted = article.get('date_extracted', '')
                        date_stored = datetime.datetime.now().isoformat()
                        source = article.get('source', 'crawler')
                        labels = json.dumps(article.get('labels', {}))
                        
                        cursor.execute("""
                        INSERT OR REPLACE INTO articles 
                        (url, title, content, domain, publish_date, date_extracted, date_stored, source, labels)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (url, title, content, domain, publish_date, date_extracted, 
                             date_stored, source, labels))
                        
                        added_count += 1
                    except KeyError as e:
                        logger.warning(f"Missing required field in article: {e}")
                    except Exception as e:
                        logger.error(f"Error adding article: {str(e)}")
                
        except sqlite3.Error as e:
            logger.error(f"Error in batch add operation: {str(e)}")
        
        return added_count
    
    def get_article(self, article_id: Union[int, str]) -> Optional[Dict[str, Any]]:
        """
        Get an article by ID or URL.
        
        Args:
            article_id: Either database ID (int) or URL (str)
            
        Returns:
            Article dictionary or None if not found
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                if isinstance(article_id, int):
                    cursor.execute("SELECT * FROM articles WHERE id = ?", (article_id,))
                else:
                    cursor.execute("SELECT * FROM articles WHERE url = ?", (article_id,))
                
                row = cursor.fetchone()
                if row:
                    # Convert to dictionary
                    article = dict(row)
                    
                    # Parse JSON fields
                    try:
                        article['labels'] = json.loads(article['labels']) if article['labels'] else {}
                    except json.JSONDecodeError:
                        article['labels'] = {}
                    
                    return article
                
                return None
                
        except sqlite3.Error as e:
            logger.error(f"Error retrieving article: {str(e)}")
            return None
    
    def update_article(self, article_id: Union[int, str], updates: Dict[str, Any]) -> bool:
        """
        Update an existing article.
        
        Args:
            article_id: Database ID or URL of the article
            updates: Dictionary of fields to update
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Prepare the SET part of the SQL query
                set_clause = []
                params = []
                
                for key, value in updates.items():
                    # Skip the ID field and non-column fields
                    if key in ('id', 'metadata'):
                        continue
                    
                    # Handle special cases
                    if key == 'labels' and isinstance(value, dict):
                        value = json.dumps(value)
                    
                    set_clause.append(f"{key} = ?")
                    params.append(value)
                
                if not set_clause:
                    logger.warning("No valid fields to update")
                    return False
                
                # Add update timestamp
                set_clause.append("date_stored = ?")
                params.append(datetime.datetime.now().isoformat())
                
                # Create the full SQL query
                if isinstance(article_id, int):
                    sql = f"UPDATE articles SET {', '.join(set_clause)} WHERE id = ?"
                else:
                    sql = f"UPDATE articles SET {', '.join(set_clause)} WHERE url = ?"
                
                params.append(article_id)
                
                # Execute the update
                cursor.execute(sql, params)
                
                return cursor.rowcount > 0
                
        except sqlite3.Error as e:
            logger.error(f"Error updating article: {str(e)}")
            return False
    
    def delete_article(self, article_id: Union[int, str]) -> bool:
        """
        Delete an article from the database.
        
        Args:
            article_id: Database ID or URL of the article
            
        Returns:
            True if deleted, False otherwise
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                if isinstance(article_id, int):
                    cursor.execute("DELETE FROM articles WHERE id = ?", (article_id,))
                else:
                    cursor.execute("DELETE FROM articles WHERE url = ?", (article_id,))
                
                return cursor.rowcount > 0
                
        except sqlite3.Error as e:
            logger.error(f"Error deleting article: {str(e)}")
            return False
    
    def search_articles(self, 
                       domain: Optional[str] = None,
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None,
                       content_filter: Optional[str] = None,
                       limit: int = 100,
                       offset: int = 0) -> List[Dict[str, Any]]:
        """
        Search articles with various filters.
        
        Args:
            domain: Filter by domain
            start_date: Filter articles extracted after this date (ISO format)
            end_date: Filter articles extracted before this date (ISO format)
            content_filter: Text to search for in title or content
            limit: Maximum number of results
            offset: Pagination offset
            
        Returns:
            List of matching article dictionaries
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Build the query
                sql = "SELECT * FROM articles WHERE 1=1"
                params = []
                
                if domain:
                    sql += " AND domain = ?"
                    params.append(domain)
                
                if start_date:
                    sql += " AND date_extracted >= ?"
                    params.append(start_date)
                
                if end_date:
                    sql += " AND date_extracted <= ?"
                    params.append(end_date)
                
                if content_filter:
                    sql += " AND (title LIKE ? OR content LIKE ?)"
                    pattern = f"%{content_filter}%"
                    params.extend([pattern, pattern])
                
                # Add sorting, limit and offset
                sql += " ORDER BY date_extracted DESC LIMIT ? OFFSET ?"
                params.extend([limit, offset])
                
                # Execute the query
                cursor.execute(sql, params)
                
                # Process results
                result = []
                for row in cursor.fetchall():
                    article = dict(row)
                    
                    # Parse JSON fields
                    try:
                        article['labels'] = json.loads(article['labels']) if article['labels'] else {}
                    except json.JSONDecodeError:
                        article['labels'] = {}
                    
                    result.append(article)
                
                return result
                
        except sqlite3.Error as e:
            logger.error(f"Error searching articles: {str(e)}")
            return []
    
    def count_articles(self,
                      domain: Optional[str] = None,
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None,
                      content_filter: Optional[str] = None) -> int:
        """
        Count articles matching the filters.
        
        Args:
            Similar to search_articles
            
        Returns:
            Count of matching articles
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Build the query
                sql = "SELECT COUNT(*) FROM articles WHERE 1=1"
                params = []
                
                if domain:
                    sql += " AND domain = ?"
                    params.append(domain)
                
                if start_date:
                    sql += " AND date_extracted >= ?"
                    params.append(start_date)
                
                if end_date:
                    sql += " AND date_extracted <= ?"
                    params.append(end_date)
                
                if content_filter:
                    sql += " AND (title LIKE ? OR content LIKE ?)"
                    pattern = f"%{content_filter}%"
                    params.extend([pattern, pattern])
                
                # Execute the query
                cursor.execute(sql, params)
                
                return cursor.fetchone()[0]
                
        except sqlite3.Error as e:
            logger.error(f"Error counting articles: {str(e)}")
            return 0
    
    def export_to_json(self, output_file: str, filters: Dict[str, Any] = None) -> int:
        """
        Export articles to a JSON file.
        
        Args:
            output_file: Path to output file
            filters: Dictionary of filters to apply
            
        Returns:
            Number of exported records
        """
        filters = filters or {}
        
        try:
            # Get the matching articles
            articles = self.search_articles(
                domain=filters.get('domain'),
                start_date=filters.get('start_date'),
                end_date=filters.get('end_date'),
                content_filter=filters.get('content_filter'),
                limit=filters.get('limit', 10000)  # Higher limit for exports
            )
            
            # Write to file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(articles, f, indent=2)
            
            logger.info(f"Exported {len(articles)} articles to {output_file}")
            return len(articles)
            
        except Exception as e:
            logger.error(f"Error exporting to JSON: {str(e)}")
            return 0
    
    def export_to_csv(self, output_file: str, filters: Dict[str, Any] = None) -> int:
        """
        Export articles to a CSV file.
        
        Args:
            output_file: Path to output file
            filters: Dictionary of filters to apply
            
        Returns:
            Number of exported records
        """
        filters = filters or {}
        
        try:
            # Get the matching articles
            articles = self.search_articles(
                domain=filters.get('domain'),
                start_date=filters.get('start_date'),
                end_date=filters.get('end_date'),
                content_filter=filters.get('content_filter'),
                limit=filters.get('limit', 10000)  # Higher limit for exports
            )
            
            if not articles:
                logger.warning("No articles found for export")
                return 0
            
            # Determine CSV fields (using first article as reference)
            fields = list(articles[0].keys())
            
            # Write to CSV
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                
                for article in articles:
                    # Convert non-string fields
                    if isinstance(article.get('labels'), dict):
                        article['labels'] = json.dumps(article['labels'])
                    writer.writerow(article)
            
            logger.info(f"Exported {len(articles)} articles to {output_file}")
            return len(articles)
            
        except Exception as e:
            logger.error(f"Error exporting to CSV: {str(e)}")
            return 0
    
    def log_crawl_session(self, start_time: str, end_time: str, urls_processed: int,
                        success_count: int, config: Dict[str, Any]) -> int:
        """
        Log a crawl session to the database.
        
        Args:
            start_time: ISO format start time
            end_time: ISO format end time
            urls_processed: Number of URLs processed
            success_count: Number of successful fetches
            config: Crawl configuration
            
        Returns:
            ID of the session log
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                INSERT INTO crawl_sessions 
                (start_time, end_time, urls_processed, success_count, config)
                VALUES (?, ?, ?, ?, ?)
                """, (start_time, end_time, urls_processed, success_count, json.dumps(config)))
                
                return cursor.lastrowid
                
        except sqlite3.Error as e:
            logger.error(f"Error logging crawl session: {str(e)}")
            return -1


def init_storage(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Initialize storage backends based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary with storage objects
    """
    config = config or {}
    storage_config = config.get('storage', {})
    
    # Initialize file store
    file_store_path = storage_config.get('file_store_path', 'data')
    file_store = FileStore(base_dir=file_store_path)
    
    # Initialize SQLite store
    db_path = storage_config.get('db_path', 'vigil_data.db')
    db_store = SQLiteStore(db_path=db_path)
    
    return {
        'file_store': file_store,
        'db_store': db_store
    }
