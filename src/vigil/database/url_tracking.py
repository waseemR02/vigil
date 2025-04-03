"""
Database operations for URL tracking.
Manages the tracking of processed URLs to avoid duplicates.
"""
import sqlite3
import logging
from datetime import datetime
from typing import List, Optional
import os
import json

from vigil.database.connection import DB_PATH

logger = logging.getLogger("vigil.database.url_tracking")

def init_url_tracking_table():
    """Create the URL tracking tables if they don't exist."""
    try:
        # Ensure database directory exists
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Create processed URLs table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS processed_urls (
            url TEXT PRIMARY KEY,
            first_processed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_processed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            process_count INTEGER DEFAULT 1
        )
        """)
        
        # Create saved queues table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS saved_queues (
            domain TEXT PRIMARY KEY,
            urls TEXT,
            saved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        conn.commit()
        conn.close()
        logger.info("URL tracking tables initialized")
        return True
    except Exception as e:
        logger.error(f"Error initializing URL tracking tables: {str(e)}")
        return False

def is_url_processed(url: str) -> bool:
    """Check if a URL has been processed before."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("SELECT url FROM processed_urls WHERE url = ?", (url,))
        result = cursor.fetchone()
        conn.close()
        
        return result is not None
    except Exception as e:
        logger.error(f"Error checking if URL is processed: {str(e)}")
        return False

def mark_url_processed(url: str) -> bool:
    """Mark a URL as processed in the database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Try to update existing record
        cursor.execute(
            """
            UPDATE processed_urls 
            SET last_processed = CURRENT_TIMESTAMP, process_count = process_count + 1 
            WHERE url = ?
            """,
            (url,)
        )
        
        # If no rows updated, insert new record
        if cursor.rowcount == 0:
            cursor.execute(
                "INSERT INTO processed_urls (url) VALUES (?)",
                (url,)
            )
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Error marking URL as processed: {str(e)}")
        return False

def get_processed_urls(domain: Optional[str] = None, limit: Optional[int] = None) -> List[str]:
    """Get list of processed URLs, optionally filtered by domain."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        sql = "SELECT url FROM processed_urls"
        params = []
        
        if domain:
            sql += " WHERE url LIKE ?"
            params.append(f"%{domain}%")
            
        if limit is not None and limit > 0:
            sql += f" LIMIT {int(limit)}"
            
        cursor.execute(sql, params)
        results = cursor.fetchall()
        conn.close()
        
        return [row[0] for row in results]
    except Exception as e:
        logger.error(f"Error getting processed URLs: {str(e)}")
        return []
        
def save_url_queue(domain: str, urls: List[str]) -> bool:
    """
    Save a list of unprocessed URLs for a domain to use in future runs.
    
    Args:
        domain: The domain name to associate with the queue
        urls: List of URLs to save for future processing
        
    Returns:
        True if successful, False otherwise
    """
    if not urls:
        logger.debug(f"No URLs to save for domain {domain}")
        return True
        
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Convert list to JSON string
        urls_json = json.dumps(urls)
        
        # Upsert the queue data
        cursor.execute(
            """
            INSERT OR REPLACE INTO saved_queues (domain, urls, saved_at) 
            VALUES (?, ?, CURRENT_TIMESTAMP)
            """,
            (domain, urls_json)
        )
        
        conn.commit()
        conn.close()
        logger.info(f"Saved {len(urls)} URLs for domain {domain}")
        return True
    except Exception as e:
        logger.error(f"Error saving URL queue: {str(e)}")
        return False

def get_url_queue(domain: str) -> List[str]:
    """
    Retrieve the saved queue for a domain.
    
    Args:
        domain: The domain to get saved URLs for
        
    Returns:
        List of URLs that were saved for future processing
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("SELECT urls FROM saved_queues WHERE domain = ?", (domain,))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            try:
                return json.loads(result[0])
            except json.JSONDecodeError:
                logger.error(f"Error parsing saved URLs for domain {domain}")
                return []
        else:
            logger.debug(f"No saved queue found for domain {domain}")
            return []
    except Exception as e:
        logger.error(f"Error retrieving URL queue: {str(e)}")
        return []
