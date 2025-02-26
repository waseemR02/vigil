"""Text processing utilities for the Vigil monitoring system."""

import re
import html
from typing import List, Optional
import unicodedata


def remove_html_tags(text: str) -> str:
    """
    Remove HTML tags from text content.
    
    Args:
        text: HTML text content
        
    Returns:
        Plain text with HTML tags removed
    """
    if not text:
        return ""
    
    # Remove HTML tags
    clean = re.sub(r'<[^>]*>', '', text)
    # Decode HTML entities
    clean = html.unescape(clean)
    return clean


def clean_text(text: str, normalize_unicode: bool = True) -> str:
    """
    Clean text by removing extra whitespace and normalizing characters.
    
    Args:
        text: The text to clean
        normalize_unicode: Whether to normalize Unicode characters
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Normalize Unicode if requested
    if normalize_unicode:
        text = unicodedata.normalize('NFKC', text)
    
    # Replace multiple whitespace with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    return text.strip()


def extract_text_from_html(html_content: str) -> str:
    """
    Extract plain text from HTML content.
    
    Args:
        html_content: HTML content
        
    Returns:
        Extracted plain text
    """
    if not html_content:
        return ""
    
    # First remove script and style elements
    html_content = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', html_content, flags=re.DOTALL)
    
    # Then remove all HTML tags
    text = remove_html_tags(html_content)
    
    # Clean up the resulting text
    return clean_text(text)


def summarize_text(text: str, sentences: int = 3) -> str:
    """
    Create a simple summary by extracting the first N sentences.
    
    Args:
        text: The text to summarize
        sentences: Number of sentences to include in the summary
        
    Returns:
        Text summary
    """
    if not text:
        return ""
    
    # Split text into sentences
    # This is a simple approach; more sophisticated sentence detection could be used
    sentence_endings = re.finditer(r'[.!?][\s\n]', text)
    
    # Get the positions of sentence endings
    end_positions = [match.end() for match in sentence_endings]
    
    if not end_positions or sentences <= 0:
        return text
    
    # Limit to the requested number of sentences or all available
    if len(end_positions) < sentences:
        return text
    
    # Extract the text up to the Nth sentence
    return text[:end_positions[sentences-1]].strip()


def extract_keywords(text: str, count: int = 5) -> List[str]:
    """
    Extract the most common meaningful words from text as keywords.
    
    Args:
        text: The text to analyze
        count: Number of keywords to extract
        
    Returns:
        List of extracted keywords
    """
    if not text:
        return []
    
    # Convert to lowercase and split into words
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    
    # Common English stopwords to filter out
    stopwords = {
        'the', 'and', 'to', 'of', 'a', 'in', 'for', 'is', 'on', 'that',
        'by', 'this', 'with', 'i', 'you', 'it', 'not', 'or', 'be', 'are',
        'from', 'at', 'as', 'your', 'all', 'have', 'new', 'more', 'was'
    }
    
    # Filter out stopwords
    filtered_words = [word for word in words if word not in stopwords]
    
    # Count word frequencies
    word_freq = {}
    for word in filtered_words:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    # Sort by frequency and return top 'count' words
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, _ in sorted_words[:count]]
