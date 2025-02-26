"""Tests for text processing utility functions."""

import pytest
from vigil.utils.text_utils import (remove_html_tags, clean_text,
                                   extract_text_from_html, summarize_text,
                                   extract_keywords)


class TestTextUtils:
    """Test cases for text utilities."""

    def test_remove_html_tags(self):
        """Test HTML tag removal."""
        # Test basic tag removal
        assert remove_html_tags("<p>Hello</p>") == "Hello"
        
        # Test nested tags
        assert remove_html_tags("<div><p>Hello <b>World</b></p></div>") == "Hello World"
        
        # Test HTML entities
        assert remove_html_tags("&lt;tag&gt; &amp; entity") == "<tag> & entity"
        
        # Test empty input
        assert remove_html_tags("") == ""

    def test_clean_text(self, sample_html):
        """Test text cleaning."""
        # Test whitespace normalization
        assert clean_text("  Multiple    spaces  ") == "Multiple spaces"
        
        # Test newline handling
        assert clean_text("Line 1\nLine 2\n\nLine 3") == "Line 1 Line 2 Line 3"
        
        # Test Unicode normalization
        assert clean_text("caf\u00E9") == "café"
        assert clean_text("caf\u0065\u0301") == "café"  # Decomposed form
        
        # Test empty input
        assert clean_text("") == ""

    def test_extract_text_from_html(self, sample_html):
        """Test HTML to text extraction."""
        text = extract_text_from_html(sample_html)
        
        # Should contain main content text
        assert "Main Heading" in text
        assert "This is a paragraph with bold text and a link." in text
        assert "Second paragraph with some additional text." in text
        assert "List item 1" in text
        
        # Should not contain HTML elements or JavaScript
        assert "<h1>" not in text
        assert "function test()" not in text
        assert "body { font-family" not in text
        assert "<!-- Comment -->" not in text
        
        # Test empty input
        assert extract_text_from_html("") == ""

    def test_summarize_text(self):
        """Test text summarization."""
        text = "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence."
        
        # Test summarizing with different sentence counts
        assert summarize_text(text, 1) == "First sentence."
        assert summarize_text(text, 3) == "First sentence. Second sentence. Third sentence."
        
        # Test with more sentences than available
        assert summarize_text(text, 10) == text
        
        # Test with invalid sentence count
        assert summarize_text(text, 0) == text
        
        # Test empty input
        assert summarize_text("") == ""

    def test_extract_keywords(self):
        """Test keyword extraction."""
        text = """
        Python is an interpreted, high-level, general-purpose programming language.
        Python's design philosophy emphasizes code readability with its notable use of significant whitespace.
        """
        
        # Test default keyword count (5)
        keywords = extract_keywords(text)
        assert len(keywords) <= 5
        assert "python" in keywords
        
        # Test custom keyword count
        keywords = extract_keywords(text, 2)
        assert len(keywords) <= 2
        
        # Test empty input
        assert extract_keywords("") == []
        
        # Test with stopwords
        keywords = extract_keywords("The and of to in for is on that")
        assert len(keywords) == 0
