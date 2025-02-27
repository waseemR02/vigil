"""
Content extraction module for extracting meaningful text and metadata from HTML pages.
"""

import datetime
import logging
import re
from urllib.parse import urlparse

from bs4 import BeautifulSoup

# Configure module logger
logger = logging.getLogger('vigil.data_collection.content_extractor')

class ContentExtractor:
    """Extract structured content from HTML pages."""
    
    # Elements that typically contain the main article content
    CONTENT_TAGS = ['article', 'main', 'div', 'section']
    CONTENT_CLASSES = ['content', 'article', 'post', 'entry', 'story', 'text', 'body']
    
    # Elements to ignore when extracting text (typically navigation, ads, etc.)
    IGNORE_CLASSES = ['nav', 'navbar', 'menu', 'header', 'footer', 'sidebar', 
                      'comment', 'ad', 'advertisement', 'social', 'share', 'related']
    IGNORE_IDS = ['nav', 'navbar', 'menu', 'header', 'footer', 'sidebar', 
                  'comment', 'ad', 'advertisement', 'social', 'share', 'related']
    
    # Date pattern for finding publication dates in text
    DATE_PATTERNS = [
        # ISO format: 2023-11-15
        r'\d{4}-\d{1,2}-\d{1,2}',
        # Common US format: 11/15/2023
        r'\d{1,2}/\d{1,2}/\d{4}',
        # Text format: November 15, 2023
        r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}',
        # Text format with abbr: Nov 15, 2023
        r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) \d{1,2},? \d{4}'
    ]
    
    def __init__(self, min_content_length=300):
        """
        Initialize the content extractor.
        
        Args:
            min_content_length (int): Minimum length of text to be considered valid content
        """
        self.min_content_length = min_content_length
    
    def extract_content(self, html_content, url):
        """
        Extract structured content from HTML.
        
        Args:
            html_content (str): Raw HTML content
            url (str): Source URL of the content
            
        Returns:
            dict: Structured content with metadata
        """
        if not html_content:
            logger.warning(f"No HTML content provided for {url}")
            return self._create_empty_result(url)
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
        except Exception as e:
            logger.error(f"Failed to parse HTML from {url}: {str(e)}")
            return self._create_empty_result(url)
        
        # Extract basic metadata
        title = self._extract_title(soup)
        publish_date = self._extract_publish_date(soup)
        
        # Extract the main content
        content = self._extract_main_content(soup)
        
        # If we couldn't find structured content, fall back to extracting all paragraphs
        if not content or len(content) < self.min_content_length:
            logger.debug(f"Using fallback content extraction for {url}")
            content = self._extract_all_paragraphs(soup)
        
        # Build the result
        result = {
            "url": url,
            "title": title,
            "content": content,
            "date_extracted": datetime.datetime.now().isoformat(),
            "metadata": {
                "publish_date": publish_date,
                "domain": urlparse(url).netloc
            }
        }
        
        return result
    
    def _create_empty_result(self, url):
        """Create an empty result template."""
        return {
            "url": url,
            "title": "",
            "content": "",
            "date_extracted": datetime.datetime.now().isoformat(),
            "metadata": {
                "publish_date": None,
                "domain": urlparse(url).netloc
            }
        }
        
    def _extract_title(self, soup):
        """Extract the article title from HTML."""
        # Try different strategies to find the title
        # 1. Look for h1 in article/main content
        for container in soup.select('article, main, [role="main"]'):
            h1 = container.find('h1')
            if h1:
                return h1.get_text().strip()
        
        # 2. Look for any h1
        h1 = soup.find('h1')
        if h1:
            return h1.get_text().strip()
        
        # 3. Fall back to the title tag
        title_tag = soup.find('title')
        if title_tag:
            title_text = title_tag.get_text().strip()
            # Remove site name from title if it contains a separator
            for separator in [' | ', ' - ', ' – ', ' — ', ' :: ', ' // ']:
                if separator in title_text:
                    return title_text.split(separator)[0].strip()
            return title_text
        
        return ""
    
    def _extract_publish_date(self, soup):
        """Extract the article publication date."""
        # 1. Look for time tags with datetime attribute
        time_tag = soup.find('time')
        if time_tag and time_tag.get('datetime'):
            return time_tag['datetime']
        
        # 2. Look for meta tags with publication date
        for meta_name in ['publishdate', 'pubdate', 'date', 'published', 'article:published_time']:
            meta = soup.find('meta', attrs={'name': meta_name}) or soup.find('meta', attrs={'property': meta_name})
            if meta and meta.get('content'):
                return meta['content']
        
        # 3. Look for elements with date-related classes
        for class_name in ['date', 'time', 'published', 'pub-date', 'post-date']:
            date_element = soup.find(class_=re.compile(class_name, re.I))
            if date_element:
                text = date_element.get_text().strip()
                # Extract date using regex patterns
                for pattern in self.DATE_PATTERNS:
                    match = re.search(pattern, text)
                    if match:
                        return match.group(0)
        
        # 4. Look for dates in any element with date-related attributes
        for attr in ['datetime', 'date-time', 'date']:
            elements = soup.find_all(attrs={attr: True})
            for element in elements:
                date_value = element[attr]
                # Check if it looks like a date
                for pattern in self.DATE_PATTERNS:
                    if re.search(pattern, date_value):
                        return date_value
        
        return None
    
    def _extract_main_content(self, soup):
        """Extract the main content of the article using common patterns."""
        # Remove script, style, and other non-content elements
        self._remove_non_content_elements(soup)
        
        # Strategy 1: Look for article containers
        for tag in self.CONTENT_TAGS:
            for class_name in self.CONTENT_CLASSES:
                content_elements = soup.find_all(tag, class_=re.compile(class_name, re.I))
                for element in content_elements:
                    # Check if this element has a reasonable amount of text
                    content = self._clean_extracted_text(element.get_text())
                    if len(content) >= self.min_content_length:
                        return content
        
        # Strategy 2: Look for article by role
        content_by_role = soup.find(attrs={"role": "main"})
        if content_by_role:
            content = self._clean_extracted_text(content_by_role.get_text())
            if len(content) >= self.min_content_length:
                return content
        
        # Strategy 3: Look for article with most paragraphs
        paragraphs_by_container = {}
        for tag in self.CONTENT_TAGS:
            elements = soup.find_all(tag)
            for element in elements:
                if self._should_ignore_element(element):
                    continue
                p_count = len(element.find_all('p'))
                if p_count >= 3:  # Only consider elements with at least 3 paragraphs
                    paragraphs_by_container[element] = p_count
        
        if paragraphs_by_container:
            main_container = max(paragraphs_by_container, key=paragraphs_by_container.get)
            content = self._clean_extracted_text(main_container.get_text())
            if len(content) >= self.min_content_length:
                return content
        
        return ""
    
    def _extract_all_paragraphs(self, soup):
        """Fallback method to extract all paragraph text from the page."""
        # Remove script, style, and non-content elements
        self._remove_non_content_elements(soup)
        
        # Get all paragraphs with substantial content
        paragraphs = []
        for p in soup.find_all('p'):
            if self._should_ignore_element(p):
                continue
            
            text = p.get_text().strip()
            if len(text) >= 40:  # Only consider paragraphs with some substantial content
                paragraphs.append(text)
        
        return self._clean_extracted_text("\n\n".join(paragraphs))
    
    def _remove_non_content_elements(self, soup):
        """Remove elements that are not part of the main content."""
        # Remove script and style elements
        for element in soup.find_all(['script', 'style', 'noscript', 'iframe', 'svg']):
            element.decompose()
        
        # Remove elements by class and id patterns
        for class_name in self.IGNORE_CLASSES:
            for element in soup.find_all(class_=re.compile(class_name, re.I)):
                element.decompose()
        
        for id_name in self.IGNORE_IDS:
            for element in soup.find_all(id=re.compile(id_name, re.I)):
                element.decompose()
    
    def _should_ignore_element(self, element):
        """Check if an element should be ignored based on its attributes."""
        # Check class
        if element.get('class'):
            for class_name in element.get('class'):
                for ignore_class in self.IGNORE_CLASSES:
                    if ignore_class.lower() in class_name.lower():
                        return True
        
        # Check id
        if element.get('id'):
            for ignore_id in self.IGNORE_IDS:
                if ignore_id.lower() in element.get('id').lower():
                    return True
        
        return False
    
    def _clean_extracted_text(self, text):
        """Clean extracted text by removing extra whitespace, etc."""
        if not text:
            return ""
        
        # Replace multiple newlines and spaces with single instances
        text = re.sub(r'\n{2,}', '\n\n', text)
        text = re.sub(r'[ \t]{2,}', ' ', text)
        
        # Normalize whitespace
        lines = [line.strip() for line in text.splitlines()]
        text = '\n'.join(line for line in lines if line)
        
        return text
