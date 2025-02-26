"""URL handling utilities for the Vigil monitoring system."""

import re
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse, ParseResult
from typing import Dict, Optional, Tuple


def normalize_url(url: str) -> str:
    """
    Normalize a URL to a canonical form.
    
    Handles:
    - Lowercasing the scheme and hostname
    - Sorting query parameters
    - Removing default ports (80 for HTTP, 443 for HTTPS)
    - Removing trailing slashes
    - Removing www. prefix
    
    Args:
        url: The URL to normalize
        
    Returns:
        A normalized URL string
    """
    if not url:
        return ""
    
    # Parse the URL
    parsed = urlparse(url)
    
    # Lowercase scheme and netloc
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    
    # Remove default ports
    if (scheme == 'http' and netloc.endswith(':80')) or (scheme == 'https' and netloc.endswith(':443')):
        netloc = netloc.rsplit(':', 1)[0]
    
    # Remove www. prefix
    if netloc.startswith('www.'):
        netloc = netloc[4:]
    
    # Sort query parameters
    query_params = parse_qsl(parsed.query)
    query_params.sort()
    query = urlencode(query_params)
    
    # Handle path - remove trailing slashes except for root path
    path = parsed.path
    if path.endswith('/') and path != '/':
        path = path[:-1]
    
    # If path is empty, set to '/' for consistency
    if not path:
        path = '/'
    
    # Rebuild the URL
    normalized = urlunparse((scheme, netloc, path, parsed.params, query, ''))
    return normalized


def extract_domain(url: str) -> str:
    """
    Extract the domain from a URL.
    
    Args:
        url: The URL to extract the domain from
        
    Returns:
        The domain name without subdomain
    """
    if not url:
        return ""
    
    parsed = urlparse(url)
    netloc = parsed.netloc.lower()
    
    # Remove port if present
    if ':' in netloc:
        netloc = netloc.split(':')[0]
        
    # Extract the main domain (last two parts)
    parts = netloc.split('.')
    if len(parts) > 2:
        # Handle special cases like co.uk, com.au
        if parts[-2] in ['co', 'com', 'org', 'net', 'gov', 'edu'] and parts[-1] in ['uk', 'au', 'jp', 'br']:
            return '.'.join(parts[-3:])
        return '.'.join(parts[-2:])
    return netloc


def is_valid_url(url: str) -> bool:
    """
    Validate if a string is a properly formatted URL.
    
    Args:
        url: The URL to validate
        
    Returns:
        True if the URL is valid, False otherwise
    """
    if not url:
        return False
    
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False


def get_url_params(url: str) -> Dict[str, str]:
    """
    Extract query parameters from a URL.
    
    Args:
        url: The URL to extract parameters from
        
    Returns:
        Dictionary of parameter names and values
    """
    if not url:
        return {}
        
    parsed = urlparse(url)
    # Use keep_blank_values=True to preserve empty parameter values
    return dict(parse_qsl(parsed.query, keep_blank_values=True))


def update_url_params(url: str, params_to_add: Dict[str, str], params_to_remove: Optional[list] = None) -> str:
    """
    Update the query parameters of a URL.
    
    Args:
        url: The URL to modify
        params_to_add: Dictionary of parameters to add or update
        params_to_remove: List of parameter names to remove
        
    Returns:
        Updated URL string
    """
    if not url:
        return ""
        
    parsed = urlparse(url)
    
    # Get existing parameters and update them
    params = dict(parse_qsl(parsed.query))
    
    # Remove parameters if specified
    if params_to_remove:
        for param in params_to_remove:
            params.pop(param, None)
    
    # Add/update parameters
    params.update(params_to_add)
    
    # Reconstruct the URL
    parts = list(parsed)
    parts[4] = urlencode(params)
    return urlunparse(parts)


def join_url_path(base_url: str, path: str) -> str:
    """
    Join a base URL and a path, handling trailing slashes correctly.
    
    Args:
        base_url: The base URL
        path: The path to append
        
    Returns:
        Combined URL
    """
    if not base_url:
        return path
    
    if not path:
        return base_url
        
    # Remove trailing slash from base_url
    if base_url.endswith('/'):
        base_url = base_url[:-1]
    
    # Remove leading slash from path
    if path.startswith('/'):
        path = path[1:]
    
    return f"{base_url}/{path}"
