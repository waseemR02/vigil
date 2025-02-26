"""
Pytest configuration file.
Contains shared fixtures and configuration for all tests.
"""

import pytest
import logging


@pytest.fixture(autouse=True)
def disable_logging():
    """
    Disable logging output during tests.
    This fixture runs automatically for all tests.
    """
    loggers = [
        logging.getLogger("vigil"),
        logging.getLogger("vigil.crawler"),
        logging.getLogger("vigil.frontier"),
        logging.getLogger("vigil.strategies"),
        logging.getLogger("vigil.executor"),
    ]

    # Save original levels to restore later
    original_levels = [logger.level for logger in loggers]
    
    # Set all to ERROR to minimize test output
    for logger in loggers:
        logger.setLevel(logging.ERROR)
    
    yield
    
    # Restore original levels
    for logger, level in zip(loggers, original_levels):
        logger.setLevel(level)
