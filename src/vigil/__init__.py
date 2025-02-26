from vigil.config import load_config
from vigil.logging import setup_logging
import logging
import os

# Create a module level logger
logger = logging.getLogger('vigil')

def main() -> None:
    """Main entry point for the VIGIL application."""
    # Load configuration
    config = load_config()
    
    # Set up logging system
    global logger
    logger = setup_logging(config.config.get('logging', {}))
    
    # Verify logging setup
    log_file = config.config.get('logging', {}).get('file')
    if log_file and os.path.isfile(log_file):
        logger.info(f"Logging to file: {log_file}")
    elif log_file:
        logger.warning(f"Log file specified but not created: {log_file}")
    
    logger.info(f"VIGIL initialized with config from {config.config_file}")
    logger.debug("Configuration details:")
    for section in config.config:
        logger.debug(f"  [{section}]")
        for key, value in config.config[section].items():
            logger.debug(f"    {key}: {value}")
    
    # Application logic follows here
    logger.info("Starting VIGIL application...")
