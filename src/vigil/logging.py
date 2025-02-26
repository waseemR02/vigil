"""
Logger module for logging application events.
"""
import logging
import os
import sys


class ColoredFormatter(logging.Formatter):
    """
    Formatter that adds color to logs in the console.
    """
    # Color codes
    RESET = "\033[0m"
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    BOLD = "\033[1m"

    # Log level colors
    COLORS = {
        logging.DEBUG: CYAN,
        logging.INFO: GREEN,
        logging.WARNING: YELLOW,
        logging.ERROR: RED,
        logging.CRITICAL: RED + BOLD,
    }

    def format(self, record):
        # Save original levelname
        original_levelname = record.levelname
        # Add color to levelname if we're in a terminal
        if sys.stdout.isatty() and record.levelno in self.COLORS:
            # Pad the levelname to 5 characters before adding color
            color = self.COLORS[record.levelno]
            record.levelname = f"{color}{original_levelname:7}{self.RESET}"
        else:
            # Just pad without color
            record.levelname = f"{original_levelname:7}"
        
        # Format the record
        result = super().format(record)
        
        # Restore original levelname
        record.levelname = original_levelname
        return result


class Logger:
    """
    Logger class for logging application events.
    """

    def __init__(self, name=None):
        """
        Initialize Logger.

        :param name: Logger name
        """
        self.name = name or __name__
        self.logger = logging.getLogger(self.name)
        
        # Base format strings
        self.standard_format = '%(asctime)s - [%(levelname)-7s] - %(message)s'
        self.debug_format = '%(asctime)s - [%(levelname)-7s] - [%(module)s:%(lineno)d:%(funcName)s] - %(message)s'
        
        # Standard formatters for file output
        self.standard_formatter = logging.Formatter(self.standard_format)
        self.debug_formatter = logging.Formatter(self.debug_format)
        
        # Colored formatters for console output
        self.colored_std_formatter = ColoredFormatter(self.standard_format)
        self.colored_debug_formatter = ColoredFormatter(self.debug_format)

    def set_level(self, level):
        """
        Set logger level.

        :param level: Logger level
        """
        # Set the level for the logger itself
        self.logger.setLevel(level)
        
        # Set the level for all handlers unless they have their own configuration
        for handler in self.logger.handlers:
            if not getattr(handler, 'custom_level_set', False):
                handler.setLevel(level)
        
        return self

    def add_console_handler(self, level=None):
        """
        Add console handler.

        :param level: Console handler level
        """
        console_handler = logging.StreamHandler()
        
        # Choose formatter based on level, using colored formatters for console
        if level == logging.DEBUG:
            console_handler.setFormatter(self.colored_debug_formatter)
        else:
            console_handler.setFormatter(self.colored_std_formatter)
        
        if level is not None:
            console_handler.setLevel(level)
            console_handler.custom_level_set = True
        else:
            console_handler.setLevel(self.logger.level)
        
        self.logger.addHandler(console_handler)
        return self

    def add_file_handler(self, filename, level=None, mode='a'):
        """
        Add file handler.

        :param filename: Log filename
        :param level: File handler level
        :param mode: File open mode ('a' for append, 'w' for overwrite)
        """
        # Ensure the directory exists
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        
        file_handler = logging.FileHandler(filename, mode=mode)
        
        # Choose formatter based on level
        if level == logging.DEBUG:
            file_handler.setFormatter(self.debug_formatter)
        else:
            file_handler.setFormatter(self.standard_formatter)
        
        if level is not None:
            file_handler.setLevel(level)
            file_handler.custom_level_set = True
        else:
            file_handler.setLevel(self.logger.level)
        
        self.logger.addHandler(file_handler)
        return self

    def get_logger(self):
        """
        Get logger.

        :return: Logger
        """
        return self.logger


def setup_logging(log_config):
    """
    Setup logging.

    :param log_config: Logging configuration
    :return: Logger
    """
    # Gracefully handle missing config
    if not log_config:
        print("Warning: Logging configuration is missing or empty. Using default settings.")
        log_config = {}

    # Get the base log level with a default of INFO
    log_level_str = log_config.get('level', 'INFO')
    
    # Handle the case where log_level is already an integer
    if isinstance(log_level_str, int):
        log_level = log_level_str
    else:
        try:
            log_level = getattr(logging, log_level_str.upper())
        except (AttributeError, TypeError):
            print(f"Warning: Invalid log level '{log_level_str}'. Using INFO level.")
            log_level = logging.INFO

    # Get console and file levels, handling cases where they might be integers already
    console_level = log_config.get('console_level', log_level)
    if not isinstance(console_level, int):
        try:
            console_level = getattr(logging, console_level.upper())
        except (AttributeError, TypeError):
            print(f"Warning: Invalid console log level. Using base log level.")
            console_level = log_level

    file_level = log_config.get('file_level', log_level)
    if not isinstance(file_level, int):
        try:
            file_level = getattr(logging, file_level.upper())
        except (AttributeError, TypeError):
            print(f"Warning: Invalid file log level. Using base log level.")
            file_level = log_level

    # The root logger should be set to the lowest level of any handler
    root_level = min(log_level, console_level, file_level)
    
    # Setup logger (removed use_color parameter)
    logger_name = log_config.get('name', 'vigil')
    logger = Logger(logger_name).set_level(root_level)
    
    # Add console handler if enabled
    if log_config.get('console', True):
        logger.add_console_handler(console_level)
    
    # Add file handler if enabled and path is provided
    if log_config.get('file'):
        log_path = log_config.get('file')
        try:
            # Create directory even if it doesn't exist
            directory = os.path.dirname(os.path.abspath(log_path))
            os.makedirs(directory, exist_ok=True)
            
            # Determine file mode - 'w' to overwrite, 'a' to append
            file_mode = 'w' if log_config.get('file_overwrite', True) else 'a'
            
            logger.add_file_handler(log_path, file_level, mode=file_mode)
            
        except Exception as e:
            print(f"Warning: Failed to set up file logging: {e}")
    
    return logger.get_logger()
