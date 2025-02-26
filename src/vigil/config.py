import os
import yaml
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration manager for VIGIL."""
    
    def __init__(self):
        self.config: Dict[str, Any] = {}
        self.config_file: str = ""
        
    def load_from_file(self, config_file: str) -> None:
        """Load configuration from a YAML file."""
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f) or {}
        self.config_file = config_file
            
    def load_from_env(self) -> None:
        """Override configuration with environment variables."""
        prefix = "VIGIL_"
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Convert VIGIL_DATABASE_URL to config['database']['url']
                parts = key[len(prefix):].lower().split('_')
                
                # Navigate to the right level in the config dict
                current = self.config
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                
                # Set the value
                current[parts[-1]] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value, optionally providing a default."""
        parts = key.split('.')
        current = self.config
        
        try:
            for part in parts:
                current = current[part]
            return current
        except (KeyError, TypeError):
            return default


def load_config() -> Config:
    """Load configuration from default locations and environment."""
    config = Config()
    
    # Default config with minimal settings
    default_config = {
        'logging': {
            'level': 'INFO',
        }
    }
    
    config.config = default_config
    
    # Look for config file in common locations
    config_locations = [
        os.environ.get('VIGIL_CONFIG'),
        os.path.join(os.getcwd(), 'config.yml'),
        os.path.expanduser('~/.config/vigil/config.yml'),
        '/etc/vigil/config.yml',
    ]
    
    for location in config_locations:
        if location and os.path.isfile(location):
            config.load_from_file(location)
            break
    else:
        # No config file found, use default
        config.config_file = "default"
    
    # Override with environment variables
    config.load_from_env()
    
    return config
