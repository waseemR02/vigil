from vigil.config import load_config

def main() -> None:
    """Main entry point for the VIGIL application."""
    config = load_config()
    print(f"VIGIL initialized with config from {config.config_file}")
    print("Configuration values:")
    for section in config.config:
        print(f"  [{section}]")
        for key, value in config.config[section].items():
            print(f"    {key}: {value}")
