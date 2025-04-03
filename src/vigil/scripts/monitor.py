"""
Script to run the Vigil monitoring system.
Provides a command-line interface to the system monitor.
"""
import argparse
import json
import time
import os
import sys
from datetime import datetime
import logging

from vigil.core.monitoring import SystemMonitor
from vigil.logging import setup_logging
from vigil.config import load_config

def export_status(monitor, export_dir, logger):
    """Export system status to a file in the specified directory."""
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(export_dir, f"status_{timestamp}.json")
    
    status = monitor.get_system_status()
    with open(filename, 'w') as f:
        json.dump(status, f, indent=2, default=str)
    
    logger.info(f"Exported status to {filename}")

def run_monitor(args):
    """Run the system monitor based on command line arguments."""
    # Load configuration
    config = load_config()
    if args.config:
        try:
            config.load_from_file(args.config)
        except Exception as e:
            print(f"Error loading config file: {e}")
            sys.exit(1)
    
    # Set up logging
    log_config = config.config.get('logging', {})
    if args.verbose:
        log_config['console_level'] = 'DEBUG'
    
    # Initialize logging system and get the logger
    logger = setup_logging(log_config)
    
    # Create monitor with the same config
    monitor = SystemMonitor(args.config)
    
    if args.interval:
        monitor.check_interval = args.interval
    
    if args.dashboard:
        # Display the interactive dashboard with the specified refresh rate
        refresh_interval = args.refresh if args.refresh else 5
        monitor.display_dashboard(refresh_interval=refresh_interval)
    
    elif args.background:
        # Run monitoring in the background
        monitor.start_monitoring()
        print(f"Background monitoring started (interval: {monitor.check_interval}s)")
        print("Press Ctrl+C to stop...")
        
        try:
            last_export_time = 0
            while True:
                time.sleep(1)
                
                # Export status periodically if requested
                if args.export_dir and time.time() - last_export_time >= args.export_interval:
                    export_status(monitor, args.export_dir, logger)
                    last_export_time = time.time()
                    
        except KeyboardInterrupt:
            monitor.stop_monitoring()
            print("Monitoring stopped")
    
    elif args.export:
        # Just export the current status
        status = monitor.get_system_status()
        
        with open(args.export, 'w') as f:
            json.dump(status, f, indent=2, default=str)
        print(f"System status exported to {args.export}")
    
    else:
        # Just print the current status
        status = monitor.get_system_status()
        print(json.dumps(status, indent=2, default=str))

def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Run Vigil system monitor")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--dashboard", action="store_true", help="Display interactive dashboard")
    parser.add_argument("--background", action="store_true", help="Run in background")
    parser.add_argument("--interval", type=int, help="Check interval in seconds")
    parser.add_argument("--refresh", type=int, default=5, help="Dashboard refresh interval in seconds")
    parser.add_argument("--export", help="Export status to file")
    parser.add_argument("--export-dir", help="Directory for periodic status exports")
    parser.add_argument("--export-interval", type=int, default=300, 
                      help="Interval between exports (seconds)")
    
    args = parser.parse_args()
    
    try:
        run_monitor(args)
    except KeyboardInterrupt:
        print("\nMonitor interrupted by user.")
    except Exception as e:
        # Initialize minimal logging if not already done
        minimal_logger = logging.getLogger('vigil.scripts.monitor')
        minimal_logger.error(f"Error running monitor: {str(e)}")
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
