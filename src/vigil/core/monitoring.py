"""
Monitoring module for the Vigil cybersecurity platform.
Provides system status monitoring and metrics.
"""
import logging
import time
import threading
from datetime import datetime
import os
import json
from typing import Dict, List

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from vigil.database import queries
from vigil.database.connection import get_engine

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("vigil.core.monitoring")

class SystemMonitor:
    """Monitors the Vigil system and collects metrics."""
    
    def __init__(self, config_path=None):
        """Initialize the system monitor."""
        self.components = {
            "database": {"status": "unknown", "last_check": None},
            "crawler": {"status": "unknown", "last_check": None},
            "pipeline": {"status": "unknown", "last_check": None}
        }
        
        self.metrics = {
            "incidents_total": 0,
            "sources_total": 0,
            "tags_total": 0,
            "last_incident": None
        }
        
        self.is_monitoring = False
        self.monitor_thread = None
        self.start_time = datetime.now()
        self.check_interval = 60  # seconds
        logger.info("System monitor initialized")
    
    def check_database(self):
        """Check database connection and get metrics."""
        try:
            # Check connection
            engine = get_engine()
            connection = engine.connect()
            connection.close()
            
            # Update component status
            self.components["database"] = {
                "status": "up",
                "last_check": datetime.now()
            }
            
            # Get metrics
            sources = queries.list_sources()
            self.metrics["sources_total"] = len(sources)
            
            tags = queries.list_tags()
            self.metrics["tags_total"] = len(tags)
            
            # Get total incidents and latest
            incidents = queries.list_incidents(per_page=1)
            self.metrics["incidents_total"] = incidents["total"]
            
            if incidents["items"]:
                latest = incidents["items"][0]
                self.metrics["last_incident"] = {
                    "id": latest.id,
                    "title": latest.title,
                    "created_at": latest.created_at.isoformat()
                }
            
            return True
        
        except Exception as e:
            logger.error(f"Database check failed: {str(e)}")
            self.components["database"] = {
                "status": "down",
                "last_check": datetime.now(),
                "error": str(e)
            }
            return False
    
    def check_all(self):
        """Check all system components."""
        self.check_database()
        
        # Crawler and pipeline are marked as up if their modules can be imported
        try:
            from vigil.data_collection.crawler import Crawler
            self.components["crawler"] = {
                "status": "up",
                "last_check": datetime.now()
            }
        except Exception as e:
            self.components["crawler"] = {
                "status": "down",
                "last_check": datetime.now(),
                "error": str(e)
            }
        
        try:
            from vigil.core.pipeline import Pipeline
            self.components["pipeline"] = {
                "status": "up",
                "last_check": datetime.now()
            }
        except Exception as e:
            self.components["pipeline"] = {
                "status": "down",
                "last_check": datetime.now(),
                "error": str(e)
            }
    
    def get_system_status(self):
        """Get the overall system status."""
        # Update status before returning
        self.check_all()
        
        # Calculate overall status
        component_statuses = [c["status"] for c in self.components.values()]
        if all(status == "up" for status in component_statuses):
            overall_status = "healthy"
        elif any(status == "down" for status in component_statuses):
            overall_status = "degraded"
        else:
            overall_status = "unknown"
        
        # Calculate uptime
        uptime_seconds = (datetime.now() - self.start_time).total_seconds()
        hours, remainder = divmod(uptime_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        uptime_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
        
        return {
            "timestamp": datetime.now().isoformat(),
            "status": overall_status,
            "uptime": uptime_str,
            "uptime_seconds": uptime_seconds,
            "components": self.components,
            "metrics": self.metrics
        }
    
    def start_monitoring(self):
        """Start monitoring in a background thread."""
        if self.is_monitoring:
            logger.warning("Monitoring is already running")
            return False
        
        self.is_monitoring = True
        
        def monitor_loop():
            logger.info("System monitoring started")
            while self.is_monitoring:
                try:
                    self.check_all()
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(self.check_interval)
        
        self.monitor_thread = threading.Thread(target=monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        return True
    
    def stop_monitoring(self):
        """Stop the monitoring thread."""
        if not self.is_monitoring:
            logger.warning("Monitoring is not running")
            return False
        
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
            self.monitor_thread = None
        
        logger.info("System monitoring stopped")
        return True
    
    def display_dashboard(self, refresh_interval=5):
        """
        Display a continuously updating status dashboard using Rich.
        
        Args:
            refresh_interval: Time between dashboard updates in seconds
        """
        console = Console()
        
        try:
            while True:
                # Clear the console
                console.clear()
                
                # Check the status
                self.check_all()
                
                # Create component status table
                status_table = Table(title="System Components")
                status_table.add_column("Component")
                status_table.add_column("Status")
                status_table.add_column("Last Check")
                
                for name, info in self.components.items():
                    status_color = "green" if info["status"] == "up" else "red"
                    last_check = info["last_check"].strftime("%H:%M:%S") if info["last_check"] else "Never"
                    
                    status_table.add_row(
                        name.capitalize(),
                        f"[{status_color}]{info['status']}[/]",
                        last_check
                    )
                
                # Create metrics panel
                metrics_text = [
                    f"Incidents: {self.metrics['incidents_total']}",
                    f"Sources: {self.metrics['sources_total']}",
                    f"Tags: {self.metrics['tags_total']}"
                ]
                
                if self.metrics["last_incident"]:
                    metrics_text.append("\nLatest Incident:")
                    metrics_text.append(f"  {self.metrics['last_incident']['title']}")
                    metrics_text.append(f"  (ID: {self.metrics['last_incident']['id']})")
                
                metrics_panel = Panel(
                    "\n".join(metrics_text),
                    title="System Metrics",
                    border_style="blue"
                )
                
                # Calculate uptime
                uptime_seconds = (datetime.now() - self.start_time).total_seconds()
                hours, remainder = divmod(uptime_seconds, 3600)
                minutes, seconds = divmod(remainder, 60)
                uptime_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
                
                # Create system info panel
                system_text = [
                    f"Status: {self.get_system_status()['status']}",
                    f"Uptime: {uptime_str}",
                    f"Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}",
                    f"\nLast refresh: {datetime.now().strftime('%H:%M:%S')}",
                    f"Refresh interval: {refresh_interval}s",
                    "\n[Press Ctrl+C to exit]"
                ]
                
                system_panel = Panel(
                    "\n".join(system_text),
                    title="System Information",
                    border_style="green"
                )
                
                # Display all elements
                console.print(status_table)
                console.print()
                console.print(metrics_panel)
                console.print()
                console.print(system_panel)
                
                # Wait for next refresh
                time.sleep(refresh_interval)
                
        except KeyboardInterrupt:
            console.print("\n[bold]Dashboard stopped.[/bold]")

def main():
    """Entry point for command line execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Vigil system monitor")
    parser.add_argument("--dashboard", action="store_true", help="Display status dashboard")
    parser.add_argument("--monitor", action="store_true", help="Start background monitoring")
    parser.add_argument("--interval", type=int, default=60, help="Monitoring interval in seconds")
    parser.add_argument("--refresh", type=int, default=5, help="Dashboard refresh interval in seconds")
    parser.add_argument("--export", help="Export system status to JSON file")
    
    args = parser.parse_args()
    
    monitor = SystemMonitor()
    
    if args.dashboard:
        monitor.display_dashboard(refresh_interval=args.refresh)
    
    elif args.monitor:
        monitor.check_interval = args.interval
        monitor.start_monitoring()
        print(f"Monitoring started (interval: {args.interval}s)")
        print("Press Ctrl+C to stop...")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            monitor.stop_monitoring()
            print("Monitoring stopped")
    
    elif args.export:
        status = monitor.get_system_status()
        with open(args.export, 'w') as f:
            json.dump(status, f, indent=2, default=str)
        print(f"System status exported to {args.export}")
    
    else:
        # Just print a simple status report
        status = monitor.get_system_status()
        print(f"System Status: {status['status']}")
        print(f"Uptime: {status['uptime']}")
        print("\nComponents:")
        for name, info in status["components"].items():
            print(f"  {name.capitalize()}: {info['status']}")
        print("\nMetrics:")
        for name, value in status["metrics"].items():
            if isinstance(value, dict):
                continue
            print(f"  {name}: {value}")

if __name__ == "__main__":
    main()
