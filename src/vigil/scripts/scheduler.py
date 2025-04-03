"""
Script to run the Vigil scheduler.
Provides a command-line interface to the scheduling system.
"""
import argparse
import json
import time
import sys
from datetime import datetime
import logging

from vigil.core.scheduler import VigilScheduler
from vigil.logging import setup_logging
from vigil.config import load_config

def run_scheduler(args):
    """Run the scheduler based on command line arguments."""
    # Load configuration
    config = load_config()
    if args.config:
        try:
            config.load_from_file(args.config)
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
            sys.exit(1)
    
    # Set up logging
    log_config = config.config.get('logging', {})
    if args.verbose:
        log_config['console_level'] = 'DEBUG'
    
    # Initialize logging system and get the logger
    logger = setup_logging(log_config)
    
    # Initialize the scheduler with model and vectorizer paths
    scheduler = VigilScheduler(model_path=args.model_path, vectorizer_path=args.vectorizer_path)
    
    # Initialize the pipeline to create URL tracking table if needed
    if hasattr(scheduler, 'pipeline') and scheduler.pipeline is None:
        from vigil.core.pipeline import Pipeline
        scheduler.pipeline = Pipeline(model_path=args.model_path, vectorizer_path=args.vectorizer_path)
    
    # Always try to load jobs from config if no explicit action is specified
    # This ensures jobs are available when starting or exporting
    if args.load_config or (args.start and not args.load_config) or (args.export and not args.load_config):
        jobs_loaded = scheduler.load_jobs_from_config()
        if jobs_loaded:
            logger.info(f"Loaded {len(scheduler.jobs)} jobs from configuration")
    
    if args.add_job:
        job_id = args.job_id or f"job-{int(time.time())}"
        if scheduler.add_job(job_id, args.add_job, args.interval, args.max_urls):
            logger.info(f"Added job {job_id}")
        else:
            logger.error("Failed to add job")
    
    elif args.remove_job:
        if scheduler.remove_job(args.remove_job):
            logger.info(f"Removed job {args.remove_job}")
        else:
            logger.error(f"Failed to remove job {args.remove_job}")
    
    elif args.list_jobs:
        jobs = scheduler.list_jobs()
        if not jobs:
            logger.info("No jobs scheduled")
        else:
            logger.info(f"Scheduled Jobs ({len(jobs)}):")
            for job in jobs:
                logger.info(f"  {job['id']}: {job['url']} (every {job['interval_minutes']} minutes)")
    
    elif args.job_history:
        history = scheduler.get_job_history(limit=args.limit)
        if not history:
            logger.info("No job history available")
        else:
            logger.info(f"Job History ({len(history)}):")
            for entry in history:
                status = entry["status"]
                logger.info(f"  {entry['job_id']}: {status} at {entry['end_time']}")
    
    elif args.run_once:
        if args.job_id:
            logger.info(f"Running job {args.job_id} once...")
            scheduler._run_job(args.job_id)
        else:
            logger.error("Job ID required for run-once operation")
    
    elif args.start:
        # Make sure we have jobs
        if not scheduler.jobs:
            logger.warning("Warning: No jobs scheduled. Use --load-config to load jobs from config file.")
            if input("Do you want to continue with no jobs? (y/n): ").lower() != 'y':
                return
        
        if scheduler.start():
            logger.info(f"Scheduler started with {len(scheduler.jobs)} jobs")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                scheduler.stop()
                logger.info("Scheduler stopped")
        else:
            logger.error("Failed to start scheduler")
    
    elif args.export:
        # Make sure we have jobs to export
        if not scheduler.jobs:
            logger.warning("No jobs to export.")
        
        jobs_data = {
            "jobs": [job for job in scheduler.list_jobs()],
            "history": scheduler.get_job_history(limit=None),
            "exported_at": datetime.now().isoformat()
        }
        
        with open(args.export, 'w') as f:
            json.dump(jobs_data, f, indent=2, default=str)
        logger.info(f"Jobs exported to {args.export} ({len(scheduler.jobs)} jobs)")
    
    else:
        logger.info("No action specified. Use --help to see available commands.")

def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Run Vigil scheduler")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--load-config", action="store_true", help="Load jobs from config file")
    
    # Job management arguments
    parser.add_argument("--add-job", help="Add a new job (URL)")
    parser.add_argument("--job-id", help="Job ID for adding or running a job")
    parser.add_argument("--remove-job", help="Remove a job by ID")
    parser.add_argument("--list-jobs", action="store_true", help="List all jobs")
    parser.add_argument("--job-history", action="store_true", help="Show job execution history")
    parser.add_argument("--limit", type=int, default=10, help="Limit for job history")
    
    # Job parameters
    parser.add_argument("--interval", type=int, default=60, help="Job interval in minutes")
    parser.add_argument("--max-urls", type=int, default=50, help="Maximum URLs per job")
    
    # Model and vectorizer paths
    parser.add_argument("--model-path", help="Path to the ML model")
    parser.add_argument("--vectorizer-path", help="Path to the vectorizer")
    
    # Execution control
    parser.add_argument("--start", action="store_true", help="Start the scheduler")
    parser.add_argument("--run-once", action="store_true", help="Run a job once (requires --job-id)")
    
    # Export/import
    parser.add_argument("--export", help="Export jobs to a JSON file")
    
    args = parser.parse_args()
    
    # Create minimal logger for initial errors
    global logger
    logger = logging.getLogger('vigil.scripts.scheduler')
    
    try:
        run_scheduler(args)
    except KeyboardInterrupt:
        logger.info("Scheduler interrupted by user.")
    except Exception as e:
        logger.error(f"Error running scheduler: {str(e)}")

if __name__ == "__main__":
    main()
