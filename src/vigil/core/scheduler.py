"""
Scheduler module for the Vigil cybersecurity platform.
Schedules and manages crawl jobs at specified intervals.
"""
import time
import threading
from datetime import datetime
import os
import sqlite3
from threading import Lock
import logging

from vigil.config import load_config
from vigil.core.pipeline import Pipeline

# Configure module logger
logger = logging.getLogger('vigil.core.scheduler')

class JobStatus:
    """Job status constants."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class SchedulerDb:
    """SQLite database manager for the scheduler."""
    
    def __init__(self, db_path):
        """Initialize the database connection."""
        self.db_path = db_path
        self.lock = Lock()  # Thread safety for database operations
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Initialize database
        self._init_db()
    
    def _init_db(self):
        """Create necessary tables if they don't exist."""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create jobs table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                url TEXT NOT NULL,
                interval_minutes INTEGER DEFAULT 60,
                max_urls INTEGER DEFAULT 50,
                next_run TEXT,
                status TEXT,
                last_run TEXT,
                run_count INTEGER DEFAULT 0,
                created_at TEXT
            )
            ''')
            
            # Create job executions table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS job_executions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL,
                url TEXT NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT NOT NULL,
                status TEXT NOT NULL,
                urls_processed INTEGER DEFAULT 0,
                relevant_found INTEGER DEFAULT 0,
                error TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            conn.commit()
            conn.close()
    
    def list_jobs(self):
        """List all scheduled jobs."""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM jobs")
            jobs = [dict(row) for row in cursor.fetchall()]
            
            conn.close()
            return jobs
    
    def get_job(self, job_id):
        """Get a job by id."""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
            job = cursor.fetchone()
            
            conn.close()
            return dict(job) if job else None
    
    def add_job(self, job_id, url, interval_minutes, max_urls, created_at, next_run, status):
        """Add a new job."""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            INSERT OR REPLACE INTO jobs
            (job_id, url, interval_minutes, max_urls, created_at, next_run, status, run_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, 0)
            ''', (job_id, url, interval_minutes, max_urls, created_at, next_run, status))
            
            conn.commit()
            conn.close()
    
    def update_job(self, job_id, next_run=None, status=None, last_run=None, run_count=None):
        """Update job fields."""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Build the update query based on provided parameters
            update_parts = []
            params = []
            
            if next_run is not None:
                update_parts.append("next_run = ?")
                params.append(next_run)
            
            if status is not None:
                update_parts.append("status = ?")
                params.append(status)
            
            if last_run is not None:
                update_parts.append("last_run = ?")
                params.append(last_run)
            
            if run_count is not None:
                update_parts.append("run_count = ?")
                params.append(run_count)
            
            if not update_parts:
                return  # Nothing to update
            
            query = f"UPDATE jobs SET {', '.join(update_parts)} WHERE job_id = ?"
            params.append(job_id)
            
            cursor.execute(query, params)
            conn.commit()
            conn.close()
    
    def remove_job(self, job_id):
        """Remove a job."""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
            
            conn.commit()
            conn.close()
            return cursor.rowcount > 0
    
    def log_execution(self, job_id, url, start_time, end_time, status, urls_processed=0, relevant_found=0, error=None):
        """Log a job execution."""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            INSERT INTO job_executions
            (job_id, url, start_time, end_time, status, urls_processed, relevant_found, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (job_id, url, start_time, end_time, status, urls_processed, relevant_found, error))
            
            last_id = cursor.lastrowid
            conn.commit()
            conn.close()
            return last_id
    
    def get_executions(self, job_id=None, limit=10):
        """Get job execution history."""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            if job_id:
                cursor.execute(
                    "SELECT * FROM job_executions WHERE job_id = ? ORDER BY end_time DESC LIMIT ?", 
                    (job_id, limit)
                )
            else:
                cursor.execute(
                    "SELECT * FROM job_executions ORDER BY end_time DESC LIMIT ?", 
                    (limit,)
                )
            
            executions = [dict(row) for row in cursor.fetchall()]
            conn.close()
            return executions


class VigilScheduler:
    """Scheduler for running cybersecurity crawl jobs at specified intervals."""
    
    def __init__(self, model_path=None, vectorizer_path=None):
        """Initialize the scheduler."""
        # Load configuration
        self.config = self._load_config()
        
        self.pipeline = Pipeline(model_path=model_path, vectorizer_path=vectorizer_path)
        self.jobs = {}
        self.is_running = False
        self.scheduler_thread = None
        self.job_history = []
        
        # Set up the separate scheduler database
        scheduler_db_path = self.config.get('scheduler', {}).get('db_path', 'data/scheduler.db')
        self.db = SchedulerDb(scheduler_db_path)
        
        # Load jobs from the scheduler database
        self._load_jobs_from_db()
        
        logger.info("Scheduler initialized")
    
    def _load_config(self):
        """Load configuration from YAML file."""
        
        try:
            return load_config()
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            return {}
    
    def _load_jobs_from_db(self):
        """Load jobs from database."""
        try:
            db_jobs = self.db.list_jobs()
            for job in db_jobs:
                self.jobs[job['job_id']] = {
                    "id": job['job_id'],
                    "url": job['url'],
                    "interval_minutes": job['interval_minutes'],
                    "max_urls": job['max_urls'],
                    "next_run": job['next_run'],
                    "status": job['status'] or JobStatus.PENDING,
                    "last_run": job['last_run'],
                    "run_count": job['run_count'] or 0,
                    "created_at": job['created_at'],
                }
            
            logger.info(f"Loaded {len(self.jobs)} jobs from database")
            return True
        except Exception as e:
            logger.error(f"Error loading jobs from database: {str(e)}")
            return False
    
    def add_job(self, job_id, url, interval_minutes=60, max_urls=50):
        """
        Add a job to the scheduler.
        
        Args:
            job_id: Unique identifier for the job
            url: Seed URL to crawl
            interval_minutes: How often to run the job (in minutes)
            max_urls: Maximum number of URLs to process
            
        Returns:
            bool: True if job was added, False otherwise
        """
        if job_id in self.jobs:
            logger.warning(f"Job {job_id} already exists")
            return False
        
        created_at = datetime.now().isoformat()
        next_run = created_at  # Default to run immediately
        
        # Create job in memory
        self.jobs[job_id] = {
            "id": job_id,
            "url": url,
            "interval_minutes": interval_minutes,
            "max_urls": max_urls,
            "next_run": next_run,
            "status": JobStatus.PENDING,
            "last_run": None,
            "run_count": 0,
            "created_at": created_at,
        }
        
        # Create job in database
        try:
            self.db.add_job(
                job_id=job_id,
                url=url,
                interval_minutes=interval_minutes,
                max_urls=max_urls,
                created_at=created_at,
                next_run=next_run,
                status=JobStatus.PENDING
            )
        except Exception as e:
            logger.error(f"Error creating job in database: {str(e)}")
            # Keep job in memory even if database fails
        
        logger.info(f"Added job {job_id} for {url} (every {interval_minutes} minutes)")
        return True
    
    def remove_job(self, job_id):
        """Remove a job from the scheduler."""
        if job_id not in self.jobs:
            logger.warning(f"Job {job_id} not found")
            return False
        
        # Remove from memory
        del self.jobs[job_id]
        
        # Remove from database
        try:
            self.db.remove_job(job_id)
        except Exception as e:
            logger.error(f"Error removing job from database: {str(e)}")
        
        logger.info(f"Removed job {job_id}")
        return True
    
    def get_job(self, job_id):
        """Get information about a job."""
        return self.jobs.get(job_id)
    
    def list_jobs(self):
        """List all scheduled jobs."""
        return list(self.jobs.values())
    
    def get_job_history(self, limit=10):
        """Get the history of job executions."""
        # Try to get from database
        try:
            return self.db.get_executions(limit=limit)
        except Exception as e:
            logger.error(f"Error getting job history from database: {str(e)}")
            # Fall back to in-memory history
            return self.job_history[-limit:] if limit else self.job_history
    
    def _run_job(self, job_id):
        """Run a job and update its status."""
        if job_id not in self.jobs:
            logger.error(f"Job {job_id} not found")
            return
        
        job = self.jobs[job_id]
        job["status"] = JobStatus.RUNNING
        job["last_run"] = datetime.now().isoformat()
        job["run_count"] += 1
        
        # Update job in database
        try:
            self.db.update_job(
                job_id=job_id,
                status=JobStatus.RUNNING,
                last_run=job["last_run"],
                run_count=job["run_count"]
            )
        except Exception as e:
            logger.error(f"Error updating job in database: {str(e)}")
        
        logger.info(f"Running job {job_id} for URL {job['url']}")
        
        start_time = datetime.now().isoformat()
        
        try:
            # Run the job using the pipeline
            results = self.pipeline.run_workflow(job["url"], job["max_urls"])
            
            # Update job status
            job["status"] = JobStatus.COMPLETED
            job["last_results"] = results
            next_run_dt = datetime.now().timestamp() + (job["interval_minutes"] * 60)
            job["next_run"] = datetime.fromtimestamp(next_run_dt).isoformat()
            
            # Update job in database
            try:
                self.db.update_job(
                    job_id=job_id,
                    status=JobStatus.COMPLETED,
                    next_run=job["next_run"]
                )
            except Exception as e:
                logger.error(f"Error updating job in database: {str(e)}")
            
            # Add to history
            execution = {
                "job_id": job_id,
                "url": job["url"],
                "start_time": start_time,
                "end_time": datetime.now().isoformat(),
                "status": JobStatus.COMPLETED,
                "urls_processed": results.get("urls_processed", 0),
                "relevant_found": results.get("relevant_found", 0)
            }
            self.job_history.append(execution)
            
            # Log execution to database
            try:
                self.db.log_execution(
                    job_id=job_id,
                    url=job["url"],
                    start_time=start_time,
                    end_time=datetime.now().isoformat(),
                    status=JobStatus.COMPLETED,
                    urls_processed=results.get("urls_processed", 0),
                    relevant_found=results.get("relevant_found", 0)
                )
            except Exception as e:
                logger.error(f"Error logging job execution to database: {str(e)}")
            
            logger.info(f"Job {job_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Error running job {job_id}: {str(e)}")
            
            # Update job status
            job["status"] = JobStatus.FAILED
            job["last_error"] = str(e)
            next_run_dt = datetime.now().timestamp() + (job["interval_minutes"] * 60)
            job["next_run"] = datetime.fromtimestamp(next_run_dt).isoformat()
            
            # Update job in database
            try:
                self.db.update_job(
                    job_id=job_id,
                    status=JobStatus.FAILED,
                    next_run=job["next_run"]
                )
            except Exception as db_error:
                logger.error(f"Error updating job in database: {str(db_error)}")
            
            # Add to history
            execution = {
                "job_id": job_id,
                "url": job["url"],
                "start_time": start_time,
                "end_time": datetime.now().isoformat(),
                "status": JobStatus.FAILED,
                "error": str(e)
            }
            self.job_history.append(execution)
            
            # Log execution to database
            try:
                self.db.log_execution(
                    job_id=job_id,
                    url=job["url"],
                    start_time=start_time,
                    end_time=datetime.now().isoformat(),
                    status=JobStatus.FAILED,
                    error=str(e)
                )
            except Exception as log_error:
                logger.error(f"Error logging job execution to database: {str(log_error)}")
    
    def start(self):
        """Start the scheduler."""
        if self.is_running:
            logger.warning("Scheduler is already running")
            return False
        
        self.is_running = True
        
        def scheduler_loop():
            logger.info("Scheduler started")
            
            while self.is_running:
                current_time = datetime.now()
                
                # Check for jobs to run
                for job_id, job in list(self.jobs.items()):
                    # Skip jobs that are currently running
                    if job["status"] == JobStatus.RUNNING:
                        continue
                    
                    # Check if it's time to run the job
                    next_run = job.get("next_run")
                    if isinstance(next_run, str):
                        next_run = datetime.fromisoformat(next_run)
                    
                    if not next_run or current_time >= next_run:
                        # Run the job in a separate thread
                        job_thread = threading.Thread(target=self._run_job, args=(job_id,))
                        job_thread.daemon = True
                        job_thread.start()
                
                # Sleep for a bit
                time.sleep(10)
        
        self.scheduler_thread = threading.Thread(target=scheduler_loop)
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()
        
        return True
    
    def stop(self):
        """Stop the scheduler."""
        if not self.is_running:
            logger.warning("Scheduler is not running")
            return False
        
        self.is_running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
            self.scheduler_thread = None
        
        logger.info("Scheduler stopped")
        return True
    
    def load_jobs_from_config(self):
        """Load jobs from the configuration file."""
        if not self.config:
            logger.error("No configuration loaded")
            return False
        
        try:
            # Load scheduled jobs if present
            scheduled_jobs = self.config.get("scheduler", {}).get("jobs", [])
            
            if not scheduled_jobs:
                logger.warning("No jobs found in configuration")
                return False
                
            job_count = 0
            for job in scheduled_jobs:
                # Verify job has required fields
                if not job.get("id") or not job.get("url"):
                    logger.warning(f"Skipping job with missing id or url: {job}")
                    continue
                    
                # Add the job
                self.add_job(
                    job["id"],
                    job["url"],
                    job.get("interval_minutes", 60),
                    job.get("max_urls", 50)
                )
                job_count += 1
            
            logger.info(f"Loaded {job_count} jobs from configuration")
            return job_count > 0
        
        except Exception as e:
            logger.error(f"Error loading jobs from configuration: {str(e)}")
            return False

def main():
    """Entry point for command line execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run the Vigil scheduler")
    parser.add_argument("--start", action="store_true", help="Start the scheduler")
    parser.add_argument("--stop", action="store_true", help="Stop the scheduler")
    parser.add_argument("--add-job", help="Add a new job (URL)")
    parser.add_argument("--remove-job", help="Remove a job by ID")
    parser.add_argument("--list-jobs", action="store_true", help="List all jobs")
    parser.add_argument("--job-history", action="store_true", help="Show job execution history")
    parser.add_argument("--interval", type=int, default=60, help="Job interval in minutes")
    parser.add_argument("--max-urls", type=int, default=50, help="Maximum URLs per job")
    parser.add_argument("--model-path", help="Path to the ML model")
    parser.add_argument("--vectorizer-path", help="Path to the vectorizer")
    
    args = parser.parse_args()
    
    scheduler = VigilScheduler(model_path=args.model_path, vectorizer_path=args.vectorizer_path)
    
    if args.add_job:
        job_id = f"job-{int(time.time())}"
        if scheduler.add_job(job_id, args.add_job, args.interval, args.max_urls):
            print(f"Added job {job_id}")
        else:
            print("Failed to add job")
    
    elif args.remove_job:
        if scheduler.remove_job(args.remove_job):
            print(f"Removed job {args.remove_job}")
        else:
            print(f"Failed to remove job {args.remove_job}")
    
    elif args.list_jobs:
        jobs = scheduler.list_jobs()
        if not jobs:
            print("No jobs scheduled")
        else:
            print(f"Scheduled Jobs ({len(jobs)}):")
            for job in jobs:
                print(f"  {job['id']}: {job['url']} (every {job['interval_minutes']} minutes)")
    
    elif args.job_history:
        history = scheduler.get_job_history()
        if not history:
            print("No job history available")
        else:
            print(f"Job History ({len(history)}):")
            for entry in history:
                status = entry["status"]
                print(f"  {entry['job_id']}: {status} at {entry['end_time']}")
    
    elif args.start:
        if scheduler.start():
            print("Scheduler started")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                scheduler.stop()
                print("Scheduler stopped")
        else:
            print("Failed to start scheduler")
    
    elif args.stop:
        if scheduler.stop():
            print("Scheduler stopped")
        else:
            print("Failed to stop scheduler")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
