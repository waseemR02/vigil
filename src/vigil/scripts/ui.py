import os
import argparse
import subprocess
import sys
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("vigil.runner")

def init_database():
    """Initialize the database if it doesn't exist."""
    try:
        from vigil.database.connection import init_db, DB_PATH
        
        # Check if database file exists
        db_path = Path(DB_PATH)
        if not db_path.exists() or db_path.stat().st_size == 0:
            logger.info(f"Database file not found or empty at {DB_PATH}. Initializing now...")
            if init_db():
                logger.info("Database initialized successfully.")
                
                # Add some sample data for testing
                add_sample_data()
            else:
                logger.error("Failed to initialize database.")
                return False
        else:
            logger.info(f"Database already exists at {DB_PATH}")
        
        return True
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        return False

def add_sample_data():
    """Add some sample data to the database."""
    try:
        from vigil.database import queries
        
        # Check if there's already data in the database
        session = queries.get_session()
        try:
            incident_count = session.query(queries.Incident).count()
            if incident_count > 0:
                logger.info(f"Database already contains {incident_count} incidents. Skipping sample data.")
                return True
        finally:
            session.close()
        
        # Add sample sources
        sources = [
            {"name": "Security Blog", "url": "https://securityblog.example.com"},
            {"name": "Threat Intelligence", "url": "https://threatintel.example.com"},
            {"name": "Internal Monitoring", "url": None}
        ]
        
        source_ids = []
        for source in sources:
            source_id = queries.create_source(source["name"], source["url"])
            source_ids.append(source_id)
            logger.info(f"Added source: {source['name']} (ID: {source_id})")
        
        # Add sample tags
        tags = ["malware", "phishing", "ransomware", "zero-day", "data breach", "social engineering"]
        for tag in tags:
            tag_id = queries.create_tag(tag)
            logger.info(f"Added tag: {tag} (ID: {tag_id})")
        
        # Add sample incidents
        incidents = [
            {
                "title": "Phishing Campaign Targeting Finance Department",
                "description": "Multiple employees in the finance department received sophisticated phishing emails attempting to steal credentials.",
                "source_id": source_ids[0],
                "source_reference": "INCIDENT-2023-001",
                "tags": ["phishing", "social engineering"]
            },
            {
                "title": "Ransomware Detected on Marketing Workstation",
                "description": "Ransomware was detected and isolated on a workstation in the marketing department before it could spread.",
                "source_id": source_ids[2],
                "source_reference": "INCIDENT-2023-002",
                "tags": ["ransomware", "malware"]
            },
            {
                "title": "Zero-day Vulnerability in Web Application",
                "description": "A previously unknown vulnerability was discovered in our web application that could allow unauthorized access.",
                "source_id": source_ids[1],
                "source_reference": "CVE-2023-12345",
                "tags": ["zero-day", "data breach"]
            }
        ]
        
        for incident in incidents:
            incident_id = queries.create_incident(
                title=incident["title"],
                description=incident["description"],
                source_id=incident["source_id"],
                source_reference=incident["source_reference"],
                tags=incident["tags"]
            )
            logger.info(f"Added incident: {incident['title']} (ID: {incident_id})")
        
        logger.info("Sample data added successfully")
        return True
    except Exception as e:
        logger.error(f"Error adding sample data: {str(e)}")
        return False

def run_api():
    """Run the Vigil API server."""
    print("📡 Starting Vigil API server...")
    subprocess.run([
        "uvicorn", 
        "vigil.api.app:app", 
        "--host", "0.0.0.0", 
        "--port", "8000", 
        "--reload"
    ])

def run_dashboard():
    """Run the Vigil Dashboard application."""
    print("🖥️ Starting Vigil Dashboard...")
    subprocess.run([
        "uvicorn", 
        "vigil.dashboard.app:app", 
        "--host", "0.0.0.0", 
        "--port", "8001", 
        "--reload"
    ])

def run_both():
    """Run both API and Dashboard in separate processes."""
    print("🚀 Starting Vigil system (API + Dashboard)...")
    
    # Start API server in a new process
    api_process = subprocess.Popen([
        "uvicorn", 
        "vigil.api.app:app", 
        "--host", "0.0.0.0", 
        "--port", "8000", 
        "--reload"
    ])
    
    # Give the API a moment to start
    print("⏳ Waiting for API to start...")
    time.sleep(3)
    
    # Start Dashboard in a new process
    dashboard_process = subprocess.Popen([
        "uvicorn", 
        "vigil.dashboard.app:app", 
        "--host", "0.0.0.0", 
        "--port", "8001", 
        "--reload"
    ])
    
    print("\n✅ Vigil system is running!")
    print("📡 API: http://localhost:8000")
    print("📡 API Documentation: http://localhost:8000/docs")
    print("🖥️ Dashboard: http://localhost:8001")
    print("\nPress Ctrl+C to stop both services...\n")
    
    try:
        # Keep the script running to allow the subprocesses to run
        api_process.wait()
    except KeyboardInterrupt:
        print("\n🛑 Stopping Vigil system...")
        api_process.terminate()
        dashboard_process.terminate()
        api_process.wait()
        dashboard_process.wait()
        print("✅ Vigil system stopped.")

def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Run Vigil Cybersecurity Dashboard System")
    parser.add_argument("--component", choices=["api", "dashboard", "both"], default="both",
                        help="Which component to run (default: both)")
    parser.add_argument("--init-db", action="store_true", 
                        help="Initialize the database (if it doesn't exist) with sample data")
    
    args = parser.parse_args()
    
    # Ensure the package is in PYTHONPATH
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    # Initialize database if requested or if running both components
    if args.init_db or args.component in ["both", "api"]:
        if not init_database():
            print("⚠️ Database initialization issues may affect system operation.")
    
    # Run the specified component(s)
    if args.component == "api":
        run_api()
    elif args.component == "dashboard":
        run_dashboard()
    else:
        run_both()

if __name__ == "__main__":
    main()