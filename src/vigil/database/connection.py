import os
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.ext.declarative import declarative_base

# Configure logger
logger = logging.getLogger('vigil.database.connection')

# Get database configuration from environment variables with defaults
DB_TYPE = os.environ.get("VIGIL_DB_TYPE", "sqlite")
DB_PATH = os.environ.get("VIGIL_DB_PATH", "/home/waseem/workspace/vigil/database-workflow/vigil.db")
DB_URL = os.environ.get("VIGIL_DB_URL", f"{DB_TYPE}:///{DB_PATH}")

# Create the directory for the database if it doesn't exist
if DB_TYPE == "sqlite" and DB_PATH:
    db_dir = os.path.dirname(DB_PATH)
    if db_dir and not os.path.exists(db_dir):
        try:
            os.makedirs(db_dir, exist_ok=True)
            logger.info(f"Created database directory: {db_dir}")
        except Exception as e:
            logger.error(f"Failed to create database directory {db_dir}: {str(e)}")

# Create SQLAlchemy engine
engine = create_engine(
    DB_URL, 
    connect_args={"check_same_thread": False} if DB_TYPE == "sqlite" else {},
    echo=os.environ.get("VIGIL_DB_ECHO", "false").lower() == "true"
)

# Create session factory
session_factory = sessionmaker(bind=engine)
Session = scoped_session(session_factory)

# Base class for all models
Base = declarative_base()

def get_engine():
    """Return the database engine."""
    return engine

def get_session():
    """Create and return a new session."""
    return Session()

def init_db():
    """Initialize the database schema."""
    # Import models to ensure they're registered with Base
    from .models import Incident, Source, Tag
    
    # Ensure data directory exists for SQLite
    if DB_TYPE == "sqlite":
        db_dir = os.path.dirname(DB_PATH)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
            logger.info(f"Ensured database directory exists: {db_dir}")
    
    # Create all tables
    Base.metadata.create_all(engine)
    logger.info("Database schema initialized")
    
    return True
