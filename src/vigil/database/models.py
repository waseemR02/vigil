from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Table
from sqlalchemy.orm import relationship
from .connection import Base

# Association table for many-to-many relationship between Incident and Tag
incident_tags = Table(
    'incident_tags',
    Base.metadata,
    Column('incident_id', Integer, ForeignKey('incidents.id'), primary_key=True),
    Column('tag_id', Integer, ForeignKey('tags.id'), primary_key=True)
)

class Incident(Base):
    """Model representing a cybersecurity incident."""
    __tablename__ = 'incidents'
    
    id = Column(Integer, primary_key=True)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    source_id = Column(Integer, ForeignKey('sources.id'), nullable=True)
    source_reference = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    source = relationship("Source", back_populates="incidents")
    tags = relationship("Tag", secondary=incident_tags, back_populates="incidents")
    
    def __repr__(self):
        return f"<Incident(id={self.id}, title='{self.title}')>"

class Source(Base):
    """Model representing a source of cybersecurity incidents."""
    __tablename__ = 'sources'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False, unique=True)
    url = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    incidents = relationship("Incident", back_populates="source")
    
    def __repr__(self):
        return f"<Source(id={self.id}, name='{self.name}')>"

class Tag(Base):
    """Model representing a tag for categorizing incidents."""
    __tablename__ = 'tags'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False, unique=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    incidents = relationship("Incident", secondary=incident_tags, back_populates="tags")
    
    def __repr__(self):
        return f"<Tag(id={self.id}, name='{self.name}')>"
