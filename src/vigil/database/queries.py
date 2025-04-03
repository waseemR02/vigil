from sqlalchemy import and_
from sqlalchemy.orm import joinedload
from datetime import datetime
from .connection import get_session
from .models import Incident, Source, Tag

# ---- Incident CRUD operations ----

def create_incident(title, description=None, source_id=None, source_reference=None, tags=None):
    """Create a new incident record."""
    session = get_session()
    try:
        incident = Incident(
            title=title,
            description=description,
            source_id=source_id,
            source_reference=source_reference
        )
        
        # Add tags if provided
        if tags:
            for tag_name in tags:
                tag = session.query(Tag).filter(Tag.name == tag_name).first()
                if not tag:
                    tag = Tag(name=tag_name)
                incident.tags.append(tag)
        
        session.add(incident)
        session.commit()
        return incident.id
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

def get_incident(incident_id):
    """Retrieve an incident by ID."""
    session = get_session()
    try:
        # Use joinedload to eagerly load relationships
        return session.query(Incident).options(
            joinedload(Incident.source),
            joinedload(Incident.tags)
        ).filter(Incident.id == incident_id).first()
    finally:
        session.close()

def update_incident(incident_id, title=None, description=None, source_id=None, 
                   source_reference=None, tags=None):
    """Update an existing incident."""
    session = get_session()
    try:
        incident = session.query(Incident).filter(Incident.id == incident_id).first()
        if not incident:
            return False
        
        if title:
            incident.title = title
        if description is not None:
            incident.description = description
        if source_id is not None:
            incident.source_id = source_id
        if source_reference is not None:
            incident.source_reference = source_reference
        
        # Update tags if provided
        if tags is not None:
            # Clear existing tags
            incident.tags = []
            
            # Add new tags
            for tag_name in tags:
                tag = session.query(Tag).filter(Tag.name == tag_name).first()
                if not tag:
                    tag = Tag(name=tag_name)
                incident.tags.append(tag)
        
        incident.updated_at = datetime.utcnow()
        session.commit()
        return True
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

def delete_incident(incident_id):
    """Delete an incident by ID."""
    session = get_session()
    try:
        incident = session.query(Incident).filter(Incident.id == incident_id).first()
        if not incident:
            return False
        session.delete(incident)
        session.commit()
        return True
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

def list_incidents(source_id=None, tag_name=None, page=1, per_page=20):
    """List incidents with optional filtering and pagination."""
    session = get_session()
    try:
        # Start with a query that eagerly loads relationships
        query = session.query(Incident).options(
            joinedload(Incident.source),
            joinedload(Incident.tags)
        )
        
        # Apply filters
        filters = []
        if source_id:
            filters.append(Incident.source_id == source_id)
        
        if tag_name:
            query = query.join(Incident.tags).filter(Tag.name == tag_name)
        
        if filters:
            query = query.filter(and_(*filters))
        
        # Apply pagination
        total = query.count()
        incidents = query.order_by(Incident.created_at.desc()) \
                        .limit(per_page) \
                        .offset((page - 1) * per_page) \
                        .all()
        
        return {
            'items': incidents,
            'total': total,
            'page': page,
            'per_page': per_page,
            'pages': (total + per_page - 1) // per_page
        }
    finally:
        session.close()

# ---- Source CRUD operations ----

def create_source(name, url=None):
    """Create a new source."""
    session = get_session()
    try:
        source = Source(name=name, url=url)
        session.add(source)
        session.commit()
        return source.id
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

def get_source(source_id):
    """Retrieve a source by ID."""
    session = get_session()
    try:
        return session.query(Source).filter(Source.id == source_id).first()
    finally:
        session.close()

def list_sources():
    """List all sources."""
    session = get_session()
    try:
        return session.query(Source).all()
    finally:
        session.close()

# ---- Tag operations ----

def create_tag(name):
    """Create a new tag."""
    session = get_session()
    try:
        tag = Tag(name=name)
        session.add(tag)
        session.commit()
        return tag.id
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

def get_tag(tag_id):
    """Retrieve a tag by ID."""
    session = get_session()
    try:
        return session.query(Tag).filter(Tag.id == tag_id).first()
    finally:
        session.close()

def list_tags():
    """List all tags."""
    session = get_session()
    try:
        return session.query(Tag).all()
    finally:
        session.close()
