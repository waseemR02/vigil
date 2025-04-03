from fastapi import APIRouter, Query
from typing import List, Optional
from pydantic import BaseModel
from sqlalchemy import or_
from vigil.database.connection import get_session
from vigil.database.models import Incident, Source

router = APIRouter(tags=["search"])

# Pydantic models
class SearchResult(BaseModel):
    id: int
    title: str
    description: Optional[str] = None
    source_name: Optional[str] = None
    
    class Config:
        orm_mode = True

@router.get("/search", response_model=List[SearchResult])
async def search_incidents(q: str = Query(..., min_length=2)):
    """Search for incidents by keyword."""
    if not q:
        return []
    
    session = get_session()
    try:
        # Simple text search across multiple fields
        search_term = f"%{q}%"
        results = session.query(
            Incident.id,
            Incident.title,
            Incident.description,
            Source.name.label('source_name')
        ).outerjoin(
            Source, Incident.source_id == Source.id
        ).filter(
            or_(
                Incident.title.ilike(search_term),
                Incident.description.ilike(search_term),
                Source.name.ilike(search_term)
            )
        ).limit(50).all()
        
        # Convert results to list of dictionaries
        return [
            {
                "id": r.id,
                "title": r.title,
                "description": r.description,
                "source_name": r.source_name
            }
            for r in results
        ]
    finally:
        session.close()
