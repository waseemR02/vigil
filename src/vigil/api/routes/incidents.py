from fastapi import APIRouter, HTTPException, Query, Path
from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from vigil.database import queries

router = APIRouter(prefix="/incidents", tags=["incidents"])

# Pydantic models for response serialization
class TagResponse(BaseModel):
    id: int
    name: str

class SourceResponse(BaseModel):
    id: int
    name: str
    url: Optional[str] = None

class IncidentBase(BaseModel):
    id: int
    title: str
    description: Optional[str] = None
    source_reference: Optional[str] = None
    created_at: datetime
    updated_at: datetime

class IncidentListItem(IncidentBase):
    source: Optional[SourceResponse] = None
    tags: List[TagResponse] = Field(default_factory=list)
    
    class Config:
        orm_mode = True

class PaginatedIncidents(BaseModel):
    items: List[IncidentListItem]
    total: int
    page: int
    per_page: int
    pages: int

@router.get("", response_model=PaginatedIncidents)
async def list_incidents(
    source_id: Optional[int] = None,
    tag_name: Optional[str] = None,
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100)
):
    """List all incidents with optional filtering and pagination."""
    result = queries.list_incidents(
        source_id=source_id,
        tag_name=tag_name,
        page=page,
        per_page=per_page
    )
    return result

@router.get("/recent", response_model=List[IncidentListItem])
async def get_recent_incidents(limit: int = Query(10, ge=1, le=50)):
    """Get the most recent incidents for the dashboard."""
    result = queries.list_incidents(page=1, per_page=limit)
    return result["items"]

@router.get("/{incident_id}", response_model=IncidentListItem)
async def get_incident(incident_id: int = Path(..., ge=1)):
    """Get a specific incident by ID."""
    incident = queries.get_incident(incident_id)
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    return incident
