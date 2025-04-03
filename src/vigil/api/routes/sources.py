from fastapi import APIRouter
from typing import List, Optional
from pydantic import BaseModel
from vigil.database import queries

router = APIRouter(tags=["metadata"])

# Pydantic models
class SourceResponse(BaseModel):
    id: int
    name: str
    url: Optional[str] = None
    
    class Config:
        orm_mode = True

class TagResponse(BaseModel):
    id: int
    name: str
    
    class Config:
        orm_mode = True

@router.get("/sources", response_model=List[SourceResponse])
async def list_sources():
    """List all incident sources."""
    return queries.list_sources()

@router.get("/tags", response_model=List[TagResponse])
async def list_tags():
    """List all incident tags."""
    return queries.list_tags()
