# backend/api/v1/endpoints/search.py (UPDATED FOR DATABASE)
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional

from config.database import get_db
from services.database_search_service import DatabaseSearchService

router = APIRouter()

class SearchRequest(BaseModel):
    query: str
    limit: Optional[int] = 5

class SearchResponse(BaseModel):
    query: str
    results: List[dict]
    total_results: int
    search_stats: Optional[dict] = None

@router.post("/search", response_model=SearchResponse)
async def search_profiles(request: SearchRequest, db: Session = Depends(get_db)):
    """Search FHIR profiles using semantic similarity and keyword matching"""
    try:
        search_service = DatabaseSearchService()
        results = search_service.search(request.query, request.limit, db)
        stats = search_service.get_search_stats(db)
        
        return SearchResponse(
            query=request.query,
            results=results,
            total_results=len(results),
            search_stats=stats
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@router.get("/search/stats")
async def get_search_stats(db: Session = Depends(get_db)):
    """Get statistics about the searchable profile database"""
    try:
        search_service = DatabaseSearchService()
        stats = search_service.get_search_stats(db)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")