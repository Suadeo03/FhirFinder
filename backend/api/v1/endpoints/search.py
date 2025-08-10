# backend/api/v1/endpoints/search.py (SIMPLIFIED)
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional

from config.database import get_db
from services.search.search_service import SearchService
from services.ultility.conversational_service import FHIRConversationalService

router = APIRouter()

class SearchRequest(BaseModel):
    query: str
    limit: Optional[int] = 5
    include_summary: Optional[bool] = True

class SearchResponse(BaseModel):
    query: str
    results: List[dict]
    total_results: int
    summary: Optional[str] = None  # Simple summary
    search_stats: Optional[dict] = None

@router.post("/search", response_model=SearchResponse)
async def search_profiles(request: SearchRequest, db: Session = Depends(get_db)):
    """Search FHIR profiles with optional AI summary"""
    try:
        # Your existing search
        search_service = SearchService()
        results = search_service.semantic_search(request.query, request.limit, db)
        stats = search_service.get_search_stats(db)
        
        # Generate simple summary
        summary = None
        if request.include_summary:
            try:
                summarizer = FHIRConversationalService()
                summary = summarizer.summarize_results(request.query, results)
            except Exception as e:
                print(f"Summary generation failed: {e}")
                # Simple fallback
                if results:
                    summary = f"Found {len(results)} FHIR profiles for '{request.query}'."

        return SearchResponse(
            query=request.query,
            results=results,
            total_results=len(results),
            summary=summary,
            search_stats=stats
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@router.get("/search/stats")
async def get_search_stats(db: Session = Depends(get_db)):
    """Get search statistics"""
    try:
        search_service = SearchService()
        stats = search_service.get_search_stats(db)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")