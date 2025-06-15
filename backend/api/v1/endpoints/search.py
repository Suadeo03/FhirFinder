from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional
from services.search_service import SimpleSearchService

router = APIRouter()
search_service = SimpleSearchService()

class SearchRequest(BaseModel):
    query: str
    limit: Optional[int] = 5

class SearchResponse(BaseModel):
    query: str
    results: List[dict]
    total_results: int

@router.post("/search", response_model=SearchResponse)
async def search_profiles(request: SearchRequest):
    results = search_service.search(request.query, request.limit)
    
    return SearchResponse(
        query=request.query,
        results=results,
        total_results=len(results)
    )