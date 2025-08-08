# backend/api/v1/endpoints/search.py (UPDATED FOR FORMS)
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from config.database import get_db
from services.form_search_service import FormLookupService

router = APIRouter()

class SearchRequest(BaseModel):
    query: str
    limit: Optional[int] = 10
    search_type: Optional[str] = "hybrid"  # semantic, traditional, hybrid
    semantic_weight: Optional[float] = 0.7
    filters: Optional[Dict[str, Any]] = None

class FormResult(BaseModel):
    id: str
    domain: str
    screening_tool: str
    question: str
    loinc_panel_code: Optional[str] = None
    loinc_panel_name: Optional[str] = None
    loinc_question_code: Optional[str] = None
    loinc_question_name_long: Optional[str] = None
    answer_concept: Optional[str] = None
    loinc_answer: Optional[str] = None
    loinc_concept: Optional[str] = None
    snomed_code_ct: Optional[str] = None
    similarity_score: float
    formset_id: str
    is_active: bool

class SearchResponse(BaseModel):
    query: str
    results: List[Dict[str, Any]]  # Using Dict to handle dynamic form data
    total_results: int
    search_type: str
    search_stats: Optional[Dict[str, Any]] = None

class FilterSearchRequest(BaseModel):
    formset_id: Optional[str] = None
    domain: Optional[str] = None
    screening_tool: Optional[str] = None
    loinc_panel_code: Optional[str] = None
    snomed_code_ct: Optional[str] = None
    limit: Optional[int] = 50

@router.post("/forms/search", response_model=SearchResponse)
async def search_forms(request: SearchRequest, db: Session = Depends(get_db)):
    """Search forms using semantic similarity, traditional text search, or hybrid approach"""
    try:
        search_service = FormLookupService()
        
 
        results = search_service.semantic_search(
                query=request.query, 
                top_k=request.limit, 
                db=db, 
                filters=request.filters
            )
      
        
        stats = search_service.get_search_stats(db)
        
        return SearchResponse(
            query=request.query,
            results=results,
            total_results=len(results),
            search_type=request.search_type or "semantic",
            search_stats=stats
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@router.get("/forms/search/simple")
async def simple_form_search(
    query: str = Query(..., description="Search query"),
    limit: int = Query(10, description="Maximum number of results"),
    search_type: str = Query("hybrid", description="Search type: semantic, traditional, or hybrid"),
    domain: Optional[str] = Query(None, description="Filter by domain"),
    screening_tool: Optional[str] = Query(None, description="Filter by screening tool"),
    formset_id: Optional[str] = Query(None, description="Filter by formset ID"),
    db: Session = Depends(get_db)
):

    try:
        search_service = FormLookupService()
        
        # Build filters from query parameters
        filters = {}
        if domain:
            filters['domain'] = domain
        if screening_tool:
            filters['screening_tool'] = screening_tool
        if formset_id:
            filters['formset_id'] = formset_id
        
        # Perform search based on type
        if search_type == "semantic":
            results = search_service.semantic_search(
                query=query, 
                top_k=limit, 
                db=db, 
                filters=filters if filters else None
            )
        elif search_type == "traditional":
            results = search_service.traditional_search(
                query=query, 
                db=db, 
                top_k=limit, 
                filters=filters if filters else None
            )
        else:  # hybrid
            results = search_service.hybrid_search(
                query=query, 
                top_k=limit, 
                db=db, 
                semantic_weight=0.7,
                filters=filters if filters else None
            )
        
        return {
            "query": query,
            "results": results,
            "total_results": len(results),
            "search_type": search_type,
            "filters_applied": filters
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@router.post("/forms/filter", response_model=SearchResponse)
async def filter_forms(request: FilterSearchRequest, db: Session = Depends(get_db)):
    """Search forms using filters only (no text query)"""
    try:
        search_service = FormLookupService()
        
        # Build filters dictionary
        filters = {}
        if request.formset_id:
            filters['formset_id'] = request.formset_id
        if request.domain:
            filters['domain'] = request.domain
        if request.screening_tool:
            filters['screening_tool'] = request.screening_tool
        if request.loinc_panel_code:
            filters['loinc_panel_code'] = request.loinc_panel_code
        if request.snomed_code_ct:
            filters['snomed_code_ct'] = request.snomed_code_ct
        
        results = search_service.search_by_filters_only(
            db=db, 
            filters=filters, 
            top_k=request.limit or 50
        )
        
        stats = search_service.get_search_stats(db)
        
        return SearchResponse(
            query="filter_only",
            results=results,
            total_results=len(results),
            search_type="filter_only",
            search_stats=stats
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Filter search failed: {str(e)}")



@router.get("/forms/stats")
async def get_search_statistics(db: Session = Depends(get_db)):
    """Get comprehensive search and form statistics"""
    try:
        search_service = FormLookupService()
        stats = search_service.get_search_stats(db)
        
        return stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")
