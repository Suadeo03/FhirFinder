# backend/api/v1/endpoints/search.py (SIMPLIFIED)
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional, Dict
from config.database import get_db
from services.search.v2_mapping_search import V2FHIRSearchService
from models.database.query_performance_model import QueryPerformance
from services.ultility.query_tracker import QueryTracker
from services.model_training.training_service_v2fhir import FeedbackTrainingV2FHIR
from datetime import datetime

feedback_service = FeedbackTrainingV2FHIR()
search_service = V2FHIRSearchService()
router = APIRouter()

class SearchRequest(BaseModel):
    query: str
    limit: Optional[int] = 5


class SearchResponse(BaseModel):
    query: str
    results: List[dict]
    total_results: int
    search_stats: Optional[dict] = None

class FeedbackV2Request(BaseModel):
    query: str
    mapping_id: str
    feedback_type: str  # 'positive', 'negative', 'neutral'
    session_id: str
    user_id: Optional[str] = None
    original_score: float
    context_info: Optional[Dict] = None

@router.post("/search/mapping", response_model=SearchResponse)
async def search_profiles(request: SearchRequest, db: Session = Depends(get_db)):

    try:
        results = search_service.semantic_search(request.query, request.limit, db)
        stats = search_service.get_search_stats(db)
        
        return SearchResponse(
            query=request.query,
            results=results,
            total_results=len(results),
            search_stats=stats
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@router.get("/search/mapping/stats")
async def get_search_stats(db: Session = Depends(get_db)):
    """Get search statistics"""
    try:

        stats = search_service.get_search_stats(db)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")
    

@router.post("/mappings/feedback")
async def record_v2_fhir_feedback(
    request: FeedbackV2Request,
    db: Session = Depends(get_db)
):
    """Record user feedback for V2-FHIR search results"""
    try:
        if request.feedback_type not in ['positive', 'negative', 'neutral']:
            raise HTTPException(
                status_code=400, 
                detail="Invalid feedback type. Must be 'positive', 'negative', or 'neutral'"
            )
        
        feedback_service.record_user_feedback(
        query=request.query,
        mapping_id=request.mapping_id,
        feedback_type=request.feedback_type,
        user_id=request.user_id or "anonymous",
        session_id=request.session_id,
        original_score=request.original_score,
        db=db,
        context_info=request.context_info
        )
        

        recent_query = db.query(QueryPerformance).filter(
            QueryPerformance.query_text == request.query,
            QueryPerformance.top_result_id == request.mapping_id,
            QueryPerformance.dataset_type == 'v2fhir'
        ).order_by(QueryPerformance.query_date.desc()).first()
        
        if recent_query:
            search_service.query_tracker.update_user_interaction(
                query_id=recent_query.id,
                interaction_type='feedback',
                feedback=request.feedback_type,
                db=db
            )
            
            return {
                "success": True,
                "message": f"Feedback '{request.feedback_type}' recorded for query '{request.query}'"
            }
        else:
            return {
                "success": False,
                "message": "No matching query found for feedback"
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to record feedback: {str(e)}")
    
@router.get("/search/v2-fhir-mappings/analytics")
async def get_search_analytics(days: int = 30, db: Session = Depends(get_db)):
    """Get V2-FHIR search analytics for the specified period"""
    try:
        search_service = V2FHIRSearchService()
        analytics = search_service.query_tracker.get_query_analytics(db, days)
        
        return {
            "analytics": analytics,
            "generated_at": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get analytics: {str(e)}")
