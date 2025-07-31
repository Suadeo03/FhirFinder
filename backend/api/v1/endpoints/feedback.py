# feedback_endpoints.py - API endpoints for handling user feedback

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional, Dict, List
from datetime import datetime
from config.database import get_db
from services.training_service import TrainingFeedback
from services.search_service import SearchService

router = APIRouter()
search_service = SearchService()

feedback_service = TrainingFeedback()

# Pydantic models for API
class FeedbackRequest(BaseModel):
    query: str
    profile_id: str
    feedback_type: str  # 'positive', 'negative', 'neutral'
    session_id: str
    user_id: Optional[str] = None
    original_score: float
    context_info: Optional[Dict] = None

class FeedbackResponse(BaseModel):
    success: bool
    message: str
    feedback_id: Optional[int] = None

class SearchWithFeedbackRequest(BaseModel):
    query: str
    top_k: int = 10
    search_type: str = "hybrid"  # 'semantic', 'traditional', 'hybrid'
    semantic_weight: float = 0.7
    filters: Optional[Dict] = None
    apply_feedback: bool = True
    session_id: str
    user_id: Optional[str] = None

class SearchResult(BaseModel):
    id: int
    name: str
    description: str
    resource_type: str
    category: str
    similarity_score: float
    original_score: Optional[float] = None
    adjustment_factor: Optional[float] = None
    feedback_boost: Optional[str] = None
    keywords: List[str]
    # Add other fields as needed

class SearchResponse(BaseModel):
    results: List[SearchResult]
    total_count: int
    query: str
    search_type: str
    applied_feedback: bool
    session_id: str

# Initialize search service


@router.post("/search/with-feedback", response_model=SearchResponse)
async def search_with_feedback(
    request: SearchWithFeedbackRequest,
    db: Session = Depends(get_db)
):
    """
    Perform search with feedback integration
    """
    try:
        # Perform the appropriate search
        if request.search_type == "semantic":
            results = search_service.semantic_search(
                query=request.query,
                top_k=request.top_k,
                db=db,
                filters=request.filters,
                apply_feedback=request.apply_feedback
            )
        elif request.search_type == "traditional":
            results = search_service.traditional_search(
                query=request.query,
                db=db,
                top_k=request.top_k,
                filters=request.filters,
                apply_feedback=request.apply_feedback
            )
        else:  # hybrid
            results = search_service.hybrid_search(
                query=request.query,
                top_k=request.top_k,
                db=db,
                semantic_weight=request.semantic_weight,
                filters=request.filters,
                apply_feedback=request.apply_feedback
            )
        
        # Convert to response format
        search_results = []
        for result in results:
            search_results.append(SearchResult(
                id=result['id'],
                name=result['name'],
                description=result['description'],
                resource_type=result['resource_type'],
                category=result['category'],
                similarity_score=result['similarity_score'],
                original_score=result.get('original_score'),
                adjustment_factor=result.get('adjustment_factor'),
                feedback_boost=result.get('feedback_boost'),
                keywords=result['keywords']
            ))
        
        return SearchResponse(
            results=search_results,
            total_count=len(search_results),
            query=request.query,
            search_type=request.search_type,
            applied_feedback=request.apply_feedback,
            session_id=request.session_id
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@router.post("/feedback/record", response_model=FeedbackResponse)
async def record_feedback(
    request: FeedbackRequest,
    db: Session = Depends(get_db)
):
    """
    Record user feedback for a search result
    """

    try:
        # Validate feedback type
        if request.feedback_type not in ['positive', 'negative', 'neutral']:
            raise HTTPException(
                status_code=400, 
                detail="Invalid feedback type. Must be 'positive', 'negative', or 'neutral'"
            )
        
        # Record the feedback
        feedback_service.record_user_feedback(
            query=request.query,
            profile_id=request.profile_id,
            feedback_type=request.feedback_type,
            user_id=request.user_id or "anonymous",
            session_id=request.session_id,
            original_score=request.original_score,
            db=db,
            context_info=request.context_info
        )
        
        return FeedbackResponse(
            success=True,
            message=f"Feedback recorded successfully for profile {request.profile_id}"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to record feedback: {str(e)}")

@router.get("/feedback/stats")
async def get_feedback_stats(
    days: int = 30,
    db: Session = Depends(get_db)
):
    """
    Get feedback statistics for monitoring search quality
    """
    try:
        stats = feedback_service.get_feedback_stats_simple(db, days)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get feedback stats: {str(e)}")

@router.post("/feedback/retrain")
async def trigger_retraining(
    batch_size: int = 100,
    db: Session = Depends(get_db)
):
    """
    Trigger retraining of embeddings based on accumulated feedback
    This should be protected and only accessible to admin users
    """
    try:
        search_service.retrain_embeddings(db, batch_size)
        return {"success": True, "message": "Retraining completed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")

@router.get("/search/quality-metrics")
async def get_search_quality_metrics(
    query: Optional[str] = None,
    profile_id: Optional[int] = None,
    limit: int = 20,
    db: Session = Depends(get_db)
):
    """
    Get search quality metrics for specific queries or profiles
    """
    try:
        # This would query your SearchQualityMetrics table
        # For now, return mock data
        return {
            "metrics": [],
            "summary": {
                "total_queries": 0,
                "avg_feedback_score": 0.0,
                "improvement_rate": 0.0
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get quality metrics: {str(e)}")

# Additional endpoint for bulk feedback (useful for A/B testing)
@router.post("/feedback/bulk")
async def record_bulk_feedback(
    feedbacks: List[FeedbackRequest],
    db: Session = Depends(get_db)
):
    """
    Record multiple feedback entries at once
    """
    try:
        results = []
        for feedback in feedbacks:
            try:
                search_service.record_user_feedback(
                    query=feedback.query,
                    profile_id=feedback.profile_id,
                    feedback_type=feedback.feedback_type,
                    user_id=feedback.user_id or "anonymous",
                    session_id=feedback.session_id,
                    original_score=feedback.original_score,
                    db=db,
                    context_info=feedback.context_info
                )
                results.append({"profile_id": feedback.profile_id, "success": True})
            except Exception as e:
                results.append({"profile_id": feedback.profile_id, "success": False, "error": str(e)})
        
        return {"results": results}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Bulk feedback recording failed: {str(e)}")

# Endpoint for getting search suggestions based on feedback
@router.get("/search/suggestions")
async def get_search_suggestions(
    query: str,
    limit: int = 5,
    db: Session = Depends(get_db)
):
    """
    Get search suggestions based on successful past searches
    """
    try:
        # This would analyze past positive feedback to suggest better queries
        # For now, return mock suggestions
        return {
            "suggestions": [
                f"{query} profile",
                f"{query} template",
                f"{query} specification"
            ],
            "based_on_feedback": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get suggestions: {str(e)}")

# Endpoint for A/B testing different search algorithms
@router.post("/search/ab-test")
async def ab_test_search(
    request: SearchWithFeedbackRequest,
    algorithm_variant: str = "standard",  # 'standard', 'experimental'
    db: Session = Depends(get_db)
):
    """
    Perform search with A/B testing variants
    """
    try:
        # Implement different search variants for testing
        if algorithm_variant == "experimental":
            # Use different weights or algorithms
            request.semantic_weight = 0.8  # More emphasis on semantic
        
        # Perform search
        if request.search_type == "hybrid":
            results = search_service.hybrid_search(
                query=request.query,
                top_k=request.top_k,
                db=db,
                semantic_weight=request.semantic_weight,
                filters=request.filters,
                apply_feedback=request.apply_feedback
            )
        elif request.search_type == "semantic":
            results = search_service.semantic_search(
                query=request.query,
                top_k=request.top_k,
                db=db,
                filters=request.filters,
                apply_feedback=request.apply_feedback
            )
        else:
            results = search_service.traditional_search(
                query=request.query,
                db=db,
                top_k=request.top_k,
                filters=request.filters,
                apply_feedback=request.apply_feedback
            )
        
        # Add A/B test metadata to results
        for result in results:
            result['ab_variant'] = algorithm_variant
        
        return {
            "results": results,
            "variant": algorithm_variant,
            "session_id": request.session_id,
            "query": request.query
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"A/B test search failed: {str(e)}")