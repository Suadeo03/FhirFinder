# feedback_endpoints.py - API endpoints for handling user feedback

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional, Dict, List
from datetime import datetime
from config.database import get_db
from services.model_training.training_service import FeedbackTraining
from models.database.feedback_models import UserFeedback, SearchQualityMetrics
from services.search.search_service import SearchService
from services.model_training.training_service_forms import FeedbackTrainingForm

router = APIRouter()
search_service = SearchService()
feedback_service = FeedbackTraining()
form_feedback_service = FeedbackTrainingForm()

# Pydantic models for API
class FeedbackRequest(BaseModel):
    query: str
    profile_id: str
    feedback_type: str  # 'positive', 'negative', 'neutral'
    session_id: str
    user_id: Optional[str] = None
    original_score: float
    context_info: Optional[Dict] = None

class FeedbackFormRequest(BaseModel):
    query: str
    form_id: str
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
    search_type: str = "semantic"  # 'semantic', 'traditional', 'hybrid'
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



@router.post("/feedback/record", response_model=FeedbackResponse)
async def record_feedback(
    request: FeedbackRequest,
    db: Session = Depends(get_db)
):
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
    
@router.post("/forms/feedback/record", response_model=FeedbackResponse)
async def record_feedback(
    request: FeedbackFormRequest,
    db: Session = Depends(get_db)
):
    try:
        # Validate feedback type
        if request.feedback_type not in ['positive', 'negative', 'neutral']:
            raise HTTPException(
                status_code=400, 
                detail="Invalid feedback type. Must be 'positive', 'negative', or 'neutral'"
            )
        
        # Record the feedback
        form_feedback_service.record_user_feedback(
            query=request.query,
            form_id=request.form_id,
            feedback_type=request.feedback_type,
            user_id=request.user_id or "anonymous",
            session_id=request.session_id,
            original_score=request.original_score,
            db=db,
            context_info=request.context_info
        )

        return FeedbackResponse(
            success=True,
            message=f"Feedback recorded successfully for profile {request.form_id}",
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


@router.get("/debug/profile/{profile_id}")
async def debug_profile_embedding(
    profile_id: str,
    db: Session = Depends(get_db)
):

    try:
        from models.database.models import Profile
        
        debug_info = {
            "profile_id": profile_id,
            "database_status": {},
            "chroma_status": {},
            "search_service_status": {},
            "feedback_service_status": {}
        }

        profile = db.query(Profile).filter(Profile.id == profile_id).first()
        if profile:
            debug_info["database_status"] = {
                "found": True,
                "name": profile.name,
                "is_active": profile.is_active,
                "dataset_id": profile.dataset_id
            }
        else:
            debug_info["database_status"] = {"found": False}
        
        try:
            search_results = search_service.collection.get(
                ids=[profile_id],
                include=['embeddings', 'metadatas']
            )
            
            debug_info["search_service_status"] = {
                "collection_available": bool(search_service.collection),
                "profile_found": bool(search_results.get('ids')),
                "has_embeddings": bool(search_results.get('embeddings')),
                "embedding_length": len(search_results['embeddings'][0]) if search_results.get('embeddings') and search_results['embeddings'][0] else 0
            }
        except Exception as e:
            debug_info["search_service_status"] = {
                "error": str(e),
                "collection_available": bool(search_service.collection)
            }

        try:
            feedback_results = feedback_service.collection.get(
                ids=[profile_id],
                include=['embeddings', 'metadatas']
            )
            
            debug_info["feedback_service_status"] = {
                "collection_available": bool(feedback_service.collection),
                "profile_found": bool(feedback_results.get('ids')),
                "has_embeddings": bool(feedback_results.get('embeddings')),
                "embedding_length": len(feedback_results['embeddings'][0]) if feedback_results.get('embeddings') and feedback_results['embeddings'][0] else 0,
                "validation_result": feedback_service._is_valid_embedding_result(feedback_results)
            }
        except Exception as e:
            debug_info["feedback_service_status"] = {
                "error": str(e),
                "collection_available": bool(feedback_service.collection)
            }

        debug_info["collection_comparison"] = {
            "search_collection": str(search_service.collection),
            "feedback_collection": str(feedback_service.collection),
            "are_same_object": search_service.collection is feedback_service.collection
        }
        
        return debug_info
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Debug failed: {str(e)}")


