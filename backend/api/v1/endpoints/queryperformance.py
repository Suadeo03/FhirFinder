from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel, field_validator
from typing import List, Optional, Union, Dict
from datetime import datetime

from config.database import get_db
from models.database.models import QueryPerformance

router = APIRouter()

class QueryPerformanceResponse(BaseModel):
    """Pydantic model for API responses"""
    id: str
    profile_id: str
    query_text: str
    query_date: datetime
    profile_name: str
    profile_oid: str
    profile_score: float
    context_score: float
    combined_score: float
    match_reasons: Optional[str] = None
    keywords: Optional[Union[Dict, List, str]] = None  # Add default None
    
    @field_validator('keywords', mode='before')
    @classmethod
    def validate_keywords(cls, v):
        """Convert keywords to consistent format"""
        if v is None:
            return None
        if isinstance(v, str):
            try:
                import json
                return json.loads(v)
            except:
                return {"raw": v}
        if isinstance(v, list):
            return {"terms": v}  # Convert list to dict
        if isinstance(v, dict):
            return v
        return {"raw": str(v)}
    
    class Config:
        from_attributes = True 

class DatasetList(BaseModel):
    items: List[QueryPerformanceResponse]
    total: int
    page: int
    size: int


@router.get("/performance", response_model=DatasetList)
async def get_performance(db: Session = Depends(get_db)):
    """List all datasets with performance scores ordered by profile score in descending order."""
    try:
        db_results = db.query(QueryPerformance).order_by(QueryPerformance.profile_score.desc()).all()
        pydantic_results = [QueryPerformanceResponse.model_validate(item) for item in db_results]
        return DatasetList(
                items=pydantic_results,
                total=len(pydantic_results),
                page=1,
                size=len(pydantic_results)
                )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")