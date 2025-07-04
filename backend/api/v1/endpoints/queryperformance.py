from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional

from config.database import get_db
from models.database.models import QueryPerformance

router = APIRouter()

class DatasetList(BaseModel):
    datasets: List[QueryPerformance]
    total: int


@router.get("/performance", response_model=QueryPerformance)
async def list_datasets(db: Session = Depends(get_db)):
    """List all datasets with performance scores ordered by profile score in descending order."""
    try:
        datasets = db.query(QueryPerformance).order_by(QueryPerformance.profile_score.desc()).all()
        return DatasetList(
            datasets=[QueryPerformance.from_orm(d) for d in datasets],
            total=len(datasets)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")