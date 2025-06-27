# backend/api/v1/endpoints/datasets.py
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime
import os
import shutil

from config.database import get_db
from models.database.models import Dataset, Profile, ProcessingJob
from services.etl_service import ETLService

router = APIRouter()
etl_service = ETLService()

# Pydantic models for API
class DatasetResponse(BaseModel):
    id: str
    name: str
    filename: str
    description: Optional[str]
    status: str
    upload_date: datetime
    record_count: int
    error_message: Optional[str]
    
    class Config:
        from_attributes = True

class DatasetList(BaseModel):
    datasets: List[DatasetResponse]
    total: int

class ProcessResponse(BaseModel):
    success: bool
    message: str
    dataset_id: str

# Create uploads directory
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.get("/datasets", response_model=DatasetList)
async def list_datasets(db: Session = Depends(get_db)):
    """List all uploaded datasets"""
    datasets = db.query(Dataset).order_by(Dataset.upload_date.desc()).all()
    return DatasetList(
        datasets=[DatasetResponse.from_orm(d) for d in datasets],
        total=len(datasets)
    )

@router.get("/datasets/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(dataset_id: str, db: Session = Depends(get_db)):
    """Get details of a specific dataset"""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return DatasetResponse.from_orm(dataset)

@router.post("/datasets/upload", response_model=ProcessResponse)
async def upload_dataset(
    file: UploadFile = File(...),
    name: str = Form(...),
    description: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """Upload a new dataset file"""
    try:
        # Validate file type
        allowed_extensions = {'.csv', '.json', '.xlsx', '.xls'}
        file_extension = os.path.splitext(file.filename)[1].lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Create dataset record
        dataset = Dataset(
            name=name,
            filename=file.filename,
            description=description,
            status="uploaded",
            file_size=0  # Will be updated after saving
        )
        
        db.add(dataset)
        db.flush()  # Get the dataset ID
        
        # Save uploaded file
        file_path = os.path.join(UPLOAD_DIR, f"{dataset.id}_{file.filename}")
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Update dataset with file info
        dataset.file_path = file_path
        dataset.file_size = len(content)
        db.commit()
        
        return ProcessResponse(
            success=True,
            message=f"Dataset '{name}' uploaded successfully. Use /datasets/{dataset.id}/process to process it.",
            dataset_id=dataset.id
        )
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
    

@router.get("/datasets/{dataset_id}/download")
async def download_dataset(
    dataset_id: str,
    db: Session = Depends(get_db)
):
    """Download a dataset file"""
    try:
        # Get dataset from database
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
            
        if not dataset.file_path:
            raise HTTPException(status_code=404, detail="No file associated with this dataset")
            
        # Check if file exists on disk
        if not os.path.exists(dataset.file_path):
            raise HTTPException(status_code=404, detail="File not found on disk")
        
        dataset.download_count = (dataset.download_count or 0) + 1
        dataset.last_downloaded = datetime.utcnow()
        db.commit()
            
        # Return file response
        return FileResponse(
            path=dataset.file_path,
            filename=dataset.filename,
            media_type='application/octet-stream'
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")

@router.post("/datasets/{dataset_id}/process", response_model=ProcessResponse)
async def process_dataset(dataset_id: str, db: Session = Depends(get_db)):
    """Process an uploaded dataset"""
    try:
        # Check if dataset exists
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        if dataset.status not in ["uploaded", "failed"]:
            raise HTTPException(
                status_code=400, 
                detail=f"Dataset is in '{dataset.status}' status and cannot be processed"
            )
        
        # Process the dataset
        success = etl_service.process_dataset(dataset_id, db)
        
        if success:
            return ProcessResponse(
                success=True,
                message=f"Dataset processed successfully. {dataset.record_count} profiles loaded.",
                dataset_id=dataset_id
            )
        else:
            return ProcessResponse(
                success=False,
                message=f"Processing failed: {dataset.error_message}",
                dataset_id=dataset_id
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@router.put("/datasets/{dataset_id}/activate", response_model=ProcessResponse)
async def activate_dataset(dataset_id: str, db: Session = Depends(get_db)):
    """Activate a processed dataset (make it the active search dataset)"""
    try:
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        if dataset.status != "ready":
            raise HTTPException(
                status_code=400,
                detail=f"Dataset must be in 'ready' status to activate. Current status: {dataset.status}"
            )
        
        # Activate the dataset
        success = etl_service.activate_dataset(dataset_id, db)
        
        if success:
            return ProcessResponse(
                success=True,
                message=f"Dataset '{dataset.name}' activated successfully. It is now the active search dataset.",
                dataset_id=dataset_id
            )
        else:
            return ProcessResponse(
                success=False,
                message="Failed to activate dataset",
                dataset_id=dataset_id
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Activation failed: {str(e)}")
    
@router.put("/datasets/{dataset_id}/deactivate", response_model=ProcessResponse)
async def deactivate_dataset(dataset_id: str, db: Session = Depends(get_db)):
    """Activate a processed dataset (make it the active search dataset)"""
    try:
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        if dataset.status != "active":
            raise HTTPException(
                status_code=400,
                detail=f"Dataset must be in 'active' status to deactivate. Current status: {dataset.status}"
            )
        
        # Activate the dataset
        success = etl_service.deactivate_dataset(dataset_id, db)
        
        if success:
            return ProcessResponse(
                success=True,
                message=f"Dataset '{dataset.name}' deactivated successfully. It is no longer in search dataset.",
                dataset_id=dataset_id
            )
        else:
            return ProcessResponse(
                success=False,
                message="Failed to activate dataset",
                dataset_id=dataset_id
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Activation failed: {str(e)}")
    


@router.delete("/datasets/{dataset_id}")
async def delete_dataset(dataset_id: str, db: Session = Depends(get_db)):
    """Delete a dataset and all its profiles"""
    try:
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Don't allow deleting active datasets
        if dataset.status == "active":
            raise HTTPException(
                status_code=400,
                detail="Cannot delete active dataset. Activate another dataset first."
            )
        
        # Delete associated file
        if dataset.file_path and os.path.exists(dataset.file_path):
            os.remove(dataset.file_path)
        
        # Delete from database (profiles will be deleted via cascade)
        db.delete(dataset)
        db.commit()
        
        return {"success": True, "message": f"Dataset '{dataset.name}' deleted successfully"}
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")

@router.get("/datasets/{dataset_id}/profiles")
async def preview_dataset_profiles(
    dataset_id: str, 
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """Preview profiles from a dataset"""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    profiles = db.query(Profile).filter(
        Profile.dataset_id == dataset_id
    ).limit(limit).all()
    
    return {
        "dataset_name": dataset.name,
        "total_profiles": dataset.record_count,
        "preview_profiles": [
            {
                "id": p.id,
                "name": p.name,
                "description": p.description[:200] + "..." if len(p.description or "") > 200 else p.description,
                "keywords": p.keywords,
                "category": p.category,
                "version": p.version,
                "resource_type": p.resource_type,
                "fhir_resource": p.fhir_resource
            }
            for p in profiles
        ]
    }