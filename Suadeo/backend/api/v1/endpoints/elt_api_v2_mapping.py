# backend/api/v1/endpoints/v2_fhir_datasets.py
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form as FormUpload
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime
import os
import shutil

from config.database import get_db
from models.database.fhir_v2_model import V2FHIRdataset, V2FHIRdata, V2ProcessingJob
from services.elt_pipeline.etl_service_V2 import V2FHIRETLService

router = APIRouter()
etl_service = V2FHIRETLService()


class V2FHIRDatasetResponse(BaseModel): 
    id: str
    name: str
    filename: str
    description: Optional[str]
    status: str
    upload_date: datetime
    processed_date: Optional[datetime]
    activated_date: Optional[datetime]
    deactivated_date: Optional[datetime]
    record_count: int
    error_message: Optional[str]
    download_count: int
    last_downloaded: Optional[datetime]
    file_size: Optional[int]
    
    class Config:
        from_attributes = True

class V2FHIRDatasetList(BaseModel): 
    datasets: List[V2FHIRDatasetResponse]  
    total: int

class V2ProcessResponse(BaseModel):
    success: bool
    message: str
    dataset_id: str  

class V2FHIRDataResponse(BaseModel):
    id: str
    local_id: str
    resource: str
    sub_detail: Optional[str]
    fhir_detail: Optional[str]
    fhir_version: Optional[str]
    hl7v2_field: Optional[str]
    hl7v2_field_detail: Optional[str]
    hl7v2_field_version: Optional[str]
    is_active: bool
    created_date: datetime
    
    class Config:
        from_attributes = True


UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.get("/v2-fhir-datasets", response_model=V2FHIRDatasetList)
async def list_v2_fhir_datasets(db: Session = Depends(get_db)):
    """List all uploaded V2 FHIR datasets"""
    datasets = db.query(V2FHIRdataset).order_by(V2FHIRdataset.upload_date.desc()).all()
    return V2FHIRDatasetList(
        datasets=[V2FHIRDatasetResponse.from_orm(d) for d in datasets],
        total=len(datasets)
    )

@router.get("/v2-fhir-datasets/{dataset_id}", response_model=V2FHIRDatasetResponse)
async def get_v2_fhir_dataset(dataset_id: str, db: Session = Depends(get_db)):
    """Get details of a specific V2 FHIR dataset"""
    dataset = db.query(V2FHIRdataset).filter(V2FHIRdataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return V2FHIRDatasetResponse.from_orm(dataset)

@router.post("/v2-fhir-datasets/upload", response_model=V2ProcessResponse)
async def upload_v2_fhir_dataset(
    file: UploadFile = File(...),
    name: str = FormUpload(...),
    description: Optional[str] = FormUpload(None),
    db: Session = Depends(get_db)
):

    """Upload a new V2 FHIR mapping dataset file"""
    try:
      
        allowed_extensions = {'.csv', '.json', '.xlsx', '.xls'}
        file_extension = os.path.splitext(file.filename)[1].lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
            )


        dataset = V2FHIRdataset(
            name=name,
            filename=file.filename,
            description=description,
            status="uploaded",
            file_size=0  
        )
        
        db.add(dataset)
        db.flush()  

   
        file_path = os.path.join(UPLOAD_DIR, f"{dataset.id}_{file.filename}")
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

    
        dataset.file_path = file_path
        dataset.file_size = len(content)
        db.commit()
        
        return V2ProcessResponse(
            success=True,
            message=f"Dataset '{name}' uploaded successfully. Use /datasets/{dataset.id}/process to process it.",
            dataset_id=dataset.id
        )
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.get("/v2-fhir-datasets/{dataset_id}/download")
async def download_v2_fhir_dataset(
    dataset_id: str,
    db: Session = Depends(get_db)
):
    """Download the original dataset file"""
    try:
        dataset = db.query(V2FHIRdataset).filter(V2FHIRdataset.id == dataset_id).first()
        
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
            
        if not dataset.file_path:
            raise HTTPException(status_code=404, detail="No file associated with this dataset")

        if not os.path.exists(dataset.file_path):
            raise HTTPException(status_code=404, detail="File not found on disk")
        
    
        dataset.download_count = (dataset.download_count or 0) + 1
        dataset.last_downloaded = datetime.utcnow()
        db.commit()

        return FileResponse(
            path=dataset.file_path,
            filename=dataset.filename,
            media_type='application/octet-stream'
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")

@router.post("/v2-fhir-datasets/{dataset_id}/process", response_model=V2ProcessResponse)
async def process_v2_fhir_dataset(dataset_id: str, db: Session = Depends(get_db)):
    """Process an uploaded V2 FHIR dataset"""
    try:
        dataset = db.query(V2FHIRdataset).filter(V2FHIRdataset.id == dataset_id).first()
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        if dataset.status not in ["uploaded", "failed"]:
            raise HTTPException(
                status_code=400, 
                detail=f"Dataset is in '{dataset.status}' status and cannot be processed"
            )
 
        
        success = etl_service.process_dataset(dataset_id, db)
        
   
        db.refresh(dataset)
        
        if success:
            return V2ProcessResponse(
                success=True,
                message=f"Dataset processed successfully. {dataset.record_count} V2-FHIR mappings loaded.",
                dataset_id=dataset_id
            )
        else:
            return V2ProcessResponse(
                success=False,
                message=f"Processing failed: {dataset.error_message}",
                dataset_id=dataset_id
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@router.put("/v2-fhir-datasets/{dataset_id}/activate", response_model=V2ProcessResponse)
async def activate_v2_fhir_dataset(dataset_id: str, db: Session = Depends(get_db)):
    """Activate a processed dataset (make its mappings searchable)"""
    try:
        dataset = db.query(V2FHIRdataset).filter(V2FHIRdataset.id == dataset_id).first()
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        if dataset.status != "ready":
            raise HTTPException(
                status_code=400,
                detail=f"Dataset must be in 'ready' status to activate. Current status: {dataset.status}"
            )
        
        
        success = etl_service.activate_dataset(dataset_id, db)
        
        if success:
            return V2ProcessResponse(
                success=True,
                message=f"Dataset '{dataset.name}' activated successfully. Its V2-FHIR mappings are now searchable.",
                dataset_id=dataset_id
            )
        else:
            return V2ProcessResponse(
                success=False,
                message="Failed to activate dataset",
                dataset_id=dataset_id
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Activation failed: {str(e)}")

@router.put("/v2-fhir-datasets/{dataset_id}/deactivate", response_model=V2ProcessResponse)
async def deactivate_v2_fhir_dataset(dataset_id: str, db: Session = Depends(get_db)):
    """Deactivate a dataset (remove its mappings from search)"""
    try:
        dataset = db.query(V2FHIRdataset).filter(V2FHIRdataset.id == dataset_id).first()
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        if dataset.status != "active":
            raise HTTPException(
                status_code=400,
                detail=f"Dataset must be in 'active' status to deactivate. Current status: {dataset.status}"
            )
        
        success = etl_service.deactivate_dataset(dataset_id, db)
        
        if success:
            return V2ProcessResponse(
                success=True,
                message=f"Dataset '{dataset.name}' deactivated successfully. Its mappings are no longer searchable.",
                dataset_id=dataset_id
            )
        else:
            return V2ProcessResponse(
                success=False,
                message="Failed to deactivate dataset",
                dataset_id=dataset_id
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Deactivation failed: {str(e)}")

@router.delete("/v2-fhir-datasets/{dataset_id}")
async def delete_v2_fhir_dataset(dataset_id: str, db: Session = Depends(get_db)):
    """Delete a dataset and all its V2-FHIR mappings"""
    try:
        dataset = db.query(V2FHIRdataset).filter(V2FHIRdataset.id == dataset_id).first()
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")

        if dataset.status == "active":
            raise HTTPException(
                status_code=400,
                detail="Cannot delete active dataset. Deactivate it first."
            )
        
        # Delete the file if it exists
        if dataset.file_path and os.path.exists(dataset.file_path):
            os.remove(dataset.file_path)
        
        # Delete from database (cascade will handle related records)
        db.delete(dataset)
        db.commit()
        
        return {"success": True, "message": f"Dataset '{dataset.name}' deleted successfully"}
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")

@router.get("/v2-fhir-datasets/{dataset_id}/mappings")
async def preview_v2_fhir_dataset_mappings(
    dataset_id: str, 
    limit: int = 10,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """Preview V2-FHIR mappings in a dataset"""
    dataset = db.query(V2FHIRdataset).filter(V2FHIRdataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Get mappings with pagination
    mappings = db.query(V2FHIRdata).filter(
        V2FHIRdata.V2FHIRdataset_id == dataset_id
    ).offset(offset).limit(limit).all()
    
    # Get total count
    total_count = db.query(V2FHIRdata).filter(
        V2FHIRdata.V2FHIRdataset_id == dataset_id
    ).count()
    
    return {
        "dataset_name": dataset.name,
        "dataset_status": dataset.status,
        "total_mappings": total_count,
        "showing": f"{offset + 1}-{min(offset + limit, total_count)} of {total_count}",
        "mappings": [
            V2FHIRDataResponse.from_orm(m) for m in mappings
        ]
    }

@router.get("/v2-fhir-datasets/{dataset_id}/stats")
async def get_v2_fhir_dataset_stats(dataset_id: str, db: Session = Depends(get_db)):
    """Get statistics for a V2 FHIR dataset"""
    dataset = db.query(V2FHIRdataset).filter(V2FHIRdataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
   
    total_mappings = db.query(V2FHIRdata).filter(
        V2FHIRdata.V2FHIRdataset_id == dataset_id
    ).count()
    
    active_mappings = db.query(V2FHIRdata).filter(
        V2FHIRdata.V2FHIRdataset_id == dataset_id,
        V2FHIRdata.is_active == True
    ).count()
    
    # Get unique resource types
    resource_types = db.query(V2FHIRdata.resource).filter(
        V2FHIRdata.V2FHIRdataset_id == dataset_id
    ).distinct().all()
    
    # Get unique FHIR versions
    fhir_versions = db.query(V2FHIRdata.fhir_version).filter(
        V2FHIRdata.V2FHIRdataset_id == dataset_id,
        V2FHIRdata.fhir_version.isnot(None)
    ).distinct().all()
    
    # Get unique HL7 V2 versions
    v2_versions = db.query(V2FHIRdata.hl7v2_field_version).filter(
        V2FHIRdata.V2FHIRdataset_id == dataset_id,
        V2FHIRdata.hl7v2_field_version.isnot(None)
    ).distinct().all()
    
    return {
        "dataset_info": {
            "id": dataset.id,
            "name": dataset.name,
            "status": dataset.status,
            "upload_date": dataset.upload_date,
            "processed_date": dataset.processed_date,
            "file_size": dataset.file_size
        },
        "mapping_stats": {
            "total_mappings": total_mappings,
            "active_mappings": active_mappings,
            "inactive_mappings": total_mappings - active_mappings
        },
        "content_overview": {
            "resource_types": [r[0] for r in resource_types if r[0]],
            "fhir_versions": [v[0] for v in fhir_versions if v[0]],
            "hl7v2_versions": [v[0] for v in v2_versions if v[0]]
        }
    }