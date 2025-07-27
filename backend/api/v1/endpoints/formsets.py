# backend/api/v1/endpoints/datasets.py
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form as FormUpload
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime
import os
import shutil

from config.database import get_db
from models.database.form_model import Formset, Form, FormProcessingJob
from services.elt_form_service import ETL_Form_Service

router = APIRouter()
elt_form_service = ETL_Form_Service()


class FormsetResponse(BaseModel): 
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

class FormsetList(BaseModel): 
    formsets: List[FormsetResponse]  
    total: int

class ProcessResponse(BaseModel):
    success: bool
    message: str
    formset_id: str  


UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.get("/formsets", response_model=FormsetList)
async def list_formsets(db: Session = Depends(get_db)):
    """List all uploaded formsets"""
    formsets = db.query(Formset).order_by(Formset.upload_date.desc()).all()
    return FormsetList(
        formsets=[FormsetResponse.from_orm(f) for f in formsets],
        total=len(formsets)
    )

@router.get("/formsets/{formset_id}", response_model=FormsetResponse)
async def get_formset(formset_id: str, db: Session = Depends(get_db)):
    """Get details of a specific formset"""
    formset = db.query(Formset).filter(Formset.id == formset_id).first()
    if not formset:
        raise HTTPException(status_code=404, detail="Formset not found")
    return FormsetResponse.from_orm(formset)

@router.post("/formsets/upload", response_model=ProcessResponse)
async def upload_formset(
    file: UploadFile = File(...),
    name: str = FormUpload(...),
    description: Optional[str] = FormUpload(None),
    db: Session = Depends(get_db)
):
    """Upload a new formset file"""
    try:
        # Validate file type
        allowed_extensions = {'.csv', '.json', '.xlsx', '.xls'}
        file_extension = os.path.splitext(file.filename)[1].lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
            )

        formset = Formset(
            name=name,
            filename=file.filename,
            description=description,
            status="uploaded",
            file_size=0  
        )
        
        db.add(formset)
        db.flush()  

        file_path = os.path.join(UPLOAD_DIR, f"{formset.id}_{file.filename}")
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        formset.file_path = file_path
        formset.file_size = len(content)
        db.commit()
        
        return ProcessResponse(
            success=True,
            message=f"Formset '{name}' uploaded successfully. Use /formsets/{formset.id}/process to process it.",
            formset_id=formset.id
        )
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
    

@router.get("/formsets/{formset_id}/download")
async def download_formset(
    formset_id: str,
    db: Session = Depends(get_db)
):
    """Download the original formset file"""
    try:
        formset = db.query(Formset).filter(Formset.id == formset_id).first()
        
        if not formset:
            raise HTTPException(status_code=404, detail="Formset not found")
            
        if not formset.file_path:
            raise HTTPException(status_code=404, detail="No file associated with this formset")

        if not os.path.exists(formset.file_path):
            raise HTTPException(status_code=404, detail="File not found on disk")
        
        formset.download_count = (formset.download_count or 0) + 1
        formset.last_downloaded = datetime.utcnow()
        db.commit()

        return FileResponse(
            path=formset.file_path,
            filename=formset.filename,
            media_type='application/octet-stream'
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")

@router.post("/formsets/{formset_id}/process", response_model=ProcessResponse)
async def process_formset(formset_id: str, db: Session = Depends(get_db)):
    """Process an uploaded formset"""
    try:
        formset = db.query(Formset).filter(Formset.id == formset_id).first()
        if not formset:
            raise HTTPException(status_code=404, detail="Formset not found")
        
        if formset.status not in ["uploaded", "failed"]:
            raise HTTPException(
                status_code=400, 
                detail=f"Formset is in '{formset.status}' status and cannot be processed"
            )
 
        success = elt_form_service.process_dataset(formset_id, db)
        
        if success:
            return ProcessResponse(
                success=True,
                message=f"Formset processed successfully. {formset.record_count} forms loaded.",
                formset_id=formset_id
            )
        else:
            return ProcessResponse(
                success=False,
                message=f"Processing failed: {formset.error_message}",
                formset_id=formset_id
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    

@router.put("/formsets/{formset_id}/activate", response_model=ProcessResponse)
async def activate_formset(formset_id: str, db: Session = Depends(get_db)):
    """Activate a processed formset (make it the active search formset)"""
    try:
        formset = db.query(Formset).filter(Formset.id == formset_id).first()
        if not formset:
            raise HTTPException(status_code=404, detail="Formset not found")
        
        if formset.status != "ready":
            raise HTTPException(
                status_code=400,
                detail=f"Formset must be in 'ready' status to activate. Current status: {formset.status}"
            )
        
        # Activate the formset
        success = elt_form_service.activate_dataset(formset_id, db)
        
        if success:
            return ProcessResponse(
                success=True,
                message=f"Formset '{formset.name}' activated successfully. It is now the active search formset.",
                formset_id=formset_id
            )
        else:
            return ProcessResponse(
                success=False,
                message="Failed to activate formset",
                formset_id=formset_id
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Activation failed: {str(e)}")
    
@router.put("/formsets/{formset_id}/deactivate", response_model=ProcessResponse)
async def deactivate_formset(formset_id: str, db: Session = Depends(get_db)):
    """Deactivate a processed formset (remove from active search)"""
    try:
        formset = db.query(Formset).filter(Formset.id == formset_id).first()
        if not formset:
            raise HTTPException(status_code=404, detail="Formset not found")
        
        if formset.status != "active":
            raise HTTPException(
                status_code=400,
                detail=f"Formset must be in 'active' status to deactivate. Current status: {formset.status}"
            )
        
        success = elt_form_service.deactivate_dataset(formset_id, db)
        
        if success:
            return ProcessResponse(
                success=True,
                message=f"Formset '{formset.name}' deactivated successfully. It is no longer in search formset.",
                formset_id=formset_id
            )
        else:
            return ProcessResponse(
                success=False,
                message="Failed to deactivate formset",
                formset_id=formset_id
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Deactivation failed: {str(e)}")
    
@router.delete("/formsets/{formset_id}")
async def delete_formset(formset_id: str, db: Session = Depends(get_db)):
    """Delete a formset and all its forms"""
    try:
        formset = db.query(Formset).filter(Formset.id == formset_id).first()
        if not formset:
            raise HTTPException(status_code=404, detail="Formset not found")

        if formset.status == "active":
            raise HTTPException(
                status_code=400,
                detail="Cannot delete active formset. Activate another formset first."
            )
        

        if formset.file_path and os.path.exists(formset.file_path):
            os.remove(formset.file_path)
        
        
        db.delete(formset)
        db.commit()
        
        return {"success": True, "message": f"Formset '{formset.name}' deleted successfully"}
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")

@router.get("/formsets/{formset_id}/forms")
async def preview_formset_forms(
    formset_id: str, 
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """Preview forms from a formset"""
    formset = db.query(Formset).filter(Formset.id == formset_id).first()
    if not formset:
        raise HTTPException(status_code=404, detail="Formset not found")
    
   
    forms = db.query(Form).filter(
        Form.formset_id == formset_id
    ).limit(limit).all()
    
    return {
        "formset_name": formset.name,
        "total_forms": formset.record_count,
        "preview_forms": [
            {
                "id": f.id,
                "question": f.question,
                "domain": f.domain,
                "screening_tool": f.screening_tool,
                "loinc_panel_code": f.loinc_panel_code,
                "answer_concept": f.answer_concept
            }
            for f in forms
        ]
    }