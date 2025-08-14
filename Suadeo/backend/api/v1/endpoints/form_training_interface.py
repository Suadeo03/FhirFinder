# backend/api/v1/endpoints/embeddings.py
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime

from config.database import get_db
from config.chroma import get_chroma_instance
from models.database.form_model import Form, Formset
from services.elt_pipeline.elt_form_service import ETL_Form_Service
from sentence_transformers import SentenceTransformer

router = APIRouter()

class EmbeddingUpdateRequest(BaseModel):
    """Request model for updating embeddings"""
    form_id: str = Field(..., description="ID of the form to update")
    embedding: Optional[List[float]] = Field(None, description="Custom embedding vector (768 dimensions)")
    search_text: Optional[str] = Field(None, description="Custom search text (will generate embedding if provided)")
    metadata_updates: Optional[Dict[str, Any]] = Field(None, description="Metadata updates for ChromaDB")
    regenerate_from_form: Optional[bool] = Field(False, description="Regenerate embedding from form data")

    @validator('embedding')
    def validate_embedding_dimensions(cls, v):
        if v is not None and len(v) != 384:  
            raise ValueError('Embedding must be 384 dimensions')
        return v

class BulkEmbeddingUpdateRequest(BaseModel):
    """Request model for bulk embedding updates"""
    updates: List[EmbeddingUpdateRequest] = Field(..., description="List of embedding updates")
    formset_id: Optional[str] = Field(None, description="Filter updates to specific formset")

class EmbeddingUpdateResponse(BaseModel):
    """Response model for embedding updates"""
    form_id: str
    success: bool
    message: str
    embedding_updated: bool = False
    metadata_updated: bool = False
    chroma_updated: bool = False

class BulkEmbeddingUpdateResponse(BaseModel):
    """Response model for bulk embedding updates"""
    total_requested: int
    successful_updates: int
    failed_updates: int
    results: List[EmbeddingUpdateResponse]

class EmbeddingRetrainRequest(BaseModel):
    """Request model for retraining embeddings"""
    formset_id: Optional[str] = Field(None, description="Retrain specific formset (all if None)")
    force_regenerate: bool = Field(False, description="Force regenerate even if embeddings exist")

@router.post("/embeddings/update", response_model=EmbeddingUpdateResponse)
async def update_form_embedding(
    request: EmbeddingUpdateRequest, 
    db: Session = Depends(get_db)
):
    """Update embedding for a specific form"""
    try:
        # Get the form
        form = db.query(Form).filter(Form.id == request.form_id).first()
        if not form:
            raise HTTPException(status_code=404, detail=f"Form {request.form_id} not found")

        # Initialize services
        etl_service = ETL_Form_Service()
        chroma = get_chroma_instance()
        
        response = EmbeddingUpdateResponse(
            form_id=request.form_id,
            success=False,
            message="Starting update..."
        )

        # Determine the embedding to use
        embedding = None
        search_text = None

        if request.embedding:
            # Use provided embedding
            embedding = request.embedding
            search_text = request.search_text or f"{form.domain} {form.question} {form.answer_concept}"
            
        elif request.search_text:
            # Generate embedding from provided search text
            search_text = request.search_text
            embedding = etl_service.model.encode([search_text])[0].tolist()
            
        elif request.regenerate_from_form:
            # Regenerate from form data
            search_text = f"{form.domain} {form.question} {form.answer_concept} {form.loinc_concept}"
            embedding = etl_service.model.encode([search_text])[0].tolist()
            
        else:
            raise HTTPException(
                status_code=400, 
                detail="Must provide either embedding, search_text, or set regenerate_from_form=true"
            )

        # Update ChromaDB if available
        if chroma.is_available() and embedding:
            try:
                # Prepare metadata
                metadata = {
                    'domain': form.domain or '',
                    'screening_tool': (form.screening_tool or '')[:1000],
                    'question': form.question or 'Unknown',
                    'answer_concept': form.answer_concept or 'Unknown',
                    'formset_id': form.formset_id or '',
                    'is_active': form.is_active,
                    'last_updated': datetime.utcnow().isoformat()
                }
                
                # Add any additional metadata updates
                if request.metadata_updates:
                    metadata.update(request.metadata_updates)

                # Update in ChromaDB
                collection = chroma.get_collection()
                collection.upsert(
                    ids=[request.form_id],
                    embeddings=[embedding],
                    documents=[search_text],
                    metadatas=[metadata]
                )
                
                response.chroma_updated = True
                response.embedding_updated = True
                
            except Exception as e:
                print(f"ChromaDB update failed: {e}")
                response.message = f"ChromaDB update failed: {str(e)}"
                return response

        # Update metadata only if requested
        elif request.metadata_updates and chroma.is_available():
            try:
                collection = chroma.get_collection()
                
                # Get current data
                results = collection.get(ids=[request.form_id], include=['metadatas'])
                if results['ids']:
                    current_metadata = results['metadatas'][0]
                    current_metadata.update(request.metadata_updates)
                    current_metadata['last_updated'] = datetime.utcnow().isoformat()
                    
                    collection.update(
                        ids=[request.form_id],
                        metadatas=[current_metadata]
                    )
                    
                    response.metadata_updated = True
                    
            except Exception as e:
                print(f"Metadata update failed: {e}")
                response.message = f"Metadata update failed: {str(e)}"
                return response

        response.success = True
        response.message = "Embedding updated successfully"
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Update failed: {str(e)}")

@router.post("/embeddings/bulk-update", response_model=BulkEmbeddingUpdateResponse)
async def bulk_update_embeddings(
    request: BulkEmbeddingUpdateRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Bulk update embeddings for multiple forms"""
    results = []
    successful = 0
    failed = 0
    
    for update_req in request.updates:
        try:
            # Filter by formset if specified
            if request.formset_id:
                form = db.query(Form).filter(
                    Form.id == update_req.form_id,
                    Form.formset_id == request.formset_id
                ).first()
                if not form:
                    results.append(EmbeddingUpdateResponse(
                        form_id=update_req.form_id,
                        success=False,
                        message=f"Form not found in formset {request.formset_id}"
                    ))
                    failed += 1
                    continue
            
            # Process individual update
            result = await update_form_embedding(update_req, db)
            results.append(result)
            
            if result.success:
                successful += 1
            else:
                failed += 1
                
        except Exception as e:
            results.append(EmbeddingUpdateResponse(
                form_id=update_req.form_id,
                success=False,
                message=f"Error: {str(e)}"
            ))
            failed += 1
    
    return BulkEmbeddingUpdateResponse(
        total_requested=len(request.updates),
        successful_updates=successful,
        failed_updates=failed,
        results=results
    )

@router.post("/embeddings/retrain-formset")
async def retrain_formset_embeddings(
    request: EmbeddingRetrainRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Retrain embeddings for a formset or all forms"""
    
    def retrain_task(formset_id: Optional[str], force_regenerate: bool):
        """Background task to retrain embeddings"""
        try:
            etl_service = ETL_Form_Service()
            chroma = get_chroma_instance()
            
            # Get forms to retrain
            query = db.query(Form)
            if formset_id:
                query = query.filter(Form.formset_id == formset_id)
            
            forms = query.all()
            
            print(f"Retraining embeddings for {len(forms)} forms")
            
            batch_size = 50
            for i in range(0, len(forms), batch_size):
                batch = forms[i:i + batch_size]
                
                # Generate embeddings for batch
                search_texts = []
                embeddings = []
                form_data = []
                
                for form in batch:
                    search_text = f"{form.domain} {form.question} {form.answer_concept} {form.loinc_concept}"
                    embedding = etl_service.model.encode([search_text])[0].tolist()
                    
                    search_texts.append(search_text)
                    embeddings.append(embedding)
                    form_data.append({
                        'id': form.id,
                        'domain': form.domain,
                        'screening_tool': form.screening_tool,
                        'question': form.question,
                        'answer_concept': form.answer_concept,
                        'formset_id': form.formset_id
                    })
                
                # Batch update ChromaDB
                if chroma.is_available():
                    etl_service._batch_add_to_chroma(form_data, search_texts, embeddings)
                
                print(f"Processed batch {i//batch_size + 1}/{(len(forms) + batch_size - 1)//batch_size}")
            
            print(f"Completed retraining {len(forms)} embeddings")
            
        except Exception as e:
            print(f"Retraining failed: {e}")
    
    # Start background task
    background_tasks.add_task(
        retrain_task, 
        request.formset_id, 
        request.force_regenerate
    )
    
    return {
        "message": "Retraining started in background",
        "formset_id": request.formset_id,
        "status": "processing"
    }

@router.get("/embeddings/status/{form_id}")
async def get_embedding_status(form_id: str, db: Session = Depends(get_db)):
    """Get embedding status for a specific form"""
    try:
        # Check if form exists in database
        form = db.query(Form).filter(Form.id == form_id).first()
        if not form:
            raise HTTPException(status_code=404, detail="Form not found")
        
        # Check if embedding exists in ChromaDB
        chroma = get_chroma_instance()
        chroma_status = False
        metadata = None
        
        if chroma.is_available():
            try:
                collection = chroma.get_collection()
                results = collection.get(ids=[form_id], include=['metadatas', 'documents'])
                
                if results['ids']:
                    chroma_status = True
                    metadata = results['metadatas'][0] if results['metadatas'] else None
                    
            except Exception as e:
                print(f"ChromaDB check failed: {e}")
        
        return {
            "form_id": form_id,
            "exists_in_db": True,
            "exists_in_chroma": chroma_status,
            "is_active": form.is_active,
            "formset_id": form.formset_id,
            "chroma_metadata": metadata,
            "form_data": {
                "domain": form.domain,
                "question": form.question,
                "answer_concept": form.answer_concept
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

@router.delete("/embeddings/{form_id}")
async def delete_embedding(form_id: str, db: Session = Depends(get_db)):
    """Delete embedding from ChromaDB (keeps form in database)"""
    try:
        # Check if form exists
        form = db.query(Form).filter(Form.id == form_id).first()
        if not form:
            raise HTTPException(status_code=404, detail="Form not found")
        
        # Delete from ChromaDB
        chroma = get_chroma_instance()
        if chroma.is_available():
            collection = chroma.get_collection()
            collection.delete(ids=[form_id])
            
            return {
                "message": f"Embedding deleted for form {form_id}",
                "form_id": form_id,
                "deleted_from_chroma": True
            }
        else:
            raise HTTPException(status_code=503, detail="ChromaDB not available")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")