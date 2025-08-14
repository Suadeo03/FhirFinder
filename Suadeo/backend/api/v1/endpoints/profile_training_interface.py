# backend/api/v1/endpoints/fhir_profile_embeddings.py
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime

from config.database import get_db
from config.chroma import get_chroma_instance
from models.database.models import Profile, Dataset
from services.elt_pipeline.etl_service import ETLService
from sentence_transformers import SentenceTransformer

router = APIRouter()

class ProfileEmbeddingUpdateRequest(BaseModel):
    """Request model for updating profile embeddings"""
    profile_id: str = Field(..., description="ID of the profile to update")
    embedding: Optional[List[float]] = Field(None, description="Custom embedding vector (384 dimensions)")
    search_text: Optional[str] = Field(None, description="Custom search text (will generate embedding if provided)")
    metadata_updates: Optional[Dict[str, Any]] = Field(None, description="Metadata updates for ChromaDB")
    regenerate_from_profile: Optional[bool] = Field(False, description="Regenerate embedding from profile data")

    @validator('embedding')
    def validate_embedding_dimensions(cls, v):
        if v is not None and len(v) != 384:  # all-MiniLM-L6-v2 produces 384-dim embeddings
            raise ValueError('Embedding must be 384 dimensions')
        return v

class BulkProfileEmbeddingUpdateRequest(BaseModel):
    """Request model for bulk profile embedding updates"""
    updates: List[ProfileEmbeddingUpdateRequest] = Field(..., description="List of embedding updates")
    dataset_id: Optional[str] = Field(None, description="Filter updates to specific dataset")

class ProfileEmbeddingUpdateResponse(BaseModel):
    """Response model for profile embedding updates"""
    profile_id: str
    success: bool
    message: str
    embedding_updated: bool = False
    metadata_updated: bool = False
    chroma_updated: bool = False

class BulkProfileEmbeddingUpdateResponse(BaseModel):
    """Response model for bulk profile embedding updates"""
    total_requested: int
    successful_updates: int
    failed_updates: int
    results: List[ProfileEmbeddingUpdateResponse]

class ProfileEmbeddingRetrainRequest(BaseModel):
    """Request model for retraining profile embeddings"""
    dataset_id: Optional[str] = Field(None, description="Retrain specific dataset (all if None)")
    force_regenerate: bool = Field(False, description="Force regenerate even if embeddings exist")

@router.post("/profiles/embeddings/update", response_model=ProfileEmbeddingUpdateResponse)
async def update_profile_embedding(
    request: ProfileEmbeddingUpdateRequest, 
    db: Session = Depends(get_db)
):
    """Update embedding for a specific FHIR profile"""
    try:
        # Get the profile
        profile = db.query(Profile).filter(Profile.id == request.profile_id).first()
        if not profile:
            raise HTTPException(status_code=404, detail=f"Profile {request.profile_id} not found")

        # Initialize services
        etl_service = ETLService()
        chroma = get_chroma_instance()
        
        response = ProfileEmbeddingUpdateResponse(
            profile_id=request.profile_id,
            success=False,
            message="Starting update..."
        )

        # Determine the embedding to use
        embedding = None
        search_text = None

        if request.embedding:
            # Use provided embedding
            embedding = request.embedding
            search_text = request.search_text or f"{profile.name} {profile.description} {' '.join(profile.keywords or [])}"
            
        elif request.search_text:
            # Generate embedding from provided search text
            search_text = request.search_text
            embedding = etl_service.model.encode([search_text])[0].tolist()
            
        elif request.regenerate_from_profile:
            # Regenerate from profile data
            search_text = f"{profile.name} {profile.description} {' '.join(profile.keywords or [])} {profile.fhir_searchable_text or ''}"
            embedding = etl_service.model.encode([search_text])[0].tolist()
            
            # Update the profile's search_text in database
            profile.search_text = search_text
            db.commit()
            
        else:
            raise HTTPException(
                status_code=400, 
                detail="Must provide either embedding, search_text, or set regenerate_from_profile=true"
            )

        # Update ChromaDB if available
        if chroma.is_available() and embedding:
            try:
                # Prepare metadata
                metadata = {
                    'name': profile.name,
                    'description': (profile.description or '')[:1000],
                    'resource_type': profile.resource_type or 'Unknown',
                    'category': profile.category or 'Unknown',
                    'dataset_id': profile.dataset_id or '',
                    'keywords': ','.join((profile.keywords or [])[:200]),
                    'use_contexts': str(profile.use_contexts or []),
                    'fhir_searchable_text': profile.fhir_searchable_text or '',
                    'is_active': profile.is_active,
                    'last_updated': datetime.utcnow().isoformat()
                }
                
        
                if request.metadata_updates:
                    metadata.update(request.metadata_updates)

              
                collection = chroma.get_collection()
                collection.upsert(
                    ids=[request.profile_id],
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

   
        elif request.metadata_updates and chroma.is_available():
            try:
                collection = chroma.get_collection()
                
                # Get current data
                results = collection.get(ids=[request.profile_id], include=['metadatas'])
                if results['ids']:
                    current_metadata = results['metadatas'][0]
                    current_metadata.update(request.metadata_updates)
                    current_metadata['last_updated'] = datetime.utcnow().isoformat()
                    
                    collection.update(
                        ids=[request.profile_id],
                        metadatas=[current_metadata]
                    )
                    
                    response.metadata_updated = True
                    
            except Exception as e:
                print(f"Metadata update failed: {e}")
                response.message = f"Metadata update failed: {str(e)}"
                return response

        response.success = True
        response.message = "Profile embedding updated successfully"
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Update failed: {str(e)}")

@router.post("/profiles/embeddings/bulk-update", response_model=BulkProfileEmbeddingUpdateResponse)
async def bulk_update_profile_embeddings(
    request: BulkProfileEmbeddingUpdateRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Bulk update embeddings for multiple profiles"""
    results = []
    successful = 0
    failed = 0
    
    for update_req in request.updates:
        try:
            # Filter by dataset if specified
            if request.dataset_id:
                profile = db.query(Profile).filter(
                    Profile.id == update_req.profile_id,
                    Profile.dataset_id == request.dataset_id
                ).first()
                if not profile:
                    results.append(ProfileEmbeddingUpdateResponse(
                        profile_id=update_req.profile_id,
                        success=False,
                        message=f"Profile not found in dataset {request.dataset_id}"
                    ))
                    failed += 1
                    continue
            
            # Process individual update
            result = await update_profile_embedding(update_req, db)
            results.append(result)
            
            if result.success:
                successful += 1
            else:
                failed += 1
                
        except Exception as e:
            results.append(ProfileEmbeddingUpdateResponse(
                profile_id=update_req.profile_id,
                success=False,
                message=f"Error: {str(e)}"
            ))
            failed += 1
    
    return BulkProfileEmbeddingUpdateResponse(
        total_requested=len(request.updates),
        successful_updates=successful,
        failed_updates=failed,
        results=results
    )

@router.post("/profiles/embeddings/retrain-dataset")
async def retrain_profile_dataset_embeddings(
    request: ProfileEmbeddingRetrainRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Retrain embeddings for a dataset or all profiles"""
    
    def retrain_task(dataset_id: Optional[str], force_regenerate: bool):
        """Background task to retrain profile embeddings"""
        try:
            etl_service = ETLService()
            chroma = get_chroma_instance()
            
            # Get profiles to retrain
            query = db.query(Profile)
            if dataset_id:
                query = query.filter(Profile.dataset_id == dataset_id)
            
            profiles = query.all()
            
            print(f"Retraining embeddings for {len(profiles)} profiles")
            
            batch_size = 50
            for i in range(0, len(profiles), batch_size):
                batch = profiles[i:i + batch_size]
                
                # Generate embeddings for batch
                search_texts = []
                embeddings = []
                profile_data = []
                
                for profile in batch:
                    # Create comprehensive search text
                    search_text = f"{profile.name} {profile.description} {' '.join(profile.keywords or [])} {profile.fhir_searchable_text or ''}"
                    embedding = etl_service.model.encode([search_text])[0].tolist()
                    
                    # Update profile search text in database
                    profile.search_text = search_text
                    
                    search_texts.append(search_text)
                    embeddings.append(embedding)
                    profile_data.append({
                        'id': profile.id,
                        'name': profile.name,
                        'description': profile.description,
                        'resource_type': profile.resource_type,
                        'category': profile.category,
                        'dataset_id': profile.dataset_id,
                        'keywords': profile.keywords,
                        'use_contexts': profile.use_contexts,
                        'fhir_searchable_text': profile.fhir_searchable_text
                    })
                
                # Commit database updates
                db.commit()
                
                # Batch update ChromaDB
                if chroma.is_available():
                    etl_service._batch_add_to_chroma(profile_data, search_texts, embeddings)
                
                print(f"Processed batch {i//batch_size + 1}/{(len(profiles) + batch_size - 1)//batch_size}")
            
            print(f"Completed retraining {len(profiles)} profile embeddings")
            
        except Exception as e:
            print(f"Profile retraining failed: {e}")
            db.rollback()
    
    # Start background task
    background_tasks.add_task(
        retrain_task, 
        request.dataset_id, 
        request.force_regenerate
    )
    
    return {
        "message": "Profile embedding retraining started in background",
        "dataset_id": request.dataset_id,
        "status": "processing"
    }

@router.get("/profiles/embeddings/status/{profile_id}")
async def get_profile_embedding_status(profile_id: str, db: Session = Depends(get_db)):
    """Get embedding status for a specific profile"""
    try:
        # Check if profile exists in database
        profile = db.query(Profile).filter(Profile.id == profile_id).first()
        if not profile:
            raise HTTPException(status_code=404, detail="Profile not found")
        
        # Check if embedding exists in ChromaDB
        chroma = get_chroma_instance()
        chroma_status = False
        metadata = None
        
        if chroma.is_available():
            try:
                collection = chroma.get_collection()
                results = collection.get(ids=[profile_id], include=['metadatas', 'documents'])
                
                if results['ids']:
                    chroma_status = True
                    metadata = results['metadatas'][0] if results['metadatas'] else None
                    
            except Exception as e:
                print(f"ChromaDB check failed: {e}")
        
        return {
            "profile_id": profile_id,
            "exists_in_db": True,
            "exists_in_chroma": chroma_status,
            "is_active": profile.is_active,
            "dataset_id": profile.dataset_id,
            "chroma_metadata": metadata,
            "profile_data": {
                "name": profile.name,
                "description": profile.description,
                "resource_type": profile.resource_type,
                "keywords": profile.keywords,
                "search_text": profile.search_text
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

@router.delete("/profiles/embeddings/{profile_id}")
async def delete_profile_embedding(profile_id: str, db: Session = Depends(get_db)):
    """Delete embedding from ChromaDB (keeps profile in database)"""
    try:
        # Check if profile exists
        profile = db.query(Profile).filter(Profile.id == profile_id).first()
        if not profile:
            raise HTTPException(status_code=404, detail="Profile not found")
        
        # Delete from ChromaDB
        chroma = get_chroma_instance()
        if chroma.is_available():
            collection = chroma.get_collection()
            collection.delete(ids=[profile_id])
            
            return {
                "message": f"Embedding deleted for profile {profile_id}",
                "profile_id": profile_id,
                "deleted_from_chroma": True
            }
        else:
            raise HTTPException(status_code=503, detail="ChromaDB not available")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")

@router.get("/profiles/embeddings/dataset/{dataset_id}/stats")
async def get_dataset_embedding_stats(dataset_id: str, db: Session = Depends(get_db)):
    """Get embedding statistics for a specific dataset"""
    try:
        # Check if dataset exists
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Get profile counts
        total_profiles = db.query(Profile).filter(Profile.dataset_id == dataset_id).count()
        active_profiles = db.query(Profile).filter(
            Profile.dataset_id == dataset_id,
            Profile.is_active == True
        ).count()
        
        # Check ChromaDB status
        chroma_count = 0
        chroma_active_count = 0
        
        chroma = get_chroma_instance()
        if chroma.is_available():
            try:
                collection = chroma.get_collection()
                # Get sample of profile IDs to check ChromaDB
                profile_ids = [p.id for p in db.query(Profile.id).filter(Profile.dataset_id == dataset_id).limit(100).all()]
                
                if profile_ids:
                    results = collection.get(ids=profile_ids, include=['metadatas'])
                    chroma_count = len(results.get('ids', []))
                    
                    # Count active embeddings
                    for metadata in results.get('metadatas', []):
                        if metadata and metadata.get('is_active', False):
                            chroma_active_count += 1
                            
            except Exception as e:
                print(f"ChromaDB stats check failed: {e}")
        
        return {
            "dataset_id": dataset_id,
            "dataset_name": dataset.name,
            "total_profiles": total_profiles,
            "active_profiles": active_profiles,
            "inactive_profiles": total_profiles - active_profiles,
            "chroma_embeddings_count": chroma_count,
            "chroma_active_embeddings": chroma_active_count,
            "embedding_coverage": round((chroma_count / total_profiles * 100) if total_profiles > 0 else 0, 2)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stats retrieval failed: {str(e)}")