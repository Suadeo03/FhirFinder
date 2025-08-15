# backend/api/v1/endpoints/v2_fhir_embeddings.py
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime

from config.database import get_db
from config.chroma import get_chroma_instance
from models.database.fhir_v2_model import V2FHIRdata, V2FHIRdataset
from services.elt_pipeline.etl_service_V2 import V2FHIRETLService
from sentence_transformers import SentenceTransformer

router = APIRouter()

class V2MappingEmbeddingUpdateRequest(BaseModel):
    """Request model for updating V2 FHIR mapping embeddings"""
    mapping_id: str = Field(..., description="ID of the mapping to update")
    embedding: Optional[List[float]] = Field(None, description="Custom embedding vector (384 dimensions)")
    search_text: Optional[str] = Field(None, description="Custom search text (will generate embedding if provided)")
    metadata_updates: Optional[Dict[str, Any]] = Field(None, description="Metadata updates for ChromaDB")
    regenerate_from_mapping: Optional[bool] = Field(False, description="Regenerate embedding from mapping data")

    @validator('embedding')
    def validate_embedding_dimensions(cls, v):
        if v is not None and len(v) != 384:  # all-MiniLM-L6-v2 produces 384-dim embeddings
            raise ValueError('Embedding must be 384 dimensions')
        return v

class BulkV2MappingEmbeddingUpdateRequest(BaseModel):
    """Request model for bulk V2 mapping embedding updates"""
    updates: List[V2MappingEmbeddingUpdateRequest] = Field(..., description="List of embedding updates")
    dataset_id: Optional[str] = Field(None, description="Filter updates to specific dataset")

class V2MappingEmbeddingUpdateResponse(BaseModel):
    """Response model for V2 mapping embedding updates"""
    mapping_id: str
    success: bool
    message: str
    embedding_updated: bool = False
    metadata_updated: bool = False
    chroma_updated: bool = False

class BulkV2MappingEmbeddingUpdateResponse(BaseModel):
    """Response model for bulk V2 mapping embedding updates"""
    total_requested: int
    successful_updates: int
    failed_updates: int
    results: List[V2MappingEmbeddingUpdateResponse]

class V2MappingEmbeddingRetrainRequest(BaseModel):
    """Request model for retraining V2 mapping embeddings"""
    dataset_id: Optional[str] = Field(None, description="Retrain specific dataset (all if None)")
    force_regenerate: bool = Field(False, description="Force regenerate even if embeddings exist")

@router.post("/v2-mappings/embeddings/update", response_model=V2MappingEmbeddingUpdateResponse)
async def update_v2_mapping_embedding(
    request: V2MappingEmbeddingUpdateRequest, 
    db: Session = Depends(get_db)
):
    """Update embedding for a specific V2 FHIR mapping"""
    try:
        # Get the mapping
        mapping = db.query(V2FHIRdata).filter(V2FHIRdata.id == request.mapping_id).first()
        if not mapping:
            raise HTTPException(status_code=404, detail=f"V2 FHIR mapping {request.mapping_id} not found")

        # Initialize services
        etl_service = V2FHIRETLService()
        chroma = get_chroma_instance()
        
        response = V2MappingEmbeddingUpdateResponse(
            mapping_id=request.mapping_id,
            success=False,
            message="Starting update..."
        )


        embedding = None
        search_text = None

        if request.embedding:
           
            embedding = request.embedding
            search_text = request.search_text or f"{mapping.resource} {mapping.fhir_detail} {mapping.hl7v2_field}"
            
        elif request.search_text:
        
            search_text = request.search_text
            embedding = etl_service.model.encode([search_text])[0].tolist()
            
        elif request.regenerate_from_mapping:

            search_text_parts = [
                mapping.resource or '',
                mapping.fhir_detail or '',
                mapping.hl7v2_field or '',
                mapping.hl7v2_field_detail or '',
                mapping.sub_detail or ''
            ]
            search_text = ' '.join([part for part in search_text_parts if part])
            embedding = etl_service.model.encode([search_text])[0].tolist()
            
        else:
            raise HTTPException(
                status_code=400, 
                detail="Must provide either embedding, search_text, or set regenerate_from_mapping=true"
            )

      
        if chroma.is_available() and embedding:
            try:
             
                metadata = {
                    'local_id': mapping.local_id or '',
                    'resource': mapping.resource or '',
                    'sub_detail': (mapping.sub_detail or '')[:500],
                    'fhir_detail': (mapping.fhir_detail or '')[:1000],
                    'fhir_version': mapping.fhir_version or 'R4',
                    'hl7v2_field': mapping.hl7v2_field or '',
                    'hl7v2_field_detail': (mapping.hl7v2_field_detail or '')[:1000],
                    'hl7v2_field_version': mapping.hl7v2_field_version or '2.5',
                    'dataset_id': mapping.V2FHIRdataset_id or '',
                    'is_active': mapping.is_active,
                    'last_updated': datetime.utcnow().isoformat()
                }
                
                if request.metadata_updates:
                    metadata.update(request.metadata_updates)

                #
                collection = chroma.get_collection()
                collection.upsert(
                    ids=[request.mapping_id],
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

                results = collection.get(ids=[request.mapping_id], include=['metadatas'])
                if results['ids']:
                    current_metadata = results['metadatas'][0]
                    current_metadata.update(request.metadata_updates)
                    current_metadata['last_updated'] = datetime.utcnow().isoformat()
                    
                    collection.update(
                        ids=[request.mapping_id],
                        metadatas=[current_metadata]
                    )
                    
                    response.metadata_updated = True
                    
            except Exception as e:
                print(f"Metadata update failed: {e}")
                response.message = f"Metadata update failed: {str(e)}"
                return response

        response.success = True
        response.message = "V2 FHIR mapping embedding updated successfully"
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Update failed: {str(e)}")

@router.post("/v2-mappings/embeddings/bulk-update", response_model=BulkV2MappingEmbeddingUpdateResponse)
async def bulk_update_v2_mapping_embeddings(
    request: BulkV2MappingEmbeddingUpdateRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Bulk update embeddings for multiple V2 FHIR mappings"""
    results = []
    successful = 0
    failed = 0
    
    for update_req in request.updates:
        try:
          
            if request.dataset_id:
                mapping = db.query(V2FHIRdata).filter(
                    V2FHIRdata.id == update_req.mapping_id,
                    V2FHIRdata.V2FHIRdataset_id == request.dataset_id
                ).first()
                if not mapping:
                    results.append(V2MappingEmbeddingUpdateResponse(
                        mapping_id=update_req.mapping_id,
                        success=False,
                        message=f"Mapping not found in dataset {request.dataset_id}"
                    ))
                    failed += 1
                    continue
            
           
            result = await update_v2_mapping_embedding(update_req, db)
            results.append(result)
            
            if result.success:
                successful += 1
            else:
                failed += 1
                
        except Exception as e:
            results.append(V2MappingEmbeddingUpdateResponse(
                mapping_id=update_req.mapping_id,
                success=False,
                message=f"Error: {str(e)}"
            ))
            failed += 1
    
    return BulkV2MappingEmbeddingUpdateResponse(
        total_requested=len(request.updates),
        successful_updates=successful,
        failed_updates=failed,
        results=results
    )

@router.post("/v2-mappings/embeddings/retrain-dataset")
async def retrain_v2_mapping_dataset_embeddings(
    request: V2MappingEmbeddingRetrainRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Retrain embeddings for a V2 FHIR dataset or all mappings"""
    
    def retrain_task(dataset_id: Optional[str], force_regenerate: bool):
        """Background task to retrain V2 mapping embeddings"""
        try:
            etl_service = V2FHIRETLService()
            chroma = get_chroma_instance()
            
            # Get mappings to retrain
            query = db.query(V2FHIRdata)
            if dataset_id:
                query = query.filter(V2FHIRdata.V2FHIRdataset_id == dataset_id)
            
            mappings = query.all()
            
            print(f"Retraining embeddings for {len(mappings)} V2 FHIR mappings")
            
            batch_size = 50
            for i in range(0, len(mappings), batch_size):
                batch = mappings[i:i + batch_size]
                

                search_texts = []
                embeddings = []
                mapping_data = []
                
                for mapping in batch:
                   
                    search_text_parts = [
                        mapping.resource or '',
                        mapping.fhir_detail or '',
                        mapping.hl7v2_field or '',
                        mapping.hl7v2_field_detail or '',
                        mapping.sub_detail or ''
                    ]
                    search_text = ' '.join([part for part in search_text_parts if part])
                    embedding = etl_service.model.encode([search_text])[0].tolist()
                    
                    search_texts.append(search_text)
                    embeddings.append(embedding)
                    mapping_data.append({
                        'id': mapping.id,
                        'local_id': mapping.local_id,
                        'resource': mapping.resource,
                        'sub_detail': mapping.sub_detail,
                        'fhir_detail': mapping.fhir_detail,
                        'fhir_version': mapping.fhir_version,
                        'hl7v2_field': mapping.hl7v2_field,
                        'hl7v2_field_detail': mapping.hl7v2_field_detail,
                        'hl7v2_field_version': mapping.hl7v2_field_version,
                        'dataset_id': mapping.V2FHIRdataset_id
                    })
                
                # Batch update ChromaDB
                if chroma.is_available():
                    etl_service._batch_add_to_chroma(mapping_data, search_texts, embeddings)
                
                print(f"Processed batch {i//batch_size + 1}/{(len(mappings) + batch_size - 1)//batch_size}")
            
            print(f"Completed retraining {len(mappings)} V2 FHIR mapping embeddings")
            
        except Exception as e:
            print(f"V2 FHIR mapping retraining failed: {e}")
            db.rollback()
    
    # Start background task
    background_tasks.add_task(
        retrain_task, 
        request.dataset_id, 
        request.force_regenerate
    )
    
    return {
        "message": "V2 FHIR mapping embedding retraining started in background",
        "dataset_id": request.dataset_id,
        "status": "processing"
    }

@router.get("/v2-mappings/embeddings/status/{mapping_id}")
async def get_v2_mapping_embedding_status(mapping_id: str, db: Session = Depends(get_db)):
    """Get embedding status for a specific V2 FHIR mapping"""
    try:

        mapping = db.query(V2FHIRdata).filter(V2FHIRdata.id == mapping_id).first()
        if not mapping:
            raise HTTPException(status_code=404, detail="V2 FHIR mapping not found")
        
 
        chroma = get_chroma_instance()
        chroma_status = False
        metadata = None
        
        if chroma.is_available():
            try:
                collection = chroma.get_collection()
                results = collection.get(ids=[mapping_id], include=['metadatas', 'documents'])
                
                if results['ids']:
                    chroma_status = True
                    metadata = results['metadatas'][0] if results['metadatas'] else None
                    
            except Exception as e:
                print(f"ChromaDB check failed: {e}")
        
        return {
            "mapping_id": mapping_id,
            "exists_in_db": True,
            "exists_in_chroma": chroma_status,
            "is_active": mapping.is_active,
            "dataset_id": mapping.V2FHIRdataset_id,
            "chroma_metadata": metadata,
            "mapping_data": {
                "local_id": mapping.local_id,
                "resource": mapping.resource,
                "fhir_detail": mapping.fhir_detail,
                "hl7v2_field": mapping.hl7v2_field,
                "hl7v2_field_detail": mapping.hl7v2_field_detail
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

@router.delete("/v2-mappings/embeddings/{mapping_id}")
async def delete_v2_mapping_embedding(mapping_id: str, db: Session = Depends(get_db)):
    """Delete embedding from ChromaDB (keeps mapping in database)"""
    try:

        mapping = db.query(V2FHIRdata).filter(V2FHIRdata.id == mapping_id).first()
        if not mapping:
            raise HTTPException(status_code=404, detail="V2 FHIR mapping not found")
        

        chroma = get_chroma_instance()
        if chroma.is_available():
            collection = chroma.get_collection()
            collection.delete(ids=[mapping_id])
            
            return {
                "message": f"Embedding deleted for V2 FHIR mapping {mapping_id}",
                "mapping_id": mapping_id,
                "deleted_from_chroma": True
            }
        else:
            raise HTTPException(status_code=503, detail="ChromaDB not available")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")

@router.get("/v2-mappings/embeddings/dataset/{dataset_id}/stats")
async def get_v2_dataset_embedding_stats(dataset_id: str, db: Session = Depends(get_db)):
    """Get embedding statistics for a specific V2 FHIR dataset"""
    try:
        # Check if dataset exists
        dataset = db.query(V2FHIRdataset).filter(V2FHIRdataset.id == dataset_id).first()
        if not dataset:
            raise HTTPException(status_code=404, detail="V2 FHIR dataset not found")
        
        # Get mapping counts
        total_mappings = db.query(V2FHIRdata).filter(V2FHIRdata.V2FHIRdataset_id == dataset_id).count()
        active_mappings = db.query(V2FHIRdata).filter(
            V2FHIRdata.V2FHIRdataset_id == dataset_id,
            V2FHIRdata.is_active == True
        ).count()
        

        chroma_count = 0
        chroma_active_count = 0
        
        chroma = get_chroma_instance()
        if chroma.is_available():
            try:
                collection = chroma.get_collection()

                mapping_ids = [m.id for m in db.query(V2FHIRdata.id).filter(V2FHIRdata.V2FHIRdataset_id == dataset_id).limit(100).all()]
                
                if mapping_ids:
                    results = collection.get(ids=mapping_ids, include=['metadatas'])
                    chroma_count = len(results.get('ids', []))
                    
           
                    for metadata in results.get('metadatas', []):
                        if metadata and metadata.get('is_active', False):
                            chroma_active_count += 1
                            
            except Exception as e:
                print(f"ChromaDB stats check failed: {e}")
        
        return {
            "dataset_id": dataset_id,
            "dataset_name": dataset.name,
            "total_mappings": total_mappings,
            "active_mappings": active_mappings,
            "inactive_mappings": total_mappings - active_mappings,
            "chroma_embeddings_count": chroma_count,
            "chroma_active_embeddings": chroma_active_count,
            "embedding_coverage": round((chroma_count / total_mappings * 100) if total_mappings > 0 else 0, 2)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stats retrieval failed: {str(e)}")

@router.get("/v2-mappings/embeddings/compare/{mapping_id}")
async def compare_v2_mapping_with_query(
    mapping_id: str, 
    query: str,
    db: Session = Depends(get_db)
):
    """Compare a V2 FHIR mapping's embedding with a query for similarity testing"""
    try:

        mapping = db.query(V2FHIRdata).filter(V2FHIRdata.id == mapping_id).first()
        if not mapping:
            raise HTTPException(status_code=404, detail="V2 FHIR mapping not found")
        

        chroma = get_chroma_instance()
        if not chroma.is_available():
            raise HTTPException(status_code=503, detail="ChromaDB not available")
        
        collection = chroma.get_collection()
        results = collection.get(ids=[mapping_id], include=['embeddings', 'documents', 'metadatas'])
        
        if not results['ids']:
            raise HTTPException(status_code=404, detail="Mapping embedding not found in ChromaDB")
  
        etl_service = V2FHIRETLService()
        query_embedding = etl_service.model.encode([query])[0].tolist()
        mapping_embedding = results['embeddings'][0]
        
     
        import numpy as np
        
        query_vec = np.array(query_embedding)
        mapping_vec = np.array(mapping_embedding)
        
    
        similarity = np.dot(query_vec, mapping_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(mapping_vec))
        
        return {
            "mapping_id": mapping_id,
            "query": query,
            "similarity_score": float(similarity),
            "mapping_document": results['documents'][0] if results['documents'] else None,
            "mapping_metadata": results['metadatas'][0] if results['metadatas'] else None,
            "mapping_info": {
                "resource": mapping.resource,
                "fhir_detail": mapping.fhir_detail,
                "hl7v2_field": mapping.hl7v2_field
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")

@router.post("/v2-mappings/embeddings/batch-compare")
async def batch_compare_v2_mappings_with_query(
    query: str,
    mapping_ids: List[str],
    db: Session = Depends(get_db)
):
    """Batch compare multiple V2 FHIR mappings with a query"""
    try:
        chroma = get_chroma_instance()
        if not chroma.is_available():
            raise HTTPException(status_code=503, detail="ChromaDB not available")
        

        etl_service = V2FHIRETLService()
        query_embedding = etl_service.model.encode([query])[0].tolist()
        

        collection = chroma.get_collection()
        results = collection.get(ids=mapping_ids, include=['embeddings', 'documents', 'metadatas'])
        
        comparisons = []
        import numpy as np
        query_vec = np.array(query_embedding)
        
        for i, mapping_id in enumerate(results['ids']):
            try:
                mapping_embedding = results['embeddings'][i]
                mapping_vec = np.array(mapping_embedding)
         
                similarity = np.dot(query_vec, mapping_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(mapping_vec))
                
                comparisons.append({
                    "mapping_id": mapping_id,
                    "similarity_score": float(similarity),
                    "document": results['documents'][i] if results['documents'] else None,
                    "metadata": results['metadatas'][i] if results['metadatas'] else None
                })
                
            except Exception as e:
                print(f"Error comparing mapping {mapping_id}: {e}")
                comparisons.append({
                    "mapping_id": mapping_id,
                    "similarity_score": 0.0,
                    "error": str(e)
                })
        
     
        comparisons.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
        
        return {
            "query": query,
            "total_comparisons": len(comparisons),
            "comparisons": comparisons
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch comparison failed: {str(e)}")