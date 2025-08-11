# backend/services/search/v2_fhir_search_service.py
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from models.database.fhir_v2_model import V2FHIRdata, V2FHIRdataset
from services.ultility.named_entity_removal_service import PHIQueryScrubber
from config.redis_cache import RedisQueryCache
from config.chroma import get_chroma_instance
import uuid
import logging
from datetime import datetime

redis_client = RedisQueryCache()

class V2FHIRSearchService:   
    def __init__(self):
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            print(f"Error loading SentenceTransformer: {e}")
            self.model = None
        try:
            self.chroma_config = get_chroma_instance()
            self.collection = self.chroma_config.get_collection()
            self.phi_scrubber = PHIQueryScrubber(model_name="en_core_web_sm")
        except Exception as e:
            print(f"V2FHIRSearchService failed to connect to Chroma: {e}")
            self.collection = None
            self.chroma_config = None

    def semantic_search(self, query: str, limit: int = 10, db: Session = None, filters: Optional[Dict] = None) -> List[Dict]:
        """Semantic search for V2-FHIR mappings"""
        # PHI scrubbing
        phi_scrubbed_query = self.phi_scrubber.scrub_query(query)
        print(f"PHI scrubbed query: '{phi_scrubbed_query}'")
        
        if not phi_scrubbed_query or not phi_scrubbed_query.strip():
            logging.warning(f"Query was entirely PHI or empty after scrubbing: '{query}'")
            return []
        
        if phi_scrubbed_query != query:
            logging.info(f"PHI scrubbed from query. Original: '{query}' -> Scrubbed: '{phi_scrubbed_query}'")

        if not self.collection:
            raise ValueError("Chroma collection not available for search")
        
        if not db:
            raise ValueError("Database session required")

        results = []
        mapping_dict = {}
        query_normalized = phi_scrubbed_query.lower().strip()
        
        # Check cache
        cached_feedback = redis_client.get_cached_feedback(f"v2fhir_{query_normalized}")
        similarity_scores = []
        mapping_ids = []        
        mappings = []
        
        try:
            if cached_feedback:
                print(f"Cache hit for V2-FHIR query: {query_normalized}") 
                cached_mapping_ids = [f['mapping_id'] for f in cached_feedback]

                mappings = db.query(V2FHIRdata).filter(
                    V2FHIRdata.id.in_(cached_mapping_ids),
                    V2FHIRdata.is_active == True
                ).limit(limit).all()
                
                active_mapping_ids = [m.id for m in mappings]
                
                inactive_count = len(cached_mapping_ids) - len(active_mapping_ids)
                if inactive_count > 0:
                    print(f"Found {inactive_count} inactive mappings in cache, filtering them out")
                    if inactive_count / len(cached_mapping_ids) > 0.3:  
                        print(f"Cache has too many inactive mappings, invalidating...")
                        redis_client.clear_cache(f"v2fhir_{query_normalized}")
                        cached_feedback = None

                if cached_feedback and len(active_mapping_ids) > 0:
                    mapping_ids = active_mapping_ids
                    mapping_dict = {m.id: m for m in mappings}
                    similarity_scores = [1.0] * len(mapping_ids)
                    print(f"Using {len(mapping_ids)} active mappings from cache")
                else:
                    cached_feedback = None

            if not cached_feedback:
                print(f"Cache miss for V2-FHIR query: {query_normalized}, performing vector search")
                
                if not self.model:
                    print("SentenceTransformer model not available")
                    return []

                query_embedding = self.model.encode([phi_scrubbed_query])[0].tolist()
                
     
                where_clause = {'is_active': True}
    

                search_results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=limit,
                    where=where_clause if where_clause else None,
                    include=['metadatas', 'distances', 'embeddings']
                )
                
                if not search_results['ids'] or not search_results['ids'][0]:
                    return []
                
                print(f"Chroma search results: {search_results}")
                
                mapping_ids = search_results['ids'][0]
                distances = search_results.get('distances', [[]])[0]
                
                if distances:
                    similarity_scores = [1 / (1 + distance) for distance in distances]
                else:
                    similarity_scores = [0.5] * len(mapping_ids)
                
                mappings = db.query(V2FHIRdata).filter(
                    V2FHIRdata.id.in_(mapping_ids),
                    V2FHIRdata.is_active == True
                ).all()
                
                mapping_dict = {m.id: m for m in mappings}
                
                if len(mapping_ids) > 0:
                    cache_data = [{'mapping_id': mid} for mid in mapping_ids]
                    redis_client.set_cached_feedback(f"v2fhir_{query_normalized}", cache_data, 3600)
                    print(f"Cached {len(mapping_ids)} V2-FHIR results for future use")

            # Ensure score and ID arrays match
            if len(similarity_scores) != len(mapping_ids):
                print(f"Score length ({len(similarity_scores)}) != Mapping ID length ({len(mapping_ids)})")
                if len(similarity_scores) < len(mapping_ids):
                    similarity_scores.extend([0.5] * (len(mapping_ids) - len(similarity_scores)))
                else:
                    similarity_scores = similarity_scores[:len(mapping_ids)]

            # Build results
            for i, mapping_id in enumerate(mapping_ids):
                if mapping_id in mapping_dict:
                    mapping = mapping_dict[mapping_id]
                    results.append({
                        'id': mapping.id,
                        'local_id': mapping.local_id,
                        'resource': mapping.resource,
                        'sub_detail': mapping.sub_detail or "",
                        'fhir_detail': mapping.fhir_detail or "",
                        'fhir_version': mapping.fhir_version or "",
                        'hl7v2_field': mapping.hl7v2_field or "",
                        'hl7v2_field_detail': mapping.hl7v2_field_detail or "",
                        'hl7v2_field_version': mapping.hl7v2_field_version or "",
                        'dataset_id': mapping.V2FHIRdataset_id,
                        'is_active': mapping.is_active,
                        'created_date': mapping.created_date,
                        'similarity_score': similarity_scores[i]
                    })
                else:
                    print(f"Mapping {mapping_id} not found in database or inactive")
            
            print(f"Returning {len(results)} V2-FHIR mapping results")
            return results
            
        except Exception as e:
            print(f"Error in V2-FHIR semantic search: {e}")
            return []

    def hybrid_search(self, query: str, limit: int = 10, db: Session = None,
                     semantic_weight: float = 0.7, filters: Optional[Dict] = None) -> List[Dict]:
        """Combine semantic search with traditional text search for V2-FHIR mappings"""
        if not db:
            raise ValueError("Database session required")
            
        results_map = {}
        
        # Semantic search component
        if semantic_weight > 0 and self.collection:
            try:
                semantic_results = self.semantic_search(query, limit, db, filters)
                for result in semantic_results:
                    results_map[result['id']] = {
                        **result,
                        'semantic_score': result['similarity_score'],
                        'text_score': 0.0
                    }
            except Exception as e:
                print(f"Semantic search failed in hybrid mode: {e}")

        # Traditional text search component
        if semantic_weight < 1.0:
            try:
                text_query = db.query(V2FHIRdata).filter(
                    V2FHIRdata.is_active == True
                ).filter(
                    # Search across V2-FHIR mapping fields
                    (V2FHIRdata.resource.ilike(f"%{query}%")) |
                    (V2FHIRdata.fhir_detail.ilike(f"%{query}%")) |
                    (V2FHIRdata.hl7v2_field.ilike(f"%{query}%")) |
                    (V2FHIRdata.hl7v2_field_detail.ilike(f"%{query}%")) |
                    (V2FHIRdata.sub_detail.ilike(f"%{query}%")) |
                    (V2FHIRdata.local_id.ilike(f"%{query}%"))
                )
                
                # Apply filters
                if filters:
                    if filters.get('resource_type'):
                        text_query = text_query.filter(V2FHIRdata.resource == filters['resource_type'])
                    if filters.get('fhir_version'):
                        text_query = text_query.filter(V2FHIRdata.fhir_version == filters['fhir_version'])
                    if filters.get('hl7v2_version'):
                        text_query = text_query.filter(V2FHIRdata.hl7v2_field_version == filters['hl7v2_version'])
                    if filters.get('dataset_id'):
                        text_query = text_query.filter(V2FHIRdata.V2FHIRdataset_id == filters['dataset_id'])
                
                text_mappings = text_query.limit(limit).all()
                
                for mapping in text_mappings:
                    if mapping.id in results_map:
                        results_map[mapping.id]['text_score'] = 0.8
                    else:
                        results_map[mapping.id] = {
                            'id': mapping.id,
                            'local_id': mapping.local_id,
                            'resource': mapping.resource,
                            'sub_detail': mapping.sub_detail or "",
                            'fhir_detail': mapping.fhir_detail or "",
                            'fhir_version': mapping.fhir_version or "",
                            'hl7v2_field': mapping.hl7v2_field or "",
                            'hl7v2_field_detail': mapping.hl7v2_field_detail or "",
                            'hl7v2_field_version': mapping.hl7v2_field_version or "",
                            'dataset_id': mapping.V2FHIRdataset_id,
                            'is_active': mapping.is_active,
                            'created_date': mapping.created_date,
                            'semantic_score': 0.0,
                            'text_score': 0.8
                        }
            except Exception as e:
                print(f"Text search failed in hybrid mode: {e}")
        
        # Calculate hybrid scores and sort
        for result in results_map.values():
            result['similarity_score'] = (
                semantic_weight * result['semantic_score'] + 
                (1 - semantic_weight) * result['text_score']
            )
        
        sorted_results = sorted(
            results_map.values(),
            key=lambda x: x['similarity_score'],
            reverse=True
        )[:limit]
        
        return sorted_results

    def traditional_search(self, query: str, db: Session = None, limit: int = 10, 
                          filters: Optional[Dict] = None) -> List[Dict]:
        """Traditional text-based search for V2-FHIR mappings using PostgreSQL only"""
        if not db:
            raise ValueError("Database session required")
            
        try:
            text_query = db.query(V2FHIRdata).filter(
                V2FHIRdata.is_active == True
            ).filter(
                # Search across V2-FHIR mapping fields
                (V2FHIRdata.resource.ilike(f"%{query}%")) |
                (V2FHIRdata.fhir_detail.ilike(f"%{query}%")) |
                (V2FHIRdata.hl7v2_field.ilike(f"%{query}%")) |
                (V2FHIRdata.hl7v2_field_detail.ilike(f"%{query}%")) |
                (V2FHIRdata.sub_detail.ilike(f"%{query}%")) |
                (V2FHIRdata.local_id.ilike(f"%{query}%")) |
                (V2FHIRdata.fhir_version.ilike(f"%{query}%")) |
                (V2FHIRdata.hl7v2_field_version.ilike(f"%{query}%"))
            )
            
            # Apply filters
            if filters:
                if filters.get('resource_type'):
                    text_query = text_query.filter(V2FHIRdata.resource == filters['resource_type'])
                if filters.get('fhir_version'):
                    text_query = text_query.filter(V2FHIRdata.fhir_version == filters['fhir_version'])
                if filters.get('hl7v2_version'):
                    text_query = text_query.filter(V2FHIRdata.hl7v2_field_version == filters['hl7v2_version'])
                if filters.get('dataset_id'):
                    text_query = text_query.filter(V2FHIRdata.V2FHIRdataset_id == filters['dataset_id'])
            
            mappings = text_query.limit(limit).all()
            
            # Format results
            results = []
            for mapping in mappings:
                results.append({
                    'id': mapping.id,
                    'local_id': mapping.local_id,
                    'resource': mapping.resource,
                    'sub_detail': mapping.sub_detail or "",
                    'fhir_detail': mapping.fhir_detail or "",
                    'fhir_version': mapping.fhir_version or "",
                    'hl7v2_field': mapping.hl7v2_field or "",
                    'hl7v2_field_detail': mapping.hl7v2_field_detail or "",
                    'hl7v2_field_version': mapping.hl7v2_field_version or "",
                    'dataset_id': mapping.V2FHIRdataset_id,
                    'is_active': mapping.is_active,
                    'created_date': mapping.created_date,
                    'similarity_score': 1.0  # No similarity score for text search
                })
            
            return results
            
        except Exception as e:
            print(f"Traditional V2-FHIR search failed: {e}")
            return []

    def search_by_filters_only(self, db: Session = None, filters: Dict = None, limit: int = 100) -> List[Dict]:
        """Search V2-FHIR mappings using only filters (no text query)"""
        if not db:
            raise ValueError("Database session required")
            
        try:
            query = db.query(V2FHIRdata).filter(V2FHIRdata.is_active == True)
            
            if filters:
                if filters.get('dataset_id'):
                    query = query.filter(V2FHIRdata.V2FHIRdataset_id == filters['dataset_id'])
                if filters.get('resource_type'):
                    query = query.filter(V2FHIRdata.resource == filters['resource_type'])
                if filters.get('fhir_version'):
                    query = query.filter(V2FHIRdata.fhir_version == filters['fhir_version'])
                if filters.get('hl7v2_version'):
                    query = query.filter(V2FHIRdata.hl7v2_field_version == filters['hl7v2_version'])
                if filters.get('hl7v2_field'):
                    query = query.filter(V2FHIRdata.hl7v2_field.ilike(f"%{filters['hl7v2_field']}%"))
            
            mappings = query.limit(limit).all()
            
            results = []
            for mapping in mappings:
                results.append({
                    'id': mapping.id,
                    'local_id': mapping.local_id,
                    'resource': mapping.resource,
                    'sub_detail': mapping.sub_detail or "",
                    'fhir_detail': mapping.fhir_detail or "",
                    'fhir_version': mapping.fhir_version or "",
                    'hl7v2_field': mapping.hl7v2_field or "",
                    'hl7v2_field_detail': mapping.hl7v2_field_detail or "",
                    'hl7v2_field_version': mapping.hl7v2_field_version or "",
                    'dataset_id': mapping.V2FHIRdataset_id,
                    'is_active': mapping.is_active,
                    'created_date': mapping.created_date,
                    'similarity_score': 1.0
                })
            
            return results
            
        except Exception as e:
            print(f"Filter-only V2-FHIR search failed: {e}")
            return []

    def find_similar_mappings(self, mapping_id: str, limit: int = 5, db: Session = None) -> List[Dict]:
        """Find mappings similar to a given mapping"""
        if not self.collection or not db:
            return []
        
        try:
            # Get the source mapping
            source_mapping = db.query(V2FHIRdata).filter(V2FHIRdata.id == mapping_id).first()
            if not source_mapping:
                return []
            
            # Create search text from the source mapping
            search_text = f"{source_mapping.resource} {source_mapping.fhir_detail or ''} {source_mapping.hl7v2_field or ''}"
            
            # Use semantic search to find similar mappings
            results = self.semantic_search(search_text, limit + 1, db)  # +1 to exclude self
            
            # Remove the source mapping from results
            similar_results = [r for r in results if r['id'] != mapping_id][:limit]
            
            return similar_results
            
        except Exception as e:
            print(f"Error finding similar mappings: {e}")
            return []

    def get_search_stats(self, db: Session = None) -> Dict[str, Any]:
        """Get search statistics for V2-FHIR mappings and datasets"""
        if not db:
            raise ValueError("Database session required")
            
        try:
            active_mappings = db.query(V2FHIRdata).filter(V2FHIRdata.is_active == True).count()
            total_mappings = db.query(V2FHIRdata).count()
            
            active_datasets = db.query(V2FHIRdataset).filter(V2FHIRdataset.status == "active").all()
            total_datasets = db.query(V2FHIRdataset).count()
            
            return {
                "active_mappings": active_mappings,
                "total_mappings": total_mappings,
                "active_datasets_count": len(active_datasets),
                "total_datasets": total_datasets,
                "active_datasets": [
                    {
                        "id": ds.id,
                        "name": ds.name,
                        "activated_date": ds.activated_date,
                        "record_count": ds.record_count
                    } for ds in active_datasets
                ]
            }
        except Exception as e:
            print(f"Error getting V2-FHIR search stats: {e}")
            return {
                "active_mappings": 0,
                "total_mappings": 0,
                "active_datasets_count": 0,
                "total_datasets": 0,
                "active_datasets": []
            }

    def get_detailed_search_stats(self, db: Session = None) -> Dict[str, Any]:
        """Get detailed search statistics including breakdowns"""
        if not db:
            raise ValueError("Database session required")
            
        try:
            base_stats = self.get_search_stats(db)
            
            # Get resource type distribution
            resource_types = db.query(V2FHIRdata.resource, func.count(V2FHIRdata.id)).filter(
                V2FHIRdata.is_active == True
            ).group_by(V2FHIRdata.resource).all()
            
            # Get FHIR version distribution
            fhir_versions = db.query(V2FHIRdata.fhir_version, func.count(V2FHIRdata.id)).filter(
                V2FHIRdata.is_active == True,
                V2FHIRdata.fhir_version.isnot(None)
            ).group_by(V2FHIRdata.fhir_version).all()
            
            # Get HL7 V2 version distribution
            hl7v2_versions = db.query(V2FHIRdata.hl7v2_field_version, func.count(V2FHIRdata.id)).filter(
                V2FHIRdata.is_active == True,
                V2FHIRdata.hl7v2_field_version.isnot(None)
            ).group_by(V2FHIRdata.hl7v2_field_version).all()
            
            # Get most recent update
            latest_mapping = db.query(V2FHIRdata).filter(
                V2FHIRdata.is_active == True
            ).order_by(desc(V2FHIRdata.created_date)).first()
            
            base_stats.update({
                "resource_types": [rt[0] for rt in resource_types if rt[0]],
                "fhir_versions": [fv[0] for fv in fhir_versions if fv[0]],
                "hl7v2_versions": [hv[0] for hv in hl7v2_versions if hv[0]],
                "last_updated": latest_mapping.created_date.isoformat() if latest_mapping else None,
                "resource_distribution": [{"resource": rt[0], "count": rt[1]} for rt in resource_types],
                "fhir_version_distribution": [{"version": fv[0], "count": fv[1]} for fv in fhir_versions],
                "hl7v2_version_distribution": [{"version": hv[0], "count": hv[1]} for hv in hl7v2_versions]
            })
            
            return base_stats
            
        except Exception as e:
            print(f"Error getting detailed V2-FHIR search stats: {e}")
            return self.get_search_stats(db)

    def get_available_filters(self, db: Session = None) -> Dict[str, List[str]]:
        """Get available filter options"""
        if not db:
            return {}
        
        try:
            resource_types = db.query(V2FHIRdata.resource).filter(
                V2FHIRdata.is_active == True,
                V2FHIRdata.resource.isnot(None)
            ).distinct().all()
            
            fhir_versions = db.query(V2FHIRdata.fhir_version).filter(
                V2FHIRdata.is_active == True,
                V2FHIRdata.fhir_version.isnot(None)
            ).distinct().all()
            
            hl7v2_versions = db.query(V2FHIRdata.hl7v2_field_version).filter(
                V2FHIRdata.is_active == True,
                V2FHIRdata.hl7v2_field_version.isnot(None)
            ).distinct().all()
            
            datasets = db.query(V2FHIRdataset.id, V2FHIRdataset.name).filter(
                V2FHIRdataset.status == "active"
            ).all()
            
            return {
                "resource_types": sorted([rt[0] for rt in resource_types if rt[0]]),
                "fhir_versions": sorted([fv[0] for fv in fhir_versions if fv[0]]),
                "hl7v2_versions": sorted([hv[0] for hv in hl7v2_versions if hv[0]]),
                "datasets": [{"id": ds[0], "name": ds[1]} for ds in datasets]
            }
        except Exception as e:
            print(f"Error getting available filters: {e}")
            return {}

  