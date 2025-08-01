# backend/services/database_search_service.py
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from services.performance_log import create_performance_log
from models.database.models import Profile, Dataset
from config.redis_cache import RedisQueryCache
from config.chroma import ChromaConfig
import uuid


class SearchService:   
    def __init__(self):
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')

        except Exception as e:
            print(f"Error loading SentenceTransformer: {e}")
            self.model = None
        try:
            self.chroma_config = ChromaConfig()
            self.collection = self.chroma_config.collection
            

        except Exception as e:
            print(f"Error initializing Chroma: {e}")
            self.collection = None
            self.chroma_config = None


    def semantic_search(self, query: str, top_k: int = 10, db: Session = None, filters: Optional[Dict] = None) -> List[Dict]:
   
        if not self.collection:
            raise ValueError("Chroma collection not available for search")
        
        if not db:
            raise ValueError("Database session required")
        
        redis_client = RedisQueryCache()
        results = []
        profile_dict = {}
        query_normalized = query.lower().strip()
        cached_feedback = redis_client.get_cached_feedback(query_normalized)
        similarity_scores = []  
        profile_ids = []  
        
        try:
        
            if cached_feedback:
                print(f"Cache hit for query: {query_normalized}") 
                profile_ids = [f['profile_id'] for f in cached_feedback]
                print(f"Cached profile IDs: {profile_ids}")
                profiles = db.query(Profile).filter(
                    Profile.id.in_(profile_ids),
                    Profile.is_active == True 
                ).limit(top_k).all()
                profile_dict = {p.id: p for p in profiles}
                
                # Create default similarity scores for cached results
                similarity_scores = [1.0] * len(profile_ids)  # Default score for cached results
            
            else:        
            
                if not self.model:
                    print("SentenceTransformer model not available")
                    return []

                query_embedding = self.model.encode([query])[0].tolist()
                
                where_clause = {'is_active': True}
                if filters:
                    for key, value in filters.items():
                        if key in ['dataset_id', 'resource_type', 'category', 'keywords']:
                            where_clause[key] = value

                search_results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    where=where_clause if where_clause else None,
                    include=['metadatas', 'distances']
                )
                
     
                if not search_results.get('ids') or not search_results['ids'] or not search_results['ids'][0]:
                    print("No search results found")
                    return []
                
                profile_ids = search_results['ids'][0]
                distances = search_results.get('distances', [[]])[0]  # Safe access with default
                
                # Ensure distances list exists and has proper length
                if distances:
                    similarity_scores = [1 / (1 + distance) for distance in distances]
                else:
                    similarity_scores = [0.5] * len(profile_ids)  # Default scores if no distances
                

                profiles = db.query(Profile).filter(
                    Profile.id.in_(profile_ids),
                    Profile.is_active == True
                ).all()
                profile_dict = {p.id: p for p in profiles}
            

            if len(similarity_scores) != len(profile_ids):
                print(f"Warning: Score length ({len(similarity_scores)}) != Profile ID length ({len(profile_ids)})")
    
                if len(similarity_scores) < len(profile_ids):
                    similarity_scores.extend([0.5] * (len(profile_ids) - len(similarity_scores)))
                else:
                    similarity_scores = similarity_scores[:len(profile_ids)]
        
            for i, profile_id in enumerate(profile_ids):
                if profile_id in profile_dict:
                    profile = profile_dict[profile_id]
                    
                    similarity_score = similarity_scores[i] if i < len(similarity_scores) else "n/a"
                    
                    results.append({
                        'id': profile.id,
                        'name': profile.name,
                        'description': profile.description or "",
                        'resource_type': profile.resource_type or "Unknown",
                        'category': profile.category or "Unknown",
                        'version': profile.version or "",
                        'keywords': profile.keywords or [],
                        'must_support': profile.must_support or [],
                        'must_have': profile.must_have or [],
                        'invariants': profile.invariants or [],
                        'resource_url': profile.resource_url,
                        'similarity_score': similarity_score,
                        'dataset_id': profile.dataset_id,
                        'oid': profile.oid or "",
                        'use_contexts': profile.use_contexts or [],
                        'fhir_resource': profile.fhir_resource or {},
                        'fhir_searchable_text': profile.fhir_searchable_text or []
                    })
            
            return results
        
        except Exception as e:
            print(f"Error in semantic search: {e}")
            import traceback
            traceback.print_exc()  # This will help debug the exact error
            return []

    def hybrid_search(self, query: str, top_k: int = 10, db: Session = None,
                     semantic_weight: float = 0.7, filters: Optional[Dict] = None) -> List[Dict]:
        """
        Combine semantic search (Chroma) with traditional text search (PostgreSQL)
        """
        if not db:
            raise ValueError("Database session required")
            
        results_map = {}
        
        if semantic_weight > 0 and self.collection:
            try:

                semantic_results = self.semantic_search(query, top_k, db, filters)
                for result in semantic_results:
                    results_map[result['id']] = {
                        **result,
                        'semantic_score': result['similarity_score'],
                        'text_score': 0.0
                    }
            except Exception as e:
                print(f"Semantic search failed in hybrid mode: {e}")

        if semantic_weight < 1.0:
            try:
                text_query = db.query(Profile).filter(
                    Profile.is_active == True,
                    Profile.search_text.ilike(f"%{query}%")
                )
                
                # Apply filters
                if filters:
                    if filters.get('dataset_id'):
                        text_query = text_query.filter(Profile.dataset_id == filters['dataset_id'])
                    if filters.get('resource_type'):
                        text_query = text_query.filter(Profile.resource_type == filters['resource_type'])
                    if filters.get('category'):
                        text_query = text_query.filter(Profile.category == filters['category'])
                
                text_profiles = text_query.limit(top_k).all()
                
                for profile in text_profiles:
                    if profile.id in results_map:
                        results_map[profile.id]['text_score'] = 0.8
                    else:
                        results_map[profile.id] = {
                            'id': profile.id,
                            'name': profile.name,
                            'description': profile.description or "",
                            'resource_type': profile.resource_type or "Unknown",
                            'category': profile.category or "Unknown",
                            'version': profile.version or "",
                            'keywords': profile.keywords or [],
                            'must_support': profile.must_support or [],
                            'must_have': profile.must_have or [],
                            'invariants': profile.invariants or [],
                            'resource_url': profile.resource_url,
                            'dataset_id': profile.dataset_id,
                            'oid': profile.oid,
                            'use_contexts': profile.use_contexts or [],
                            'fhir_resource': profile.fhir_resource or {},
                            'fhir_searchable_text': profile.fhir_searchable_text or [],
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
        )[:top_k]
        
        return sorted_results

    def traditional_search(self, query: str, db: Session = None, top_k: int = 10, 
                          filters: Optional[Dict] = None) -> List[Dict]:
        """
        Traditional text-based search using PostgreSQL only
        """
        if not db:
            raise ValueError("Database session required")
            
        try:
   
            text_query = db.query(Profile).filter(
                Profile.is_active == True,
                Profile.search_text.ilike(f"%{query}%")
            )

            if filters:
                if filters.get('dataset_id'):
                    text_query = text_query.filter(Profile.dataset_id == filters['dataset_id'])
                if filters.get('resource_type'):
                    text_query = text_query.filter(Profile.resource_type == filters['resource_type'])
                if filters.get('category'):
                    text_query = text_query.filter(Profile.category == filters['category'])
            
            profiles = text_query.limit(top_k).all()
            
            # Format results
            results = []
            for profile in profiles:
                results.append({
                    'id': profile.id,
                    'name': profile.name,
                    'description': profile.description or "",
                    'resource_type': profile.resource_type or "Unknown",
                    'category': profile.category or "Unknown",
                    'version': profile.version or "",
                    'keywords': profile.keywords or [],
                    'must_support': profile.must_support or [],
                    'must_have': profile.must_have or [],
                    'invariants': profile.invariants or [],
                    'resource_url': profile.resource_url,
                    'similarity_score': 1.0,  # No similarity score for text search
                    'dataset_id': profile.dataset_id,
                    'oid': profile.oid,
                    'use_contexts': profile.use_contexts or [],
                    'fhir_resource': profile.fhir_resource or {},
                    'fhir_searchable_text': profile.fhir_searchable_text or []
                })
            
            """if results:
                for res in results:
                    create_performance_log(
                        profile_id=res['id'],
                        query_text=query,
                        profile_name=res['name'],
                        profile_oid=res['oid'],
                        keywords=res['keywords'],
                        db=db
                    )    """
            return results
            
        except Exception as e:
            print(f"Traditional search failed: {e}")
            return []

    def search_by_filters_only(self, db: Session = None, filters: Dict = None, top_k: int = 100) -> List[Dict]:
        """
        Search using only filters (no text query)
        """
        if not db:
            raise ValueError("Database session required")
            
        try:
            query = db.query(Profile).filter(Profile.is_active == True)
            
            if filters:
                if filters.get('dataset_id'):
                    query = query.filter(Profile.dataset_id == filters['dataset_id'])
                if filters.get('resource_type'):
                    query = query.filter(Profile.resource_type == filters['resource_type'])
                if filters.get('category'):
                    query = query.filter(Profile.category == filters['category'])
                if filters.get('keywords'):
                    # Search for any of the provided keywords
                    keyword_filter = False
                    for keyword in filters['keywords']:
                        keyword_filter |= Profile.keywords.op('?')(keyword)
                    if keyword_filter:
                        query = query.filter(keyword_filter)
            
            profiles = query.limit(top_k).all()
            
            results = []
            for profile in profiles:
                results.append({
                    'id': profile.id,
                    'name': profile.name,
                    'description': profile.description or "",
                    'resource_type': profile.resource_type or "Unknown",
                    'category': profile.category or "Unknown",
                    'version': profile.version or "",
                    'keywords': profile.keywords or [],
                    'must_support': profile.must_support or [],
                    'must_have': profile.must_have or [],
                    'invariants': profile.invariants or [],
                    'resource_url': profile.resource_url,
                    'similarity_score': 1.0,
                    'dataset_id': profile.dataset_id,
                    'oid': profile.oid,
                    'use_contexts': profile.use_contexts or []
                })
            
            return results
            
        except Exception as e:
            print(f"Filter-only search failed: {e}")
            return []

    def get_search_stats(self, db: Session = None) -> Dict[str, Any]:
        """Get search statistics"""
        if not db:
            raise ValueError("Database session required")
            
        active_profiles = db.query(Profile).filter(Profile.is_active == True).count()
        total_profiles = db.query(Profile).count()

        active_dataset = db.query(Dataset).filter(Dataset.status == "active").first()
        
        return {
            "active_profiles": active_profiles,
            "total_profiles": total_profiles,
            "active_dataset": {
                "id": active_dataset.id if active_dataset else None,
                "name": active_dataset.name if active_dataset else None,
                "activated_date": active_dataset.activated_date if active_dataset else None
            } if active_dataset else None
        }