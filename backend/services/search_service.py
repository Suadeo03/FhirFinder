# backend/services/database_search_service.py
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from services.performance_log import create_performance_log
from models.database.models import Profile, Dataset
from config.redis_cache import RedisQueryCache
from config.chroma import get_chroma_instance, is_chroma_available
import uuid
from services.training_service import FeedbackTraining
redis_client = RedisQueryCache()

class SearchService:   
    def __init__(self):
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            print(f"Error loading SentenceTransformer: {e}")
            self.model = None
        try:
            self.chroma_config = get_chroma_instance()
            self.collection = self.chroma_config.get_collection()
            print("SearchService connected to Chroma singleton")
            
            # Initialize feedback training system with same singleton
            self.feedback_trainer = FeedbackTraining()
        except Exception as e:
            print(f"SearchService failed to connect to Chroma: {e}")
            self.collection = None
            self.chroma_config = None
            self.feedback_trainer = None

    def semantic_search(self, query: str, top_k: int = 10, db: Session = None, 
                    filters: Optional[Dict] = None) -> List[Dict]:
        """Fixed semantic search using Chroma singleton"""
        

        if not is_chroma_available():
            raise ValueError("Chroma singleton not available for search")
        
        if not db:
            raise ValueError("Database session required")


        try:
            collection = self.collection
        except Exception as e:
            print(f"Failed to get collection from singleton: {e}")
            return []

        results = []
        profile_dict = {}
        query_normalized = query.lower().strip()
        
        cached_feedback = redis_client.get_cached_feedback(query_normalized)
        similarity_scores = []
        profile_ids = []

        try:

            if cached_feedback:
                print(f"Cache hit for query: {query_normalized}") 
                cached_profile_ids = [f['profile_id'] for f in cached_feedback]

                profiles = db.query(Profile).filter(
                    Profile.id.in_(cached_profile_ids),
                    Profile.is_active == True
                ).limit(top_k).all()
                
                active_profile_ids = [p.id for p in profiles]
       
                inactive_count = len(cached_profile_ids) - len(active_profile_ids)
                if inactive_count > 0:
                    print(f"Found {inactive_count} inactive profiles in cache, filtering them out")
                    if inactive_count / len(cached_profile_ids) > 0.3:  
                        print(f"Cache has too many inactive profiles, invalidating...")
                        redis_client.clear_cache(query_normalized)
                        cached_feedback = None  

                if cached_feedback and len(active_profile_ids) > 0:
                    profile_ids = active_profile_ids
                    profile_dict = {p.id: p for p in profiles}
                    similarity_scores = [1.0] * len(profile_ids)
                    print(f"Using {len(profile_ids)} active profiles from cache")
                else:
                    cached_feedback = None  

            if not cached_feedback:
                print(f"Cache miss or invalidated for query: {query_normalized}, performing vector search")
                
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
                    include=['metadatas', 'distances', 'embeddings']
                )
                print(f"Chroma search results: {search_results}")
                if not search_results.get('ids') or not search_results['ids'] or not search_results['ids'][0]:
                    print("No results returned from Chroma")
                    return []
                
                profile_ids = search_results['ids'][0]
                distances = search_results.get('distances', [[]])[0]

                
                print(f"Chroma returned {len(profile_ids)} profiles: {profile_ids[:3]}...")
                
                if distances:
                    similarity_scores = [1 / (1 + distance) for distance in distances]
                else:
                    similarity_scores = [0.5] * len(profile_ids)
                profiles = db.query(Profile).filter(
                    Profile.id.in_(profile_ids),
                    Profile.is_active == True
                ).all()
                profile_dict = {p.id: p for p in profiles}
                
                print(f"Found {len(profiles)} active profiles in database")
                
                if len(profile_ids) > 0:
                    cache_data = [{'profile_id': pid} for pid in profile_ids]
                    redis_client.set_cached_feedback(query_normalized, cache_data, 3600)
                    print(f"Cached {len(profile_ids)} results for future use")

            if len(similarity_scores) != len(profile_ids):
                print(f"Score length ({len(similarity_scores)}) != Profile ID length ({len(profile_ids)})")
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
                else:
                    print(f"Profile {profile_id} not found in database or inactive")
            
            print(f"Returning {len(results)} total results")
            return results
        
        except Exception as e:
            print(f"Error in semantic search: {e}")
            import traceback
            traceback.print_exc()
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
    def record_feedback(self, query: str, profile_id: str, feedback_type: str, 
                       user_id: str, session_id: str, original_score: float, 
                       db: Session, context_info: Optional[Dict] = None):
   
        if not self.feedback_trainer:
            print("Feedback trainer not available")
            return {"status": "error", "message": "Feedback system not initialized"}
        try:
            self.feedback_trainer.record_user_feedback(
                query, profile_id, feedback_type, user_id, session_id, 
                original_score, db, context_info
            )
            self._handle_cache_invalidation(query, feedback_type)
            return {
                "status": "success", 
                "message": f"{feedback_type} feedback recorded and embeddings updated"
            }
        except Exception as e:
            print(f"Error recording feedback: {e}")
            return {"status": "error", "message": str(e)}

    def _handle_cache_invalidation(self, query: str, feedback_type: str):
        """Smart cache invalidation based on feedback type"""
        try:

            query_normalized = query.lower().strip()
            
            if feedback_type == 'negative':
                redis_client.clear_cache(query_normalized)
                print(f"Cache invalidated for negative feedback on query: {query}")
            elif feedback_type == 'positive':
                cached_data = redis_client.get_cached_feedback(query_normalized)
                if cached_data:
                    redis_client.set_cached_feedback(query_normalized, cached_data, 86400)
                    print(f"Cache extended for positive feedback on query: {query}")
        except Exception as e:
            print(f"Error handling cache invalidation: {e}")

    def run_batch_retraining(self, db: Session, days_back: int = 30):
        """Run batch retraining of embeddings"""
        if self.feedback_trainer:
            self.feedback_trainer.batch_retrain_embeddings(db, days_back)
            try:
                redis_client.clear_all_cache()
                print("All cache cleared after batch retraining")
            except Exception as e:
                print(f"Error clearing cache after retraining: {e}")
        else:
            print("Feedback trainer not available for batch retraining")

    def get_learning_stats(self, db: Session) -> Dict:
        """Get statistics about the learning system"""
        if self.feedback_trainer:
            return self.feedback_trainer.get_learning_stats(db)
        else:
            return {"error": "Feedback trainer not available"}
    def get_feedback_stats(self, db: Session, days: int = 30) -> Dict:
        if self.feedback_trainer:
            return self.feedback_trainer.get_feedback_stats_simple(db, days)
        else:
            return {"error": "Feedback trainer not available"}
    
    def cleanup_inactive_profiles_from_cache(db: Session):
        try: 
            if not redis_client.is_connected():
                print("❌ Redis not connected")
                return
            cache_keys = redis_client.keyspy("query:*")
            cleaned_count = 0
            
            for cache_key in cache_keys:
                try:
                    cached_data = redis_client.get_cached_feedback(cache_key.replace("query:", ""))
                    if not cached_data:
                        continue
                    profile_ids = [f['profile_id'] for f in cached_data]
                    active_profiles = db.query(Profile.id).filter(
                        Profile.id.in_(profile_ids),
                        Profile.is_active == True
                    ).all()
                    active_profile_ids = [p.id for p in active_profiles]
                    if len(active_profile_ids) < len(profile_ids):
                        redis_client.redis_client.delete(cache_key)
                        cleaned_count += 1
                        print(f"Cleaned cache key: {cache_key}")
                    
                except Exception as e:
                    print(f"Error cleaning cache key {cache_key}: {e}")
                    continue
            
            print(f"✅ Cleaned {cleaned_count} cache entries with inactive profiles")
            
        except Exception as e:
            print(f"Error in cache cleanup: {e}")