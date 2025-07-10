# backend/services/database_search_service.py
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from services.performance_log import create_performance_log
from models.database.models import Profile
from config.redis import get_redis_client
from config.chroma import ChromaConfig
from services.cache_service import RedisQueryCache

class DatabaseSearchService:   
    def __init__(self):
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ SentenceTransformer model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading SentenceTransformer: {e}")
            self.model = None
        
        try:
            self.chroma_config = ChromaConfig()
            self.collection = self.chroma_config.collection
            
            # Test the connection
            if self.collection:
                info = self.chroma_config.get_client_info()
                print(f"✅ Chroma initialized: {info}")
                
                # Test basic functionality
                if self.chroma_config.test_connection():
                    print("✅ Chroma is ready for semantic search")
                else:
                    print("⚠️  Chroma connection unstable")
            else:
                print("❌ Chroma collection not available - falling back to traditional search only")
                
        except Exception as e:
            print(f"❌ Error initializing Chroma: {e}")
            self.collection = None
            self.chroma_config = None
    """ 
    def search(self, query: str, limit: int = 5, db: Session = None) -> List[Dict]:

        if not db:
            # This should be passed from the endpoint
            raise ValueError("Database session required")
        
        if not get_redis_client():
            raise ValueError("Redis client not initialized")
        else:
            try:
                
                redis_client = RedisQueryCache()
                redis_client.search_query_in_cache(query)

            except Exception as e:
                print(f"Error connecting to Redis: {e}")

        
        # add pinecone check
        



        # Get active profiles
        active_profiles = db.query(Profile).filter(Profile.is_active == True).all()
        if not active_profiles:
            return [] 
        query_embedding = self.model.encode([query])[0]
        results = []
        
        for profile in active_profiles:
            # Calculate similarities
            profile_similarity = self._calculate_profile_similarity(query_embedding, profile)
            context_similarity, best_context = self._calculate_context_similarity(query_embedding, profile)
            
            # Combined scoring
            combined_score = self._calculate_combined_score(
                profile_similarity, 
                context_similarity, 
                query, 
                profile
            )
            
            # Threshold setting
            if combined_score > 0.3:
                result = {
                    "id": profile.id,
                    "oid": profile.oid,
                    "name": profile.name,
                    "description": profile.description,
                    "must_have": profile.must_have or [],
                    "must_support": profile.must_support or [],
                    "invariants": profile.invariants or [],
                    "resource_url": profile.resource_url or [],
                    "keywords": profile.keywords or [],
                    "category": profile.category,
                    "resource_type": profile.resource_type,
                    "use_contexts": profile.use_contexts or [],
                    "version": profile.version,
                    "fhir_resource": profile.fhir_resource or {},
                    "fhir_searchable_text": profile.fhir_searchable_text or [],
                    "confidence_score": float(combined_score),
                    "profile_similarity": float(profile_similarity),
                    "context_similarity": float(context_similarity),
                    "match_reasons": self._generate_match_reasons(
                        profile_similarity, 
                        context_similarity, 
                        best_context,
                        query,
                        profile
                    )
                }
                results.append(result)
            if results:
                for res in results:
                    create_performance_log(
                        profile_id=res['id'],
                        query_text=query,
                        profile_name=res['name'],
                        profile_oid=res['oid'],
                        profile_score=res['profile_similarity'],
                        context_score=res['context_similarity'],
                        combined_score=res['confidence_score'],
                        match_reasons=res['match_reasons'],
                        keywords=res['keywords'],
                        db=db
                    )    
        
        # Sort by combined score
        results.sort(key=lambda x: x['confidence_score'], reverse=True)
        return results[:limit]
    
    def _calculate_profile_similarity(self, query_embedding: np.ndarray, profile: Profile) -> float:
     
        if profile.embedding_vector:
            profile_embedding = np.array(profile.embedding_vector)
            return cosine_similarity([query_embedding], [profile_embedding])[0][0]
        else:
            profile_text = f"{profile.name} {profile.description} {' '.join(profile.keywords or [])} {profile.category or ''} {profile.resource_type or ''} {profile.fhir_resource or ''} " 
            profile_embedding = self.model.encode([profile_text])[0]
            return cosine_similarity([query_embedding], [profile_embedding])[0][0]
    
    def _calculate_context_similarity(self, query_embedding: np.ndarray, profile: Profile) -> tuple:
  
        max_context_similarity = 0.0
        best_matching_context = None
        
        for context in profile.use_contexts or []:
            context_text = f"{context.get('scenario', '')} {' '.join(context.get('keywords', []))}"
            if context_text.strip():
                context_embedding = self.model.encode([context_text])[0]
                similarity = cosine_similarity([query_embedding], [context_embedding])[0][0]
                
                if similarity > max_context_similarity:
                    max_context_similarity = similarity
                    best_matching_context = context
        
        return max_context_similarity, best_matching_context
    
    def _calculate_combined_score(self, profile_sim: float, context_sim: float, query: str, profile: Profile) -> float:
        
        profile_weight = 0.6
        context_weight = 0.4
        keyword_boost = self._calculate_keyword_boost(query, profile)
        
        combined = (profile_sim * profile_weight + 
                   context_sim * context_weight + 
                   keyword_boost * 0.2)
        
        return min(combined, 1.0)  # Cap at 1.0
    
    def _calculate_keyword_boost(self, query: str, profile: Profile) -> float:
    
        query_words = set(query.lower().split())
        profile_keywords = set([kw.lower() for kw in profile.keywords or []])
        
        # Add context keywords
        for context in profile.use_contexts or []:
            context_keywords = set([kw.lower() for kw in context.get('keywords', [])])
            profile_keywords.update(context_keywords)
        
        matches = query_words.intersection(profile_keywords)
        return len(matches) / len(query_words) if query_words else 0
    
    def _generate_match_reasons(self, profile_sim: float, context_sim: float, best_context: Dict, query: str, profile: Profile) -> List[str]:
    
        reasons = []
        
        if profile_sim > 0.5:
            reasons.append("Strong semantic match with profile content")
        elif profile_sim > 0.3:
            reasons.append("Moderate semantic match with profile content")
        
        if context_sim > 0.5 and best_context:
            reasons.append(f"High relevance to use case: '{best_context.get('scenario', 'Unknown')}'")
        elif context_sim > 0.3 and best_context:
            reasons.append(f"Matches use case: '{best_context.get('scenario', 'Unknown')}'")
        
        # Check for exact keyword matches
        query_words = set(query.lower().split())
        profile_keywords = set([kw.lower() for kw in profile.keywords or []])
        matches = query_words.intersection(profile_keywords)
        
        if matches:
            reasons.append(f"Exact keyword matches: {', '.join(matches)}")
        
        return reasons if reasons else ["General similarity match"]
    
    def get_search_stats(self, db: Session) -> Dict[str, Any]:
 
        active_profiles = db.query(Profile).filter(Profile.is_active == True).count()
        total_profiles = db.query(Profile).count()
        
        # Get active dataset info
        from models.database.models import Dataset
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

    """
    def semantic_search(self, query: str, top_k: int = 10, db: Session = None, filters: Optional[Dict] = None) -> List[Dict]:
        """
        Perform semantic search using Chroma vectors + PostgreSQL metadata
        """
        if not self.collection:
            raise ValueError("Chroma collection not available for search")
        
        if not db:
            raise ValueError("Database session required")
        
        try:
            # Generate embedding for the search query
            query_embedding = self.model.encode([query])[0].tolist()
            
            # Prepare filter metadata for Chroma
            where_clause = {'is_active': True}
            if filters:
                for key, value in filters.items():
                    if key in ['dataset_id', 'resource_type', 'category', 'keywords']:
                        where_clause[key] = value
            
            # Search in Chroma
            search_results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_clause if where_clause else None,
                include=['metadatas', 'distances']
            )
            
            if not search_results['ids'] or not search_results['ids'][0]:
                return []
            
            # Get profile IDs and scores from Chroma results
            profile_ids = search_results['ids'][0]
            distances = search_results['distances'][0]
            
            # Convert distances to similarity scores
            similarity_scores = [1 / (1 + distance) for distance in distances]
            
            # Get full profile data from PostgreSQL
            profiles = db.query(Profile).filter(
                Profile.id.in_(profile_ids),
                Profile.is_active == True
            ).all()
            
            # Create lookup dictionary
            profile_dict = {p.id: p for p in profiles}
            
            # Combine results maintaining order from Chroma
            results = []
            for i, profile_id in enumerate(profile_ids):
                if profile_id in profile_dict:
                    profile = profile_dict[profile_id]
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
                        'similarity_score': similarity_scores[i],
                        'dataset_id': profile.dataset_id,
                        'oid': profile.oid,
                        'use_contexts': profile.use_contexts or [],
                        'fhir_resource': profile.fhir_resource or {},
                        'fhir_searchable_text': profile.fhir_searchable_text or []
                    })
            
            return results
            
        except Exception as e:
            print(f"Error in semantic search: {e}")
            return []

    def hybrid_search(self, query: str, top_k: int = 10, db: Session = None,
                     semantic_weight: float = 0.7, filters: Optional[Dict] = None) -> List[Dict]:
        """
        Combine semantic search (Chroma) with traditional text search (PostgreSQL)
        """
        if not db:
            raise ValueError("Database session required")
            
        results_map = {}
        
        # Semantic search component
        if semantic_weight > 0 and self.collection:
            try:
                # Correct parameter order:
                semantic_results = self.semantic_search(query, top_k, db, filters)
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
            # Build query
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
        
        # Get active dataset info
        from models.database.models import Dataset
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