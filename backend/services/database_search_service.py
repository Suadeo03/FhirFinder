# backend/services/database_search_service.py
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from services.performancelog import create_performance_log
from models.database.models import Profile
from config.database import get_db
from config.redis import get_redis_client

class DatabaseSearchService:   
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
   
    def search(self, query: str, limit: int = 5, db: Session = None) -> List[Dict]:

        if not db:
            # This should be passed from the endpoint
            raise ValueError("Database session required")
        
        if not get_redis_client():
            raise ValueError('Cannot connect to redis')
        
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
        """Calculate similarity with main profile content"""
        if profile.embedding_vector:
            profile_embedding = np.array(profile.embedding_vector)
            return cosine_similarity([query_embedding], [profile_embedding])[0][0]
        else:
            profile_text = f"{profile.name} {profile.description} {' '.join(profile.keywords or [])} {profile.category or ''} {profile.resource_type or ''} {profile.fhir_resource or ''} " 
            profile_embedding = self.model.encode([profile_text])[0]
            return cosine_similarity([query_embedding], [profile_embedding])[0][0]
    
    def _calculate_context_similarity(self, query_embedding: np.ndarray, profile: Profile) -> tuple:
        """Calculate similarity with use contexts"""
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
        """Calculate weighted combined score"""
        # Weights
        profile_weight = 0.6
        context_weight = 0.4
        
        # Keyword boost
        keyword_boost = self._calculate_keyword_boost(query, profile)
        
        combined = (profile_sim * profile_weight + 
                   context_sim * context_weight + 
                   keyword_boost * 0.2)
        
        return min(combined, 1.0)  # Cap at 1.0
    
    def _calculate_keyword_boost(self, query: str, profile: Profile) -> float:
        """Boost score for exact keyword matches"""
        query_words = set(query.lower().split())
        profile_keywords = set([kw.lower() for kw in profile.keywords or []])
        
        # Add context keywords
        for context in profile.use_contexts or []:
            context_keywords = set([kw.lower() for kw in context.get('keywords', [])])
            profile_keywords.update(context_keywords)
        
        matches = query_words.intersection(profile_keywords)
        return len(matches) / len(query_words) if query_words else 0
    
    def _generate_match_reasons(self, profile_sim: float, context_sim: float, best_context: Dict, query: str, profile: Profile) -> List[str]:
        """Generate human-readable match explanations"""
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
        """Get statistics about searchable profiles"""
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