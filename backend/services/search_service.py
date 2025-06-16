# backend/services/enhanced_search_service.py (FIXED)
import json
import os
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

class EnhancedSearchService:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load profiles with use contexts
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        profiles_path = os.path.join(project_root, 'data', 'samples', 'sample_profiles.json')
        
        if not os.path.exists(profiles_path):
            self._create_enhanced_sample_data(profiles_path)
        
        with open(profiles_path, 'r') as f:
            self.profiles = json.load(f)
        
        # Pre-compute embeddings for profiles AND use contexts
        self.profile_embeddings = self._compute_profile_embeddings()
        self.context_embeddings = self._compute_context_embeddings()
    
    def _create_enhanced_sample_data(self, profiles_path):
        """Create sample data with rich use contexts"""
        os.makedirs(os.path.dirname(profiles_path), exist_ok=True)
        
        
        print(f"Created enhanced sample data at: {profiles_path}")
    
    def _compute_profile_embeddings(self):
        """Compute embeddings for main profile content"""
        texts = []
        for profile in self.profiles:
            text = f"{profile['name']} {profile['description']} {' '.join(profile['keywords'])}"
            texts.append(text)
        return self.model.encode(texts)
    
    def _compute_context_embeddings(self):
        """Compute embeddings for use context scenarios - FIXED VERSION"""
        context_embeddings = []
        
        for profile in self.profiles:
            profile_context_embeddings = []
            
            for context in profile.get('use_contexts', []):
                context_text = f"{context['scenario']} {' '.join(context['keywords'])}"
                # Encode individual context and store as numpy array
                context_embedding = self.model.encode([context_text])[0]  # Get first (and only) embedding
                profile_context_embeddings.append(context_embedding)
            
            context_embeddings.append(profile_context_embeddings)
        
        return context_embeddings
    
    def search(self, query: str, limit: int = 5) -> List[Dict]:
        """Enhanced search using both profile content and use contexts"""
        query_embedding = self.model.encode([query])[0]  # Get the embedding vector
        
        results = []
        
        for i, profile in enumerate(self.profiles):
            # Calculate profile content similarity
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Main profile similarity
            profile_similarity = cosine_similarity(
                [query_embedding], 
                [self.profile_embeddings[i]]
            )[0][0]
            
            # Use context similarity
            max_context_similarity = 0.0
            best_matching_context = None
            
            # Check each use context for this profile
            for j, context in enumerate(profile.get('use_contexts', [])):
                if j < len(self.context_embeddings[i]):  # Make sure we have embeddings for this context
                    context_embedding = self.context_embeddings[i][j]
                    
                    context_similarity = cosine_similarity(
                        [query_embedding], 
                        [context_embedding]
                    )[0][0]
                    
                    if context_similarity > max_context_similarity:
                        max_context_similarity = context_similarity
                        best_matching_context = context
            
            # Combined scoring
            combined_score = self._calculate_combined_score(
                profile_similarity, 
                max_context_similarity, 
                query, 
                profile
            )
            
            # Only include if above threshold
            if combined_score > 0.3:
                result = profile.copy()
                result['confidence_score'] = float(combined_score)
                result['profile_similarity'] = float(profile_similarity)
                result['context_similarity'] = float(max_context_similarity)
                result['match_reasons'] = self._generate_match_reasons(
                    profile_similarity, 
                    max_context_similarity, 
                    best_matching_context,
                    query,
                    profile
                )
                results.append(result)
        
        # Sort by combined score
        results.sort(key=lambda x: x['confidence_score'], reverse=True)
        return results[:limit]
    
    def _calculate_combined_score(self, profile_sim, context_sim, query, profile):
        """Calculate weighted combined score"""
        # Weights: profile content vs use context
        profile_weight = 0.6
        context_weight = 0.4
        
        # Keyword boost
        keyword_boost = self._calculate_keyword_boost(query, profile)
        
        combined = (profile_sim * profile_weight + 
                   context_sim * context_weight + 
                   keyword_boost * 0.2)
        
        return min(combined, 1.0)  # Cap at 1.0
    
    def _calculate_keyword_boost(self, query, profile):
        """Boost score for exact keyword matches"""
        query_words = set(query.lower().split())
        profile_keywords = set([kw.lower() for kw in profile.get('keywords', [])])
        
        # Add context keywords
        for context in profile.get('use_contexts', []):
            context_keywords = set([kw.lower() for kw in context.get('keywords', [])])
            profile_keywords.update(context_keywords)
        
        matches = query_words.intersection(profile_keywords)
        return len(matches) / len(query_words) if query_words else 0
    
    def _generate_match_reasons(self, profile_sim, context_sim, best_context, query, profile):
        """Generate human-readable match explanations"""
        reasons = []
        
        if profile_sim > 0.5:
            reasons.append("Strong semantic match with profile content")
        elif profile_sim > 0.3:
            reasons.append("Moderate semantic match with profile content")
        
        if context_sim > 0.5 and best_context:
            reasons.append(f"High relevance to use case: '{best_context['scenario']}'")
        elif context_sim > 0.3 and best_context:
            reasons.append(f"Matches use case: '{best_context['scenario']}'")
        
        # Check for exact keyword matches
        query_words = set(query.lower().split())
        profile_keywords = set([kw.lower() for kw in profile.get('keywords', [])])
        matches = query_words.intersection(profile_keywords)
        
        if matches:
            reasons.append(f"Exact keyword matches: {', '.join(matches)}")
        
        return reasons if reasons else ["General similarity match"]