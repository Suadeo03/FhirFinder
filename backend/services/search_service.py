import json
import os
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

class SimpleSearchService:
    def __init__(self):
        # Load embedding model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Get correct path to profiles - go up one level from backend to project root
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))  # Go up 2 levels from services/
        profiles_path = os.path.join(project_root, 'data', 'samples', 'sample_profiles.json')
        
        # Create sample data if it doesn't exist
        if not os.path.exists(profiles_path):
            self._create_sample_data(profiles_path)
        
        # Load profiles
        with open(profiles_path, 'r') as f:
            self.profiles = json.load(f)
        # Pre-compute embeddings
        self.profile_embeddings = self._compute_embeddings()
    
    def _compute_embeddings(self):
        texts = []
        for profile in self.profiles:
            text = f"{profile['name']} {profile['description']} {' '.join(profile['keywords'])}"
            texts.append(text)
        return self.model.encode(texts)
    
    def search(self, query: str, limit: int = 5) -> List[Dict]:
        # Encode query
        query_embedding = self.model.encode([query])
        
        # Calculate similarities
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(query_embedding, self.profile_embeddings)[0]
        
        # Rank results
        results = []
        for i, similarity in enumerate(similarities):
            if similarity > 0.3:  # Threshold
                result = self.profiles[i].copy()
                result['confidence_score'] = float(similarity)
                result['match_reasons'] = ["Semantic similarity"]
                results.append(result)
        
        # Sort and limit
        results.sort(key=lambda x: x['confidence_score'], reverse=True)
        return results[:limit]