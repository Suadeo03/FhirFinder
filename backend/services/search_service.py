import json
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

class SimpleSearchService:
    def __init__(self):
        # Load embedding model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load profiles
        with open('/data/samples/sample_profiles.json', 'r') as f:
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