import hashlib
import json
import redis
import numpy as np
import logging
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional, Dict, Any



class HybridCache:
    def __init__(self, redis_client):
        self.redis = redis_client
        self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.semantic_threshold = 0.9
    
    def search_with_cache(self, query, vector_search_func):
        # Level 1: Exact cache (fastest)
        exact_key = f"exact:{hashlib.md5(query.encode()).hexdigest()}"
        cached = self.redis.get(exact_key)
        if cached:
            return json.loads(cached), "exact_hit"
        
        # Level 2: Normalized cache (fast)
        normalized = self.normalize_query(query)
        norm_key = f"norm:{hashlib.md5(normalized.encode()).hexdigest()}"
        cached = self.redis.get(norm_key)
        if cached:
            return json.loads(cached), "normalized_hit"
        
        # Level 3: Semantic cache (slower, but broader coverage)
        semantic_result = self.check_semantic_cache(query)
        if semantic_result:
            return semantic_result, "semantic_hit"
        
        # Level 4: Vector search (slowest)
        results = vector_search_func(query)
        
        # Cache at all levels
        self.cache_results(exact_key, norm_key, query, results)
        return results, "vector_search"
    
    def normalize_query(self, query):
        """Normalized preprocessing"""
        return re.sub(r'[^\w\s]', '', query.lower().strip())
    
    def check_semantic_cache(self, query):
        """Check for semantically similar cached queries"""
        query_vector = self.embeddings_model.encode([query])[0]
        
        # Get all cached vectors
        cached_vectors = self.redis.hgetall("semantic_vectors")
        
        for cached_hash, vector_json in cached_vectors.items():
            cached_vector = np.array(json.loads(vector_json))
            similarity = cosine_similarity([query_vector], [cached_vector])[0][0]
            
            if similarity >= self.semantic_threshold:
                cached_results = self.redis.get(f"sem_results:{cached_hash}")
                if cached_results:
                    return json.loads(cached_results)
        
        return None
    
    def cache_results(self, exact_key, norm_key, query, results):
        """Cache results at multiple levels"""
        # Cache exact and normalized
        self.redis.setex(exact_key, 3600, json.dumps(results))
        self.redis.setex(norm_key, 3600, json.dumps(results))
        
        # Cache semantic
        query_vector = self.embeddings_model.encode([query])[0]
        sem_hash = hashlib.md5(query.encode()).hexdigest()
        
        self.redis.hset("semantic_vectors", sem_hash, json.dumps(query_vector.tolist()))
        self.redis.setex(f"sem_results:{sem_hash}", 3600, json.dumps(results))

# review
class SearchCacheService:
    def __init__(self, redis_client, database):
        self.cache = redis_client
        self.db = database
    
    def search_with_cache(self, query, include_content=True):
        cache_key = f"search:{hashlib.md5(query.encode()).hexdigest()}"
        
        # Check cache for search metadata
        cached_data = self.cache.get(cache_key)
        if cached_data:
            print("Cache hit")
            search_metadata = json.loads(cached_data)
            
            if include_content:
                # Fetch fresh content from database
                return self.hydrate_results(search_metadata)
            else:
                # Return just metadata (fast)
                return search_metadata
        
        # Cache miss - perform vector search
        print("Cache miss - performing vector search")
        full_results = self.vector_search(query)
        
        # Extract metadata for caching
        search_metadata = [
            {
                'id': result['id'],
                'score': result['score'],
                'rank': i,
                'timestamp': time.time()
            }
            for i, result in enumerate(full_results)
        ]
        
        # Cache metadata
        self.cache.setex(cache_key, 3600, json.dumps(search_metadata))
        
        return full_results
    
    def hydrate_results(self, search_metadata):
        """Fetch fresh content and merge with cached metadata"""
        doc_ids = [item['id'] for item in search_metadata]
        fresh_documents = self.db.get_documents_by_ids(doc_ids)
        
        # Create lookup for fast merging
        doc_lookup = {doc['id']: doc for doc in fresh_documents}
        
        # Merge cached metadata with fresh content
        hydrated_results = []
        for metadata in search_metadata:
            doc_id = metadata['id']
            if doc_id in doc_lookup:
                result = {
                    **doc_lookup[doc_id],  # Fresh content
                    'search_score': metadata['score'],
                    'search_rank': metadata['rank']
                }
                hydrated_results.append(result)
        
        return hydrated_results