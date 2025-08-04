# config/redis_cache.py - Fixed version
import redis
import json
from typing import List, Dict, Any, Optional
from datetime import datetime

class RedisQueryCache:
    def __init__(self):
        try:
            self.redis_client = redis.Redis(
                host='localhost',
                port=6379,
                db=0,
                decode_responses=True  # This ensures strings are returned, not bytes
            )
            # Test connection
            self.redis_client.ping()
            self._connected = True


        except Exception as e:
            print(f"Redis connection failed: {e}")
            self._connected = False
            self.redis_client = None

    def is_connected(self) -> bool:
        return self._connected and self.redis_client is not None

    def get_cached_feedback(self, query_key: str) -> Optional[List[Dict]]:
        """Get cached feedback data for a query"""
        if not self.is_connected():
            return None
        
        try:
            cached_data = self.redis_client.get(f"query:{query_key}")
            if cached_data:
                # Deserialize JSON string back to Python object
                return json.loads(cached_data)
            return None
        except Exception as e:
            print(f"Error getting cached feedback for {query_key}: {e}")
            return None

    def set_cached_feedback(self, query_key: str, feedback_data: List[Dict], expiration_seconds: int = 3600):
        """Set cached feedback data for a query"""
        if not self.is_connected():
            return False
        
        try:
            # Serialize Python object to JSON string
            json_data = json.dumps(feedback_data)
            result = self.redis_client.setex(
                f"query:{query_key}", 
                expiration_seconds, 
                json_data
            )
            return result
        except Exception as e:
            print(f"Error setting cached feedback for {query_key}: {e}")
            return False

    def set_feedback(self, feedback_key: str, feedback_value: Any, expiration_seconds: int = 86400):
        """Set individual feedback data"""
        if not self.is_connected():
            return False
        
        try:
            # Convert value to JSON string if it's not already a simple type
            if isinstance(feedback_value, (dict, list)):
                value_to_store = json.dumps(feedback_value)
            elif isinstance(feedback_value, (str, int, float, bool)):
                value_to_store = str(feedback_value)
            else:
                # For other types (like datetime), convert to string
                value_to_store = str(feedback_value)
            
            result = self.redis_client.setex(
                feedback_key, 
                expiration_seconds, 
                value_to_store
            )
            print(f"Successfully set feedback key {feedback_key}")
            return result
        except Exception as e:
            print(f"Error setting feedback key {feedback_key}: {e}")
            return False

    def get_feedback(self, feedback_key: str) -> Optional[str]:
        """Get individual feedback data"""
        if not self.is_connected():
            return None
        
        try:
            return self.redis_client.get(feedback_key)
        except Exception as e:
            print(f"Error getting feedback key {feedback_key}: {e}")
            return None

    def exists_feedback(self, feedback_key: str) -> bool:
        """Check if feedback key exists"""
        if not self.is_connected():
            return False
        
        try:
            return bool(self.redis_client.exists(feedback_key))
        except Exception as e:
            print(f"Error checking feedback key {feedback_key}: {e}")
            return False

    def delete_cached_feedback(self, query_key: str) -> bool:
        """Delete cached feedback for a query"""
        if not self.is_connected():
            return False
        
        try:
            result = self.redis_client.delete(f"query:{query_key}")
            return bool(result)
        except Exception as e:
            print(f"Error deleting cached feedback for {query_key}: {e}")
            return False

    def clear_cache(self, query_key: str) -> bool:
        """Clear cache for a specific query"""
        return self.delete_cached_feedback(query_key)

    def clear_all_cache(self) -> bool:
        """Clear all cache entries"""
        if not self.is_connected():
            return False
        
        try:
            # Get all query keys
            query_keys = self.redis_client.keys("query:*")
            if query_keys:
                result = self.redis_client.delete(*query_keys)
                print(f"Cleared {result} cache entries")
                return True
            return True
        except Exception as e:
            print(f"Error clearing all cache: {e}")
            return False

    def get_ttl(self, query_key: str) -> int:
        """Get time-to-live for a cache key"""
        if not self.is_connected():
            return -1
        
        try:
            return self.redis_client.ttl(f"query:{query_key}")
        except Exception as e:
            print(f"Error getting TTL for {query_key}: {e}")
            return -1

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self.is_connected():
            return {"connected": False}
        
        try:
            info = self.redis_client.info()
            query_keys = self.redis_client.keys("query:*")
            feedback_keys = self.redis_client.keys("*feedback*")
            
            return {
                "connected": True,
                "total_query_cache_entries": len(query_keys),
                "total_feedback_entries": len(feedback_keys),
                "memory_used": info.get('used_memory_human', 'unknown'),
                "total_keys": info.get('db0', {}).get('keys', 0) if 'db0' in info else 0
            }
        except Exception as e:
            print(f"Error getting cache stats: {e}")
            return {"connected": True, "error": str(e)}

