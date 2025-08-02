import redis
import json
import hashlib
from typing import Any, Optional
from datetime import datetime

try:
    REDIS_AVAILABLE = True
except ImportError:
    print("❌ Redis library not installed. Install with: pip install redis")
    REDIS_AVAILABLE = False

class RedisQueryCache:
    def __init__(self, host='localhost', port=6379, db=0):
        """Initialize Redis cache with connection parameters"""
        if not REDIS_AVAILABLE:
            print("❌ Redis library not available")
            self.client = None
            return
            
        self.host = host
        self.port = port
        self.db = db
        self.client = None
        self._connect()
    
    def _connect(self):
        """Establish Redis connection"""
        if not REDIS_AVAILABLE:
            print("❌ Redis library not available")
            self.client = None
            return
            
        try:
                
            self.client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                decode_responses=True,
                socket_connect_timeout=5,  
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            # Test connection
            self.client.ping()
            print(f"✅ Redis connected to {self.host}:{self.port}")
        except Exception as e:
            print(f"❌ Redis connection failed to {self.host}:{self.port}")
            print(f"Error details: {type(e).__name__}: {e}")
            print("Make sure Redis server is running and accessible")
            self.client = None
    
    def is_connected(self) -> bool:
        """Check if Redis is connected and working"""
        try:
            if self.client:
                self.client.ping()
                return True
        except:
            pass
        return False
    
    def _generate_cache_key(self, query: str, filters: dict = None) -> str:
        """Generate a consistent cache key from query and filters"""
        cache_data = {
            'query': query.lower().strip(),
            'filters': filters or {}
        }
        cache_string = json.dumps(cache_data, sort_keys=True)
        return f"search:{hashlib.md5(cache_string.encode()).hexdigest()}"
    
    def get_cached_feedback(self, query: str):
        """Get cached feedback data for a query"""
        if not self.is_connected():
                return None      
        try:
                pattern = f"positive_feedback:{query.lower().strip()}:*"
                feedback_data = []
                
                for key in self.client.scan_iter(match=pattern, count=1000):
                    # Get the timestamp value
                    timestamp_str = self.client.get(key)
                    if timestamp_str:
                        # Extract profile_id from key
                        parts = key.split(':')
                        profile_id = parts[2] if len(parts) >= 3 else None
                        
                        feedback_data.append({
                            'profile_id': profile_id,
                            'timestamp': timestamp_str,
                            'key': key
                        })
                
                if feedback_data:
                    print(f"Found {len(feedback_data)} feedback entries for query: {query}")
                    return feedback_data
                else:
                    print(f"No feedback found for query: {query}")
                    return None

        except Exception as e:
                print(f"Error getting cached search: {e}")
                return None
    
    def cache_search_results(self, query: str, results: dict, filters: dict = None, ttl: int = 3600):
        """Cache search results with TTL (default 1 hour)"""
        if not self.is_connected():
            return False
        
        try:
            cache_key = self._generate_cache_key(query, filters)
            cache_data = {
                'query': query,
                'results': results,
                'filters': filters,
                'cached_at': datetime.utcnow().isoformat()
            }
            
            self.client.setex(
                cache_key, 
                ttl, 
                json.dumps(cache_data, default=str)
            )
            print(f"Cached results for query: {query}")
            return True
            
        except Exception as e:
            print(f"Error caching search results: {e}")
            return False
    
    def clear_cache(self, pattern: str = "search:*") -> bool:
        """Clear cache entries matching pattern (default: all search cache)"""

        
        try:
            keys = self.client.keys(pattern)
            if keys:
                deleted = self.client.delete(*keys)
                print(f"✅ Cleared {deleted} cache entries matching pattern: {pattern}")
                return True
            else:
                print(f"No cache entries found matching pattern: {pattern}")
                return True
                
        except Exception as e:
            print(f"Error clearing cache: {e}")
            return False
    
    def clear_all_cache(self) -> bool:
        """Clear ALL cache entries (use with caution!)"""
        if not self.is_connected():
            print("Cannot clear cache - Redis not connected")
            return False
        
        try:
            self.client.flushdb()
            print("Cleared ALL cache entries from database")
            return True
        except Exception as e:
            print(f"Error clearing all cache: {e}")
            return False
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics"""
        if not self.is_connected():
            return {"error": "Redis not connected"}
        
        try:
            info = self.client.info()
            search_keys = len(self.client.keys("search:*"))
            feedback_keys = len(self.client.keys("feedback:*"))
            
            return {
                "connected": True,
                "total_keys": info.get('db0', {}).get('keys', 0),
                "search_cache_keys": search_keys,
                "feedback_keys": feedback_keys,
                "memory_used": info.get('used_memory_human', 'unknown'),
                "uptime_seconds": info.get('uptime_in_seconds', 0)
            }
        except Exception as e:
            return {"error": str(e)}
    
    # Feedback methods
    def set_feedback(self, key: str, value: str, expire_seconds: int = None) -> bool:
        """Store feedback data in Redis"""
        if not self.is_connected():
            return False
        
        try:
            feedback_key = key
            if expire_seconds:
                self.client.set(feedback_key, value, ex=expire_seconds)
            else:
                self.client.set(feedback_key, value)
            return True
        except Exception as e:
            print(f"Error setting feedback key {key}: {e}")
            return False

    def get_feedback(self, key: str) -> Optional[str]:
        """Get feedback data from Redis"""
        if not self.is_connected():
            return None
        
        try:
            feedback_key = f"feedback:{key}"
            return self.client.get(feedback_key)
        except Exception as e:
            print(f"Error getting feedback key {key}: {e}")
            return None

    def exists_feedback(self, key: str) -> bool:
        """Check if feedback key exists"""
        if not self.is_connected():
            return False
        
        try:
            feedback_key = f"feedback:{key}"
            return self.client.exists(feedback_key) > 0
        except Exception as e:
            print(f"Error checking feedback key {key}: {e}")
            return False
    
    def close(self):
        """Close Redis connection"""
        if self.client:
            self.client.close()
            print("Redis connection closed")



