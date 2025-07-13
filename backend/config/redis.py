import redis
import json
import hashlib
from typing import Any, Optional

class RedisQueryCache:
    def __init__(self, host='localhost', port=6379, db=0):
        """Initialize Redis cache with connection parameters"""
        self.host = host
        self.port = port
        self.db = db
        self.client = None
        self._connect()
    
    def _connect(self):
        """Establish Redis connection"""
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
        except redis.ConnectionError:
            print(f"❌ Redis connection failed to {self.host}:{self.port} - is Redis running?")
            self.client = None
        except Exception as e:
            print(f"❌ Redis error: {e}")
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
    
    def ping(self) -> bool:
        """Ping Redis server"""
        try:
            if self.client:
                return self.client.ping()
        except Exception as e:
            print(f"Redis ping failed: {e}")
            return False
        return False
    
    def _generate_cache_key(self, query: str, filters: dict = None) -> str:
        """Generate a consistent cache key from query and filters"""
        cache_data = {
            'query': query.lower().strip(),
            'filters': filters or {}
        }
        cache_string = json.dumps(cache_data, sort_keys=True)
        return f"search:{hashlib.md5(cache_string.encode()).hexdigest()}"
    
    def get_cached_search(self, query: str, filters: dict = None) -> Optional[dict]:
        """Get cached search results"""
        if not self.is_connected():
            return None
        
        try:
            cache_key = self._generate_cache_key(query, filters)
            cached_result = self.client.get(cache_key)
            
            if cached_result:
                print(f"✅ Cache hit for query: {query}")
                return json.loads(cached_result)
            else:
                print(f"❌ Cache miss for query: {query}")
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
                'cached_at': self._get_timestamp()
            }
            
            self.client.setex(
                cache_key, 
                ttl, 
                json.dumps(cache_data, default=str)
            )
            print(f"✅ Cached results for query: {query}")
            return True
            
        except Exception as e:
            print(f"Error caching search results: {e}")
            return False
    
    def search_query_in_cache(self, query: str, filters: dict = None) -> Optional[dict]:
        """Search for query in cache (alias for get_cached_search)"""
        return self.get_cached_search(query, filters)
    
    def invalidate_cache(self, pattern: str = "search:*"):
        """Invalidate cache entries matching pattern"""
        if not self.is_connected():
            return False
        
        try:
            keys = self.client.keys(pattern)
            if keys:
                deleted = self.client.delete(*keys)
                print(f"✅ Invalidated {deleted} cache entries")
                return True
            else:
                print("No cache entries to invalidate")
                return True
                
        except Exception as e:
            print(f"Error invalidating cache: {e}")
            return False
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics"""
        if not self.is_connected():
            return {"error": "Redis not connected"}
        
        try:
            info = self.client.info()
            search_keys = len(self.client.keys("search:*"))
            
            return {
                "connected": True,
                "total_keys": info.get('db0', {}).get('keys', 0),
                "search_cache_keys": search_keys,
                "memory_used": info.get('used_memory_human', 'unknown'),
                "uptime": info.get('uptime_in_seconds', 0)
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _get_timestamp(self):
        """Get current timestamp"""
        from datetime import datetime
        return datetime.utcnow().isoformat()
    
    def close(self):
        """Close Redis connection"""
        if self.client:
            self.client.close()
            print("Redis connection closed")


    def set_feedback(self, key: str, value: str, expire_seconds: int = None):
        """Store feedback data in Redis"""
        if not self.is_connected():
            return False
        
        try:
            if expire_seconds:
                self.client.set(key, value, ex=expire_seconds)
            else:
                self.client.set(key, value)
            return True
        except Exception as e:
            print(f"Error setting Redis key {key}: {e}")
            return False

    def get_feedback(self, key: str):
        """Get feedback data from Redis"""
        if not self.is_connected():
            return None
        
        try:
            return self.client.get(key)
        except Exception as e:
            print(f"Error getting Redis key {key}: {e}")
            return None

    def exists_feedback(self, key: str):
        """Check if feedback key exists"""
        if not self.is_connected():
            return False
        
        try:
            return self.client.exists(key) > 0
        except Exception as e:
            print(f"Error checking Redis key {key}: {e}")
            return False