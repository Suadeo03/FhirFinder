from config.redis import get_redis_client
import hashlib
import logging
from typing import Optional

# Configure logging
#logging.basicConfig(level=logging.INFO)
#logger = logging.getLogger(__name__)


# conditions for caching high frequcency (last 30 days) with no "negative" feedback
class RedisQueryCache:
    def __init__(self):
        self.redis_client = get_redis_client()
        
        # Test connection
        try:
            self.redis_client.ping()
            logger.info("Connected to Redis successfully")
        except self.redis_client.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    def _hash_query(self, query: str) -> str:
        normalized_query = query.lower().strip()
        return hashlib.sha256(normalized_query.encode('utf-8')).hexdigest()

    def search_query_in_cache(self, user_query: str) -> Optional[str]:
        """
        Search for a user query in Redis cache and return the UUID if found
        
        Args:
            user_query (str): The search query to look for
            
        Returns:
            Optional[str]: Returns UUID if found, None if not found
            
        Raises:
            redis.RedisError: If there's an error with Redis operations
        """
        try:
            # Hash the user query to create a consistent key
            hashed_query = self._hash_query(user_query)
            
            # Try to get the UUID from Redis using the hashed query as key
            cached_uuid = self.redis_client.get(hashed_query)
            
            if cached_uuid:
                print(f'Cache hit for query: "{user_query}"')
                return cached_uuid
            else:
                print(F'Cache miss for query: "{user_query}"')
                return None
                
        except self.redis.RedisError as e:
            print(f'Error searching in Redis cache: {e}')
            raise

    def cache_query_uuid(self, user_query: str, uuid: str, ttl: int = 3600) -> bool:
        """
        Store a query-UUID pair in Redis cache
        
        Args:
            user_query (str): The search query
            uuid (str): The UUID to associate with the query
            ttl (int): Time to live in seconds (default 3600, 0 for no expiration)
            
        Returns:
            bool: True if successful, False otherwise
            
        Raises:
            redis.RedisError: If there's an error with Redis operations
        """
        try:
            hashed_query = self._hash_query(user_query)
            
            if ttl > 0:
                result = self.redis_client.setex(hashed_query, ttl, uuid)
            else:
                result = self.redis_client.set(hashed_query, uuid)
            
            if result:
                logger.info(f'Cached query: "{user_query}" with UUID: {uuid}')
                return True
            else:
                logger.warning(f'Failed to cache query: "{user_query}"')
                return False
                
        except self.redis.RedisError as e:
            logger.error(f'Error caching query in Redis: {e}')
            raise

    def delete_query_from_cache(self, user_query: str) -> bool:
        """
        Delete a query from the cache
        
        Args:
            user_query (str): The query to delete
            
        Returns:
            bool: True if deleted, False if not found
        """
        try:
            hashed_query = self._hash_query(user_query)
            result = self.redis_client.delete(hashed_query)
            
            if result:
                logger.info(f'Deleted query from cache: "{user_query}"')
                return True
            else:
                logger.info(f'Query not found in cache for deletion: "{user_query}"')
                return False
                
        except self.redis.RedisError as e:
            logger.error(f'Error deleting query from Redis cache: {e}')
            raise

    def close_connection(self):
        """Close the Redis connection"""
        try:
            self.redis_client.close()
            self.logger.info("Redis connection closed")
        except Exception as e:
            self.logger.error(f"Error closing Redis connection: {e}")


# Standalone functions for simple usage
def search_query_in_cache(user_query: str, redis_host='localhost', redis_port=6379) -> Optional[str]:
    """
    Standalone function to search for a query in Redis cache
    
    Args:
        user_query (str): The search query to look for
        redis_host (str): Redis host
        redis_port (int): Redis port
        
    Returns:
        Optional[str]: Returns UUID if found, None if not found
    """
    cache = RedisQueryCache(host=redis_host, port=redis_port)
    try:
        return cache.search_query_in_cache(user_query)
    finally:
        cache.close_connection()


# Example usage
def example_usage():
    """Example of how to use the Redis query cache"""
    
    # Initialize the cache
    cache = RedisQueryCache(host='localhost', port=6379)
    
    try:
        # Example 1: Search for a query
        user_query = "how to use redis with python"
        uuid = cache.search_query_in_cache(user_query)
        
        if uuid:
            print(f"Found UUID: {uuid} - now lookup in PostgreSQL")
            # Here you would make your PostgreSQL query using the UUID
        else:
            print("Query not found in cache")
            
        # Example 2: Cache a new query-UUID pair
        new_uuid = "123e4567-e89b-12d3-a456-426614174000"
        cache.cache_query_uuid(user_query, new_uuid, ttl=3600)
        
        # Example 3: Search again to verify it's cached
        found_uuid = cache.search_query_in_cache(user_query)
        if found_uuid:
            print(f"Successfully cached and retrieved UUID: {found_uuid}")
            
    except self.redis_client.RedisError as e:
        print(f"Redis error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        cache.close_connection()


