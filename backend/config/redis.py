import redis


def get_redis_client():
    try:
        client = redis.Redis(
            host='localhost',
            port=6379,
            db=0,
            decode_responses=True,
            socket_connect_timeout=5,  # Add timeout
            socket_timeout=5
        )
        # Test connection
        client.ping()
        print("Redis connected!")
        return client
    except redis.ConnectionError:
        print("Redis connection failed - is Redis running?")
        return None
    except Exception as e:
        print(f"Redis error: {e}")
        return None


