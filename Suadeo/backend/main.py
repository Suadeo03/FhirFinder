from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from sqlalchemy import text
import logging as logger
import os
import psutil
import httpx

# Import routers
from datetime import datetime
from api.v1.endpoints.search import router as search_router
from api.v1.endpoints.datasets import router as datasets_router
from api.v1.endpoints.form_training_interface import router as form_trainer
from api.v1.endpoints.V2_training_interface import router as v2_trainer
from api.v1.endpoints.profile_training_interface import router as profile_trainer
from api.v1.endpoints.feedback import router as feedback_router
from api.v1.endpoints.formsets import router as formsets_router
from api.v1.endpoints.formLookup import router as formlookup_router
from api.v1.endpoints.v2_mapping_search import router as v2_mapping_router
from api.v1.endpoints.elt_api_v2_mapping import router as elt_api_v2_mapping_router
from api.v1.endpoints.query_performance import router as query_performance_router

from config.database import init_database, get_db
from config.chroma import ChromaConfig
from config.redis_cache import RedisQueryCache
from services.ultility.conversational_service import FHIRConversationalService

# Initialize database
init_database()

app = FastAPI(
    title="FHIR Profile Recommender",
    description="AI-powered FHIR profile search and recommendation system",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8080", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(search_router, prefix="/api/v1", tags=["search"])
app.include_router(datasets_router, prefix="/api/v1", tags=["elt_service"])
app.include_router(query_performance_router, prefix="/api/v1", tags=["metrics"])
app.include_router(feedback_router, prefix="/api/v1", tags=["feedback"])
app.include_router(formsets_router, prefix="/api/v1", tags=["etl_service"])
app.include_router(formlookup_router, prefix="/api/v1", tags=["search"])
app.include_router(v2_mapping_router, prefix="/api/v1", tags=["search"])
app.include_router(elt_api_v2_mapping_router, prefix="/api/v1", tags=["elt_service"])
app.include_router(form_trainer, prefix="/api/v1", tags=["model_training"])
app.include_router(profile_trainer, prefix="/api/v1", tags=["model_training"])
app.include_router(v2_trainer, prefix="/api/v1", tags=["model_training"])

static_dir = "static"
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    logger.info(f"Static files mounted from {static_dir}")
else:
    logger.warning(f"Static directory {static_dir} not found!")

# Initialize services
cache = RedisQueryCache()
convo = FHIRConversationalService()
chroma_config = ChromaConfig()
def health_check_db():
    """Check PostgreSQL database connectivity"""
    try:
        # Get database session
        db = next(get_db())
        
        # Simple query to test connection
        result = db.execute(text("SELECT 1")).fetchone()
        db.close()
        
        if result and result[0] == 1:
            return "connected"
        else:
            return "error"
    except Exception as e:
        print(f"Database health check failed: {e}")
        return "disconnected"

def redis_health():
    """Check Redis cache connectivity"""
    try:
        # Use the correct attribute name from RedisQueryCache
        if not cache.is_connected():
            return "disconnected"
        
        # Test Redis with a simple operation
        test_key = "health_check"
        cache.redis_client.set(test_key, "ok", ex=5)
        result = cache.redis_client.get(test_key)
        
        if result and result == "ok":  # decode_responses=True returns strings
            cache.redis_client.delete(test_key)
            return "connected"
        else:
            return "test_failed"
    except Exception as e:
        print(f"Redis health check failed: {e}")
        return "disconnected"

def chroma_health():
    """Check ChromaDB vector database connectivity"""
    try:
        if not chroma_config.is_available():
            return "unavailable"
            
        if not chroma_config.get_client():
            return "disconnected"
            
        # Test Chroma with a simple operation
        collection = chroma_config.get_collection()
        if collection:
            # Try to get collection info
            count = collection.count()
            return f"connected ({count} items)"
        else:
            return "no_collection"
    except Exception as e:
        print(f"Chroma health check failed: {e}")
        return "disconnected"

async def ollama_health():
    """Check Ollama LLM service connectivity"""
    try:
        ollama_url = os.getenv('OLLAMA_URL', 'http://localhost:11434')
        
        # Test Ollama API endpoint
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{ollama_url}/api/tags")
            
            if response.status_code == 200:
                data = response.json()
                models = data.get('models', [])
                model_names = [model.get('name', 'unknown') for model in models]
                return f"connected ({len(models)} models: {', '.join(model_names[:3])})"
            else:
                return f"error (status: {response.status_code})"
                
    except httpx.TimeoutException:
        return "timeout"
    except httpx.ConnectError:
        return "disconnected"
    except Exception as e:
        print(f"Ollama health check failed: {e}")
        return "error"

def get_system_stats():
    """Get basic system information"""
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            "cpu_usage": f"{cpu_percent}%",
            "memory_usage": f"{memory.percent}%",
            "disk_usage": f"{disk.percent}%",
            "available_memory": f"{memory.available / (1024**3):.1f}GB"
        }
    except ImportError:
        return {"note": "psutil not installed - no system stats available"}
    except Exception as e:
        return {"error": f"System stats error: {str(e)}"}

@app.get("/")
async def root():
    """Serve the main index page"""
    try:
        # Check if static/index.html exists
        index_path = "static/index.html"
        if os.path.exists(index_path):
            return FileResponse(index_path)
        else:
            # Fallback: return a simple HTML response
            return {
                "message": "FHIR Registry API is running",
                "docs": "/docs",
                "health": "/health",
                "note": "Frontend files not found in static/ directory"
            }
    except Exception as e:
        logger.error(f"Error serving index page: {e}")
        raise HTTPException(status_code=500, detail=f"Error serving index page: {str(e)}")

@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint for container orchestration"""
    try:
        # Check all services
        db_status = health_check_db()
        redis_status = redis_health()
        chroma_status = chroma_health()
        ollama_status = await ollama_health()
        
        # Determine overall health
        critical_services = [db_status, redis_status]
        is_healthy = all(status in ["connected", "available"] or "connected" in status for status in critical_services)
        
        # Get system stats
        system_stats = get_system_stats()
        
        response = {
            "status": "healthy" if is_healthy else "degraded",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "2.0.0",
            "environment": os.getenv("ENVIRONMENT", "development"),
            "services": {
                "api": "running",
                "database": db_status,
                "cache": redis_status,
                "vector_db": chroma_status,
                "ollama": ollama_status
            },
            "system": system_stats
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e),
            "services": {
                "api": "error",
                "database": "unknown",
                "cache": "unknown", 
                "vector_db": "unknown",
                "ollama": "unknown"
            }
        }

@app.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with more information"""
    try:
        # Database details
        db_details = {}
        try:
            db = next(get_db())
            
            # Check query_performance table
            result = db.execute(text("SELECT COUNT(*) FROM query_performance")).fetchone()
            db_details["query_performance_count"] = result[0] if result else 0
            
            # Check V2FHIRdata table if it exists
            try:
                result = db.execute(text('SELECT COUNT(*) FROM "V2fhirdata" WHERE is_active = true')).fetchone()
                db_details["active_v2_mappings"] = result[0] if result else 0
            except:
                db_details["active_v2_mappings"] = "table_not_found"
            
            db.close()
        except Exception as e:
            db_details["error"] = str(e)
        
        # Redis details
        redis_details = {}
        try:
            if cache.is_connected():
                info = cache.redis_client.info()
                redis_details = {
                    "connected_clients": info.get("connected_clients", 0),
                    "used_memory_human": info.get("used_memory_human", "unknown"),
                    "keyspace": info.get("db0", {}),
                    "cache_stats": cache.get_cache_stats()
                }
            else:
                redis_details = {"error": "not_connected"}
        except Exception as e:
            redis_details["error"] = str(e)
        
        # Chroma details
        chroma_details = {}
        try:
            if is_chroma_available():
                chroma_config = get_chroma_instance()
                collection = chroma_config.get_collection()
                if collection:
                    chroma_details = {
                        "collection_count": collection.count(),
                        "collection_name": collection.name if hasattr(collection, 'name') else "unknown"
                    }
        except Exception as e:
            chroma_details["error"] = str(e)
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "detailed_status": {
                "database": db_details,
                "redis": redis_details,
                "chroma": chroma_details,
                "system": get_system_stats()
            }
        }
        
    except Exception as e:
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }

# Only run uvicorn if this file is executed directly (not when imported)
#if __name__ == "__main__":
#    import uvicorn
#    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)