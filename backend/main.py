# backend/main.py (UPDATED)
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
import logging as logger
import os

# Import routers
from api.v1.endpoints.search import router as search_router
from api.v1.endpoints.datasets import router as datasets_router
from api.v1.endpoints.query_performance import router as queryperformance_router

from config.database import init_database, get_db
from config.redis import RedisQueryCache
from services.search_service import SearchService

# Initialize database on startup
init_database()
cache = RedisQueryCache()


app = FastAPI(
    title="FHIR Profile Recommender",
    description="AI-powered FHIR profile search and recommendation system",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(search_router, prefix="/api/v1", tags=["search"])
app.include_router(datasets_router, prefix="/api/v1", tags=["datasets"])
app.include_router(queryperformance_router, prefix="/api/v1", tags=["performance"])

static_dir = "static"
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    logger.info(f"📁 Static files mounted from {static_dir}")





@app.get("/")
async def root():
    return {
        "message": "FHIR Profile Recommender API v2.0",
        "features": [
            "AI-powered profile search",
            "Multi-format dataset upload (CSV, JSON, Excel)",
            "ETL pipeline for data processing",
        ]
    }

@app.get("/health/db")
async def health(db: Session = Depends(get_db)):
    """Health check with database connectivity and search stats"""
    try:
        search_service = SearchService()
        stats = search_service.get_search_stats(db)
        
        return {
            "status": "healthy",
            "database": "connected",
            "search_stats": stats
        }
    except Exception as e:
        return {
            "status": "degraded",
            "database": "error",
            "error": str(e)
        }
@app.get("/health/redis")
def redis_health():
    """Health check with database connectivity and search stats"""
    cache = RedisQueryCache()
    if cache.is_connected():
        return {"status": "healthy", "redis": "connected"}
    else:
        return {"status": "unhealthy", "redis": "disconnected"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)