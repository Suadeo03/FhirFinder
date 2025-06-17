from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.v1.endpoints.search import router as search_router
from api.v1.endpoints.datasets import router as datasets_router
from sqlalchemy.orm import Session

from config.database import init_database, get_db
from services.database_search_service import DatabaseSearchService

init_database()
app = FastAPI(title="FHIR Profile Recommender")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(search_router, prefix="/api/v1", tags=["search"])
app.include_router(datasets_router, prefix="/api/v1", tags=["datasets"])

@app.get("/")
async def root():
    return {
        "message": "FHIR Profile Recommender API v2.0",
        "features": [
            "Database-powered profile search",
            "Multi-format dataset upload (CSV, JSON, Excel)",
            "ETL pipeline for data processing",
            "Dataset versioning and management"
        ]
    }

@app.get("/health")
async def health(db: Session = Depends(get_db)):
    """Health check with database connectivity and search stats"""
    try:
        search_service = DatabaseSearchService()
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)