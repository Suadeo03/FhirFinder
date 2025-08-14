# backend/api/v1/endpoints/query_performance.py
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, asc
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, timedelta
import csv
import io
import json

from config.database import get_db
from config.chroma import get_chroma_instance
from models.database.models import QueryPerformance  # Try this path instead
from sentence_transformers import SentenceTransformer

router = APIRouter()

# Add a simple test endpoint to verify the router is working
@router.get("/test")
async def test_endpoint():
    """Test endpoint to verify router is working"""
    return {"message": "Query performance router is working"}

@router.get("/query-performance/debug/schema")
async def debug_query_performance_schema(db: Session = Depends(get_db)):
    """Debug endpoint to show what columns exist in QueryPerformance"""
    try:
        # Get a sample record to see what attributes are available
        sample_record = db.query(QueryPerformance).first()
        
        if sample_record:
            # Get all attributes of the model instance
            attributes = [attr for attr in dir(sample_record) 
                         if not attr.startswith('_') and not callable(getattr(sample_record, attr))]
            
            # Get column names from the model
            column_names = [column.name for column in QueryPerformance.__table__.columns]
            
            return {
                "message": "QueryPerformance schema debug info",
                "table_columns": column_names,
                "instance_attributes": attributes,
                "sample_data": {
                    "id": getattr(sample_record, 'id', None),
                    "query_text": getattr(sample_record, 'query_text', None),
                    "has_query_normalized": hasattr(sample_record, 'query_normalized'),
                    "has_dataset_type": hasattr(sample_record, 'dataset_type'),
                    "has_search_type": hasattr(sample_record, 'search_type')
                }
            }
        else:
            return {
                "message": "No QueryPerformance records found",
                "table_columns": [column.name for column in QueryPerformance.__table__.columns],
                "total_records": db.query(QueryPerformance).count()
            }
            
    except Exception as e:
        return {"error": f"Debug failed: {str(e)}"}

class QueryPerformanceResponse(BaseModel):
    """Response model for query performance data"""
    id: str
    profile_id: str
    query_text: str
    query_normalized: Optional[str]
    query_date: datetime
    profile_name: Optional[str]
    profile_oid: Optional[str]
    profile_score: Optional[float]
    context_score: Optional[float]
    combined_score: Optional[float]
    match_reasons: Optional[str]
    keywords: Optional[Union[Dict[str, Any], List[Any]]]
    search_type: Optional[str]
    dataset_type: Optional[str]
    user_session: Optional[str]
    result_count: int
    has_results: bool
    top_result_id: Optional[str]
    top_result_score: Optional[float]
    top_result_type: Optional[str]
    search_duration_ms: Optional[int]
    embedding_duration_ms: Optional[int]
    db_query_duration_ms: Optional[int]
    chroma_query_duration_ms: Optional[int]
    similarity_scores: Optional[Union[Dict[str, Any], List[Any]]]
    result_metadata: Optional[Union[Dict[str, Any], List[Any]]]
    filters_applied: Optional[Union[Dict[str, Any], List[Any]]]
    result_clicked: bool
    feedback_given: Optional[str]
    time_to_feedback: Optional[int]

class QueryPerformanceWithEmbeddingResponse(BaseModel):
    """Response model including ChromaDB embedding data"""
    query_performance: QueryPerformanceResponse
    chroma_data: Optional[Dict[str, Any]] = None
    embedding_available: bool = False
    embedding_similarity: Optional[float] = None

class QueryAnalyticsResponse(BaseModel):
    """Response model for query analytics"""
    total_queries: int
    unique_queries: int
    avg_search_duration: Optional[float]
    avg_embedding_duration: Optional[float]
    avg_results_count: Optional[float]
    top_queries: List[Dict[str, Any]]
    top_profiles: List[Dict[str, Any]]
    search_types_breakdown: Dict[str, int]
    dataset_types_breakdown: Dict[str, int]
    feedback_breakdown: Dict[str, int]
    performance_trends: List[Dict[str, Any]]

class EmbeddingSimilarityResponse(BaseModel):
    """Response model for embedding similarity analysis"""
    query_id: str
    query_text: str
    original_profile_id: str
    similar_queries: List[Dict[str, Any]]
    embedding_cluster_info: Optional[Dict[str, Any]]

@router.get("/query-performance", response_model=List[QueryPerformanceResponse])
async def get_query_performance_data(
    skip: int = Query(0, description="Number of records to skip"),
    limit: int = Query(100, description="Maximum number of records to return"),
    query_text: Optional[str] = Query(None, description="Filter by query text (partial match)"),
    profile_id: Optional[str] = Query(None, description="Filter by profile ID"),
    dataset_type: Optional[str] = Query(None, description="Filter by dataset type"),
    search_type: Optional[str] = Query(None, description="Filter by search type"),
    has_results: Optional[bool] = Query(None, description="Filter by whether query had results"),
    start_date: Optional[datetime] = Query(None, description="Filter queries after this date"),
    end_date: Optional[datetime] = Query(None, description="Filter queries before this date"),
    order_by: Optional[str] = Query("query_date", description="Order by field"),
    order_desc: bool = Query(True, description="Order descending"),
    db: Session = Depends(get_db)
):
    """Get query performance data with filtering and pagination"""
    try:
        # Build query with filters
        query = db.query(QueryPerformance)
        
        if query_text:
            query = query.filter(QueryPerformance.query_text.ilike(f"%{query_text}%"))
        
        if profile_id:
            query = query.filter(QueryPerformance.profile_id == profile_id)
        
        if dataset_type:
            query = query.filter(QueryPerformance.dataset_type == dataset_type)
        
        if search_type:
            query = query.filter(QueryPerformance.search_type == search_type)
        
        if has_results is not None:
            query = query.filter(QueryPerformance.has_results == has_results)
        
        if start_date:
            query = query.filter(QueryPerformance.query_date >= start_date)
        
        if end_date:
            query = query.filter(QueryPerformance.query_date <= end_date)
        
        # Apply ordering
        order_field = getattr(QueryPerformance, order_by, QueryPerformance.query_date)
        if order_desc:
            query = query.order_by(desc(order_field))
        else:
            query = query.order_by(asc(order_field))
        
        # Apply pagination
        results = query.offset(skip).limit(limit).all()
        
        # Convert to response models
        response_data = []
        for result in results:
            response_data.append(QueryPerformanceResponse(
                id=result.id,
                profile_id=result.profile_id,
                query_text=result.query_text,
                query_normalized=result.query_normalized,
                query_date=result.query_date,
                profile_name=result.profile_name,
                profile_oid=result.profile_oid,
                profile_score=result.profile_score,
                context_score=result.context_score,
                combined_score=result.combined_score,
                match_reasons=result.match_reasons,
                keywords=result.keywords,
                search_type=result.search_type,
                dataset_type=result.dataset_type,
                user_session=result.user_session,
                result_count=result.result_count,
                has_results=result.has_results,
                top_result_id=result.top_result_id,
                top_result_score=result.top_result_score,
                top_result_type=result.top_result_type,
                search_duration_ms=result.search_duration_ms,
                embedding_duration_ms=result.embedding_duration_ms,
                db_query_duration_ms=result.db_query_duration_ms,
                chroma_query_duration_ms=result.chroma_query_duration_ms,
                similarity_scores=result.similarity_scores,
                result_metadata=result.result_metadata,
                filters_applied=result.filters_applied,
                result_clicked=result.result_clicked,
                feedback_given=result.feedback_given,
                time_to_feedback=result.time_to_feedback
            ))
        
        return response_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve query performance data: {str(e)}")

@router.get("/query-performance/export/csv")
async def export_query_performance_to_csv(
    query_text: Optional[str] = Query(None, description="Filter by query text (partial match)"),
    profile_id: Optional[str] = Query(None, description="Filter by profile ID"),
    dataset_type: Optional[str] = Query(None, description="Filter by dataset type"),
    search_type: Optional[str] = Query(None, description="Filter by search type"),
    has_results: Optional[bool] = Query(None, description="Filter by whether query had results"),
    start_date: Optional[datetime] = Query(None, description="Filter queries after this date"),
    end_date: Optional[datetime] = Query(None, description="Filter queries before this date"),
    include_embeddings: bool = Query(False, description="Include ChromaDB embedding similarity data"),
    max_records: int = Query(1000, description="Maximum number of records to export"),
    db: Session = Depends(get_db)
):
    """Export query performance data to CSV format"""
    try:
        # Build query with same filters as get_query_performance_data
        query = db.query(QueryPerformance)
        
        if query_text:
            query = query.filter(QueryPerformance.query_text.ilike(f"%{query_text}%"))
        
        if profile_id:
            query = query.filter(QueryPerformance.profile_id == profile_id)
        
        if dataset_type and hasattr(QueryPerformance, 'dataset_type'):
            query = query.filter(QueryPerformance.dataset_type == dataset_type)
        
        if search_type and hasattr(QueryPerformance, 'search_type'):
            query = query.filter(QueryPerformance.search_type == search_type)
        
        if has_results is not None and hasattr(QueryPerformance, 'has_results'):
            query = query.filter(QueryPerformance.has_results == has_results)
        
        if start_date and hasattr(QueryPerformance, 'query_date'):
            query = query.filter(QueryPerformance.query_date >= start_date)
        
        if end_date and hasattr(QueryPerformance, 'query_date'):
            query = query.filter(QueryPerformance.query_date <= end_date)
        
        # Order by date and limit
        if hasattr(QueryPerformance, 'query_date'):
            query = query.order_by(desc(QueryPerformance.query_date))
        query = query.limit(max_records)
        results = query.all()
        
        # Check if we have any data
        if not results:
            total_count = db.query(QueryPerformance).count()
            return {"message": f"No data found with the given filters. Total records in DB: {total_count}"}
        
        # Helper function to safely get attribute
        def safe_get(obj, attr, default=''):
            return getattr(obj, attr, default) if hasattr(obj, attr) else default
        
        # Create CSV content in memory (not streaming to avoid async issues)
        output = io.StringIO()
        
        # Define CSV headers based on what columns actually exist
        available_columns = [column.name for column in QueryPerformance.__table__.columns]
        
        # Use only columns that exist in the database
        base_headers = []
        for col in ['id', 'profile_id', 'query_text', 'query_normalized', 'query_date',
                   'profile_name', 'profile_oid', 'profile_score', 'context_score', 'combined_score',
                   'match_reasons', 'keywords', 'search_type', 'dataset_type', 'user_session',
                   'result_count', 'has_results', 'top_result_id', 'top_result_score', 'top_result_type',
                   'search_duration_ms', 'embedding_duration_ms', 'db_query_duration_ms', 'chroma_query_duration_ms',
                   'similarity_scores', 'result_metadata', 'filters_applied', 'result_clicked',
                   'feedback_given', 'time_to_feedback']:
            if col in available_columns:
                base_headers.append(col)
        
        # Add embedding headers if requested
        if include_embeddings:
            base_headers.extend([
                'embedding_available', 'embedding_similarity', 'chroma_document', 'chroma_metadata'
            ])
        
        writer = csv.writer(output)
        writer.writerow(base_headers)
        
        chroma = None
        model = None
        if include_embeddings:
            try:
                chroma = get_chroma_instance()
                if chroma.is_available():
                    model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                print(f"ChromaDB not available for embedding analysis: {e}")
        
     
        for result in results:
        
            row_data = []
            
            for col in base_headers:
                if col in ['embedding_available', 'embedding_similarity', 'chroma_document', 'chroma_metadata']:
                    continue  # Skip embedding columns for now
                
                if col == 'query_date':
                    value = result.query_date.isoformat() if hasattr(result, 'query_date') and result.query_date else ''
                elif col in ['keywords', 'similarity_scores', 'result_metadata', 'filters_applied']:
                    attr_value = safe_get(result, col)
                    value = json.dumps(attr_value) if attr_value else ''
                else:
                    value = safe_get(result, col)
                
                row_data.append(value)
            
            # Add embedding data if requested
            if include_embeddings:
                embedding_available = False
                embedding_similarity = None
                chroma_document = ''
                chroma_metadata = ''
                
                if chroma and chroma.is_available() and safe_get(result, 'top_result_id') and model:
                    try:
                        collection = chroma.get_collection()
                        chroma_results = collection.get(
                            ids=[safe_get(result, 'top_result_id')],
                            include=['embeddings', 'documents', 'metadatas']
                        )
                        
                        if chroma_results['ids']:
                            embedding_available = True
                            chroma_document = chroma_results['documents'][0] if chroma_results['documents'] else ''
                            chroma_metadata = json.dumps(chroma_results['metadatas'][0]) if chroma_results['metadatas'] else ''
      
                            if chroma_results['embeddings'] and safe_get(result, 'query_text'):
                                try:
                                    query_embedding = model.encode([safe_get(result, 'query_text')])[0]
                                    result_embedding = chroma_results['embeddings'][0]
                                    
                                    import numpy as np
                                    query_vec = np.array(query_embedding)
                                    result_vec = np.array(result_embedding)
                                    
                                    embedding_similarity = float(np.dot(query_vec, result_vec) / 
                                                               (np.linalg.norm(query_vec) * np.linalg.norm(result_vec)))
                                except Exception as e:
                                    print(f"Error calculating similarity for {safe_get(result, 'id')}: {e}")
                                    
                    except Exception as e:
                        print(f"Error retrieving ChromaDB data for {safe_get(result, 'id')}: {e}")
                
                row_data.extend([
                    embedding_available,
                    embedding_similarity,
                    chroma_document,
                    chroma_metadata
                ])
            
            writer.writerow(row_data)

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        embedding_suffix = "_with_embeddings" if include_embeddings else ""
        filename = f"query_performance_{len(results)}records_{timestamp}{embedding_suffix}.csv"

        csv_content = output.getvalue()
        
        from fastapi.responses import Response
        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

@router.get("/query-performance/{query_id}/with-embedding", response_model=QueryPerformanceWithEmbeddingResponse)
async def get_query_performance_with_embedding(
    query_id: str,
    db: Session = Depends(get_db)
):
    """Get query performance data along with ChromaDB embedding information"""
    try:
        # Get query performance data
        query_perf = db.query(QueryPerformance).filter(QueryPerformance.id == query_id).first()
        if not query_perf:
            raise HTTPException(status_code=404, detail="Query performance record not found")
        
        # Convert to response model
        performance_response = QueryPerformanceResponse(
            id=query_perf.id,
            profile_id=query_perf.profile_id,
            query_text=query_perf.query_text,
            query_normalized=query_perf.query_normalized,
            query_date=query_perf.query_date,
            profile_name=query_perf.profile_name,
            profile_oid=query_perf.profile_oid,
            profile_score=query_perf.profile_score,
            context_score=query_perf.context_score,
            combined_score=query_perf.combined_score,
            match_reasons=query_perf.match_reasons,
            keywords=query_perf.keywords,
            search_type=query_perf.search_type,
            dataset_type=query_perf.dataset_type,
            user_session=query_perf.user_session,
            result_count=query_perf.result_count,
            has_results=query_perf.has_results,
            top_result_id=query_perf.top_result_id,
            top_result_score=query_perf.top_result_score,
            top_result_type=query_perf.top_result_type,
            search_duration_ms=query_perf.search_duration_ms,
            embedding_duration_ms=query_perf.embedding_duration_ms,
            db_query_duration_ms=query_perf.db_query_duration_ms,
            chroma_query_duration_ms=query_perf.chroma_query_duration_ms,
            similarity_scores=query_perf.similarity_scores,
            result_metadata=query_perf.result_metadata,
            filters_applied=query_perf.filters_applied,
            result_clicked=query_perf.result_clicked,
            feedback_given=query_perf.feedback_given,
            time_to_feedback=query_perf.time_to_feedback
        )
        
        # Get ChromaDB data
        chroma_data = None
        embedding_available = False
        embedding_similarity = None
        
        try:
            chroma = get_chroma_instance()
            if chroma.is_available() and query_perf.top_result_id:
                collection = chroma.get_collection()
                
                # Get embedding data for the top result
                results = collection.get(
                    ids=[query_perf.top_result_id],
                    include=['embeddings', 'documents', 'metadatas']
                )
                
                if results['ids']:
                    embedding_available = True
                    chroma_data = {
                        'document': results['documents'][0] if results['documents'] else None,
                        'metadata': results['metadatas'][0] if results['metadatas'] else None,
                        'embedding_dimension': len(results['embeddings'][0]) if results['embeddings'] else 0
                    }
                    
                    # Calculate similarity between query and result embedding
                    if results['embeddings'] and query_perf.query_text:
                        model = SentenceTransformer('all-MiniLM-L6-v2')
                        query_embedding = model.encode([query_perf.query_text])[0]
                        result_embedding = results['embeddings'][0]
                        
                        import numpy as np
                        query_vec = np.array(query_embedding)
                        result_vec = np.array(result_embedding)
                        
                        # Cosine similarity
                        embedding_similarity = float(np.dot(query_vec, result_vec) / 
                                                   (np.linalg.norm(query_vec) * np.linalg.norm(result_vec)))
                        
        except Exception as e:
            print(f"Error retrieving ChromaDB data: {e}")
        
        return QueryPerformanceWithEmbeddingResponse(
            query_performance=performance_response,
            chroma_data=chroma_data,
            embedding_available=embedding_available,
            embedding_similarity=embedding_similarity
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve query data with embedding: {str(e)}")

@router.get("/query-performance/analytics", response_model=QueryAnalyticsResponse)
async def get_query_analytics(
    days_back: int = Query(30, description="Number of days to analyze"),
    db: Session = Depends(get_db)
):
    """Get comprehensive query performance analytics"""
    try:
        start_date = datetime.utcnow() - timedelta(days=days_back)
        
        # Basic statistics
        total_queries = db.query(QueryPerformance).filter(
            QueryPerformance.query_date >= start_date
        ).count()
        
        unique_queries = db.query(func.count(func.distinct(QueryPerformance.query_normalized))).filter(
            QueryPerformance.query_date >= start_date
        ).scalar()
        
        # Average performance metrics
        avg_stats = db.query(
            func.avg(QueryPerformance.search_duration_ms).label('avg_search'),
            func.avg(QueryPerformance.embedding_duration_ms).label('avg_embedding'),
            func.avg(QueryPerformance.result_count).label('avg_results')
        ).filter(QueryPerformance.query_date >= start_date).first()
        
        # Top queries
        top_queries = db.query(
            QueryPerformance.query_normalized,
            func.count(QueryPerformance.id).label('count'),
            func.avg(QueryPerformance.profile_score).label('avg_score')
        ).filter(
            QueryPerformance.query_date >= start_date
        ).group_by(
            QueryPerformance.query_normalized
        ).order_by(desc('count')).limit(10).all()
        
        # Top profiles
        top_profiles = db.query(
            QueryPerformance.profile_id,
            QueryPerformance.profile_name,
            func.count(QueryPerformance.id).label('search_count'),
            func.avg(QueryPerformance.profile_score).label('avg_score')
        ).filter(
            QueryPerformance.query_date >= start_date
        ).group_by(
            QueryPerformance.profile_id,
            QueryPerformance.profile_name
        ).order_by(desc('search_count')).limit(10).all()
        
        # Search types breakdown
        search_types = db.query(
            QueryPerformance.search_type,
            func.count(QueryPerformance.id).label('count')
        ).filter(
            QueryPerformance.query_date >= start_date
        ).group_by(QueryPerformance.search_type).all()
        
        # Dataset types breakdown
        dataset_types = db.query(
            QueryPerformance.dataset_type,
            func.count(QueryPerformance.id).label('count')
        ).filter(
            QueryPerformance.query_date >= start_date
        ).group_by(QueryPerformance.dataset_type).all()
        
        # Feedback breakdown
        feedback_breakdown = db.query(
            QueryPerformance.feedback_given,
            func.count(QueryPerformance.id).label('count')
        ).filter(
            QueryPerformance.query_date >= start_date
        ).group_by(QueryPerformance.feedback_given).all()
        
        # Performance trends (daily)
        trends = db.query(
            func.date(QueryPerformance.query_date).label('date'),
            func.count(QueryPerformance.id).label('query_count'),
            func.avg(QueryPerformance.search_duration_ms).label('avg_duration'),
            func.avg(QueryPerformance.profile_score).label('avg_score')
        ).filter(
            QueryPerformance.query_date >= start_date
        ).group_by(
            func.date(QueryPerformance.query_date)
        ).order_by('date').all()
        
        return QueryAnalyticsResponse(
            total_queries=total_queries,
            unique_queries=unique_queries or 0,
            avg_search_duration=avg_stats.avg_search,
            avg_embedding_duration=avg_stats.avg_embedding,
            avg_results_count=avg_stats.avg_results,
            top_queries=[
                {
                    "query": q.query_normalized,
                    "count": q.count,
                    "avg_score": float(q.avg_score) if q.avg_score else 0.0
                }
                for q in top_queries
            ],
            top_profiles=[
                {
                    "profile_id": p.profile_id,
                    "profile_name": p.profile_name,
                    "search_count": p.search_count,
                    "avg_score": float(p.avg_score) if p.avg_score else 0.0
                }
                for p in top_profiles
            ],
            search_types_breakdown={
                st.search_type or "unknown": st.count for st in search_types
            },
            dataset_types_breakdown={
                dt.dataset_type or "unknown": dt.count for dt in dataset_types
            },
            feedback_breakdown={
                fb.feedback_given or "none": fb.count for fb in feedback_breakdown
            },
            performance_trends=[
                {
                    "date": str(trend.date),
                    "query_count": trend.query_count,
                    "avg_duration": float(trend.avg_duration) if trend.avg_duration else 0.0,
                    "avg_score": float(trend.avg_score) if trend.avg_score else 0.0
                }
                for trend in trends
            ]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate analytics: {str(e)}")

@router.get("/query-performance/embedding-similarity/{query_id}", response_model=EmbeddingSimilarityResponse)
async def analyze_embedding_similarity(
    query_id: str,
    similarity_threshold: float = Query(0.7, description="Minimum similarity threshold"),
    max_results: int = Query(10, description="Maximum number of similar queries to return"),
    db: Session = Depends(get_db)
):
    """Analyze embedding similarity for a query against other queries"""
    try:
        # Get the source query
        source_query = db.query(QueryPerformance).filter(QueryPerformance.id == query_id).first()
        if not source_query:
            raise HTTPException(status_code=404, detail="Query not found")
        
        chroma = get_chroma_instance()
        if not chroma.is_available():
            raise HTTPException(status_code=503, detail="ChromaDB not available")
        
        # Generate embedding for source query
        model = SentenceTransformer('all-MiniLM-L6-v2')
        source_embedding = model.encode([source_query.query_text])[0]
        
        # Get other queries from the same time period
        comparison_queries = db.query(QueryPerformance).filter(
            QueryPerformance.id != query_id,
            QueryPerformance.query_date >= source_query.query_date - timedelta(days=30),
            QueryPerformance.has_results == True
        ).limit(1000).all()  # Limit for performance
        
        similar_queries = []
        
        for comp_query in comparison_queries:
            try:
                # Generate embedding for comparison query
                comp_embedding = model.encode([comp_query.query_text])[0]
                
                # Calculate similarity
                import numpy as np
                source_vec = np.array(source_embedding)
                comp_vec = np.array(comp_embedding)
                
                similarity = float(np.dot(source_vec, comp_vec) / 
                                 (np.linalg.norm(source_vec) * np.linalg.norm(comp_vec)))
                
                if similarity >= similarity_threshold:
                    similar_queries.append({
                        "query_id": comp_query.id,
                        "query_text": comp_query.query_text,
                        "similarity_score": similarity,
                        "profile_id": comp_query.profile_id,
                        "profile_score": comp_query.profile_score,
                        "query_date": comp_query.query_date.isoformat(),
                        "result_count": comp_query.result_count
                    })
                    
            except Exception as e:
                print(f"Error comparing query {comp_query.id}: {e}")
                continue
        
        # Sort by similarity and limit results
        similar_queries.sort(key=lambda x: x['similarity_score'], reverse=True)
        similar_queries = similar_queries[:max_results]
        
        # Basic cluster analysis
        cluster_info = None
        if len(similar_queries) > 0:
            similarities = [q['similarity_score'] for q in similar_queries]
            cluster_info = {
                "cluster_size": len(similar_queries),
                "avg_similarity": sum(similarities) / len(similarities),
                "min_similarity": min(similarities),
                "max_similarity": max(similarities),
                "common_terms": _extract_common_terms([source_query.query_text] + [q['query_text'] for q in similar_queries])
            }
        
        return EmbeddingSimilarityResponse(
            query_id=query_id,
            query_text=source_query.query_text,
            original_profile_id=source_query.profile_id,
            similar_queries=similar_queries,
            embedding_cluster_info=cluster_info
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Similarity analysis failed: {str(e)}")

def _extract_common_terms(queries: List[str], min_frequency: int = 2) -> List[str]:
    """Extract common terms from a list of queries"""
    try:
        from collections import Counter
        import re
        
        # Simple term extraction (split on whitespace and clean)
        all_terms = []
        for query in queries:
            terms = re.findall(r'\b\w+\b', query.lower())
            all_terms.extend(terms)
        
        # Count term frequency
        term_counts = Counter(all_terms)
        
        # Return terms that appear at least min_frequency times
        common_terms = [term for term, count in term_counts.items() 
                       if count >= min_frequency and len(term) > 2]
        
        return sorted(common_terms, key=lambda x: term_counts[x], reverse=True)[:10]
        
    except Exception:
        return []

@router.delete("/query-performance/{query_id}")
async def delete_query_performance_record(
    query_id: str,
    db: Session = Depends(get_db)
):
    """Delete a specific query performance record"""
    try:
        query_record = db.query(QueryPerformance).filter(QueryPerformance.id == query_id).first()
        if not query_record:
            raise HTTPException(status_code=404, detail="Query performance record not found")
        
        db.delete(query_record)
        db.commit()
        
        return {"message": f"Query performance record {query_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to delete record: {str(e)}")

@router.delete("/query-performance/cleanup")
async def cleanup_old_query_performance_data(
    days_to_keep: int = Query(90, description="Number of days of data to keep"),
    dry_run: bool = Query(True, description="Preview deletion without actually deleting"),
    db: Session = Depends(get_db)
):
    """Clean up old query performance data"""
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
        
        # Count records to be deleted
        records_to_delete = db.query(QueryPerformance).filter(
            QueryPerformance.query_date < cutoff_date
        ).count()
        
        if dry_run:
            return {
                "message": "Dry run completed",
                "records_to_delete": records_to_delete,
                "cutoff_date": cutoff_date.isoformat(),
                "days_to_keep": days_to_keep
            }
        else:
            # Actually delete the records
            deleted_count = db.query(QueryPerformance).filter(
                QueryPerformance.query_date < cutoff_date
            ).delete()
            
            db.commit()
            
            return {
                "message": "Cleanup completed",
                "records_deleted": deleted_count,
                "cutoff_date": cutoff_date.isoformat(),
                "days_to_keep": days_to_keep
            }
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")


@router.get("/query-performance/debug/autoload")
async def debug_with_autoload(db: Session = Depends(get_db)):
    """Test with autoloaded model"""
    try:
        from sqlalchemy import MetaData, Table
        from sqlalchemy.orm import sessionmaker
        from config.database import engine
        
        # Create autoloaded table
        metadata = MetaData()
        autoloaded_table = Table('query_performance', metadata, autoload_with=engine)
        
        # Query using the autoloaded table
        result = db.execute(autoloaded_table.select().limit(1)).first()
        
        if result:
            # Convert to dict to see all fields
            result_dict = dict(result._mapping)
            
            return {
                "message": "Autoloaded table test",
                "autoloaded_columns": [col.name for col in autoloaded_table.columns],
                "autoloaded_columns_count": len(autoloaded_table.columns),
                "sample_data_keys": list(result_dict.keys()),
                "sample_data": {k: str(v)[:50] if v else v for k, v in result_dict.items()}
            }
        else:
            return {
                "message": "No records found",
                "autoloaded_columns": [col.name for col in autoloaded_table.columns],
                "autoloaded_columns_count": len(autoloaded_table.columns)
            }
            
    except Exception as e:
        return {"error": f"Autoload test failed: {str(e)}"}
    
@router.get("/query-performance/debug/force-reflect")
async def debug_with_reflection(db: Session = Depends(get_db)):
    """Force SQLAlchemy to reflect the actual database schema"""
    try:
        from sqlalchemy import MetaData, Table, inspect
        from config.database import engine
        
        # Method 1: Check what's in the database directly
        inspector = inspect(engine)
        db_columns = inspector.get_columns('query_performance')
        
        # Method 2: Force reflection of the table
        metadata = MetaData()
        reflected_table = Table('query_performance', metadata, autoload_with=engine)
        
        # Method 3: Check current model columns
        model_columns = [column.name for column in QueryPerformance.__table__.columns]
        
        # Method 4: Try to get a record with reflection
        result = db.execute(reflected_table.select().limit(1)).first()
        
        return {
            "database_columns_count": len(db_columns),
            "database_columns": [col['name'] for col in db_columns],
            "reflected_table_columns": [col.name for col in reflected_table.columns],
            "model_columns": model_columns,
            "model_columns_count": len(model_columns),
            "sample_record_keys": list(result.keys()) if result else "No records",
            "issue_identified": len(db_columns) != len(model_columns)
        }
        
    except Exception as e:
        return {"error": f"Reflection failed: {str(e)}"}