# backend/api/v1/endpoints/query_performance.py
from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, asc
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, timedelta
import csv
import io
import json
import numpy as np
from config.database import get_db
from config.chroma import get_chroma_instance
from models.database.query_performance_model import QueryPerformance
from models.database.feedback_models import UserFeedback, SearchQualityMetrics
from sentence_transformers import SentenceTransformer

router = APIRouter()


@router.get("/test")
async def test_endpoint():
    """Test endpoint to verify router is working"""
    return {"message": "Query performance router is working"}

@router.get("/query-performance/debug/schema")
async def debug_query_performance_schema(db: Session = Depends(get_db)):
    """Debug endpoint to show what columns exist in QueryPerformance"""
    try:
        sample_record = db.query(QueryPerformance).first()
        
        if sample_record:

            attributes = [attr for attr in dir(sample_record) 
                         if not attr.startswith('_') and not callable(getattr(sample_record, attr))]
            

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
 
        order_field = getattr(QueryPerformance, order_by, QueryPerformance.query_date)
        if order_desc:
            query = query.order_by(desc(order_field))
        else:
            query = query.order_by(asc(order_field))
        
      
        results = query.offset(skip).limit(limit).all()
        
        
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
        
 
        if hasattr(QueryPerformance, 'query_date'):
            query = query.order_by(desc(QueryPerformance.query_date))
        query = query.limit(max_records)
        results = query.all()
        
        if not results:
            total_count = db.query(QueryPerformance).count()
            return {"message": f"No data found with the given filters. Total records in DB: {total_count}"}
 
        def safe_get(obj, attr, default=''):
            return getattr(obj, attr, default) if hasattr(obj, attr) else default

        output = io.StringIO()
        
        
        available_columns = [column.name for column in QueryPerformance.__table__.columns]
        
   
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
                    continue  
                
                if col == 'query_date':
                    value = result.query_date.isoformat() if hasattr(result, 'query_date') and result.query_date else ''
                elif col in ['keywords', 'similarity_scores', 'result_metadata', 'filters_applied']:
                    attr_value = safe_get(result, col)
                    value = json.dumps(attr_value) if attr_value else ''
                else:
                    value = safe_get(result, col)
                
                row_data.append(value)

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
                    
                  
                    if results['embeddings'] and query_perf.query_text:
                        model = SentenceTransformer('all-MiniLM-L6-v2')
                        query_embedding = model.encode([query_perf.query_text])[0]
                        result_embedding = results['embeddings'][0]
                        
                        import numpy as np
                        query_vec = np.array(query_embedding)
                        result_vec = np.array(result_embedding)
                      
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
        

        avg_stats = db.query(
            func.avg(QueryPerformance.search_duration_ms).label('avg_search'),
            func.avg(QueryPerformance.embedding_duration_ms).label('avg_embedding'),
            func.avg(QueryPerformance.result_count).label('avg_results')
        ).filter(QueryPerformance.query_date >= start_date).first()
        

        top_queries = db.query(
            QueryPerformance.query_normalized,
            func.count(QueryPerformance.id).label('count'),
            func.avg(QueryPerformance.profile_score).label('avg_score')
        ).filter(
            QueryPerformance.query_date >= start_date
        ).group_by(
            QueryPerformance.query_normalized
        ).order_by(desc('count')).limit(10).all()
        

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
        

        search_types = db.query(
            QueryPerformance.search_type,
            func.count(QueryPerformance.id).label('count')
        ).filter(
            QueryPerformance.query_date >= start_date
        ).group_by(QueryPerformance.search_type).all()
        
 
        dataset_types = db.query(
            QueryPerformance.dataset_type,
            func.count(QueryPerformance.id).label('count')
        ).filter(
            QueryPerformance.query_date >= start_date
        ).group_by(QueryPerformance.dataset_type).all()
        
        feedback_breakdown = db.query(
            QueryPerformance.feedback_given,
            func.count(QueryPerformance.id).label('count')
        ).filter(
            QueryPerformance.query_date >= start_date
        ).group_by(QueryPerformance.feedback_given).all()
        

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
    
        source_query = db.query(QueryPerformance).filter(QueryPerformance.id == query_id).first()
        if not source_query:
            raise HTTPException(status_code=404, detail="Query not found")
        
        chroma = get_chroma_instance()
        if not chroma.is_available():
            raise HTTPException(status_code=503, detail="ChromaDB not available")
     
        model = SentenceTransformer('all-MiniLM-L6-v2')
        source_embedding = model.encode([source_query.query_text])[0]
        
     
        comparison_queries = db.query(QueryPerformance).filter(
            QueryPerformance.id != query_id,
            QueryPerformance.query_date >= source_query.query_date - timedelta(days=30),
            QueryPerformance.has_results == True
        ).limit(1000).all()  
        
        similar_queries = []
        
        for comp_query in comparison_queries:
            try:
     
                comp_embedding = model.encode([comp_query.query_text])[0]
 
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
        

        similar_queries.sort(key=lambda x: x['similarity_score'], reverse=True)
        similar_queries = similar_queries[:max_results]
   
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
        

        all_terms = []
        for query in queries:
            terms = re.findall(r'\b\w+\b', query.lower())
            all_terms.extend(terms)
        
 
        term_counts = Counter(all_terms)
        

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

@router.get("/query-performance/export/csv-with-feedback")
async def export_query_performance_with_feedback_analysis(
    query_text: Optional[str] = Query(None, description="Filter by query text"),
    profile_id: Optional[str] = Query(None, description="Filter by profile ID"),
    feedback_type: Optional[str] = Query(None, description="Filter by feedback type"),
    start_date: Optional[datetime] = Query(None, description="Filter queries after this date"),
    end_date: Optional[datetime] = Query(None, description="Filter queries before this date"),
    include_embeddings: bool = Query(True, description="Include embedding vectors"),
    include_feedback_analysis: bool = Query(True, description="Include feedback effectiveness analysis"),
    max_records: int = Query(1000, description="Maximum number of records to export"),
    db: Session = Depends(get_db)
):
    """Export query performance data with comprehensive feedback analysis"""
    try:

        query = db.query(QueryPerformance).outerjoin(
            UserFeedback, QueryPerformance.id == UserFeedback.query_performance_id
        ).outerjoin(
            SearchQualityMetrics, 
            QueryPerformance.query_normalized == SearchQualityMetrics.query_normalized
        )

        if query_text:
            query = query.filter(QueryPerformance.query_text.ilike(f"%{query_text}%"))
        if profile_id:
            query = query.filter(QueryPerformance.profile_id == profile_id)
        if feedback_type:
            query = query.filter(UserFeedback.feedback_type == feedback_type)
        if start_date:
            query = query.filter(QueryPerformance.query_date >= start_date)
        if end_date:
            query = query.filter(QueryPerformance.query_date <= end_date)
        
        query = query.order_by(desc(QueryPerformance.query_date)).limit(max_records)
        results = query.all()
        
        if not results:
            raise HTTPException(status_code=404, detail="No data found")
        
        base_headers = [
            'id', 'query_text', 'query_normalized', 'query_date', 'profile_id', 'profile_name',
            'result_count', 'has_results', 'top_result_score', 'search_duration_ms'
        ]

        feedback_headers = [
            'has_feedback', 'feedback_type', 'feedback_timestamp', 'feedback_user_id',
            'original_score_at_feedback', 'score_improvement', 'feedback_delay_seconds',
            'cumulative_positive_feedback', 'cumulative_negative_feedback', 'feedback_ratio',
            'query_learning_effectiveness', 'profile_feedback_count'
        ]

        embedding_headers = []
        if include_embeddings:
            embedding_headers = [
                'query_embedding', 'result_embedding_current', 'result_embedding_original',
                'embedding_similarity_current', 'embedding_similarity_original', 'embedding_drift',
                'feedback_driven_similarity_change'
            ]
        
        all_headers = base_headers + feedback_headers + embedding_headers
        
        chroma = None
        model = None
        if include_embeddings:
            try:
                chroma = get_chroma_instance()
                model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                print(f"ChromaDB not available: {e}")
 
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(all_headers)
        
        for result in results:
            row_data = []

            row_data.extend([
                result.id,
                result.query_text,
                result.query_normalized,
                result.query_date.isoformat() if result.query_date else '',
                result.profile_id,
                result.profile_name,
                result.result_count,
                result.has_results,
                result.top_result_score,
                result.search_duration_ms
            ])
            
    
            feedback_data = get_feedback_analysis_for_record(result, db)
            row_data.extend([
                feedback_data['has_feedback'],
                feedback_data['feedback_type'],
                feedback_data['feedback_timestamp'],
                feedback_data['feedback_user_id'],
                feedback_data['original_score_at_feedback'],
                feedback_data['score_improvement'],
                feedback_data['feedback_delay_seconds'],
                feedback_data['cumulative_positive_feedback'],
                feedback_data['cumulative_negative_feedback'],
                feedback_data['feedback_ratio'],
                feedback_data['query_learning_effectiveness'],
                feedback_data['profile_feedback_count']
            ])
            

            if include_embeddings and chroma and model:
                embedding_data = get_embedding_analysis_for_record(result, chroma, model)
                row_data.extend([
                    json.dumps(embedding_data['query_embedding']) if embedding_data['query_embedding'] else '',
                    json.dumps(embedding_data['result_embedding_current']) if embedding_data['result_embedding_current'] else '',
                    json.dumps(embedding_data['result_embedding_original']) if embedding_data['result_embedding_original'] else '',
                    embedding_data['embedding_similarity_current'],
                    embedding_data['embedding_similarity_original'],
                    embedding_data['embedding_drift'],
                    embedding_data['feedback_driven_similarity_change']
                ])
            elif include_embeddings:
                row_data.extend([''] * len(embedding_headers))
            
            writer.writerow(row_data)
        

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"query_performance_with_feedback_{len(results)}records_{timestamp}.csv"
        
        return Response(
            content=output.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

# Add these missing functions to your endpoint file:
def get_feedback_analysis_for_record_fixed(record, db: Session) -> Dict:
    """Get comprehensive feedback analysis for a single record - using correct schema"""
    try:
        from models.database.feedback_models import UserFeedback
        
        # Get feedback for this record using your actual schema
        feedback = db.query(UserFeedback).filter(
            UserFeedback.query_normalized == record.query_normalized,
            UserFeedback.profile_id == record.profile_id
        ).order_by(UserFeedback.created_at).all()
        
        if not feedback:
            return {
                'has_feedback': False,
                'feedback_type': '',
                'feedback_timestamp': '',
                'feedback_user_id': '',
                'original_score_at_feedback': '',
                'score_improvement': '',
                'feedback_delay_seconds': '',
                'cumulative_positive_feedback': 0,
                'cumulative_negative_feedback': 0,
                'feedback_ratio': 0.0,
                'query_learning_effectiveness': 0.0,
                'profile_feedback_count': 0
            }
        
        latest_feedback = feedback[-1]
        
      
        positive_count = sum(1 for f in feedback if f.feedback_type == 'positive')
        negative_count = sum(1 for f in feedback if f.feedback_type == 'negative')
        total_feedback = positive_count + negative_count
        feedback_ratio = positive_count / total_feedback if total_feedback > 0 else 0.0
        
        score_improvement = ''
        if len(feedback) > 1:
            first_score = feedback[0].original_score
            latest_score = latest_feedback.original_score
            if first_score is not None and latest_score is not None:
                score_improvement = latest_score - first_score
        
   
        feedback_delay = ''
        if record.query_date and latest_feedback.created_at:
            delay = (latest_feedback.created_at - record.query_date).total_seconds()
            feedback_delay = delay
        

        scores = [f.original_score for f in feedback if f.original_score is not None]
        learning_effectiveness = calculate_learning_effectiveness_simple(scores)
        
        return {
            'has_feedback': True,
            'feedback_type': latest_feedback.feedback_type,
            'feedback_timestamp': latest_feedback.created_at.isoformat() if latest_feedback.created_at else '',
            'feedback_user_id': latest_feedback.user_id,
            'original_score_at_feedback': latest_feedback.original_score,
            'score_improvement': score_improvement,
            'feedback_delay_seconds': feedback_delay,
            'cumulative_positive_feedback': positive_count,
            'cumulative_negative_feedback': negative_count,
            'feedback_ratio': feedback_ratio,
            'query_learning_effectiveness': learning_effectiveness,
            'profile_feedback_count': len(feedback)
        }
        
    except Exception as e:
        print(f"Error analyzing feedback for record {record.id}: {e}")
        return {
            'has_feedback': False,
            'feedback_type': '',
            'feedback_timestamp': '',
            'feedback_user_id': '',
            'original_score_at_feedback': '',
            'score_improvement': '',
            'feedback_delay_seconds': '',
            'cumulative_positive_feedback': 0,
            'cumulative_negative_feedback': 0,
            'feedback_ratio': 0.0,
            'query_learning_effectiveness': 0.0,
            'profile_feedback_count': 0
        }

def calculate_engagement_rate_fixed(db: Session, start_date: datetime, end_date: datetime) -> float:
    """Calculate user engagement rate with feedback - using correct schema"""
    try:
        from models.database.feedback_models import UserFeedback
        
        # Total queries in period
        total_queries = db.query(QueryPerformance).filter(
            QueryPerformance.query_date >= start_date,
            QueryPerformance.query_date <= end_date
        ).count()
        
        # Unique queries that received feedback
        queries_with_feedback = db.query(QueryPerformance).join(
            UserFeedback, 
            QueryPerformance.query_normalized == UserFeedback.query_normalized
        ).filter(
            QueryPerformance.query_date >= start_date,
            QueryPerformance.query_date <= end_date,
            UserFeedback.created_at >= start_date,
            UserFeedback.created_at <= end_date
        ).distinct().count()
        
        return (queries_with_feedback / total_queries * 100) if total_queries > 0 else 0.0
        
    except Exception as e:
        print(f"Error calculating engagement rate: {e}")
        return 0.0

def analyze_score_improvements_fixed(feedback_data, db: Session) -> Dict:
    """Analyze how scores improve after feedback - using correct schema"""
    improvements = []
   
    query_profile_groups = {}
    for feedback in feedback_data:
        key = (feedback.query_normalized, feedback.profile_id)
        if key not in query_profile_groups:
            query_profile_groups[key] = []
        query_profile_groups[key].append(feedback)
    
    for (query, profile_id), feedback_list in query_profile_groups.items():
        if len(feedback_list) < 2:
            continue
        

        feedback_list.sort(key=lambda x: x.created_at)
        

        first_score = feedback_list[0].original_score
        last_score = feedback_list[-1].original_score
        
        if first_score is not None and last_score is not None:
            improvement = last_score - first_score
            time_span_days = (feedback_list[-1].created_at - feedback_list[0].created_at).days
            
            improvements.append({
                'query': query,
                'profile_id': profile_id,
                'initial_score': first_score,
                'final_score': last_score,
                'improvement': improvement,
                'feedback_count': len(feedback_list),
                'time_span_days': time_span_days
            })
    
    if not improvements:
        return {
            'average_improvement': 0.0,
            'positive_improvements': 0,
            'negative_improvements': 0,
            'total_analyzed': 0,
            'best_improvements': [],
            'worst_improvements': []
        }
    
    avg_improvement = sum(imp['improvement'] for imp in improvements) / len(improvements)
    positive_improvements = sum(1 for imp in improvements if imp['improvement'] > 0)
    negative_improvements = sum(1 for imp in improvements if imp['improvement'] < 0)
    

    improvements.sort(key=lambda x: x['improvement'], reverse=True)
    
    return {
        'average_improvement': avg_improvement,
        'positive_improvements': positive_improvements,
        'negative_improvements': negative_improvements,
        'total_analyzed': len(improvements),
        'best_improvements': improvements[:10],
        'worst_improvements': improvements[-10:],
        'improvement_distribution': {
            'significant_positive': sum(1 for imp in improvements if imp['improvement'] > 0.1),
            'moderate_positive': sum(1 for imp in improvements if 0.05 < imp['improvement'] <= 0.1),
            'slight_positive': sum(1 for imp in improvements if 0 < imp['improvement'] <= 0.05),
            'no_change': sum(1 for imp in improvements if imp['improvement'] == 0),
            'slight_negative': sum(1 for imp in improvements if -0.05 <= imp['improvement'] < 0),
            'moderate_negative': sum(1 for imp in improvements if -0.1 <= imp['improvement'] < -0.05),
            'significant_negative': sum(1 for imp in improvements if imp['improvement'] < -0.1)
        }
    }

def analyze_query_learning_patterns_fixed(feedback_data) -> Dict:
    """Analyze which types of queries learn best from feedback - using correct schema"""
    query_analysis = {}
    
    for feedback in feedback_data:
        query = feedback.query_normalized
        if query not in query_analysis:
            query_analysis[query] = {
                'total_feedback': 0,
                'positive_feedback': 0,
                'negative_feedback': 0,
                'unique_profiles': set(),
                'score_changes': [],
                'feedback_scores': []
            }
        
        query_analysis[query]['total_feedback'] += 1
        if feedback.feedback_type == 'positive':
            query_analysis[query]['positive_feedback'] += 1
        elif feedback.feedback_type == 'negative':
            query_analysis[query]['negative_feedback'] += 1
        
        query_analysis[query]['unique_profiles'].add(feedback.profile_id)
        if feedback.original_score:
            query_analysis[query]['score_changes'].append(feedback.original_score)
        if feedback.feedback_score:
            query_analysis[query]['feedback_scores'].append(feedback.feedback_score)
    

    query_patterns = []
    for query, data in query_analysis.items():
        total_sentiment = data['positive_feedback'] + data['negative_feedback']
        satisfaction_rate = (data['positive_feedback'] / total_sentiment) * 100 if total_sentiment > 0 else 0
        
        avg_feedback_score = sum(data['feedback_scores']) / len(data['feedback_scores']) if data['feedback_scores'] else 0
        
        query_patterns.append({
            'query': query,
            'total_feedback': data['total_feedback'],
            'satisfaction_rate': satisfaction_rate,
            'unique_profiles_affected': len(data['unique_profiles']),
            'average_feedback_score': avg_feedback_score,
            'learning_effectiveness': calculate_learning_effectiveness_simple(data['score_changes'])
        })
    
   
    query_patterns.sort(key=lambda x: x['total_feedback'], reverse=True)
    
    return {
        'top_feedback_queries': query_patterns[:20],
        'most_satisfied_queries': sorted(query_patterns, key=lambda x: x['satisfaction_rate'], reverse=True)[:10],
        'least_satisfied_queries': sorted(query_patterns, key=lambda x: x['satisfaction_rate'])[:10],
        'best_learning_queries': sorted(query_patterns, key=lambda x: x['learning_effectiveness'], reverse=True)[:10]
    }

def analyze_temporal_trends_fixed(feedback_data, days_back: int) -> Dict:
    """Analyze feedback trends over time - using correct schema"""
    try:
     
        daily_stats = {}
        
        for feedback in feedback_data:
            day = feedback.created_at.date()
            if day not in daily_stats:
                daily_stats[day] = {
                    'positive': 0, 
                    'negative': 0, 
                    'neutral': 0, 
                    'total': 0,
                    'total_score': 0.0,
                    'feedback_scores': []
                }
            
            daily_stats[day]['total'] += 1
            if feedback.feedback_type == 'positive':
                daily_stats[day]['positive'] += 1
            elif feedback.feedback_type == 'negative':
                daily_stats[day]['negative'] += 1
            elif feedback.feedback_type == 'neutral':
                daily_stats[day]['neutral'] += 1
                
            if feedback.original_score:
                daily_stats[day]['total_score'] += feedback.original_score
            if feedback.feedback_score:
                daily_stats[day]['feedback_scores'].append(feedback.feedback_score)
        
     
        daily_trends = []
        for day, stats in daily_stats.items():
            sentiment_total = stats['positive'] + stats['negative']
            satisfaction_rate = (stats['positive'] / sentiment_total) * 100 if sentiment_total > 0 else 0
            
            avg_original_score = stats['total_score'] / stats['total'] if stats['total'] > 0 else 0
            avg_feedback_score = sum(stats['feedback_scores']) / len(stats['feedback_scores']) if stats['feedback_scores'] else 0
            
            daily_trends.append({
                'date': day.isoformat(),
                'total_feedback': stats['total'],
                'positive_feedback': stats['positive'],
                'negative_feedback': stats['negative'],
                'neutral_feedback': stats['neutral'],
                'satisfaction_rate': satisfaction_rate,
                'average_original_score': avg_original_score,
                'average_feedback_score': avg_feedback_score
            })
        
        daily_trends.sort(key=lambda x: x['date'])
        
        return {
            'daily_trends': daily_trends,
            'trend_summary': {
                'total_days_with_feedback': len(daily_trends),
                'average_daily_feedback': sum(d['total_feedback'] for d in daily_trends) / len(daily_trends) if daily_trends else 0,
                'peak_feedback_day': max(daily_trends, key=lambda x: x['total_feedback']) if daily_trends else None,
                'best_satisfaction_day': max(daily_trends, key=lambda x: x['satisfaction_rate']) if daily_trends else None
            }
        }
        
    except Exception as e:
        print(f"Error analyzing temporal trends: {e}")
        return {'daily_trends': [], 'trend_summary': {}}

@router.get("/feedback/analysis/effectiveness")
async def analyze_feedback_effectiveness(
    days_back: int = Query(30, description="Days to analyze"),
    profile_id: Optional[str] = Query(None, description="Specific profile to analyze"),
    db: Session = Depends(get_db)
):
    """Comprehensive analysis of feedback learning effectiveness - FIXED VERSION"""
    try:
        from models.database.feedback_models import UserFeedback
        
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days_back)
        
        # Base query using created_at field
        query = db.query(UserFeedback).filter(
            UserFeedback.created_at >= start_date
        )
        
        if profile_id:
            query = query.filter(UserFeedback.profile_id == profile_id)
        
        feedback_data = query.all()
        
        # Overall metrics
        total_feedback = len(feedback_data)
        positive_feedback = sum(1 for f in feedback_data if f.feedback_type == 'positive')
        negative_feedback = sum(1 for f in feedback_data if f.feedback_type == 'negative')
        neutral_feedback = sum(1 for f in feedback_data if f.feedback_type == 'neutral')
        
        # Average scores
        original_scores = [f.original_score for f in feedback_data if f.original_score is not None]
        feedback_scores = [f.feedback_score for f in feedback_data if f.feedback_score is not None]
        
        avg_original_score = sum(original_scores) / len(original_scores) if original_scores else 0
        avg_feedback_score = sum(feedback_scores) / len(feedback_scores) if feedback_scores else 0
        
        return {
            "analysis_period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "days_analyzed": days_back
            },
            "overall_metrics": {
                "total_feedback_count": total_feedback,
                "positive_feedback_count": positive_feedback,
                "negative_feedback_count": negative_feedback,
                "neutral_feedback_count": neutral_feedback,
                "satisfaction_rate": (positive_feedback / (positive_feedback + negative_feedback)) * 100 if (positive_feedback + negative_feedback) > 0 else 0,
                "average_original_score": avg_original_score,
                "average_feedback_score": avg_feedback_score,
                "feedback_engagement_rate": calculate_engagement_rate_fixed(db, start_date, end_date)
            },
            "score_improvement_analysis": analyze_score_improvements_fixed(feedback_data, db),
            "query_learning_patterns": analyze_query_learning_patterns_fixed(feedback_data),
            "temporal_trends": analyze_temporal_trends_fixed(feedback_data, days_back),
            "recommendations": generate_learning_recommendations_fixed(feedback_data, positive_feedback, negative_feedback, total_feedback)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

def generate_learning_recommendations_fixed(feedback_data, positive_count, negative_count, total_count) -> List[str]:
    """Generate actionable recommendations based on analysis"""
    recommendations = []
    
    if total_count == 0:
        recommendations.append("No feedback data found - consider implementing user feedback collection")
        return recommendations
    
    satisfaction_rate = (positive_count / (positive_count + negative_count)) * 100 if (positive_count + negative_count) > 0 else 0
    
    if satisfaction_rate < 30:
        recommendations.append("Low satisfaction rate (<30%) - review search algorithm and result relevance")
    elif satisfaction_rate < 50:
        recommendations.append("Moderate satisfaction rate - consider fine-tuning search parameters")
    elif satisfaction_rate > 80:
        recommendations.append("High satisfaction rate - system is performing well")
    
    if negative_count > positive_count * 2:
        recommendations.append("High negative feedback ratio - investigate common negative feedback patterns")
    
    if total_count < 50:
        recommendations.append("Low feedback volume - consider incentivizing user feedback or making feedback UI more prominent")
    

    recent_feedback = [f for f in feedback_data if (datetime.utcnow() - f.created_at).days <= 7]
    if len(recent_feedback) < total_count * 0.3:
        recommendations.append("Declining feedback activity - engage users to maintain feedback flow")
    
    if not recommendations:
        recommendations.append("System appears to be learning effectively from feedback")
    
    return recommendations

def analyze_score_improvements(feedback_data, db: Session) -> Dict:
    """Analyze how scores improve after feedback"""
    improvements = []

    query_profile_groups = {}
    for feedback in feedback_data:
        key = (feedback.query_normalized, feedback.profile_id)
        if key not in query_profile_groups:
            query_profile_groups[key] = []
        query_profile_groups[key].append(feedback)
    
    for (query, profile_id), feedback_list in query_profile_groups.items():
        if len(feedback_list) < 2:
            continue
        

        feedback_list.sort(key=lambda x: x.timestamp)

        first_score = feedback_list[0].original_score
        last_score = feedback_list[-1].original_score
        
        if first_score is not None and last_score is not None:
            improvement = last_score - first_score
            improvements.append({
                'query': query,
                'profile_id': profile_id,
                'initial_score': first_score,
                'final_score': last_score,
                'improvement': improvement,
                'feedback_count': len(feedback_list),
                'time_span_days': (feedback_list[-1].timestamp - feedback_list[0].timestamp).days
            })
    
    if not improvements:
        return {
            'average_improvement': 0.0,
            'positive_improvements': 0,
            'negative_improvements': 0,
            'total_analyzed': 0,
            'best_improvements': [],
            'worst_improvements': []
        }
    
    avg_improvement = sum(imp['improvement'] for imp in improvements) / len(improvements)
    positive_improvements = sum(1 for imp in improvements if imp['improvement'] > 0)
    negative_improvements = sum(1 for imp in improvements if imp['improvement'] < 0)
    
    # Sort for best/worst
    improvements.sort(key=lambda x: x['improvement'], reverse=True)
    
    return {
        'average_improvement': avg_improvement,
        'positive_improvements': positive_improvements,
        'negative_improvements': negative_improvements,
        'total_analyzed': len(improvements),
        'best_improvements': improvements[:10],
        'worst_improvements': improvements[-10:],
        'improvement_distribution': {
            'significant_positive': sum(1 for imp in improvements if imp['improvement'] > 0.1),
            'moderate_positive': sum(1 for imp in improvements if 0.05 < imp['improvement'] <= 0.1),
            'slight_positive': sum(1 for imp in improvements if 0 < imp['improvement'] <= 0.05),
            'no_change': sum(1 for imp in improvements if imp['improvement'] == 0),
            'slight_negative': sum(1 for imp in improvements if -0.05 <= imp['improvement'] < 0),
            'moderate_negative': sum(1 for imp in improvements if -0.1 <= imp['improvement'] < -0.05),
            'significant_negative': sum(1 for imp in improvements if imp['improvement'] < -0.1)
        }
    }

def analyze_query_learning_patterns(feedback_data) -> Dict:
    """Analyze which types of queries learn best from feedback"""
    query_analysis = {}

    for feedback in feedback_data:
        query = feedback.query_normalized
        if query not in query_analysis:
            query_analysis[query] = {
                'total_feedback': 0,
                'positive_feedback': 0,
                'negative_feedback': 0,
                'unique_profiles': set(),
                'score_changes': []
            }
            
        query_analysis[query]['total_feedback'] += 1
        if feedback.feedback_type == 'positive':
            query_analysis[query]['positive_feedback'] += 1
        elif feedback.feedback_type == 'negative':
            query_analysis[query]['negative_feedback'] += 1
        
        query_analysis[query]['unique_profiles'].add(feedback.profile_id)
        if feedback.original_score:
            query_analysis[query]['score_changes'].append(feedback.original_score)

  
    query_patterns = []
    for query, data in query_analysis.items():
        satisfaction_rate = (data['positive_feedback'] / 
                        (data['positive_feedback'] + data['negative_feedback'])) * 100 \
                        if (data['positive_feedback'] + data['negative_feedback']) > 0 else 0
        
        query_patterns.append({
            'query': query,
            'total_feedback': data['total_feedback'],
            'satisfaction_rate': satisfaction_rate,
            'unique_profiles_affected': len(data['unique_profiles']),
            'learning_effectiveness': calculate_learning_effectiveness_simple(data['score_changes'])
        })


    query_patterns.sort(key=lambda x: x['total_feedback'], reverse=True)

    return {
        'top_feedback_queries': query_patterns[:20],
        'most_satisfied_queries': sorted(query_patterns, key=lambda x: x['satisfaction_rate'], reverse=True)[:10],
        'least_satisfied_queries': sorted(query_patterns, key=lambda x: x['satisfaction_rate'])[:10],
        'best_learning_queries': sorted(query_patterns, key=lambda x: x['learning_effectiveness'], reverse=True)[:10]
    }

def calculate_learning_effectiveness_simple(scores) -> float:
    """Simple learning effectiveness calculation"""
    if len(scores) < 2:
        return 0.0


    improvements = []
    for i in range(1, len(scores)):
        improvements.append(scores[i] - scores[i-1])

    return sum(improvements) / len(improvements) if improvements else 0.0

def get_embedding_analysis_for_record(record, chroma, model) -> Dict:
    """Get embedding analysis including before/after feedback changes"""
    try:
        if not record.top_result_id:
            return {
                'query_embedding': None,
                'result_embedding_current': None,
                'result_embedding_original': None,
                'embedding_similarity_current': None,
                'embedding_similarity_original': None,
                'embedding_drift': None,
                'feedback_driven_similarity_change': None
            }
        

        collection = chroma.get_collection()
        current_results = collection.get(
            ids=[record.top_result_id],
            include=['embeddings', 'metadatas']
        )
        
        if not current_results['ids']:
            return {
                'query_embedding': None,
                'result_embedding_current': None,
                'result_embedding_original': None,
                'embedding_similarity_current': None,
                'embedding_similarity_original': None,
                'embedding_drift': None,
                'feedback_driven_similarity_change': None
            }
        

        query_embedding = model.encode([record.query_text])[0]
        current_embedding = current_results['embeddings'][0]
        
        current_similarity = np.dot(query_embedding, current_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(current_embedding)
        )
        
 
        original_embedding = current_embedding  
        original_similarity = current_similarity
        embedding_drift = 0.0  # not setup 
        similarity_change = 0.0 #not setup
        
        return {
            'query_embedding': query_embedding.tolist(),
            'result_embedding_current': current_embedding.tolist() if hasattr(current_embedding, 'tolist') else list(current_embedding),
            'result_embedding_original': original_embedding.tolist() if hasattr(original_embedding, 'tolist') else list(original_embedding),
            'embedding_similarity_current': float(current_similarity),
            'embedding_similarity_original': float(original_similarity),
            'embedding_drift': float(embedding_drift),
            'feedback_driven_similarity_change': float(similarity_change)
        }
        
    except Exception as e:
        print(f"Error analyzing embeddings for record {record.id}: {e}")
        return {
            'query_embedding': None,
            'result_embedding_current': None,
            'result_embedding_original': None,
            'embedding_similarity_current': None,
            'embedding_similarity_original': None,
            'embedding_drift': None,
            'feedback_driven_similarity_change': None
        }

def get_feedback_analysis_for_record(record, db: Session) -> Dict:
    """Get comprehensive feedback analysis for a single record"""
    try:
        from models.database.feedback_models import UserFeedback
   
        feedback = db.query(UserFeedback).filter(
            UserFeedback.query_normalized == record.query_normalized,
            UserFeedback.profile_id == record.profile_id
        ).order_by(UserFeedback.created_at).all()
        
        if not feedback:
            return {
                'has_feedback': False,
                'feedback_type': '',
                'feedback_timestamp': '',
                'feedback_user_id': '',
                'original_score_at_feedback': '',
                'score_improvement': '',
                'feedback_delay_seconds': '',
                'cumulative_positive_feedback': 0,
                'cumulative_negative_feedback': 0,
                'feedback_ratio': 0.0,
                'query_learning_effectiveness': 0.0,
                'profile_feedback_count': 0
            }
        
        latest_feedback = feedback[-1]
        

        positive_count = sum(1 for f in feedback if f.feedback_type == 'positive')
        negative_count = sum(1 for f in feedback if f.feedback_type == 'negative')
        total_feedback = positive_count + negative_count
        feedback_ratio = positive_count / total_feedback if total_feedback > 0 else 0.0

        score_improvement = ''
        if len(feedback) > 1:
            first_score = feedback[0].original_score
            latest_score = latest_feedback.original_score
            if first_score is not None and latest_score is not None:
                score_improvement = latest_score - first_score

        feedback_delay = ''
        if record.query_date and latest_feedback.created_at:
            delay = (latest_feedback.created_at - record.query_date).total_seconds()
            feedback_delay = delay

        scores = [f.original_score for f in feedback if f.original_score is not None]
        learning_effectiveness = calculate_learning_effectiveness_simple(scores)
        
        return {
            'has_feedback': True,
            'feedback_type': latest_feedback.feedback_type,
            'feedback_timestamp': latest_feedback.created_at.isoformat() if latest_feedback.created_at else '',
            'feedback_user_id': latest_feedback.user_id or '',
            'original_score_at_feedback': latest_feedback.original_score,
            'score_improvement': score_improvement,
            'feedback_delay_seconds': feedback_delay,
            'cumulative_positive_feedback': positive_count,
            'cumulative_negative_feedback': negative_count,
            'feedback_ratio': feedback_ratio,
            'query_learning_effectiveness': learning_effectiveness,
            'profile_feedback_count': len(feedback)
        }
        
    except Exception as e:
        print(f"Error analyzing feedback for record {record.id}: {e}")
        return {
            'has_feedback': False,
            'feedback_type': '',
            'feedback_timestamp': '',
            'feedback_user_id': '',
            'original_score_at_feedback': '',
            'score_improvement': '',
            'feedback_delay_seconds': '',
            'cumulative_positive_feedback': 0,
            'cumulative_negative_feedback': 0,
            'feedback_ratio': 0.0,
            'query_learning_effectiveness': 0.0,
            'profile_feedback_count': 0
        }

@router.get("/query-performance/export/csv-with-feedback-fixed")
async def export_query_performance_with_feedback_analysis_fixed(
    query_text: Optional[str] = Query(None, description="Filter by query text"),
    profile_id: Optional[str] = Query(None, description="Filter by profile ID"),
    feedback_type: Optional[str] = Query(None, description="Filter by feedback type"),
    start_date: Optional[datetime] = Query(None, description="Filter queries after this date"),
    end_date: Optional[datetime] = Query(None, description="Filter queries before this date"),
    include_embeddings: bool = Query(True, description="Include embedding vectors"),
    include_feedback_analysis: bool = Query(True, description="Include feedback effectiveness analysis"),
    max_records: int = Query(1000, description="Maximum number of records to export"),
    db: Session = Depends(get_db)
):
    """Export query performance data with comprehensive feedback analysis - FIXED"""
    try:
        from models.database.feedback_models import UserFeedback
        

        query = db.query(QueryPerformance)

        if query_text:
            query = query.filter(QueryPerformance.query_text.ilike(f"%{query_text}%"))
        if profile_id:
            query = query.filter(QueryPerformance.profile_id == profile_id)
        if start_date:
            query = query.filter(QueryPerformance.query_date >= start_date)
        if end_date:
            query = query.filter(QueryPerformance.query_date <= end_date)
        
        query = query.order_by(desc(QueryPerformance.query_date)).limit(max_records)
        results = query.all()

        if feedback_type:
            filtered_results = []
            for result in results:
                feedback_exists = db.query(UserFeedback).filter(
                    UserFeedback.query_normalized == result.query_normalized,
                    UserFeedback.profile_id == result.profile_id,
                    UserFeedback.feedback_type == feedback_type
                ).first()
                if feedback_exists:
                    filtered_results.append(result)
            results = filtered_results
        
        if not results:
            raise HTTPException(status_code=404, detail="No data found")
        
        # Prepare CSV headers
        base_headers = [
            'id', 'query_text', 'query_normalized', 'query_date', 'profile_id', 'profile_name',
            'result_count', 'has_results', 'top_result_score', 'search_duration_ms'
        ]
        
        # Add feedback analysis headers
        feedback_headers = []
        if include_feedback_analysis:
            feedback_headers = [
                'has_feedback', 'feedback_type', 'feedback_timestamp', 'feedback_user_id',
                'original_score_at_feedback', 'feedback_score', 'score_improvement', 'feedback_delay_seconds',
                'cumulative_positive_feedback', 'cumulative_negative_feedback', 'feedback_ratio',
                'profile_feedback_count'
            ]

        embedding_headers = []
        if include_embeddings:
            embedding_headers = [
                'query_embedding', 'result_embedding_current', 'embedding_similarity_current'
            ]
        
        all_headers = base_headers + feedback_headers + embedding_headers

        chroma = None
        model = None
        if include_embeddings:
            try:
                chroma = get_chroma_instance()
                model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                print(f"ChromaDB not available: {e}")

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(all_headers)
        
        for result in results:
            row_data = []
 
            row_data.extend([
                result.id,
                result.query_text,
                result.query_normalized or '',
                result.query_date.isoformat() if result.query_date else '',
                result.profile_id,
                result.profile_name or '',
                result.result_count,
                result.has_results,
                result.top_result_score,
                result.search_duration_ms
            ])
            
    
            if include_feedback_analysis:

                feedback_records = db.query(UserFeedback).filter(
                    UserFeedback.query_normalized == result.query_normalized,
                    UserFeedback.profile_id == result.profile_id
                ).order_by(UserFeedback.created_at).all()
                
                if feedback_records:
                    latest_feedback = feedback_records[-1]
                    positive_count = sum(1 for f in feedback_records if f.feedback_type == 'positive')
                    negative_count = sum(1 for f in feedback_records if f.feedback_type == 'negative')
                    total_sentiment = positive_count + negative_count
                    feedback_ratio = positive_count / total_sentiment if total_sentiment > 0 else 0.0
   
                    score_improvement = ''
                    if len(feedback_records) > 1:
                        first_score = feedback_records[0].original_score
                        latest_score = latest_feedback.original_score
                        if first_score is not None and latest_score is not None:
                            score_improvement = latest_score - first_score
                    
                
                    feedback_delay = ''
                    if result.query_date and latest_feedback.created_at:
                        delay = (latest_feedback.created_at - result.query_date).total_seconds()
                        feedback_delay = delay
                    
                    row_data.extend([
                        True,  
                        latest_feedback.feedback_type,
                        latest_feedback.created_at.isoformat() if latest_feedback.created_at else '',
                        latest_feedback.user_id or '',
                        latest_feedback.original_score,
                        latest_feedback.feedback_score,
                        score_improvement,
                        feedback_delay,
                        positive_count,
                        negative_count,
                        feedback_ratio,
                        len(feedback_records)
                    ])
                else:
                 
                    row_data.extend([
                        False,  # has_feedback
                        '', '', '', '', '', '', '',  # empty feedback fields
                        0, 0, 0.0, 0  
                    ])
            

            if include_embeddings and chroma and model:
                try:
                    query_embedding = model.encode([result.query_text])[0]
                    
                    result_embedding = None
                    similarity = None
                    
                    if result.top_result_id:
                        collection = chroma.get_collection()
                        chroma_results = collection.get(
                            ids=[result.top_result_id],
                            include=['embeddings']
                        )
                        if chroma_results['ids']:
                            result_embedding = chroma_results['embeddings'][0]
                        
                            similarity = np.dot(query_embedding, result_embedding) / (
                                np.linalg.norm(query_embedding) * np.linalg.norm(result_embedding)
                            )
                    
                    row_data.extend([
                        json.dumps(query_embedding.tolist()),
                        json.dumps(result_embedding.tolist()) if result_embedding is not None else '',
                        float(similarity) if similarity is not None else ''
                    ])
                except Exception as e:
                    print(f"Error getting embeddings for {result.id}: {e}")
                    row_data.extend(['', '', ''])
            elif include_embeddings:
                row_data.extend(['', '', ''])
            
            writer.writerow(row_data)
        
       
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"query_performance_with_feedback_{len(results)}records_{timestamp}.csv"
        
        return Response(
            content=output.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")
    

@router.get("/feedback/analysis/effectiveness/export")
async def export_feedback_effectiveness_analysis(
    days_back: int = Query(30, description="Days to analyze"),
    profile_id: Optional[str] = Query(None, description="Specific profile to analyze"),
    analysis_type: str = Query("summary", description="Type of analysis: summary, query_patterns, profile_performance, temporal"),
    db: Session = Depends(get_db)
):
    """Export feedback effectiveness analysis to CSV"""
    try:
        from models.database.feedback_models import UserFeedback
        
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days_back)
        
 
        query = db.query(UserFeedback).filter(
            UserFeedback.created_at >= start_date
        )
        
        if profile_id:
            query = query.filter(UserFeedback.profile_id == profile_id)
        
        feedback_data = query.all()
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        if analysis_type == "summary":
        
            headers = [
                'metric', 'value', 'description'
            ]
            writer.writerow(headers)
            
            total_feedback = len(feedback_data)
            positive_feedback = sum(1 for f in feedback_data if f.feedback_type == 'positive')
            negative_feedback = sum(1 for f in feedback_data if f.feedback_type == 'negative')
            neutral_feedback = sum(1 for f in feedback_data if f.feedback_type == 'neutral')
            
            satisfaction_rate = (positive_feedback / (positive_feedback + negative_feedback)) * 100 if (positive_feedback + negative_feedback) > 0 else 0
            
            original_scores = [f.original_score for f in feedback_data if f.original_score is not None]
            feedback_scores = [f.feedback_score for f in feedback_data if f.feedback_score is not None]
            
            avg_original_score = sum(original_scores) / len(original_scores) if original_scores else 0
            avg_feedback_score = sum(feedback_scores) / len(feedback_scores) if feedback_scores else 0
            
            metrics = [
                ['total_feedback_count', total_feedback, 'Total number of feedback records'],
                ['positive_feedback_count', positive_feedback, 'Number of positive feedback'],
                ['negative_feedback_count', negative_feedback, 'Number of negative feedback'],
                ['neutral_feedback_count', neutral_feedback, 'Number of neutral feedback'],
                ['satisfaction_rate_percent', round(satisfaction_rate, 2), 'Percentage of positive feedback'],
                ['average_original_score', round(avg_original_score, 3), 'Average original search score'],
                ['average_feedback_score', round(avg_feedback_score, 3), 'Average user-provided feedback score'],
                ['analysis_period_days', days_back, 'Number of days analyzed'],
                ['start_date', start_date.isoformat(), 'Analysis start date'],
                ['end_date', end_date.isoformat(), 'Analysis end date']
            ]
            
            for metric in metrics:
                writer.writerow(metric)
        
        elif analysis_type == "query_patterns":
            
            headers = [
                'query_normalized', 'total_feedback', 'positive_feedback', 'negative_feedback', 
                'neutral_feedback', 'satisfaction_rate', 'unique_profiles_affected', 
                'average_original_score', 'average_feedback_score'
            ]
            writer.writerow(headers)
            
            query_analysis = {}
            for feedback in feedback_data:
                query = feedback.query_normalized
                if query not in query_analysis:
                    query_analysis[query] = {
                        'total_feedback': 0,
                        'positive_feedback': 0,
                        'negative_feedback': 0,
                        'neutral_feedback': 0,
                        'unique_profiles': set(),
                        'original_scores': [],
                        'feedback_scores': []
                    }
                
                query_analysis[query]['total_feedback'] += 1
                if feedback.feedback_type == 'positive':
                    query_analysis[query]['positive_feedback'] += 1
                elif feedback.feedback_type == 'negative':
                    query_analysis[query]['negative_feedback'] += 1
                elif feedback.feedback_type == 'neutral':
                    query_analysis[query]['neutral_feedback'] += 1
                
                query_analysis[query]['unique_profiles'].add(feedback.profile_id)
                if feedback.original_score:
                    query_analysis[query]['original_scores'].append(feedback.original_score)
                if feedback.feedback_score:
                    query_analysis[query]['feedback_scores'].append(feedback.feedback_score)
            
            
            for query, data in query_analysis.items():
                total_sentiment = data['positive_feedback'] + data['negative_feedback']
                satisfaction_rate = (data['positive_feedback'] / total_sentiment) * 100 if total_sentiment > 0 else 0
                
                avg_original_score = sum(data['original_scores']) / len(data['original_scores']) if data['original_scores'] else 0
                avg_feedback_score = sum(data['feedback_scores']) / len(data['feedback_scores']) if data['feedback_scores'] else 0
                
                writer.writerow([
                    query,
                    data['total_feedback'],
                    data['positive_feedback'],
                    data['negative_feedback'],
                    data['neutral_feedback'],
                    round(satisfaction_rate, 2),
                    len(data['unique_profiles']),
                    round(avg_original_score, 3),
                    round(avg_feedback_score, 3)
                ])
        
        elif analysis_type == "profile_performance":
           
            headers = [
                'profile_id', 'total_feedback', 'positive_feedback', 'negative_feedback', 
                'neutral_feedback', 'satisfaction_rate', 'unique_queries', 
                'average_original_score', 'average_feedback_score'
            ]
            writer.writerow(headers)
            
            profile_analysis = {}
            for feedback in feedback_data:
                profile_id = feedback.profile_id
                if profile_id not in profile_analysis:
                    profile_analysis[profile_id] = {
                        'total_feedback': 0,
                        'positive_feedback': 0,
                        'negative_feedback': 0,
                        'neutral_feedback': 0,
                        'unique_queries': set(),
                        'original_scores': [],
                        'feedback_scores': []
                    }
                
                profile_analysis[profile_id]['total_feedback'] += 1
                if feedback.feedback_type == 'positive':
                    profile_analysis[profile_id]['positive_feedback'] += 1
                elif feedback.feedback_type == 'negative':
                    profile_analysis[profile_id]['negative_feedback'] += 1
                elif feedback.feedback_type == 'neutral':
                    profile_analysis[profile_id]['neutral_feedback'] += 1
                
                profile_analysis[profile_id]['unique_queries'].add(feedback.query_normalized)
                if feedback.original_score:
                    profile_analysis[profile_id]['original_scores'].append(feedback.original_score)
                if feedback.feedback_score:
                    profile_analysis[profile_id]['feedback_scores'].append(feedback.feedback_score)
            
            for profile_id, data in profile_analysis.items():
                total_sentiment = data['positive_feedback'] + data['negative_feedback']
                satisfaction_rate = (data['positive_feedback'] / total_sentiment) * 100 if total_sentiment > 0 else 0
                
                avg_original_score = sum(data['original_scores']) / len(data['original_scores']) if data['original_scores'] else 0
                avg_feedback_score = sum(data['feedback_scores']) / len(data['feedback_scores']) if data['feedback_scores'] else 0
                
                writer.writerow([
                    profile_id,
                    data['total_feedback'],
                    data['positive_feedback'],
                    data['negative_feedback'],
                    data['neutral_feedback'],
                    round(satisfaction_rate, 2),
                    len(data['unique_queries']),
                    round(avg_original_score, 3),
                    round(avg_feedback_score, 3)
                ])
        
        elif analysis_type == "temporal":
         
            headers = [
                'date', 'total_feedback', 'positive_feedback', 'negative_feedback', 
                'neutral_feedback', 'satisfaction_rate', 'average_original_score', 
                'average_feedback_score'
            ]
            writer.writerow(headers)
            
            daily_stats = {}
            for feedback in feedback_data:
                day = feedback.created_at.date()
                if day not in daily_stats:
                    daily_stats[day] = {
                        'total_feedback': 0,
                        'positive_feedback': 0,
                        'negative_feedback': 0,
                        'neutral_feedback': 0,
                        'original_scores': [],
                        'feedback_scores': []
                    }
                
                daily_stats[day]['total_feedback'] += 1
                if feedback.feedback_type == 'positive':
                    daily_stats[day]['positive_feedback'] += 1
                elif feedback.feedback_type == 'negative':
                    daily_stats[day]['negative_feedback'] += 1
                elif feedback.feedback_type == 'neutral':
                    daily_stats[day]['neutral_feedback'] += 1
                
                if feedback.original_score:
                    daily_stats[day]['original_scores'].append(feedback.original_score)
                if feedback.feedback_score:
                    daily_stats[day]['feedback_scores'].append(feedback.feedback_score)
     
            daily_rows = []
            for day, data in daily_stats.items():
                total_sentiment = data['positive_feedback'] + data['negative_feedback']
                satisfaction_rate = (data['positive_feedback'] / total_sentiment) * 100 if total_sentiment > 0 else 0
                
                avg_original_score = sum(data['original_scores']) / len(data['original_scores']) if data['original_scores'] else 0
                avg_feedback_score = sum(data['feedback_scores']) / len(data['feedback_scores']) if data['feedback_scores'] else 0
                
                daily_rows.append([
                    day.isoformat(),
                    data['total_feedback'],
                    data['positive_feedback'],
                    data['negative_feedback'],
                    data['neutral_feedback'],
                    round(satisfaction_rate, 2),
                    round(avg_original_score, 3),
                    round(avg_feedback_score, 3)
                ])
            
   
            daily_rows.sort(key=lambda x: x[0])
            for row in daily_rows:
                writer.writerow(row)

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"feedback_effectiveness_{analysis_type}_{days_back}days_{timestamp}.csv"
        
        return Response(
            content=output.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Effectiveness export failed: {str(e)}")