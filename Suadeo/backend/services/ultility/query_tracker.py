import time
from typing import Dict, List, Optional, Any
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, Integer
from models.database.query_performance_model import QueryPerformance
from datetime import datetime
import uuid

class QueryTracker:
    """Service for tracking search query performance and analytics"""
    
    def __init__(self):
        self.current_query_id = None
        self.start_time = None
        self.embedding_start = None
        self.db_start = None
        self.chroma_start = None
    
    def start_query_tracking(self, query_text: str, search_type: str = 'semantic', 
                           dataset_type: str = 'v2fhir', user_session: str = 'default') -> str:
        """Start tracking a new query"""
        self.current_query_id = str(uuid.uuid4())
        self.start_time = time.time()
        
        self.query_data = {
            'id': self.current_query_id,
            'query_text': query_text,
            'query_normalized': query_text.lower().strip(),
            'search_type': search_type,
            'dataset_type': dataset_type,
            'user_session': user_session,
            'query_date': datetime.utcnow()
        }
        
        return self.current_query_id
    
    def mark_embedding_start(self):
        """Mark the start of embedding generation"""
        self.embedding_start = time.time()
    
    def mark_embedding_end(self):
        """Mark the end of embedding generation"""
        if self.embedding_start:
            duration = int((time.time() - self.embedding_start) * 1000)
            self.query_data['embedding_duration_ms'] = duration
    
    def mark_chroma_start(self):
        """Mark the start of Chroma vector search"""
        self.chroma_start = time.time()
    
    def mark_chroma_end(self):
        """Mark the end of Chroma vector search"""
        if self.chroma_start:
            duration = int((time.time() - self.chroma_start) * 1000)
            self.query_data['chroma_query_duration_ms'] = duration
    
    def mark_db_start(self):
        """Mark the start of database query"""
        self.db_start = time.time()
    
    def mark_db_end(self):
        """Mark the end of database query"""
        if self.db_start:
            duration = int((time.time() - self.db_start) * 1000)
            self.query_data['db_query_duration_ms'] = duration
    
    def complete_query_tracking(self, results: List[Dict], filters: Optional[Dict] = None, 
                              db: Session = None) -> bool:
        """Complete query tracking and save to database"""
        if not self.start_time or not db:
            return False
        
        try:
            
            total_duration = int((time.time() - self.start_time) * 1000)
            
          
            result_count = len(results) if results else 0
            top_result = results[0] if results else None
            
           
            self.query_data.update({
                'search_duration_ms': total_duration,
                'result_count': result_count,
                'has_results': result_count > 0,
                'similarity_scores': [r.get('similarity_score', 0) for r in results] if results else [],
                'filters_applied': filters or {},

                'context_score': None,  
                'combined_score': None,  
                'match_reasons': None, 
                'keywords': None,  
            })
            

            if top_result:
                self.query_data.update({
                    'profile_id': top_result.get('id'),
                    'profile_name': top_result.get('name', top_result.get('resource', 'Unknown')),
                    'profile_oid': top_result.get('oid', top_result.get('local_id', '')),
                    'profile_score': top_result.get('similarity_score', 0),
                    'top_result_id': top_result.get('id'),
                    'top_result_score': top_result.get('similarity_score', 0),
                    'top_result_type': top_result.get('resource', top_result.get('resource_type', 'unknown')),
                    
                    'keywords': top_result.get('keywords', top_result.get('metadata', {}).get('keywords', [])),
              
                    'match_reasons': top_result.get('match_reason', top_result.get('explanation'))
                })
                

                if self.query_data.get('profile_score') and filters:
         
                    base_score = self.query_data['profile_score']
                    context_bonus = 0.1 if filters.get('filter_applied') else 0
                    self.query_data['combined_score'] = min(1.0, base_score + context_bonus)
                    self.query_data['context_score'] = context_bonus
                
                
                metadata = {
                    'local_id': top_result.get('local_id'),
                    'fhir_version': top_result.get('fhir_version'),
                    'hl7v2_version': top_result.get('hl7v2_field_version'),
                    'dataset_id': top_result.get('dataset_id')
                }
                self.query_data['result_metadata'] = {k: v for k, v in metadata.items() if v}
            else:
      
                self.query_data.update({
                    'profile_id': 'no_result',
                    'profile_name': 'No Results Found',
                    'profile_oid': '',
                    'profile_score': 0.0,
                    'keywords': [],  # Add this
                    'match_reasons': 'No results found'  # Add this
                })
            
         
            query_performance = QueryPerformance(**self.query_data)
            db.add(query_performance)
            db.commit()
            
            print(f"✅ Query tracking saved: {self.query_data['query_text']} - {result_count} results in {total_duration}ms")
            return True
            
        except Exception as e:
            print(f"❌ Error saving query tracking: {e}")
            if db:
                db.rollback()
            return False
        finally:
       
            self._reset_tracking()
    
    def _reset_tracking(self):
        """Reset tracking state for next query"""
        self.current_query_id = None
        self.start_time = None
        self.embedding_start = None
        self.db_start = None
        self.chroma_start = None
        self.query_data = {}
    
    def get_query_analytics(self, db: Session, days: int = 30) -> Dict[str, Any]:
        """Get query analytics for the specified period"""
        try:
            from sqlalchemy import func, desc
            from datetime import timedelta
            
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            # Basic query stats
            total_queries = db.query(QueryPerformance).filter(
                QueryPerformance.query_date >= cutoff_date
            ).count()
            
            successful_queries = db.query(QueryPerformance).filter(
                QueryPerformance.query_date >= cutoff_date,
                QueryPerformance.has_results == True
            ).count()
            
            # Average performance metrics
            avg_duration = db.query(func.avg(QueryPerformance.search_duration_ms)).filter(
                QueryPerformance.query_date >= cutoff_date
            ).scalar() or 0
            
            # Most common queries (using normalized for better grouping)
            common_queries = db.query(
                QueryPerformance.query_normalized,
                func.count(QueryPerformance.id).label('count')
            ).filter(
                QueryPerformance.query_date >= cutoff_date,
                QueryPerformance.query_normalized.isnot(None)
            ).group_by(QueryPerformance.query_normalized).order_by(desc('count')).limit(10).all()
            
            # Success rate by dataset type
            dataset_stats = db.query(
                QueryPerformance.dataset_type,
                func.count(QueryPerformance.id).label('total'),
                func.sum(func.cast(QueryPerformance.has_results, Integer)).label('successful')
            ).filter(
                QueryPerformance.query_date >= cutoff_date
            ).group_by(QueryPerformance.dataset_type).all()
            
            return {
                'period_days': days,
                'total_queries': total_queries,
                'successful_queries': successful_queries,
                'success_rate': successful_queries / total_queries if total_queries > 0 else 0,
                'avg_duration_ms': round(avg_duration, 2),
                'common_queries': [{'query': q[0], 'count': q[1]} for q in common_queries],
                'dataset_performance': [
                    {
                        'dataset': ds[0],
                        'total': ds[1],
                        'successful': ds[2] or 0,
                        'success_rate': (ds[2] or 0) / ds[1] if ds[1] > 0 else 0
                    } for ds in dataset_stats
                ]
            }
            
        except Exception as e:
            print(f"❌ Error getting query analytics: {e}")
            return {}