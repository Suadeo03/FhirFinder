from datetime import datetime, timedelta
from typing import List, Dict, Optional
from sqlalchemy.orm import Session
from sqlalchemy import func
import numpy as np
from sentence_transformers import SentenceTransformer
from models.database.feedback_models import UserFeedback, SearchQualityMetrics
from models.database.models import Profile
from config.chroma import get_chroma_instance, is_chroma_available

class FeedbackTraining:
    def __init__(self):
        """Initialize with singleton Chroma collection"""
        try:
            self.chroma_config = get_chroma_instance()
            self.collection = self.chroma_config.get_collection()
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ FeedbackTraining connected to Chroma singleton")
            
            self.feedback_weights = {
                'positive': 0.1,    
                'negative': -0.05,  
            }
        except Exception as e:
            print(f"FeedbackTraining failed to connect to Chroma: {e}")
            self.collection = None
            self.chroma_config = None

        
    def record_user_feedback(self, query: str, profile_id: str, feedback_type: str, 
                       user_id: str, session_id: str, original_score: float, 
                       db: Session, context_info: Optional[Dict] = None):

    
        profile = db.query(Profile).filter(Profile.id == profile_id).first()
        
        if not profile:
            print(f"Profile {profile_id} not found in database")
            return {"status": "error", "message": "Profile not found"}
        
        if not profile.is_active:
            print(f"Profile {profile_id} is inactive, skipping embedding update")
            self._store_feedback_record(query, profile_id, feedback_type, user_id, 
                                    session_id, original_score, db, context_info)
            return {"status": "recorded", "message": "Feedback recorded for inactive profile (no embedding update)"}
        

        self._store_feedback_record(query, profile_id, feedback_type, user_id, 
                                session_id, original_score, db, context_info)
        
        if feedback_type in ['positive', 'negative']:
            self._update_profile_embedding(query, profile_id, feedback_type, db)
        
        if feedback_type == 'negative':
            self._invalidate_query_cache(query)
            print(f"Cache invalidated due to negative feedback")
        elif feedback_type == 'positive':
            self._extend_cache_if_exists(query)
            print(f"Cache extended due to positive feedback")
        
        return {"status": "success", "message": "Feedback recorded and processed"}
            
    def _extend_cache_if_exists(self, query: str):
        """Extend cache lifetime for positive feedback"""
        try:
            from config.redis_cache import RedisQueryCache
            redis_client = RedisQueryCache()
            query_normalized = query.lower().strip()

            cached_data = redis_client.get_cached_feedback(query_normalized)
            if cached_data:

                redis_client.set_feedback(query_normalized, cached_data, 86400)
                print(f"Extended cache for positive feedback on: {query}")
        except Exception as e:
            print(f"Error extending cache: {e}")

    def _update_profile_embedding(self, query: str, profile_id: str, feedback_type: str, db: Session):
        try:
            if not self.collection:
                print(f"No collection available")
                return

            current_results = self.collection.get(
                ids=[profile_id],
                include=['embeddings', 'metadatas']
            )
            
            if not current_results or current_results.get('embeddings') is None:
                print(f"No results or embeddings from Chroma")
                return
                
  
            embeddings = current_results.get('embeddings')
            if isinstance(embeddings, np.ndarray) and embeddings.size == 0:
                print(f"Empty embeddings array")
                return
            
            if not self._is_valid_embedding_result(current_results):
                print(f"Invalid embedding result for {profile_id}")
                return
                
            current_embedding = np.array(current_results['embeddings'][0])
            
            # Validate the embedding array
            if current_embedding.size == 0:
                print(f"Empty embedding array for profile {profile_id}")
                return
                
            if np.any(np.isnan(current_embedding)):
                print(f"NaN values in embedding for profile {profile_id}")
                return
            

            metadatas = current_results.get('metadatas')
            if metadatas is not None and len(metadatas) > 0:
                metadata = metadatas[0] if metadatas[0] is not None else {}
            else:
                metadata = {}

            query_embedding = np.array(self.model.encode([query])[0])
            
   
            if query_embedding.size == 0 or np.any(np.isnan(query_embedding)):
                print(f"Invalid query embedding for query: {query}")
                return           
            if feedback_type == 'positive':

                direction = query_embedding - current_embedding
                weight = self.feedback_weights['positive']
            else:  
                direction = current_embedding - query_embedding
                weight = abs(self.feedback_weights['negative'])

            updated_embedding = current_embedding + (weight * direction)

            embedding_norm = np.linalg.norm(updated_embedding)
            if embedding_norm > 1e-8:  
                updated_embedding = updated_embedding / embedding_norm
            else:
                print(f"Warning: Near-zero embedding norm for profile {profile_id}, keeping original")
                updated_embedding = current_embedding

            if np.any(np.isnan(updated_embedding)) or np.any(np.isinf(updated_embedding)):
                print(f"Invalid updated embedding for profile {profile_id}, keeping original")
                updated_embedding = current_embedding

            updated_metadata = {
                **metadata,
                'last_feedback_update': datetime.utcnow().isoformat(),
                'feedback_count': metadata.get('feedback_count', 0) + 1
            }

            self.collection.update(
                ids=[profile_id],
                embeddings=[updated_embedding.tolist()],
                metadatas=[updated_metadata]
            )
            
            print(f"Successfully updated embedding for profile {profile_id} based on {feedback_type} feedback")
                
        except Exception as e:
                print(f"Error updating profile embedding: {e}")
                import traceback
                traceback.print_exc()

    def _is_valid_embedding_result(self, results) -> bool:
        try:
            if results is None:
                print("No results returned from Chroma")
                return False
            embeddings = results.get('embeddings')
            if embeddings is None:
                print("No embeddings found in results")
                return False
            if isinstance(embeddings, np.ndarray):
                first_embedding = embeddings
            elif isinstance(embeddings, (list, tuple)):
                if len(embeddings) == 0:
                    print("Empty embeddings list")
                    return False
                first_embedding = embeddings[0]
            else:
                return False
            
            if first_embedding is None:
                print("First embedding is None")
                return False
            
            if isinstance(first_embedding, np.ndarray):
                embedding_array = first_embedding
            else:
                embedding_array = np.array(first_embedding)
                
            if embedding_array.size == 0:
                print("Embedding array is empty")
                return False
                
            if np.any(np.isnan(embedding_array)):
                print("Embedding contains NaN values")
                return False
                
            if np.any(np.isinf(embedding_array)):
                print("Embedding contains Inf values")
                return False
                
            print(f"Embedding validation passed (dimension: {embedding_array.shape[0]})")
            return True
        
        except Exception as e:
            print(f"❌ Validation error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def batch_retrain_embeddings(self, db: Session, days_back: int = 30):
        """Batch retraining based on accumulated feedback"""
        try:

            cutoff_date = datetime.utcnow() - timedelta(days=days_back)
            
            feedback_data = db.query(UserFeedback).filter(
                UserFeedback.timestamp >= cutoff_date,
                UserFeedback.feedback_type.in_(['positive', 'negative'])
            ).all()
            
            profile_feedback = {}
            for feedback in feedback_data:
                profile_id = feedback.profile_id
                if profile_id not in profile_feedback:
                    profile_feedback[profile_id] = {'positive': [], 'negative': []}
                
                profile_feedback[profile_id][feedback.feedback_type].append({
                    'query': feedback.query_text,
                    'timestamp': feedback.timestamp,
                    'score': feedback.original_score
                })
        
            for profile_id, feedback in profile_feedback.items():
                self._batch_update_profile_embedding(profile_id, feedback, db)
                
            print(f"Batch retraining completed for {len(profile_feedback)} profiles")
            
        except Exception as e:
            print(f"Error in batch retraining: {e}")
    
    def _batch_update_profile_embedding(self, profile_id: str, feedback_data: Dict, db: Session):

        try:

            current_results = self.collection.get(
                ids=[profile_id],
                include=['embeddings', 'metadatas']
            )
            
            if current_results is None:
                print(f"No results for batch update of profile {profile_id}")
                return
                
            embeddings = current_results.get('embeddings')
            if embeddings is None or len(embeddings) == 0 or embeddings[0] is None:
                print(f"No valid embeddings for batch update of profile {profile_id}")
                return
            
            current_embedding = np.array(embeddings[0])
            

            if current_embedding.size == 0 or np.any(np.isnan(current_embedding)):
                print(f"Invalid current embedding for profile {profile_id}")
                return
            
            metadatas = current_results.get('metadatas')
            metadata = metadatas[0] if (metadatas and len(metadatas) > 0 and metadatas[0]) else {}
            total_update = np.zeros_like(current_embedding)
            update_count = 0
            
            for pos_feedback in feedback_data.get('positive', []):
                try:
                    query_embedding = np.array(self.model.encode([pos_feedback['query']])[0])
                    
                    if np.any(np.isnan(query_embedding)):
                        print(f"Skipping invalid query embedding for: {pos_feedback['query']}")
                        continue
                    
                    direction = query_embedding - current_embedding

                    days_old = (datetime.utcnow() - pos_feedback['timestamp']).days
                    time_weight = np.exp(-days_old / 14)  # 2-week decay
                    
                    total_update += self.feedback_weights['positive'] * time_weight * direction
                    update_count += 1
                    
                except Exception as e:
                    print(f"Error processing positive feedback: {e}")
                    continue

            for neg_feedback in feedback_data.get('negative', []):
                try:
                    query_embedding = np.array(self.model.encode([neg_feedback['query']])[0])
                    
                    if np.any(np.isnan(query_embedding)):
                        print(f"Skipping invalid query embedding for: {neg_feedback['query']}")
                        continue
                    
                    direction = current_embedding - query_embedding
                    
                    days_old = (datetime.utcnow() - neg_feedback['timestamp']).days
                    time_weight = np.exp(-days_old / 14)
                    
                    total_update += abs(self.feedback_weights['negative']) * time_weight * direction
                    update_count += 1
                    
                except Exception as e:
                    print(f"Error processing negative feedback: {e}")
                    continue

            if update_count > 0:

                avg_update = total_update / update_count
                updated_embedding = current_embedding + avg_update
                embedding_norm = np.linalg.norm(updated_embedding)
                if embedding_norm > 1e-8:
                    updated_embedding = updated_embedding / embedding_norm
                else:
                    print(f"Warning: Near-zero norm in batch update for {profile_id}")
                    updated_embedding = current_embedding
                if np.any(np.isnan(updated_embedding)) or np.any(np.isinf(updated_embedding)):
                    print(f"Invalid final embedding for profile {profile_id}, keeping original")
                    updated_embedding = current_embedding
                updated_metadata = {
                    **metadata,
                    'last_batch_update': datetime.utcnow().isoformat(),
                    'total_feedback_processed': metadata.get('total_feedback_processed', 0) + update_count
                }
                self.collection.update(
                    ids=[profile_id],
                    embeddings=[updated_embedding.tolist()],
                    metadatas=[updated_metadata]
                )
                
                print(f"Successfully batch updated profile {profile_id} with {update_count} feedback items")
            else:
                print(f"No valid feedback to process for profile {profile_id}")
        
        except Exception as e:
            print(f"Error in batch update for profile {profile_id}: {e}")
            import traceback
            traceback.print_exc()

    
   
    def _invalidate_query_cache(self, query: str):
        """Remove cached results for this query to force fresh vector search"""
        try:
            from config.redis_cache import RedisQueryCache
            redis_client = RedisQueryCache()
            query_normalized = query.lower().strip()
            redis_client.clear_cache(query_normalized)
            print(f"Invalidated cache for query: {query_normalized}")
        except Exception as e:
            print(f"Error invalidating cache: {e}")
    
    def _store_feedback_record(self, query: str, profile_id: str, feedback_type: str, 
                             user_id: str, session_id: str, original_score: float, 
                             db: Session, context_info: Optional[Dict] = None):
        """Store feedback record in database (existing logic)"""
        try:
            feedback = UserFeedback()
            feedback.query_text = query
            feedback.query_normalized = query.lower().strip()
            feedback.profile_id = profile_id
            feedback.feedback_type = feedback_type
            feedback.user_id = user_id
            feedback.session_id = session_id
            feedback.timestamp = datetime.utcnow()
            feedback.original_score = original_score
            feedback.context_info = context_info
            db.add(feedback)
            db.commit()

            self._update_search_quality_metrics(query, profile_id, feedback_type, db)
            
        except Exception as e:
            print(f"Error storing feedback record: {e}")

    def _update_search_quality_metrics(self, query: str, profile_id: str, feedback_type: str, db: Session):

        try:
            query_normalized = query.lower().strip()
            
            # First, try to get existing record
            quality_metric = db.query(SearchQualityMetrics).filter(
                SearchQualityMetrics.query_normalized == query_normalized,
                SearchQualityMetrics.profile_id == str(profile_id)
            ).first()
            
            # If doesn't exist, create new one with proper defaults
            if not quality_metric:
                quality_metric = SearchQualityMetrics(
                    query_normalized=query_normalized,
                    profile_id=str(profile_id),  
                    positive_feedback_count=0,
                    negative_feedback_count=0,
                    neutral_feedback_count=0,
                    total_feedback_count=0,
                    feedback_ratio=0.0
                )
                db.add(quality_metric)
            
            positive_count = quality_metric.positive_feedback_count or 0
            negative_count = quality_metric.negative_feedback_count or 0
            neutral_count = quality_metric.neutral_feedback_count or 0

            if feedback_type == 'positive':
                quality_metric.positive_feedback_count = positive_count + 1
            elif feedback_type == 'negative':
                quality_metric.negative_feedback_count = negative_count + 1
            elif feedback_type == 'neutral':
                quality_metric.neutral_feedback_count = neutral_count + 1

            quality_metric.total_feedback_count = (
                (quality_metric.positive_feedback_count or 0) + 
                (quality_metric.negative_feedback_count or 0) + 
                (quality_metric.neutral_feedback_count or 0)
            )
            
            positive_final = quality_metric.positive_feedback_count or 0
            negative_final = quality_metric.negative_feedback_count or 0
            total_feedback = positive_final + negative_final
            
            quality_metric.feedback_ratio = (
                positive_final / total_feedback if total_feedback > 0 else 0.0
            )
            
            quality_metric.last_updated = datetime.utcnow()
            db.commit()
            
            print(f"Updated metrics for query '{query}' and profile {profile_id}")
            
        except Exception as e:
            db.rollback()
            print(f"Error updating search quality metrics: {e}")
            import traceback
            traceback.print_exc()

    
    def get_learning_stats(self, db: Session) -> Dict:
        """Get statistics about the learning system"""
        try:

            total_profiles = self.collection.count()
            
            # Get profiles with feedback updates
            profiles_with_metadata = self.collection.get(
                include=['metadatas']
            )
            
            updated_profiles = 0
            total_feedback_processed = 0
            
            for metadata in profiles_with_metadata['metadatas']:
                if metadata and 'feedback_count' in metadata:
                    updated_profiles += 1
                    total_feedback_processed += metadata.get('feedback_count', 0)

            recent_feedback = db.query(UserFeedback).filter(
                UserFeedback.timestamp >= datetime.utcnow() - timedelta(days=7)
            ).count()
            
            return {
                'total_profiles_in_vectordb': total_profiles,
                'profiles_updated_by_feedback': updated_profiles,
                'total_feedback_processed': total_feedback_processed,
                'recent_feedback_count': recent_feedback,
                'learning_rate_positive': self.feedback_weights['positive'],
                'learning_rate_negative': abs(self.feedback_weights['negative']),
                'last_batch_update': None  
            }
            
        except Exception as e:
            print(f"Error getting learning stats: {e}")
            return {}
    def get_feedback_stats_simple(self, db: Session, days: int = 30) -> Dict[str, any]:

        try:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)
            
            feedback_stats = db.query(
                func.count(SearchQualityMetrics.id).label('total'),
                func.coalesce(func.sum(SearchQualityMetrics.positive_feedback_count), 0).label('positive'),
                func.coalesce(func.sum(SearchQualityMetrics.negative_feedback_count), 0).label('negative'),
                func.coalesce(func.sum(SearchQualityMetrics.neutral_feedback_count), 0).label('neutral')
            ).filter(
                SearchQualityMetrics.last_updated >= start_date
            ).first()

            total = int(feedback_stats.total) if feedback_stats.total else 0
            positive = int(feedback_stats.positive) if feedback_stats.positive else 0
            negative = int(feedback_stats.negative) if feedback_stats.negative else 0
            neutral = int(feedback_stats.neutral) if feedback_stats.neutral else 0

            total_sentiment_feedback = positive + negative  
            satisfaction_rate = (positive / total_sentiment_feedback * 100) if total_sentiment_feedback > 0 else 0.0
            
            return {
                "total_feedback_count": total,
                "positive_feedback_count": positive,
                "negative_feedback_count": negative,
                "neutral_feedback_count": neutral,
                "satisfaction_rate": round(satisfaction_rate, 2),
                "feedback_rate": 0.0,  
                "average_score_improvement": 0.0,
                "most_improved_profiles": [],
                "most_problematic_profiles": []
            }
            
        except Exception as e:
            print(f"Error getting feedback stats: {e}")
            import traceback
            traceback.print_exc()
            return {
                "total_feedback_count": 0,
                "positive_feedback_count": 0,
                "negative_feedback_count": 0,
                "neutral_feedback_count": 0,
                "satisfaction_rate": 0.0,
                "feedback_rate": 0.0,
                "average_score_improvement": 0.0,
                "most_improved_profiles": [],
                "most_problematic_profiles": []
            }
        
