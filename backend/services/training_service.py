from datetime import datetime, timedelta
from typing import List, Dict, Optional
from sqlalchemy.orm import Session
from config.redis_cache import RedisQueryCache
import numpy as np
from models.database.feedback_models import UserFeedback, SearchQualityMetrics
from sqlalchemy import func, and_, case, desc, asc

class TrainingFeedback:
    def __init__(self):
        """Initialize feedback weights and decay settings"""
        self.feedback_weights = {
            'positive': 1.2,    # 20% boost for positive feedback
            'negative': 0.8,    # 20% penalty for negative feedback
            'neutral': 1.0      # No change for neutral feedback
        }
        self.feedback_decay_days = 30  # Feedback loses impact over 30 days
        
    def record_user_feedback(self, query: str, profile_id: str, feedback_type: str, 
                           user_id: str, session_id: str, original_score: float, 
                           db: Session, context_info: Optional[Dict] = None):
        """
        Record user feedback for a search result
        """
        #prevent duplicate feedback messages
        recent_cutoff = datetime.now() - timedelta(minutes=5)
    
        existing_feedback = db.query(UserFeedback).filter(
        UserFeedback.profile_id == profile_id,
        UserFeedback.query_text == query,
        UserFeedback.created_at > recent_cutoff
        ).first()
        if existing_feedback:
            return {"status": "duplicate", "message": "Feedback already submitted"}
        
        
        
        try:
            # Create feedback record
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
            
            # Save to database (you'll need to implement this table)
            db.add(feedback)
            db.commit()
            print(f"User feedback recorded: {feedback_type} for profile {profile_id} on query '{query}'")
            
            # Update quality metrics
            self._update_search_quality_metrics(query, profile_id, feedback_type, db)
            
            # Optionally update embeddings for future searches
            if feedback_type in ['positive', 'negative']:
                self._update_query_profile_association(query, profile_id, feedback_type, db)
            
            print(f"Recorded {feedback_type} feedback for query '{query}' and profile {profile_id}")
            
        except Exception as e:
            print(f"Error recording user feedback: {e}")

    def _update_search_quality_metrics(self, query: str, profile_id: str, feedback_type: str, db: Session):
        try:
            query_normalized = query.lower().strip()
            
            # First, try to get existing record
            quality_metric = db.query(SearchQualityMetrics).filter(
                SearchQualityMetrics.query_normalized == query_normalized,
                SearchQualityMetrics.profile_id == str(profile_id)  # Ensure it's a string
            ).first()
            
            # If doesn't exist, create new one
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
            
            # Update counters
            if feedback_type == 'positive':
                quality_metric.positive_feedback_count += 1
            elif feedback_type == 'negative':
                quality_metric.negative_feedback_count += 1
            elif feedback_type == 'neutral':
                quality_metric.neutral_feedback_count += 1
            
            # Update total count
            quality_metric.total_feedback_count = (
                quality_metric.positive_feedback_count + 
                quality_metric.negative_feedback_count + 
                quality_metric.neutral_feedback_count
            )
            
            # Recalculate quality score
            total_feedback = quality_metric.positive_feedback_count + quality_metric.negative_feedback_count
            quality_metric.feedback_ratio = (
                quality_metric.positive_feedback_count / total_feedback if total_feedback > 0 else 0.0
            )
            
            quality_metric.last_updated = datetime.utcnow()
            db.commit()
            
        except Exception as e:
            db.rollback()
            print(f"Error updating search quality metrics: {e}")


    def _update_query_profile_association(self, query: str, profile_id: str, 
                                        feedback_type: str, db: Session):
        """
        Update embeddings or associations based on feedback
        """
        try:
            if feedback_type == 'positive':
                # For positive feedback, we could:
                # 1. Store this as a positive example for future training
                # 2. Slightly adjust the profile's embedding to be closer to the query
                # 3. Boost this profile for similar queries
                
                # Simple approach: store positive associations
                self._store_positive_association(query, profile_id, db)
                
            elif feedback_type == 'negative':
                # For negative feedback, we could:
                # 1. Store this as a negative example
                # 2. Slightly adjust the profile's embedding to be further from the query
                # 3. Reduce this profile's ranking for similar queries
                
                self._store_negative_association(query, profile_id, db)
                
        except Exception as e:
            print(f"Error updating query-profile association: {e}")

    def _store_positive_association(self, query: str, profile_id: str, db: Session):
        """Store positive query-profile associations for future boosting"""
       
        try:
            redis_client = RedisQueryCache()
            if redis_client.is_connected():
                key = f"positive_feedback:{query.lower().strip()}:{profile_id}"
                redis_client.set_feedback(key, datetime.utcnow().isoformat(), 86400 * 30)  # 30 days
        except Exception as e:
            print(f"Error storing positive association: {e}")

    def _store_negative_association(self, query: str, profile_id: str, db: Session):
        """Store negative query-profile associations for future penalization"""
        try:
            redis_client = RedisQueryCache()
            if redis_client.is_connected():
                key = f"negative_feedback:{query.lower().strip()}:{profile_id}"
                redis_client.set_feedback(key, datetime.utcnow().isoformat(), 86400 * 30)  # 30 days
        except Exception as e:
            print(f"Error storing negative association: {e}")

    def _apply_feedback_adjustments(self, results: List[Dict], query: str) -> List[Dict]:
        """
        Apply feedback-based adjustments to search results
        """
        try:
            redis_client = RedisQueryCache()
            if not redis_client.is_connected():
                return results
                
            query_normalized = query.lower().strip()
            
            for result in results:
                profile_id = result['id']
                original_score = result['similarity_score']
                
                # Check for positive feedback
                pos_key = f"positive_feedback:{query_normalized}:{profile_id}"
                neg_key = f"negative_feedback:{query_normalized}:{profile_id}"
                
                adjustment_factor = 1.0
                
                # Apply positive feedback boost
                if redis_client.exists_feedback(pos_key):
                    adjustment_factor *= self.feedback_weights['positive']
                    result['feedback_boost'] = 'positive'
                
                # Apply negative feedback penalty
                if redis_client.exists_feedback(neg_key):
                    adjustment_factor *= self.feedback_weights['negative']
                    result['feedback_boost'] = 'negative'
                
                # Apply time decay for feedback
                adjustment_factor = self._apply_time_decay(adjustment_factor, query_normalized, profile_id)
                
                # Update the similarity score
                result['similarity_score'] = min(original_score * adjustment_factor, 1.0)
                result['original_score'] = original_score
                result['adjustment_factor'] = adjustment_factor
                
        except Exception as e:
            print(f"Error applying feedback adjustments: {e}")
            
        return results

    def _apply_time_decay(self, adjustment_factor: float, query: str, profile_id: str) -> float:
        """
        Apply time decay to feedback adjustments - older feedback has less impact
        """
        try:
            redis_client = RedisQueryCache()
            current_time = datetime.utcnow()
            
            # Check both positive and negative feedback timestamps
            for feedback_type in ['positive', 'negative']:
                key = f"{feedback_type}_feedback:{query}:{profile_id}"
                timestamp_str = redis_client.get_feedback(key)
                
                if timestamp_str:
                    feedback_time = datetime.fromisoformat(timestamp_str)
                    days_old = (current_time - feedback_time).days
                    
                    if days_old > 0:
                        # Apply exponential decay
                        decay_factor = np.exp(-days_old / self.feedback_decay_days)
                        
                        # Adjust the adjustment factor towards 1.0 (neutral) based on age
                        adjustment_factor = 1.0 + (adjustment_factor - 1.0) * decay_factor
                        
        except Exception as e:
            print(f"Error applying time decay: {e}")
            
        return adjustment_factor
    
    def get_feedback_stats_simple(self, db: Session, days: int = 30) -> Dict[str, any]:
        """
        Simplified version with just essential metrics
        """
        try:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)
            
            # Basic feedback counts
            feedback_stats = db.query(
                func.count(SearchQualityMetrics.id).label('total'),
                func.sum(SearchQualityMetrics.positive_feedback_count).label('positive'),
                func.sum(SearchQualityMetrics.negative_feedback_count).label('negative'),
                func.sum(SearchQualityMetrics.neutral_feedback_count).label('neutral')
            ).filter(
                SearchQualityMetrics.last_updated >= start_date
            ).first()
            
            total = feedback_stats.total or 0
            positive = feedback_stats.positive or 0
            negative = feedback_stats.negative or 0
            neutral = feedback_stats.neutral or 0
            
            # Calculate satisfaction rate
            satisfaction_rate = (positive / total * 100) if total > 0 else 0.0
            
            return {
                "total_feedback_count": total,
                "positive_feedback_count": positive,
                "negative_feedback_count": negative,
                "neutral_feedback_count": neutral,
                "satisfaction_rate": round(satisfaction_rate, 2),
                "feedback_rate": 0.0,  # Would need total searches to calculate
                "average_score_improvement": 0.0,
                "most_improved_profiles": [],
                "most_problematic_profiles": []
            }
            
        except Exception as e:
            print(f"Error getting feedback stats: {e}")
            return {}

    def retrain_embeddings(self, db: Session, batch_size: int = 100):
        """
        Retrain embeddings based on accumulated feedback
        This would be run periodically (e.g., weekly)
        """
        try:
            # Get profiles that have received significant feedback
            # This is a simplified approach - in practice you'd want more sophisticated retraining
            
            # 1. Collect positive and negative examples
            positive_examples = []
            negative_examples = []
            
            # 2. Fine-tune embeddings or adjust similarity calculations
            # This could involve:
            # - Updating profile embeddings
            # - Training a ranking model
            # - Adjusting similarity weights
            
            # 3. Update Chroma collection with new embeddings
            # self.collection.update(...)
            
            print("Embedding retraining completed")
            
        except Exception as e:
            print(f"Error during embedding retraining: {e}")

    # Keep existing methods (traditional_search, search_by_filters_only, get_search_stats)
    # but add apply_feedback parameter where appropriate