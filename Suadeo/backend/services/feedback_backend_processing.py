# services/feedback_processor.py - Background service for processing feedback

import asyncio
import numpy as np
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_
from celery import Celery
from sentence_transformers import SentenceTransformer
import logging
from typing import Optional, List, Tuple
from config.database import get_db
from dataclasses import dataclass

from models.feedback_models import (
    UserFeedback, SearchQualityMetrics, QueryEmbedding, 
    FeedbackTrainingBatch, SearchExperiment
)
from models.database.models import Profile
from config.redis_cache import RedisQueryCache
from config.chroma import ChromaConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Celery for background tasks
app = Celery('feedback_processor')
app.conf.update(
    broker_url='redis://localhost:6379/0',
    result_backend='redis://localhost:6379/0',
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
)

@dataclass
class FeedbackProcessingConfig:
    """Configuration for feedback processing"""
    min_feedback_count: int = 5
    feedback_decay_days: int = 30
    embedding_update_threshold: int = 10
    retraining_batch_size: int = 100
    quality_score_weight: float = 0.3
    recency_weight: float = 0.2

class FeedbackProcessor:
    """Main service for processing user feedback and updating models"""
    
    def __init__(self, config: FeedbackProcessingConfig = None):
        self.config = config or FeedbackProcessingConfig()
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.chroma_config = ChromaConfig()
        self.redis_client = RedisQueryCache()
        
    def process_feedback_batch(self, db: Session, batch_size: int = 100) -> Dict[str, Any]:
        """
        Process a batch of recent feedback to update quality metrics
        """
        try:
            # Get recent unprocessed feedback
            cutoff_time = datetime.utcnow() - timedelta(hours=1)
            
            recent_feedback = db.query(UserFeedback).filter(
                UserFeedback.created_at >= cutoff_time
            ).limit(batch_size).all()
            
            if not recent_feedback:
                logger.info("No recent feedback to process")
                return {"processed": 0, "updated_metrics": 0}
            
            processed_count = 0
            updated_metrics = 0
            
            # Group feedback by query-profile pairs
            feedback_groups = {}
            for feedback in recent_feedback:
                key = (feedback.query_normalized, feedback.profile_id)
                if key not in feedback_groups:
                    feedback_groups[key] = []
                feedback_groups[key].append(feedback)
            
            # Update quality metrics for each group
            for (query, profile_id), feedbacks in feedback_groups.items():
                self._update_quality_metrics(db, query, profile_id, feedbacks)
                updated_metrics += 1
                processed_count += len(feedbacks)
            
            # Update embeddings if needed
            embedding_updates = self._update_query_embeddings(db, feedback_groups.keys())
            
            db.commit()
            
            logger.info(f"Processed {processed_count} feedback items, updated {updated_metrics} metrics")
            
            return {
                "processed": processed_count,
                "updated_metrics": updated_metrics,
                "embedding_updates": embedding_updates
            }
            
        except Exception as e:
            logger.error(f"Error processing feedback batch: {e}")
            db.rollback()
            return {"error": str(e)}
    
    def _update_quality_metrics(self, db: Session, query: str, profile_id: int, 
                               feedbacks: List[UserFeedback]):
        """
        Update aggregated quality metrics for a query-profile pair
        """
        try:
            # Get or create quality metrics record
            metric = db.query(SearchQualityMetrics).filter(
                and_(
                    SearchQualityMetrics.query_normalized == query,
                    SearchQualityMetrics.profile_id == profile_id
                )
            ).first()
            
            if not metric:
                metric = SearchQualityMetrics(
                    query_normalized=query,
                    profile_id=profile_id
                )
                db.add(metric)
            
            # Update feedback counts
            for feedback in feedbacks:
                metric.total_feedback_count += 1
                
                if feedback.feedback_type == 'positive':
                    metric.positive_feedback_count += 1
                elif feedback.feedback_type == 'negative':
                    metric.negative_feedback_count += 1
                elif feedback.feedback_type == 'neutral':
                    metric.neutral_feedback_count += 1
            
            # Calculate feedback ratio
            total_actionable = metric.positive_feedback_count + metric.negative_feedback_count
            if total_actionable > 0:
                metric.feedback_ratio = metric.positive_feedback_count / total_actionable
            
            # Calculate confidence score based on volume and consistency
            metric.confidence_score = self._calculate_confidence_score(metric)
            
            # Update timestamps
            metric.last_feedback_at = max(f.created_at for f in feedbacks)
            if metric.first_feedback_at is None:
                metric.first_feedback_at = min(f.created_at for f in feedbacks)
            
            metric.last_updated = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Error updating quality metrics: {e}")
    
    def _calculate_confidence_score(self, metric: SearchQualityMetrics) -> float:
        """
        Calculate confidence score based on feedback volume and consistency
        """
        try:
            # Base confidence on total feedback count
            volume_factor = min(metric.total_feedback_count / self.config.min_feedback_count, 1.0)
            
            # Consistency factor (higher when feedback is consistent)
            if metric.total_feedback_count > 0:
                positive_rate = metric.positive_feedback_count / metric.total_feedback_count
                negative_rate = metric.negative_feedback_count / metric.total_feedback_count
                
                # Higher confidence when feedback is consistently positive or negative
                consistency_factor = max(positive_rate, negative_rate)
            else:
                consistency_factor = 0.0
            
            # Time decay factor (more recent feedback has higher confidence)
            if metric.last_feedback_at:
                days_since_last = (datetime.utcnow() - metric.last_feedback_at).days
                time_factor = np.exp(-days_since_last / self.config.feedback_decay_days)
            else:
                time_factor = 0.0
            
            # Combined confidence score
            confidence = (
                volume_factor * 0.4 + 
                consistency_factor * 0.4 + 
                time_factor * 0.2
            )
            
            return min(confidence, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating confidence score: {e}")
            return 0.0
    
    def _update_query_embeddings(self, db: Session, query_profile_pairs: List[tuple]) -> int:
        """
        Update query embeddings based on feedback patterns
        """
        try:
            updated_count = 0
            unique_queries = set(query for query, _ in query_profile_pairs)
            
            for query in unique_queries:
                # Get existing embedding
                query_embedding = db.query(QueryEmbedding).filter(
                    QueryEmbedding.query_normalized == query
                ).first()
                
                if not query_embedding:
                    # Create new embedding
                    embedding_vector = self.model.encode([query])[0].tolist()
                    query_embedding = QueryEmbedding(
                        query_normalized=query,
                        embedding_vector=embedding_vector
                    )
                    db.add(query_embedding)
                    updated_count += 1
                else:
                    # Update search count
                    query_embedding.search_count += 1
                    query_embedding.last_searched = datetime.utcnow()
                    
                    # Check if embedding needs updating based on feedback
                    if self._should_update_embedding(db, query):
                        new_embedding = self._generate_feedback_adjusted_embedding(db, query)
                        if new_embedding is not None:
                            query_embedding.embedding_vector = new_embedding
                            query_embedding.embedding_version += 1
                            query_embedding.updated_at = datetime.utcnow()
                            updated_count += 1
            
            return updated_count
            
        except Exception as e:
            logger.error(f"Error updating query embeddings: {e}")
            return 0
    
    def _should_update_embedding(self, db: Session, query: str) -> bool:
        """
        Determine if an embedding should be updated based on feedback volume
        """
        try:
            feedback_count = db.query(func.count(UserFeedback.id)).filter(
                UserFeedback.query_normalized == query,
                UserFeedback.created_at >= datetime.utcnow() - timedelta(days=7)
            ).scalar()
            
            return feedback_count >= self.config.embedding_update_threshold
            
        except Exception as e:
            logger.error(f"Error checking embedding update criteria: {e}")
            return False
    
    def _generate_feedback_adjusted_embedding(self, db: Session, query: str) -> Optional[List[float]]:
        """
        Generate a feedback-adjusted embedding for a query
        """
        try:
            # Get positive and negative feedback examples
            positive_profiles = db.query(Profile).join(UserFeedback).filter(
                and_(
                    UserFeedback.query_normalized == query,
                    UserFeedback.feedback_type == 'positive',
                    UserFeedback.created_at >= datetime.utcnow() - timedelta(days=30)
                )
            ).all()
            
            negative_profiles = db.query(Profile).join(UserFeedback).filter(
                and_(
                    UserFeedback.query_normalized == query,
                    UserFeedback.feedback_type == 'negative',
                    UserFeedback.created_at >= datetime.utcnow() - timedelta(days=30)
                )
            ).all()
            
            if not positive_profiles and not negative_profiles:
                return None
            
            # Create adjusted embedding
            base_embedding = self.model.encode([query])[0]
            
            # Calculate centroid of positive examples
            if positive_profiles:
                positive_embeddings = []
                for profile in positive_profiles:
                    if profile.embedding_vector:
                        positive_embeddings.append(np.array(profile.embedding_vector))
                
                if positive_embeddings:
                    positive_centroid = np.mean(positive_embeddings, axis=0)
                    # Move base embedding slightly towards positive examples
                    base_embedding = base_embedding + 0.1 * (positive_centroid - base_embedding)
            
            # Move away from negative examples
            if negative_profiles:
                negative_embeddings = []
                for profile in negative_profiles:
                    if profile.embedding_vector:
                        negative_embeddings.append(np.array(profile.embedding_vector))
                
                if negative_embeddings:
                    negative_centroid = np.mean(negative_embeddings, axis=0)
                    # Move base embedding away from negative examples
                    base_embedding = base_embedding - 0.05 * (negative_centroid - base_embedding)
            
            # Normalize the embedding
            norm = np.linalg.norm(base_embedding)
            if norm > 0:
                base_embedding = base_embedding / norm
            
            return base_embedding.tolist()
            
        except Exception as e:
            logger.error(f"Error generating feedback-adjusted embedding: {e}")
            return None

# Celery tasks
@app.task
def process_feedback_batch_task(batch_size: int = 100):
    """
    Celery task to process feedback batch in background
    """
    from database import get_db
    
    processor = FeedbackProcessor()
    db = next(get_db())
    
    try:
        result = processor.process_feedback_batch(db, batch_size)
        return result
    finally:
        db.close()

@app.task
def update_profile_feedback_scores():
    """
    Update profile-level feedback scores periodically
    """
    
    db = next(get_db())
    
    try:
        # Update profiles with aggregated feedback data
        profiles_updated = 0
        
        # Get profiles with recent feedback
        profiles_with_feedback = db.query(Profile.id).join(UserFeedback).filter(
            UserFeedback.created_at >= datetime.utcnow() - timedelta(days=1)
        ).distinct().all()
        
        for (profile_id,) in profiles_with_feedback:
            # Calculate aggregated feedback score
            feedback_stats = db.query(
                func.count(UserFeedback.id).label('total'),
                func.sum(func.case([(UserFeedback.feedback_type == 'positive', 1)], else_=0)).label('positive'),
                func.sum(func.case([(UserFeedback.feedback_type == 'negative', 1)], else_=0)).label('negative')
            ).filter(
                and_(
                    UserFeedback.profile_id == profile_id,
                    UserFeedback.created_at >= datetime.utcnow() - timedelta(days=30)
                )
            ).first()
            
            if feedback_stats.total > 0:
                positive_rate = (feedback_stats.positive or 0) / feedback_stats.total
                
                # Update profile
                db.query(Profile).filter(Profile.id == profile_id).update({
                    'feedback_score': positive_rate,
                    'positive_feedback_count': feedback_stats.positive or 0,
                    'negative_feedback_count': feedback_stats.negative or 0,
                    'last_feedback_date': datetime.utcnow()
                })
                
                profiles_updated += 1
        
        db.commit()
        
        return {"profiles_updated": profiles_updated}
        
    except Exception as e:
        logger.error(f"Error updating profile feedback scores: {e}")
        db.rollback()
        return {"error": str(e)}
    finally:
        db.close()

@app.task
def retrain_search_model():
    """
    Periodic task to retrain search models based on accumulated feedback
    """

    
    db = next(get_db())
    processor = FeedbackProcessor()
    
    try:
        # Create training batch record
        batch = FeedbackTrainingBatch(
            training_type='embedding_update',
            feedback_start_date=datetime.utcnow() - timedelta(days=7),
            feedback_end_date=datetime.utcnow(),
            training_status='running'
        )
        db.add(batch)
        db.commit()
        
        try:
            # Count feedback for this batch
            feedback_count = db.query(func.count(UserFeedback.id)).filter(
                and_(
                    UserFeedback.created_at >= batch.feedback_start_date,
                    UserFeedback.created_at <= batch.feedback_end_date
                )
            ).scalar()
            
            positive_count = db.query(func.count(UserFeedback.id)).filter(
                and_(
                    UserFeedback.created_at >= batch.feedback_start_date,
                    UserFeedback.created_at <= batch.feedback_end_date,
                    UserFeedback.feedback_type == 'positive'
                )
            ).scalar()
            
            negative_count = db.query(func.count(UserFeedback.id)).filter(
                and_(
                    UserFeedback.created_at >= batch.feedback_start_date,
                    UserFeedback.created_at <= batch.feedback_end_date,
                    UserFeedback.feedback_type == 'negative'
                )
            ).scalar()
            
            # Update batch with counts
            batch.feedback_count = feedback_count
            batch.positive_examples = positive_count
            batch.negative_examples = negative_count
            
            if feedback_count < processor.config.min_feedback_count:
                batch.training_status = 'skipped'
                batch.completed_at = datetime.utcnow()
                db.commit()
                return {"status": "skipped", "reason": "insufficient_feedback", "feedback_count": feedback_count}
            
            # Perform actual retraining using the processor instance
            improvement_score = processor._perform_model_update(db, batch)
            
            # Update batch status
            batch.training_status = 'completed'
            batch.performance_improvement = improvement_score
            batch.completed_at = datetime.utcnow()
            batch.model_version_after = f"v{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            db.commit()
            
            return {
                "status": "completed",
                "feedback_count": feedback_count,
                "improvement_score": improvement_score,
                "batch_id": str(batch.batch_id)
            }
            
        except Exception as e:
            batch.training_status = 'failed'
            batch.completed_at = datetime.utcnow()
            batch.training_logs = str(e)
            db.commit()
            raise e
            
    except Exception as e:
        logger.error(f"Error in model retraining: {e}")
        db.rollback()
        return {"error": str(e)}
    finally:
        db.close()

def _perform_model_update(self, db: Session, batch: FeedbackTrainingBatch) -> float:
        """
        Perform the actual model update based on feedback
        """
        try:
            # Get queries with significant feedback
            significant_queries = db.query(SearchQualityMetrics).filter(
                and_(
                    SearchQualityMetrics.total_feedback_count >= self.config.min_feedback_count,
                    SearchQualityMetrics.last_feedback_at >= batch.feedback_start_date
                )
            ).all()
            
            updated_embeddings = 0
            total_improvement = 0.0
            
            for metric in significant_queries:
                # Update query embedding if it has strong feedback signal
                if metric.confidence_score > 0.7:
                    new_embedding = self._generate_feedback_adjusted_embedding(
                        db, metric.query_normalized
                    )
                    
                    if new_embedding:
                        # Update in ChromaDB if available
                        if self.chroma_config and self.chroma_config.collection:
                            try:
                                # This would update the embedding in Chroma
                                # Implementation depends on your Chroma setup
                                pass
                            except Exception as e:
                                logger.warning(f"Failed to update Chroma embedding: {e}")
                        
                        # Update in PostgreSQL
                        query_emb = db.query(QueryEmbedding).filter(
                            QueryEmbedding.query_normalized == metric.query_normalized
                        ).first()
                        
                        if query_emb:
                            query_emb.embedding_vector = new_embedding
                            query_emb.embedding_version += 1
                            updated_embeddings += 1
                        
                        # Calculate improvement estimate
                        if metric.feedback_ratio > 0.6:
                            total_improvement += metric.confidence_score * metric.feedback_ratio
            
            # Calculate average improvement
            improvement_score = total_improvement / max(len(significant_queries), 1)
            
            logger.info(f"Updated {updated_embeddings} embeddings with improvement score {improvement_score}")
            
            return improvement_score
            
        except Exception as e:
            logger.error(f"Error performing model update: {e}")
            return 0.0

@app.task
def cleanup_old_feedback(days_to_keep: int = 90):
    """
    Clean up old feedback data to manage database size
    """
    
    
    db = next(get_db())
    
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
        
        # Archive or delete old feedback
        old_feedback_count = db.query(func.count(UserFeedback.id)).filter(
            UserFeedback.created_at < cutoff_date
        ).scalar()
        
        if old_feedback_count > 0:
            # In production, you might want to archive instead of delete
            db.query(UserFeedback).filter(
                UserFeedback.created_at < cutoff_date
            ).delete()
            
            # Clean up orphaned quality metrics
            db.query(SearchQualityMetrics).filter(
                SearchQualityMetrics.last_feedback_at < cutoff_date
            ).delete()
            
            db.commit()
            
            logger.info(f"Cleaned up {old_feedback_count} old feedback records")
        
        return {"cleaned_records": old_feedback_count}
        
    except Exception as e:
        logger.error(f"Error cleaning up old feedback: {e}")
        db.rollback()
        return {"error": str(e)}
    finally:
        db.close()

@app.task
def generate_feedback_analytics_report():
    """
    Generate analytics report on feedback trends
    """
 
    
    db = next(get_db())
    redis_client = RedisQueryCache()
    
    try:
        # Get feedback trends over the last 30 days
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        
        # Daily feedback counts
        daily_feedback = db.query(
            func.date(UserFeedback.created_at).label('date'),
            UserFeedback.feedback_type,
            func.count(UserFeedback.id).label('count')
        ).filter(
            UserFeedback.created_at >= thirty_days_ago
        ).group_by(
            func.date(UserFeedback.created_at),
            UserFeedback.feedback_type
        ).all()
        
        # Top performing queries
        top_queries = db.query(
            SearchQualityMetrics.query_normalized,
            SearchQualityMetrics.feedback_ratio,
            SearchQualityMetrics.total_feedback_count,
            SearchQualityMetrics.confidence_score
        ).filter(
            SearchQualityMetrics.total_feedback_count >= 5
        ).order_by(
            SearchQualityMetrics.feedback_ratio.desc(),
            SearchQualityMetrics.confidence_score.desc()
        ).limit(20).all()
        
        # Problematic queries (high negative feedback)
        problematic_queries = db.query(
            SearchQualityMetrics.query_normalized,
            SearchQualityMetrics.feedback_ratio,
            SearchQualityMetrics.total_feedback_count,
            SearchQualityMetrics.negative_feedback_count
        ).filter(
            and_(
                SearchQualityMetrics.total_feedback_count >= 5,
                SearchQualityMetrics.feedback_ratio < 0.3
            )
        ).order_by(
            SearchQualityMetrics.feedback_ratio.asc()
        ).limit(10).all()
        
        # Overall statistics
        total_feedback = db.query(func.count(UserFeedback.id)).filter(
            UserFeedback.created_at >= thirty_days_ago
        ).scalar()
        
        positive_feedback = db.query(func.count(UserFeedback.id)).filter(
            and_(
                UserFeedback.created_at >= thirty_days_ago,
                UserFeedback.feedback_type == 'positive'
            )
        ).scalar()
        
        overall_satisfaction = positive_feedback / max(total_feedback, 1)
        
        # Compile report
        report = {
            "report_date": datetime.utcnow().isoformat(),
            "period": "last_30_days",
            "overall_stats": {
                "total_feedback": total_feedback,
                "positive_feedback": positive_feedback,
                "satisfaction_rate": overall_satisfaction
            },
            "daily_trends": [
                {
                    "date": str(row.date),
                    "feedback_type": row.feedback_type,
                    "count": row.count
                }
                for row in daily_feedback
            ],
            "top_performing_queries": [
                {
                    "query": row.query_normalized,
                    "feedback_ratio": row.feedback_ratio,
                    "total_feedback": row.total_feedback_count,
                    "confidence": row.confidence_score
                }
                for row in top_queries
            ],
            "problematic_queries": [
                {
                    "query": row.query_normalized,
                    "feedback_ratio": row.feedback_ratio,
                    "total_feedback": row.total_feedback_count,
                    "negative_count": row.negative_feedback_count
                }
                for row in problematic_queries
            ]
        }
        
        # Store report in Redis for quick access
        if redis_client and redis_client.is_connected():
            redis_client.set(
                "feedback_analytics_report",
                json.dumps(report),
                ex=86400  # Expire after 24 hours
            )
        
        return report
        
    except Exception as e:
        logger.error(f"Error generating analytics report: {e}")
        return {"error": str(e)}
    finally:
        db.close()

# Periodic task scheduling
@app.on_after_configure.connect
def setup_periodic_tasks(sender, **kwargs):
    """
    Setup periodic tasks for feedback processing
    """
    # Process feedback every 10 minutes
    sender.add_periodic_task(
        600.0,  # 10 minutes
        process_feedback_batch_task.s(),
        name='process_feedback_batch'
    )
    
    # Update profile feedback scores every hour
    sender.add_periodic_task(
        3600.0,  # 1 hour
        update_profile_feedback_scores.s(),
        name='update_profile_scores'
    )
    
    # Retrain models daily
    sender.add_periodic_task(
        86400.0,  # 24 hours
        retrain_search_model.s(),
        name='retrain_search_model'
    )
    
    # Generate analytics report daily
    sender.add_periodic_task(
        86400.0,  # 24 hours
        generate_feedback_analytics_report.s(),
        name='analytics_report'
    )
    
    # Clean up old data weekly
    sender.add_periodic_task(
        604800.0,  # 1 week
        cleanup_old_feedback.s(),
        name='cleanup_old_feedback'
    )

if __name__ == "__main__":
    # For testing purposes
    processor = FeedbackProcessor()
    print("Feedback processor initialized successfully")