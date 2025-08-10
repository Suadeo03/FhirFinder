# database/models/feedback_models.py 

from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, Boolean, Text, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
import uuid

Base = declarative_base()

class UserFeedback(Base):
    """
    Store individual user feedback events for search results
    """
    __tablename__ = "user_feedback"
    
    id = Column(Integer, primary_key=True, index=True) 
    query_text = Column(Text, nullable=False, index=True)
    query_normalized = Column(Text, nullable=False, index=True)  
    profile_id = Column(String(255), nullable=False, index=True)
    feedback_type = Column(String(20), nullable=False)  # 'positive', 'negative', 'neutral'
    feedback_score = Column(Float, default=0.0) 
    feedback_comment = Column(Text, nullable=True)  
    user_id = Column(String(255), nullable=True, index=True)
    session_id = Column(String(255), nullable=False, index=True)
    original_score = Column(Float, nullable=False)
    search_type = Column(String(50), nullable=True)  
    search_rank = Column(Integer, nullable=True)  
    context_info = Column(JSON, nullable=True)  
    user_agent = Column(String(500), nullable=True)
    ip_address = Column(String(45), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_feedback_query_profile', 'query_normalized', 'profile_id'),
        Index('idx_feedback_user_session', 'user_id', 'session_id'),
        Index('idx_feedback_created_type', 'created_at', 'feedback_type'),
        Index('idx_feedback_profile_type_date', 'profile_id', 'feedback_type', 'created_at'),
    )

class SearchQualityMetrics(Base):
    """
    Aggregated metrics for query-profile pairs to optimize search performance
    """
    __tablename__ = "search_quality_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    query_normalized = Column(Text, nullable=False, index=True)
    profile_id = Column(String(255), nullable=False, index=True) # oid
    base_semantic_score = Column(Float, default=0.0)
    base_text_score = Column(Float, default=0.0)
    base_hybrid_score = Column(Float, default=0.0)
    adjusted_semantic_score = Column(Float, default=0.0)
    adjusted_text_score = Column(Float, default=0.0)
    adjusted_hybrid_score = Column(Float, default=0.0)
    total_feedback_count = Column(Integer, default=0)
    positive_feedback_count = Column(Integer, default=0)
    negative_feedback_count = Column(Integer, default=0)
    neutral_feedback_count = Column(Integer, default=0)
    feedback_ratio = Column(Float, default=0.0)  # positive / (positive + negative)
    avg_feedback_score = Column(Float, default=0.0)
    confidence_score = Column(Float, default=0.0)  # Based on feedback volume and consistency
    click_through_rate = Column(Float, default=0.0)
    dwell_time_avg = Column(Float, default=0.0)  # Average time spent on result
    conversion_rate = Column(Float, default=0.0)  # If applicable to your use case
    first_feedback_at = Column(DateTime, nullable=True)
    last_feedback_at = Column(DateTime, nullable=True)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Unique constraint on query-profile pair
    __table_args__ = (
        Index('idx_quality_query_profile', 'query_normalized', 'profile_id', unique=True),
        Index('idx_quality_feedback_ratio', 'feedback_ratio'),
        Index('idx_quality_confidence', 'confidence_score'),
    
    )

class SearchSession(Base):
    """
    Track search sessions for analytics and personalization
    """
    __tablename__ = "search_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(255), nullable=False, unique=True, index=True)
    user_id = Column(String(255), nullable=True, index=True)

    start_time = Column(DateTime, default=datetime.utcnow, nullable=False)
    end_time = Column(DateTime, nullable=True)
    total_queries = Column(Integer, default=0)
    total_results_viewed = Column(Integer, default=0)
    total_feedback_given = Column(Integer, default=0)

    user_agent = Column(String(500), nullable=True)
    ip_address = Column(String(45), nullable=True)
    referrer = Column(String(500), nullable=True)
 
    avg_query_time = Column(Float, default=0.0)
    successful_searches = Column(Integer, default=0)  # Searches that got positive feedback
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class FeedbackTrainingEmbeddings(Base):
    __tablename__ = "feedback_training_embeddings"
    
    id = Column(Integer, primary_key=True, index=True)
    query_normalized = Column(Text, nullable=False, index=True)  # Remove unique - multiple profiles per query
    query_embedding_vector = Column(JSON, nullable=False)
    
    # Profile/Resource info
    resource_id = Column(String(255), nullable=False, index=True)
    original_embedding_vector = Column(JSON, nullable=False)     # ← BASELINE (never changes)
    current_embedding_vector = Column(JSON, nullable=False)      # ← LEARNED (changes with feedback)
    
    # Feedback metadata
    feedback_type = Column(String(50), nullable=False)           # positive/negative/neutral
    feedback_count = Column(Integer, default=1)                  # How many times this query-profile pair got feedback
    learning_magnitude = Column(Float, default=0.0)              # How much the embedding has shifted
    
    # Timestamps
    first_feedback_at = Column(DateTime, default=datetime.utcnow)
    last_feedback_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Composite index for efficient queries
    __table_args__ = (
        Index('ix_query_resource', 'query_normalized', 'resource_id'),
    )

class FeedbackTrainingBatch(Base):
    """
    Track training batches for model updates
    """
    __tablename__ = "feedback_training_batches"
    
    id = Column(Integer, primary_key=True, index=True)
    batch_id = Column(UUID(as_uuid=True), default=uuid.uuid4, unique=True, index=True)
    
    # Batch details
    training_type = Column(String(50), nullable=False)  # 'embedding_update', 'ranking_model', etc.
    feedback_count = Column(Integer, nullable=False)
    positive_examples = Column(Integer, default=0)
    negative_examples = Column(Integer, default=0)
    
    # Date range of feedback used
    feedback_start_date = Column(DateTime, nullable=False)
    feedback_end_date = Column(DateTime, nullable=False)
    
    # Training results
    training_status = Column(String(20), default='pending')  # 'pending', 'running', 'completed', 'failed'
    performance_improvement = Column(Float, nullable=True)
    validation_score = Column(Float, nullable=True)
    
    # Model versioning
    model_version_before = Column(String(50), nullable=True)
    model_version_after = Column(String(50), nullable=True)
    
    # Timestamps
    started_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime, nullable=True)
    
    # Training configuration
    training_config = Column(JSON, nullable=True)
    training_logs = Column(Text, nullable=True)

