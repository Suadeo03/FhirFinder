from sqlalchemy import Column, String, Integer, DateTime, Boolean, Text, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import JSONB
from datetime import datetime
import uuid

Base = declarative_base()

class QueryPerformance(Base):
    """Track search query performance and analytics"""
    __tablename__ = "query_performance"
    

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    profile_id = Column(String(255), nullable=False)
    query_text = Column(String(500), nullable=False)
    query_date = Column(DateTime, default=datetime.utcnow)
    profile_name = Column(String(500))
    profile_oid = Column(String(75))
    profile_score = Column(Float)
    context_score = Column(Float)
    combined_score = Column(Float)
    match_reasons = Column(Text)
    keywords = Column(JSONB)
    query_normalized = Column(String(500))
    search_type = Column(String(50))
    dataset_type = Column(String(50))
    user_session = Column(String(100))
    result_count = Column(Integer, default=0)
    has_results = Column(Boolean, default=False)
    top_result_id = Column(String(255))
    top_result_score = Column(Float)
    top_result_type = Column(String(100))
    search_duration_ms = Column(Integer)
    embedding_duration_ms = Column(Integer)
    db_query_duration_ms = Column(Integer)
    chroma_query_duration_ms = Column(Integer)
    similarity_scores = Column(JSONB)
    result_metadata = Column(JSONB)
    filters_applied = Column(JSONB)
    result_clicked = Column(Boolean, default=False)
    feedback_given = Column(String(255))
    time_to_feedback = Column(Integer)
    
    def __repr__(self):
        return f"<QueryPerformance(id='{self.id}', query='{self.query_text[:50]}...', results={self.result_count})>"