from datetime import datetime
import uuid 
from models.database.feedback_models import UserFeedback, FeedbackTrainingEmbeddings


def export_learned_model():
    """Export everything needed to recreate learned state"""
    
    # 1. All feedback training data
    training_data = db.query(FeedbackTrainingEmbeddings).all()
    
    # 2. Current state of all profile embeddings in Chroma
    current_embeddings = {}
    all_profiles = chroma_collection.get(include=['embeddings', 'metadatas'])
    for profile_id, embedding in zip(all_profiles['ids'], all_profiles['embeddings']):
        current_embeddings[profile_id] = embedding
    
    # 3. System configuration
    config = {
        "base_model": "sentence-transformers/all-MiniLM-L6-v2",
        "feedback_weights": {"positive": 0.1, "negative": -0.05},
        "learning_algorithm": "vector_adjustment_v1"
    }
    
    return {
        "training_history": [
            {
                "query_normalized": record.query_normalized,
                "query_embedding": record.query_embedding_vector,
                "resource_id": record.resource_id,
                "original_embedding": record.original_embedding_vector,
                "current_embedding": record.current_embedding_vector,
                "feedback_type": record.feedback_type,
                "feedback_count": record.feedback_count,
                "learning_magnitude": record.learning_magnitude
            }
            for record in training_data
        ],
        "current_embeddings": current_embeddings,
        "config": config,
        "export_timestamp": datetime.utcnow(),
        "total_feedback_events": len(training_data)
    }