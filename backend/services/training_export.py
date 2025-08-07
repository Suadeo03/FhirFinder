from datetime import datetime
import uuid 
from models.database.feedback_models import UserFeedback, FeedbackTrainingEmbeddings


class TrainingExportService:
    def record_feedback_enhanced(self, query, profile_id, feedback_type, db):

        

        query_embedding = self.model.encode([query])[0].tolist()
        
        enhanced_feedback = FeedbackTrainingEmbeddings(
            query_text=query,
            query_embedding=query_embedding,  
            profile_id=profile_id,
            feedback_type=feedback_type,
            original_profile_embedding=get_profile_embedding(profile_id),  # ← NEW!
            updated_profile_embedding=None,  # Will be filled after update
            timestamp=datetime.utcnow()
        )
        
        # Update embedding
        self.update_embedding(...)
        
        # Store the final state
        enhanced_feedback.updated_profile_embedding = get_profile_embedding(profile_id)
        db.add(enhanced_feedback)