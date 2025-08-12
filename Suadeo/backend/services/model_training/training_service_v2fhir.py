# backend/services/model_training/training_service_v2_fhir.py
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from sqlalchemy.orm import Session
from sqlalchemy import func
import numpy as np
from sentence_transformers import SentenceTransformer
from models.database.feedback_models import UserFeedback, SearchQualityMetrics
from models.database.fhir_v2_model import V2FHIRdata
from config.chroma import get_chroma_instance, is_chroma_available

class FeedbackTrainingV2FHIR:
    def __init__(self):
        try:
            self.chroma_config = get_chroma_instance()
            self.collection = self.chroma_config.get_collection()
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ FeedbackTrainingV2FHIR connected to Chroma singleton")
            
            self.feedback_weights = {
                'positive': 0.1,    
                'negative': -0.05,  
            }
        except Exception as e:
            print(f"FeedbackTrainingV2FHIR failed to connect to Chroma: {e}")
            self.collection = None
            self.chroma_config = None

    def record_user_feedback(self, query: str, mapping_id: str, feedback_type: str, 
                           user_id: str, session_id: str, original_score: float, 
                           db: Session, context_info: Optional[Dict] = None):
        """Record user feedback for V2-FHIR mappings"""
        
        # Get the V2 FHIR mapping instead of Form
        mapping = db.query(V2FHIRdata).filter(V2FHIRdata.id == mapping_id).first()
        
        if not mapping:
            print(f"V2 FHIR mapping {mapping_id} not found in database")
            return {"status": "error", "message": "V2 FHIR mapping not found"}
        
        if not mapping.is_active:
            print(f"V2 FHIR mapping {mapping_id} is inactive, skipping embedding update")
            self._store_feedback_record(query, mapping_id, feedback_type, user_id, 
                                      session_id, original_score, db, context_info)
            return {"status": "recorded", "message": "Feedback recorded for inactive mapping (no embedding update)"}
        
        # Store feedback record
        self._store_feedback_record(query, mapping_id, feedback_type, user_id, 
                                  session_id, original_score, db, context_info)
        
        # Update embedding based on feedback
        if feedback_type in ['positive', 'negative']:
            self._update_mapping_embedding(query, mapping_id, feedback_type, db)
        
        # Handle cache based on feedback type
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

            # Use V2 FHIR cache key format
            cached_data = redis_client.get_cached_feedback(f"v2fhir_{query_normalized}")
            if cached_data:
                redis_client.set_cached_feedback(f"v2fhir_{query_normalized}", cached_data, 86400)
                print(f"Extended V2 FHIR cache for positive feedback on: {query}")
        except Exception as e:
            print(f"Error extending cache: {e}")

    def _update_mapping_embedding(self, query: str, mapping_id: str, feedback_type: str, db: Session):
        """Update V2 FHIR mapping embedding based on feedback"""
        try:
            if not self.collection:
                print(f"No collection available")
                return

            current_results = self.collection.get(
                ids=[mapping_id],
                include=['embeddings', 'metadatas']
            )
            
            if not current_results or current_results.get('embeddings') is None:
                print(f"No results or embeddings from Chroma for mapping {mapping_id}")
                return
                
            embeddings = current_results.get('embeddings')
            if isinstance(embeddings, np.ndarray) and embeddings.size == 0:
                print(f"Empty embeddings array for mapping {mapping_id}")
                return
            
            if not self._is_valid_embedding_result(current_results):
                print(f"Invalid embedding result for mapping {mapping_id}")
                return
                
            current_embedding = np.array(current_results['embeddings'][0])
            
            # Validate the embedding array
            if current_embedding.size == 0:
                print(f"Empty embedding array for mapping {mapping_id}")
                return
                
            if np.any(np.isnan(current_embedding)):
                print(f"NaN values in embedding for mapping {mapping_id}")
                return
            
            # Get metadata
            metadatas = current_results.get('metadatas')
            if metadatas is not None and len(metadatas) > 0:
                metadata = metadatas[0] if metadatas[0] is not None else {}
            else:
                metadata = {}

            # Generate query embedding
            query_embedding = np.array(self.model.encode([query])[0])
            
            # Validate query embedding
            if query_embedding.size == 0 or np.any(np.isnan(query_embedding)):
                print(f"Invalid query embedding for query: {query}")
                return           
            
            # Calculate embedding update direction
            if feedback_type == 'positive':
                # Move mapping embedding closer to query
                direction = query_embedding - current_embedding
                weight = self.feedback_weights['positive']
            else:  # negative
                # Move mapping embedding away from query
                direction = current_embedding - query_embedding
                weight = abs(self.feedback_weights['negative'])

     
            updated_embedding = current_embedding + (weight * direction)

            embedding_norm = np.linalg.norm(updated_embedding)
            if embedding_norm > 1e-8:  
                updated_embedding = updated_embedding / embedding_norm
            else:
                print(f"Warning: Near-zero embedding norm for mapping {mapping_id}, keeping original")
                updated_embedding = current_embedding

            if np.any(np.isnan(updated_embedding)) or np.any(np.isinf(updated_embedding)):
                print(f"Invalid updated embedding for mapping {mapping_id}, keeping original")
                updated_embedding = current_embedding

            updated_metadata = {
                **metadata,
                'last_feedback_update': datetime.utcnow().isoformat(),
                'feedback_count': metadata.get('feedback_count', 0) + 1,
                'mapping_type': 'v2_fhir'  
            }

            self.collection.update(
                ids=[mapping_id],
                embeddings=[updated_embedding.tolist()],
                metadatas=[updated_metadata]
            )
            
            print(f"Successfully updated embedding for V2 FHIR mapping {mapping_id} based on {feedback_type} feedback")
                
        except Exception as e:
            print(f"Error updating mapping embedding: {e}")
            import traceback
            traceback.print_exc()

    def _is_valid_embedding_result(self, results) -> bool:
        """Validate embedding result structure"""
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
        """Batch retraining based on accumulated feedback for V2 FHIR mappings"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_back)
            
            # Get feedback data - note: we're still using the same feedback tables
            # but now the profile_id actually refers to mapping_id
            feedback_data = db.query(UserFeedback).filter(
                UserFeedback.timestamp >= cutoff_date,
                UserFeedback.feedback_type.in_(['positive', 'negative'])
            ).all()
            
            mapping_feedback = {}
            for feedback in feedback_data:
                mapping_id = feedback.profile_id  # This now refers to V2 FHIR mapping ID
                if mapping_id not in mapping_feedback:
                    mapping_feedback[mapping_id] = {'positive': [], 'negative': []}
                
                mapping_feedback[mapping_id][feedback.feedback_type].append({
                    'query': feedback.query_text,
                    'timestamp': feedback.timestamp,
                    'score': feedback.original_score
                })
        
            for mapping_id, feedback in mapping_feedback.items():
                self._batch_update_mapping_embedding(mapping_id, feedback, db)
                
            print(f"Batch retraining completed for {len(mapping_feedback)} V2 FHIR mappings")
            
        except Exception as e:
            print(f"Error in batch retraining: {e}")
    
    def _batch_update_mapping_embedding(self, mapping_id: str, feedback_data: Dict, db: Session):
        """Batch update embedding for a V2 FHIR mapping"""
        try:
            current_results = self.collection.get(
                ids=[mapping_id],
                include=['embeddings', 'metadatas']
            )
            
            if current_results is None:
                print(f"No results for batch update of mapping {mapping_id}")
                return
                
            embeddings = current_results.get('embeddings')
            if embeddings is None or len(embeddings) == 0 or embeddings[0] is None:
                print(f"No valid embeddings for batch update of mapping {mapping_id}")
                return
            
            current_embedding = np.array(embeddings[0])
            
            if current_embedding.size == 0 or np.any(np.isnan(current_embedding)):
                print(f"Invalid current embedding for mapping {mapping_id}")
                return
            
            metadatas = current_results.get('metadatas')
            metadata = metadatas[0] if (metadatas and len(metadatas) > 0 and metadatas[0]) else {}
            total_update = np.zeros_like(current_embedding)
            update_count = 0
            
            # Process positive feedback
            for pos_feedback in feedback_data.get('positive', []):
                try:
                    query_embedding = np.array(self.model.encode([pos_feedback['query']])[0])
                    
                    if np.any(np.isnan(query_embedding)):
                        print(f"Skipping invalid query embedding for: {pos_feedback['query']}")
                        continue
                    
                    direction = query_embedding - current_embedding

                    # Time-based decay for older feedback
                    days_old = (datetime.utcnow() - pos_feedback['timestamp']).days
                    time_weight = np.exp(-days_old / 14)  # 2-week decay
                    
                    total_update += self.feedback_weights['positive'] * time_weight * direction
                    update_count += 1
                    
                except Exception as e:
                    print(f"Error processing positive feedback: {e}")
                    continue

            # Process negative feedback
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
                # Apply averaged update
                avg_update = total_update / update_count
                updated_embedding = current_embedding + avg_update
                
                # Normalize
                embedding_norm = np.linalg.norm(updated_embedding)
                if embedding_norm > 1e-8:
                    updated_embedding = updated_embedding / embedding_norm
                else:
                    print(f"Warning: Near-zero norm in batch update for mapping {mapping_id}")
                    updated_embedding = current_embedding
                
                # Final validation
                if np.any(np.isnan(updated_embedding)) or np.any(np.isinf(updated_embedding)):
                    print(f"Invalid final embedding for mapping {mapping_id}, keeping original")
                    updated_embedding = current_embedding
                
                # Update metadata
                updated_metadata = {
                    **metadata,
                    'last_batch_update': datetime.utcnow().isoformat(),
                    'total_feedback_processed': metadata.get('total_feedback_processed', 0) + update_count,
                    'mapping_type': 'v2_fhir'
                }
                
                # Update in Chroma
                self.collection.update(
                    ids=[mapping_id],
                    embeddings=[updated_embedding.tolist()],
                    metadatas=[updated_metadata]
                )
                
                print(f"Successfully batch updated V2 FHIR mapping {mapping_id} with {update_count} feedback items")
            else:
                print(f"No valid feedback to process for mapping {mapping_id}")
        
        except Exception as e:
            print(f"Error in batch update for mapping {mapping_id}: {e}")
            import traceback
            traceback.print_exc()
   
    def _invalidate_query_cache(self, query: str):
        """Remove cached results for this query to force fresh vector search"""
        try:
            from config.redis_cache import RedisQueryCache
            redis_client = RedisQueryCache()
            query_normalized = query.lower().strip()
            
            # Clear both regular and V2 FHIR cache keys
            redis_client.clear_cache(query_normalized)
            redis_client.clear_cache(f"v2fhir_{query_normalized}")
            print(f"Invalidated cache for V2 FHIR query: {query_normalized}")
        except Exception as e:
            print(f"Error invalidating cache: {e}")
    
    def _store_feedback_record(self, query: str, mapping_id: str, feedback_type: str, 
                             user_id: str, session_id: str, original_score: float, 
                             db: Session, context_info: Optional[Dict] = None):
        """Store feedback record in database"""
        try:
            feedback = UserFeedback()
            feedback.query_text = query
            feedback.query_normalized = query.lower().strip()
            feedback.profile_id = mapping_id  # Store mapping_id in profile_id field
            feedback.feedback_type = feedback_type
            feedback.user_id = user_id
            feedback.session_id = session_id
            feedback.timestamp = datetime.utcnow()
            feedback.original_score = original_score
            feedback.context_info = context_info or {'mapping_type': 'v2_fhir'}  # Add context
            
            db.add(feedback)
            db.commit()

            self._update_search_quality_metrics(query, mapping_id, feedback_type, db)
            
        except Exception as e:
            print(f"Error storing feedback record: {e}")
            db.rollback()

    def _update_search_quality_metrics(self, query: str, mapping_id: str, feedback_type: str, db: Session):
        """Update search quality metrics for V2 FHIR mappings"""
        try:
            query_normalized = query.lower().strip()
            
            # Get or create quality metric record
            quality_metric = db.query(SearchQualityMetrics).filter(
                SearchQualityMetrics.query_normalized == query_normalized,
                SearchQualityMetrics.profile_id == str(mapping_id)
            ).first()
            
            if not quality_metric:
                quality_metric = SearchQualityMetrics(
                    query_normalized=query_normalized,
                    profile_id=str(mapping_id),  # mapping_id stored in profile_id field
                    positive_feedback_count=0,
                    negative_feedback_count=0,
                    neutral_feedback_count=0,
                    total_feedback_count=0,
                    feedback_ratio=0.0
                )
                db.add(quality_metric)
            
            # Update counts
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
            
            # Calculate feedback ratio
            positive_final = quality_metric.positive_feedback_count or 0
            negative_final = quality_metric.negative_feedback_count or 0
            total_feedback = positive_final + negative_final
            
            quality_metric.feedback_ratio = (
                positive_final / total_feedback if total_feedback > 0 else 0.0
            )
            
            quality_metric.last_updated = datetime.utcnow()
            db.commit()
            
            print(f"Updated V2 FHIR quality metrics for query '{query}' and mapping {mapping_id}")
            
        except Exception as e:
            db.rollback()
            print(f"Error updating search quality metrics: {e}")
            import traceback
            traceback.print_exc()
    
    def get_learning_stats(self, db: Session) -> Dict:
        """Get statistics about the V2 FHIR learning system"""
        try:
            total_mappings = self.collection.count()
            
            # Get mappings with feedback updates
            mappings_with_metadata = self.collection.get(
                include=['metadatas']
            )
            
            updated_mappings = 0
            total_feedback_processed = 0
            
            if mappings_with_metadata.get('metadatas'):
                for metadata in mappings_with_metadata['metadatas']:
                    if metadata and 'feedback_count' in metadata:
                        updated_mappings += 1
                        total_feedback_processed += metadata.get('feedback_count', 0)

            # Get recent feedback count
            recent_feedback = db.query(UserFeedback).filter(
                UserFeedback.timestamp >= datetime.utcnow() - timedelta(days=7)
            ).count()
            
            return {
                'total_mappings_in_vectordb': total_mappings,
                'mappings_updated_by_feedback': updated_mappings,
                'total_feedback_processed': total_feedback_processed,
                'recent_feedback_count': recent_feedback,
                'learning_rate_positive': self.feedback_weights['positive'],
                'learning_rate_negative': abs(self.feedback_weights['negative']),
                'mapping_type': 'v2_fhir',
                'last_batch_update': None  
            }
            
        except Exception as e:
            print(f"Error getting V2 FHIR learning stats: {e}")
            return {}
    
    def get_feedback_stats_simple(self, db: Session, days: int = 30) -> Dict[str, any]:
        """Get simple feedback statistics for V2 FHIR mappings"""
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
                "most_improved_mappings": [],
                "most_problematic_mappings": [],
                "mapping_type": "v2_fhir"
            }
            
        except Exception as e:
            print(f"Error getting V2 FHIR feedback stats: {e}")
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
                "most_improved_mappings": [],
                "most_problematic_mappings": [],
                "mapping_type": "v2_fhir"
            }