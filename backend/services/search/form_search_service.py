# backend/services/database_search_service.py
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import func
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from services.model_training.training_service_forms import FeedbackTrainingForm
from models.database.form_model import Form, Formset
from services.ultility.named_entity_removal_service import PHIQueryScrubber
from config.redis_cache import RedisQueryCache
from config.chroma import get_chroma_instance
import uuid
import logging
redis_client = RedisQueryCache()

class FormLookupService:   
    def __init__(self):
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            print(f"Error loading SentenceTransformer: {e}")
            self.model = None
        try:
            self.chroma_config = get_chroma_instance()
            self.collection = self.chroma_config.get_collection()
            self.phi_scrubber = PHIQueryScrubber(model_name="en_core_web_sm")
            self.feedback_trainer = FeedbackTrainingForm()
        except Exception as e:
            print(f"SearchService failed to connect to Chroma: {e}")
            self.collection = None
            self.chroma_config = None
            self.feedback_trainer = None

  
    def semantic_search(self, query: str, top_k: int = 10, db: Session = None, filters: Optional[Dict] = None) -> List[Dict]:
        phi_scrubbed_query = self.phi_scrubber.scrub_query(query)
        print(f"PHI scrubbed query: '{phi_scrubbed_query}'")
        if not phi_scrubbed_query or not phi_scrubbed_query.strip():
            logging.warning(f"Query was entirely PHI or empty after scrubbing: '{query}'")
            return []  # Return empty results instead of proceeding
        
      
        if phi_scrubbed_query != query:
            logging.info(f"PHI scrubbed from query. Original: '{query}' -> Scrubbed: '{phi_scrubbed_query}'")

        if not self.collection:
            raise ValueError("Chroma collection not available for search")
        
        if not db:
            raise ValueError("Database session required")
        

        results = []
        form_dict = {}
        query_normalized = phi_scrubbed_query.lower().strip()
        
        cached_feedback = redis_client.get_cached_feedback(query_normalized)
        similarity_scores = []
        form_ids = []        
        forms = []
        
            
        try:
            if cached_feedback:
                print(f"Cache hit for query: {query_normalized}") 
                cached_profile_ids = [f['form_id'] for f in cached_feedback]

                forms = db.query(Form).filter(
                    Form.id.in_(cached_profile_ids),
                    Form.is_active == True
                ).limit(top_k).all()
                
                active_profile_ids = [f.id for f in forms]
                
                inactive_count = len(cached_profile_ids) - len(active_profile_ids)
                if inactive_count > 0:
                    print(f"Found {inactive_count} inactive profiles in cache, filtering them out")
                    if inactive_count / len(cached_profile_ids) > 0.3:  
                        print(f"Cache has too many inactive profiles, invalidating...")
                        redis_client.clear_cache(query_normalized)
                        cached_feedback = None  

                if cached_feedback and len(active_profile_ids) > 0:
                    form_ids= active_profile_ids
                    form_dict = {f.id: f for f in forms}
                    similarity_scores = [1.0] * len(form_ids)
                    print(f"Using {len(form_ids)} active profiles from cache")
                else:
                    cached_feedback = None  

            if not cached_feedback:
                print(f"Cache miss or invalidated for query: {query_normalized}, performing vector search")
                
                if not self.model:
                        print("SentenceTransformer model not available")
                        return []

                query_embedding = self.model.encode([phi_scrubbed_query])[0].tolist()
                where_clause = {'is_active': True}
                if filters:
                    for key, value in filters.items():
                        if key in ['formset_id', 'domain', 'screening_tool','question','answer_concept']:
                            where_clause[key] = value
                

                search_results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    where=where_clause if where_clause else None,
                    include=['metadatas', 'distances', 'embeddings']
                )
                
                if not search_results['ids'] or not search_results['ids'][0]:
                    return []
                
                print(f"Chroma search results: {search_results}")
                if not search_results.get('ids') or not search_results['ids'] or not search_results['ids'][0]:
                    print("No results returned from Chroma")
                    return []
                
                form_ids = search_results['ids'][0]
                distances = search_results.get('distances', [[]])[0]
                
                if distances:
                        similarity_scores = [1 / (1 + distance) for distance in distances]
                else:
                    similarity_scores = [0.5] * len(form_ids)
                forms = db.query(Form).filter(
                        Form.id.in_(form_ids),
                        Form.is_active == True
                    ).all()
                form_dict = {f.id: f for f in forms}
                print(forms)
                if len(form_ids) > 0:
                        cache_data = [{'form_id': fid} for fid in form_ids]
                        redis_client.set_cached_feedback(query_normalized, cache_data, 3600)
                        print(f"Cached {len(form_ids)} results for future use")

            if len(similarity_scores) != len(form_ids):
                print(f"Score length ({len(similarity_scores)}) != Profile ID length ({len(form_ids)})")
                if len(similarity_scores) < len(form_ids):
                    similarity_scores.extend([0.5] * (len(form_ids) - len(similarity_scores)))
                else:
                    similarity_scores = similarity_scores[:len(form_ids)]

            

            for i, form_id in enumerate(form_ids):
                if form_id in form_dict:
                    form = form_dict[form_id]
                    results.append({
                        'id': form.id,
                        'domain': form.domain,
                        'screening_tool': form.screening_tool,
                        'question': form.question or "",
                        'loinc_panel_code': form.loinc_panel_code or "",
                        'loinc_panel_name': form.loinc_panel_name or "",
                        'loinc_question_code': form.loinc_question_code or "",
                        'loinc_question_name_long': form.loinc_question_name_long or "",
                        'answer_concept': form.answer_concept or "",
                        'loinc_answer': form.loinc_answer or "",
                        'loinc_concept': form.loinc_concept or "",
                        'snomed_code_ct': form.snomed_code_ct or "",
                        'similarity_score': similarity_scores[i],
                        'formset_id': form.formset_id,
                        'is_active': form.is_active,
                        'created_date': form.created_date
                    })
                else:
                    print(f"Profile {form_id} not found in database or inactive")
            
            print(f"Returning {len(results)} total results")
            
            return results
            
        except Exception as e:
            print(f"Error in semantic search: {e}")
            return []

    def hybrid_search(self, query: str, top_k: int = 10, db: Session = None,
                     semantic_weight: float = 0.7, filters: Optional[Dict] = None) -> List[Dict]:
        """
        Combine semantic search (Chroma) with traditional text search (PostgreSQL)
        """
        if not db:
            raise ValueError("Database session required")
            
        results_map = {}
        
        # Semantic search component
        if semantic_weight > 0 and self.collection:
            try:
                semantic_results = self.semantic_search(query, top_k, db, filters)
                for result in semantic_results:
                    results_map[result['id']] = {
                        **result,
                        'semantic_score': result['similarity_score'],
                        'text_score': 0.0
                    }
            except Exception as e:
                print(f"Semantic search failed in hybrid mode: {e}")

        # Traditional text search component
        if semantic_weight < 1.0:
            try:
                # Create a search text from multiple fields
                text_query = db.query(Form).filter(
                    Form.is_active == True
                ).filter(
                    # Search across multiple text fields
                    (Form.question.ilike(f"%{query}%")) |
                    (Form.domain.ilike(f"%{query}%")) |
                    (Form.screening_tool.ilike(f"%{query}%")) |
                    (Form.answer_concept.ilike(f"%{query}%")) |
                    (Form.loinc_panel_name.ilike(f"%{query}%")) |
                    (Form.loinc_question_name_long.ilike(f"%{query}%"))
                )
                
                # Apply filters
                if filters:
                    if filters.get('formset_id'):
                        text_query = text_query.filter(Form.formset_id == filters['formset_id'])
                    if filters.get('domain'):
                        text_query = text_query.filter(Form.domain == filters['domain'])
                    if filters.get('screening_tool'):
                        text_query = text_query.filter(Form.screening_tool.ilike(f"%{filters['screening_tool']}%"))
                
                text_forms = text_query.limit(top_k).all()
                
                for form in text_forms:
                    if form.id in results_map:
                        results_map[form.id]['text_score'] = 0.8
                    else:
                        results_map[form.id] = {
                            'id': form.id,
                            'domain': form.domain,
                            'screening_tool': form.screening_tool,
                            'question': form.question or "",
                            'loinc_panel_code': form.loinc_panel_code or "",
                            'loinc_panel_name': form.loinc_panel_name or "",
                            'loinc_question_code': form.loinc_question_code or "",
                            'loinc_question_name_long': form.loinc_question_name_long or "",
                            'answer_concept': form.answer_concept or "",
                            'loinc_answer': form.loinc_answer or "",
                            'loinc_concept': form.loinc_concept or "",
                            'snomed_code_ct': form.snomed_code_ct or "",
                            'formset_id': form.formset_id,
                            'is_active': form.is_active,
                            'created_date': form.created_date,
                            'semantic_score': 0.0,
                            'text_score': 0.8
                        }
            except Exception as e:
                print(f"Text search failed in hybrid mode: {e}")
        
        # Calculate hybrid scores and sort
        for result in results_map.values():
            result['similarity_score'] = (
                semantic_weight * result['semantic_score'] + 
                (1 - semantic_weight) * result['text_score']
            )
        
        sorted_results = sorted(
            results_map.values(),
            key=lambda x: x['similarity_score'],
            reverse=True
        )[:top_k]
        
        return sorted_results

    def traditional_search(self, query: str, db: Session = None, top_k: int = 10, 
                          filters: Optional[Dict] = None) -> List[Dict]:
        """
        Traditional text-based search using PostgreSQL only
        """
        if not db:
            raise ValueError("Database session required")
            
        try:
            # Build query - search across multiple text fields
            text_query = db.query(Form).filter(
                Form.is_active == True
            ).filter(
                # Search across multiple text fields
                (Form.question.ilike(f"%{query}%")) |
                (Form.domain.ilike(f"%{query}%")) |
                (Form.screening_tool.ilike(f"%{query}%")) |
                (Form.answer_concept.ilike(f"%{query}%")) |
                (Form.loinc_panel_name.ilike(f"%{query}%")) |
                (Form.loinc_question_name_long.ilike(f"%{query}%")) |
                (Form.loinc_concept.ilike(f"%{query}%"))
            )
            
            # Apply filters
            if filters:
                if filters.get('formset_id'):
                    text_query = text_query.filter(Form.formset_id == filters['formset_id'])
                if filters.get('domain'):
                    text_query = text_query.filter(Form.domain == filters['domain'])
                if filters.get('screening_tool'):
                    text_query = text_query.filter(Form.screening_tool.ilike(f"%{filters['screening_tool']}%"))
            
            forms = text_query.limit(top_k).all()
            
            # Format results
            results = []
            for form in forms:
                results.append({
                    'id': form.id,
                    'domain': form.domain,
                    'screening_tool': form.screening_tool,
                    'question': form.question or "",
                    'loinc_panel_code': form.loinc_panel_code or "",
                    'loinc_panel_name': form.loinc_panel_name or "",
                    'loinc_question_code': form.loinc_question_code or "",
                    'loinc_question_name_long': form.loinc_question_name_long or "",
                    'answer_concept': form.answer_concept or "",
                    'loinc_answer': form.loinc_answer or "",
                    'loinc_concept': form.loinc_concept or "",
                    'snomed_code_ct': form.snomed_code_ct or "",
                    'similarity_score': 1.0,  # No similarity score for text search
                    'formset_id': form.formset_id,
                    'is_active': form.is_active,
                    'created_date': form.created_date
                })
            
            # Optional: Log performance data
            """
            if results:
                for res in results:
                    create_performance_log(
                        form_id=res['id'],
                        query_text=query,
                        form_question=res['question'],
                        domain=res['domain'],
                        screening_tool=res['screening_tool'],
                        db=db
                    )    
            """
            return results
            
        except Exception as e:
            print(f"Traditional search failed: {e}")
            return []

    def search_by_filters_only(self, db: Session = None, filters: Dict = None, top_k: int = 100) -> List[Dict]:
        """
        Search using only filters (no text query)
        """
        if not db:
            raise ValueError("Database session required")
            
        try:
            query = db.query(Form).filter(Form.is_active == True)
            
            if filters:
                if filters.get('formset_id'):
                    query = query.filter(Form.formset_id == filters['formset_id'])
                if filters.get('domain'):
                    query = query.filter(Form.domain == filters['domain'])
                if filters.get('screening_tool'):
                    query = query.filter(Form.screening_tool.ilike(f"%{filters['screening_tool']}%"))
                if filters.get('loinc_panel_code'):
                    query = query.filter(Form.loinc_panel_code == filters['loinc_panel_code'])
                if filters.get('snomed_code_ct'):
                    query = query.filter(Form.snomed_code_ct.ilike(f"%{filters['snomed_code_ct']}%"))
            
            forms = query.limit(top_k).all()
            
            results = []
            for form in forms:
                results.append({
                    'id': form.id,
                    'domain': form.domain,
                    'screening_tool': form.screening_tool,
                    'question': form.question or "",
                    'loinc_panel_code': form.loinc_panel_code or "",
                    'loinc_panel_name': form.loinc_panel_name or "",
                    'loinc_question_code': form.loinc_question_code or "",
                    'loinc_question_name_long': form.loinc_question_name_long or "",
                    'answer_concept': form.answer_concept or "",
                    'loinc_answer': form.loinc_answer or "",
                    'loinc_concept': form.loinc_concept or "",
                    'snomed_code_ct': form.snomed_code_ct or "",
                    'similarity_score': 1.0,
                    'formset_id': form.formset_id,
                    'is_active': form.is_active,
                    'created_date': form.created_date
                })
            
            return results
            
        except Exception as e:
            print(f"Filter-only search failed: {e}")
            return []

   
    def get_search_stats(self, db: Session = None) -> Dict[str, Any]:
        """Get search statistics for forms and formsets"""
        if not db:
            raise ValueError("Database session required")
            
        try:
            active_forms = db.query(Form).filter(Form.is_active == True).count()
            total_forms = db.query(Form).count()
            
            active_formset = db.query(Formset).filter(Formset.status == "active").first()
            

            
            return {
                "active_forms": active_forms,
                "total_forms": total_forms,
                "active_formset": {
                    "id": active_formset.id if active_formset else None,
                    "name": active_formset.name if active_formset else None,
                    "activated_date": active_formset.activated_date if active_formset else None,
                    "record_count": active_formset.record_count if active_formset else 0
                } if active_formset else None
            }
        

        except Exception as e:
            print(f"Error getting search stats: {e}")
            return {
                "active_forms": 0,
                "total_forms": 0,
                "active_formset": None,
                "domain_distribution": [],
                "screening_tool_distribution": []
            }

    def record_feedback(self, query: str, form_id: str, feedback_type: str, 
                       user_id: str, session_id: str, original_score: float, 
                       db: Session, context_info: Optional[Dict] = None):
   
        if not self.feedback_trainer:
            print("Feedback trainer not available")
            return {"status": "error", "message": "Feedback system not initialized"}
        try:
            self.feedback_trainer.record_user_feedback(
                query, form_id, feedback_type, user_id, session_id, 
                original_score, db, context_info
            )
            self._handle_cache_invalidation(query, feedback_type)
            return {
                "status": "success", 
                "message": f"{feedback_type} feedback recorded and embeddings updated"
            }
        except Exception as e:
            print(f"Error recording feedback: {e}")
            return {"status": "error", "message": str(e)}

    def _handle_cache_invalidation(self, query: str, feedback_type: str):
        """Smart cache invalidation based on feedback type"""
        try:

            query_normalized = query.lower().strip()
            
            if feedback_type == 'negative':
                redis_client.clear_cache(query_normalized)
                print(f"Cache invalidated for negative feedback on query: {query}")
            elif feedback_type == 'positive':
                cached_data = redis_client.get_cached_feedback(query_normalized)
                if cached_data:
                    redis_client.set_cached_feedback(query_normalized, cached_data, 86400)
                    print(f"Cache extended for positive feedback on query: {query}")
        except Exception as e:
            print(f"Error handling cache invalidation: {e}")