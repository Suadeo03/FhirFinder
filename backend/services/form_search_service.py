# backend/services/database_search_service.py
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import functions as func
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from services.performance_log import create_performance_log
from models.database.form_model import Form, Formset, FormProcessingJob
from config.redis import RedisQueryCache
from config.chroma import ChromaConfig
import uuid


class FormLookupService:   
    def __init__(self):
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            print("SentenceTransformer model loaded successfully")
        except Exception as e:
            print(f"Error loading SentenceTransformer: {e}")
            self.model = None
        try:
            self.chroma_config = ChromaConfig()
            self.collection = self.chroma_config.collection
            
            if self.collection:
                info = self.chroma_config.get_client_info()
                print(f"Chroma initialized: {info}")

                if self.chroma_config.test_connection():
                    print("Chroma is ready for semantic search")
                else:
                    print("Chroma connection unstable")
            else:
                print("Chroma collection not available - falling back to traditional search only")

        except Exception as e:
            print(f"Error initializing Chroma: {e}")
            self.collection = None
            self.chroma_config = None

  
    def semantic_search(self, query: str, top_k: int = 10, db: Session = None, filters: Optional[Dict] = None) -> List[Dict]:
        """
        Semantic search using Chroma vector database for form data
        """
        if not self.collection:
            raise ValueError("Chroma collection not available for search")
        
        if not db:
            raise ValueError("Database session required")
        
        redis_client = RedisQueryCache()

        if redis_client.is_connected():
            query_normalized = query.lower().strip()
            
            cached_results = redis_client.get_cached_search(query_normalized)
            if cached_results:
                print(f"Cache hit for query: {query}")
                return cached_results
            else:
                print(f"Cache miss for query: {query}")
            
        try:
            # Generate embedding for the search query
            query_embedding = self.model.encode([query])[0].tolist()
            
            # Prepare filter metadata for Chroma
            where_clause = {'is_active': True}
            if filters:
                for key, value in filters.items():
                    if key in ['formset_id', 'domain', 'screening_tool','question','answer_concept']:
                        where_clause[key] = value
            
            # Search in Chroma
            search_results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_clause if where_clause else None,
                include=['metadatas', 'distances']
            )
            
            if not search_results['ids'] or not search_results['ids'][0]:
                return []
            
            form_ids = search_results['ids'][0]
            distances = search_results['distances'][0]
            
            similarity_scores = [1 / (1 + distance) for distance in distances]
            
            # Get forms from database
            forms = db.query(Form).filter(
                Form.id.in_(form_ids),
                Form.is_active == True
            ).all()
            
            form_dict = {f.id: f for f in forms}
            
            results = []
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
            
            # Cache results if Redis is available
            if redis_client.is_connected():
                redis_client.cache_search(query_normalized, results)
            
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

    def search_by_domain(self, domain: str, db: Session = None, top_k: int = 50) -> List[Dict]:
        """
        Search forms by domain (e.g., 'Homelessness', 'Mental Health', etc.)
        """
        if not db:
            raise ValueError("Database session required")
        
        try:
            forms = db.query(Form).filter(
                Form.is_active == True,
                Form.domain.ilike(f"%{domain}%")
            ).limit(top_k).all()
            
            results = []
            for form in forms:
                results.append({
                    'id': form.id,
                    'domain': form.domain,
                    'screening_tool': form.screening_tool,
                    'question': form.question or "",
                    'loinc_panel_code': form.loinc_panel_code or "",
                    'loinc_panel_name': form.loinc_panel_name or "",
                    'answer_concept': form.answer_concept or "",
                    'similarity_score': 1.0,
                    'formset_id': form.formset_id
                })
            
            return results
            
        except Exception as e:
            print(f"Domain search failed: {e}")
            return []

    def get_search_stats(self, db: Session = None) -> Dict[str, Any]:
        """Get search statistics for forms and formsets"""
        if not db:
            raise ValueError("Database session required")
            
        try:
            active_forms = db.query(Form).filter(Form.is_active == True).count()
            total_forms = db.query(Form).count()
            
            active_formset = db.query(Formset).filter(Formset.status == "active").first()
            
            # Get domain statistics
            domain_stats = db.query(Form.domain, db.func.count(Form.id)).filter(
                Form.is_active == True
            ).group_by(Form.domain).all()
            
            # Get screening tool statistics
            tool_stats = db.query(Form.screening_tool, db.func.count(Form.id)).filter(
                Form.is_active == True
            ).group_by(Form.screening_tool).all()
            
            return {
                "active_forms": active_forms,
                "total_forms": total_forms,
                "active_formset": {
                    "id": active_formset.id if active_formset else None,
                    "name": active_formset.name if active_formset else None,
                    "activated_date": active_formset.activated_date if active_formset else None,
                    "record_count": active_formset.record_count if active_formset else 0
                } if active_formset else None,
                "domain_distribution": [{"domain": domain, "count": count} for domain, count in domain_stats],
                "screening_tool_distribution": [{"tool": tool[:50], "count": count} for tool, count in tool_stats]  # Truncate long tool names
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

    def get_available_domains(self, db: Session = None) -> List[str]:
        """Get list of available domains"""
        if not db:
            raise ValueError("Database session required")
        
        try:
            domains = db.query(Form.domain).filter(
                Form.is_active == True
            ).distinct().all()
            
            return [domain[0] for domain in domains if domain[0]]
        except Exception as e:
            print(f"Error getting domains: {e}")
            return []

    def get_available_screening_tools(self, db: Session = None) -> List[str]:
        """Get list of available screening tools"""
        if not db:
            raise ValueError("Database session required")
        
        try:
            tools = db.query(Form.screening_tool).filter(
                Form.is_active == True
            ).distinct().all()
            
            return [tool[0] for tool in tools if tool[0]]
        except Exception as e:
            print(f"Error getting screening tools: {e}")
            return []