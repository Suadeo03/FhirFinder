# backend/services/etl_service.py
import pandas as pd
import json
import uuid
from typing import List, Dict, Any, Union, Optional
from sqlalchemy.orm import Session
from models.database.form_model import Form, Formset, FormProcessingJob
from config.chroma import ChromaConfig
from sentence_transformers import SentenceTransformer
import re
from datetime import datetime


class ETL_Form_Service:

    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.collection = ChromaConfig().collection
    
    def process_dataset(self, dataset_id: str, db: Session) -> bool:
        """Process a formset dataset"""
        try:
            # Get formset (using the correct model)
            formset = db.query(Formset).filter(Formset.id == dataset_id).first()
            if not formset:
                raise ValueError(f"Formset {dataset_id} not found")
            
            formset.status = "processing"
            db.commit()

            raw_data = self._load_file(formset.file_path)
            form_data = self._transform_data(raw_data, formset.filename)
            validated_data = self._validate_forms(form_data)
            form_count = self._load_forms(validated_data, dataset_id, db) 
            
            formset.status = "ready"
            formset.processed_date = datetime.utcnow()
            formset.record_count = form_count
            db.commit()
            
            print(f"Successfully processed {form_count} forms for formset {formset.name}")
            return True
            
        except Exception as e:
            formset.status = "failed"
            formset.error_message = str(e)
            db.commit()
            print(f"Error processing formset {dataset_id}: {e}")
            return False
    
    def _load_file(self, file_path: str) -> List[Dict]:
        """Load data from various file formats"""
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            return df.to_dict('records')
        
        elif file_path.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data if isinstance(data, list) else [data]
        
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
            return df.to_dict('records')
        
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
    
    def _transform_data(self, raw_data: List[Dict], filename: str) -> List[Dict]:
        """Transform raw data into standardized form format"""
        transformed = []
        
        for i, row in enumerate(raw_data):
            try:
                form = self._transform_single_form(row, i)
                if form:
                    transformed.append(form)
            except Exception as e:
                print(f"Error transforming row {i}: {e}")
                continue
        
        return transformed
    
    def _transform_single_form(self, row: Dict, row_index: int) -> Optional[Dict]:
        """
        Transform a single row into form format:
            id : unique ID
            domain 
            screening_tool 
            loinc_panel_code 
            loinc_panel_name 
            question 
            loinc_question_code 
            loinc_question_name_long 
            answer_concept 
            loinc_answer 
            loinc_concept 
            snomed_code_ct 
        """
        form = {}
        form['id'] = self._extract_field(row, ['id'])
        if not form['id']:
            form['id'] = f"F_{uuid.uuid4().hex[:8]}_{row_index}"

        form['domain'] = self._extract_field(row, ['domain']) or 'Unknown'
        form['screening_tool'] = self._extract_field(row, ['screening_tool']) or 'Unknown'
        form['loinc_panel_code'] = self._extract_field(row, ['loinc_panel_code']) or "no code available"
        form['loinc_panel_name'] = self._extract_field(row, ['loinc_panel_name']) or ""
        form['question'] = self._extract_field(row, ['question']) or "no question available"
        
        # Fixed the typo in the field name
        form['loinc_question_code'] = self._extract_field(row, ['loinc_question_code']) or "no question code"
        form['loinc_question_name_long'] = self._extract_field(row, ['loinc_question_name_long']) or ""
        
        answer_raw = self._extract_field(row, ['answer_concept'])
        form['answer_concept'] = self._parse_keywords(answer_raw)

        loinc_answer_raw = self._extract_field(row, ['loinc_answer'])
        form['loinc_answer'] = self._parse_keywords(loinc_answer_raw) or "None available"
        
        loinc_concept_raw = self._extract_field(row, ['loinc_answer_concept'])
        form['loinc_answer_concept'] = self._parse_keywords(loinc_concept_raw) or "None available"

        snomed_raw = self._extract_field(row, ['snomed_code_ct'])
        form['snomed_code_ct'] = self._parse_keywords(snomed_raw) or "No code available"

        return form
    
    def get_dataset_fields(self, raw_data: List[Dict]) -> str:
        """Get comma-separated list of all field names in dataset"""
        all_keys = set()
        for row in raw_data:
            if isinstance(row, dict):
                all_keys.update(row.keys())
        return ", ".join(sorted(all_keys))
    
    def get_fhir_dataset_fields(self, raw_data: List[Dict]) -> str:
        """Get comma-separated list of all FHIR field names in dataset"""
        all_fhir_fields = set()
        
        for row in raw_data:
            if isinstance(row, dict):
                fhir_resource_raw = self._extract_field(row, ['fhir_resource', 'fhir', 'resource', 'structure_definition'])
                if fhir_resource_raw:
                    try:
                        if isinstance(fhir_resource_raw, str):
                            fhir_data = json.loads(fhir_resource_raw)
                        else:
                            fhir_data = fhir_resource_raw
                        
                        self._collect_fhir_field_names(fhir_data, all_fhir_fields)
                    except Exception as e:
                        print(f"Error parsing FHIR resource: {e}")
                        continue
        
        return ", ".join(sorted(all_fhir_fields))

    def _collect_fhir_field_names(self, data: Union[Dict, List, Any], field_names: set, max_depth: int = 5, current_depth: int = 0):
        """Recursively collect all field names from FHIR structure"""
        if current_depth >= max_depth:
            return
        
        if isinstance(data, dict):
            for key, value in data.items():
                field_names.add(key)
                if isinstance(value, (dict, list)):
                    self._collect_fhir_field_names(value, field_names, max_depth, current_depth + 1)
        
        elif isinstance(data, list):
            for item in data[:5]:  
                self._collect_fhir_field_names(item, field_names, max_depth, current_depth + 1)
    
    def _extract_field(self, row: Dict, possible_keys: List[str]) -> Optional[str]:
        """Extract field value from row using possible key names"""
        if not row or not isinstance(row, dict):
            return None
        if not possible_keys:
            return None
            
        for key in possible_keys:
            try:
                if key in row and row[key] is not None:
                    value = str(row[key]).strip()
                    if value and value.lower() not in ['nan', 'null', 'none', '']:
                        return value
            except (KeyError, AttributeError, TypeError):
                continue
        return None
    
    def _parse_keywords(self, keywords_raw: Any) -> str:
        """Parse keywords from various formats and return as string"""
        if not keywords_raw:
            return ""
        
        keywords_str = str(keywords_raw).strip()
        if not keywords_str or keywords_str.lower() == 'nan':
            return ""
        
        # Try to parse as JSON list first
        try:
            if keywords_str.startswith('['):
                parsed_list = json.loads(keywords_str)
                return ", ".join(str(item) for item in parsed_list if item)
        except:
            pass

        # Try different delimiters
        for delimiter in [',', ';', '|', '\n']:
            if delimiter in keywords_str:
                keywords = [kw.strip() for kw in keywords_str.split(delimiter)]
                return ", ".join(kw for kw in keywords if kw)
        
        return keywords_str if keywords_str else ""
    
    def _validate_forms(self, form_data: List[Dict]) -> List[Dict]:
        """Validate form data"""
        validated = []
        
        for i, form in enumerate(form_data):
            try:
                if not form.get('id'):
                    raise ValueError(f"Missing ID for form {i}")
                if not form.get('question'):
                    raise ValueError(f"Missing question for form {i}")

                validated.append(form)
                
            except Exception as e:
                print(f"Validation error for form {i}: {e}")
                continue
        
        return validated
    
    def _load_forms(self, form_data: List[Dict], formset_id: str, db: Session) -> int:
        """Load validated forms into database"""
        loaded_count = 0
        search_texts = []
        embeddings = []

        try:
            db.rollback()
        except:
            pass
        
        for form_dict in form_data:
            try:
                # Create search text from the form dict
                base_search_text = f"{form_dict.get('domain', '')} {form_dict.get('question', '')} {form_dict.get('answer_concept', '')} "

                # Generate embedding
                embedding = self.model.encode([base_search_text])[0].tolist()

                # Create Form object with correct field names
                form_obj = Form(
                    id=form_dict['id'],
                    domain=form_dict.get('domain', ''),  
                    screening_tool=form_dict.get('screening_tool', ''),
                    loinc_panel_code=form_dict.get('loinc_panel_code', ''),
                    loinc_panel_name=form_dict.get('loinc_panel_name', ''),  
                    question=form_dict.get('question', ''), 
                    loinc_question_code=form_dict.get('loinc_question_code', ''),
                    loinc_question_name_long=form_dict.get('loinc_question_name_long', ''),  
                    answer_concept=form_dict.get('answer_concept', ''),    
                    loinc_answer=form_dict.get('loinc_answer', ''),
                    loinc_concept=form_dict.get('loinc_concept', ''),
                    snomed_code_ct=form_dict.get('snomed_code_ct', ''),
                    formset_id=formset_id, 
                    is_active=False  
                )
                
                
                existing = db.query(Form).filter(Form.id == form_dict['id']).first()
                if existing:
                    # Update existing form
                    existing.domain = form_dict.get('domain', '')
                    existing.screening_tool = form_dict.get('screening_tool', '')
                    existing.loinc_panel_code = form_dict.get('loinc_panel_code', '')
                    existing.loinc_panel_name = form_dict.get('loinc_panel_name', '')
                    existing.question = form_dict.get('question', '')
                    existing.loinc_question_code = form_dict.get('loinc_question_code', '')
                    existing.loinc_question_name_long = form_dict.get('loinc_question_name_long', '')
                    existing.answer_concept = form_dict.get('answer_concept', '')
                    existing.loinc_answer = form_dict.get('loinc_answer', '')
                    existing.loinc_concept = form_dict.get('loinc_concept', '')
                    existing.snomed_code_ct = form_dict.get('snomed_code_ct', '')
                    existing.formset_id = formset_id
                else:
                    # Add new form
                    db.add(form_obj)
                
                search_texts.append(base_search_text)
                embeddings.append(embedding)
                form_dict['formset_id'] = formset_id  # For Chroma metadata
                
                loaded_count += 1
                
            except Exception as e:
                print(f"Error loading form {form_dict.get('id', 'unknown')}: {e}")
                db.rollback()
                continue

        try:
            db.commit()
        except Exception as e:
            print(f"Commit failed: {e}")
            db.rollback()
            raise
        
        # Add to Chroma if available
        if self.collection and form_data:
            self._batch_add_to_chroma(form_data, search_texts, embeddings)
            
        return loaded_count
    
    def activate_dataset(self, formset_id: str, db: Session) -> bool:
        """Activate a formset (make forms searchable)"""
        try:
            formset = db.query(Formset).filter(Formset.id == formset_id).first()
            if not formset:
                raise ValueError(f"Formset {formset_id} not found")
            
            # Update forms to active
            forms_updated = db.query(Form).filter(
                Form.formset_id == formset_id  # Use correct field name
            ).update({"is_active": True})
            
            formset.status = "active"
            formset.activated_date = datetime.utcnow()
            db.commit()
            
            # Update Chroma metadata
            if self.collection:
                form_ids = [f.id for f in db.query(Form.id).filter(Form.formset_id == formset_id).all()]
                for form_id in form_ids:
                    self._update_chroma_metadata(form_id, True)

            print(f"Activated {forms_updated} forms from formset {formset.name}")
            return True
            
        except Exception as e:
            print(f"Error activating formset {formset_id}: {e}")
            db.rollback()
            return False
        
    def deactivate_dataset(self, formset_id: str, db: Session) -> bool:
        """Deactivate a formset (remove from search)"""
        try:
            formset = db.query(Formset).filter(Formset.id == formset_id).first()
            if not formset:
                raise ValueError(f"Formset {formset_id} not found")
            
           
            forms_updated = db.query(Form).filter(
                Form.formset_id == formset_id  
            ).update({"is_active": False})
            
            formset.status = "inactive"
            formset.deactivated_date = datetime.utcnow()
            db.commit()
            
            # Update Chroma metadata
            if self.collection:
                form_ids = [f.id for f in db.query(Form.id).filter(Form.formset_id == formset_id).all()]
                for form_id in form_ids:
                    self._update_chroma_metadata(form_id, False)

            print(f"Deactivated {forms_updated} forms from formset {formset.name}")
            return True
            
        except Exception as e:
            print(f"Error deactivating formset {formset_id}: {e}")
            db.rollback()
            return False
    

    def _add_to_chroma(self, form_data: Dict, search_text: str, embedding: List[float]) -> bool:
        """Add a single form's embedding to Chroma"""
        if not self.collection:
            return False
        
        try:
            self.collection.upsert(
                ids=[form_data['id']],
                embeddings=[embedding],
                documents=[search_text],
                metadatas=[{
                    'domain': form_data.get('domain', ''),
                    'screening_tool': (form_data.get('screening_tool') or '')[:1000],
                    'question': form_data.get('question', 'Unknown'),
                    'answer_concept': form_data.get('answer_concept', 'Unknown'),
                    'formset_id': form_data.get('formset_id', ''), 
                    'is_active': True
                }]
            )
            return True
        except Exception as e:
            print(f"Error adding to Chroma: {e}")
            return False
    
    def _batch_add_to_chroma(self, forms_data: List[Dict], search_texts: List[str], 
                            embeddings: List[List[float]]) -> bool:
        """Batch add forms to Chroma"""
        if not self.collection or not forms_data:
            return False
        
        try:
            ids = [f['id'] for f in forms_data]
            metadatas = []
            
            for form_data in forms_data:
                metadatas.append({
                    'domain': form_data.get('domain', ''),
                    'screening_tool': (form_data.get('screening_tool') or '')[:1000],
                    'question': form_data.get('question', 'Unknown'),
                    'answer_concept': form_data.get('answer_concept', 'Unknown'),
                    'formset_id': form_data.get('formset_id', ''),  # Use correct field name
                    'is_active': True
                })
            
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=search_texts,
                metadatas=metadatas
            )
            
            print(f"Successfully added {len(forms_data)} forms to Chroma")
            return True
            
        except Exception as e:
            print(f"Error in batch add to Chroma: {e}")
            return False
    
    def _update_chroma_metadata(self, form_id: str, is_active: bool) -> bool:
        """Update metadata for a form in Chroma"""
        if not self.collection:
            return False
        
        try:
    
            results = self.collection.get(ids=[form_id], include=['metadatas'])
            if results['ids']:
                metadata = results['metadatas'][0]
                metadata['is_active'] = is_active
                
                self.collection.update(
                    ids=[form_id],
                    metadatas=[metadata]
                )
                return True
            return False
            
        except Exception as e:
            print(f"Error updating Chroma metadata: {e}")
            return False
    
    def _delete_from_chroma(self, form_ids: List[str]) -> bool:
        """Delete forms from Chroma"""
        if not self.collection or not form_ids:
            return False
        
        try:
            self.collection.delete(ids=form_ids)
            print(f"Deleted {len(form_ids)} forms from Chroma")
            return True
        except Exception as e:
            print(f"Error deleting from Chroma: {e}")
            return False