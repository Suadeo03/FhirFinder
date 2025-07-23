# backend/services/etl_service.py
import pandas as pd
import json
import uuid
from typing import List, Dict, Any, Union, Optional
from sqlalchemy.orm import Session
from models.database.models import Dataset, Profile, ProcessingJob
from config.chroma import ChromaConfig
from sentence_transformers import SentenceTransformer
import re
from datetime import datetime

class ETL_Form_Service:

    
    def __init__(self):
 
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.collection = ChromaConfig().collection
    
    def process_dataset(self, dataset_id: str, db: Session) -> bool:

        try:
            # Get dataset
            dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
            if not dataset:
                raise ValueError(f"Dataset {dataset_id} not found")
            

            dataset.status = "processing"
            db.commit()

            raw_data = self._load_file(dataset.file_path)

            profiles_data = self._transform_data(raw_data, dataset.filename)

            validated_data = self._validate_profiles(profiles_data)

            profile_count = self._load_profiles(validated_data, dataset_id, db)
            
  
            dataset.status = "ready"
            dataset.processed_date = datetime.utcnow()
            dataset.record_count = profile_count
            db.commit()
            
            print(f"Successfully processed {profile_count} profiles for dataset {dataset.name}")
            return True
            
        except Exception as e:

            dataset.status = "failed"
            dataset.error_message = str(e)
            db.commit()
            print(f"Error processing dataset {dataset_id}: {e}")
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
        """Transform raw data into standardized profile format"""
        transformed = []
        
        for i, row in enumerate(raw_data):
            try:
                profile = self._transform_single_profile(row, i)
                if profile:
                    transformed.append(profile)
            except Exception as e:
                print(f"Error transforming row {i}: {e}")
                continue
        
        return transformed
    
    def _transform_single_profile(self, row: Dict, row_index: int) -> Optional[Dict]:
        '''
        Current format:
            "id": "generated_id",
            "name": "Profile Name" # usually name of the profile or resource type by IG,
            "oid": "oid", #optional, if not present will be generated
            "resource_url": "url to resource", #optional, if not present will be generated
            "must_have": "1. field1, 2. field2", #optional, if not present will be empty
            "must_support": "1. field1, 2. field2", #fields that must be supported by implementations
            "invariants": "1. invariant1, 2. invariant2", #warings and errors that apply to implementation
            "resource_url": "url to the full specification of the profile"
            "description": "Profile description",
            "keywords": ["keyword1", "keyword2"],
            "category": "type of format", #JSON_FHIR
            "version": "version of resource",
            "resource_type": "Resource Type", ## e.g. "Patient", "Observation"/ ADT
            "use_contexts": [{"scenario": "Use Case", "keywords": ["context1", "context2"]}],
            "fhir_resource": {"resourceType": "StructureDefinition", ...}
        
        '''
        profile = {}
        profile['id'] = self._extract_field(row, ['id'])
        if not profile['id']:
            profile['id'] = f"HL7_{uuid.uuid4().hex[:8]}_{row_index}"

        profile['oid'] = self._extract_field(row, ['oid']) or f"OID_{uuid.uuid4().hex[:8]}_{row_index}"

        profile['name'] = self._extract_field(row, ['name'])
        if not profile['name']:
            raise ValueError(f"Missing required 'name' field in row {row_index}")
        profile['description'] = self._extract_field(row, ['description']) or ""

        keywords_raw = self._extract_field(row, ['keywords'])
        profile['keywords'] = self._parse_keywords(keywords_raw)

        must_have_raw = self._extract_field(row, ['must_have'])
        profile['must_have'] = self._parse_keywords(must_have_raw)

        must_support_raw = self._extract_field(row, ['must_support'])
        profile['must_support'] = self._parse_keywords(must_support_raw) or "None available"

        invariants_raw = self._extract_field(row, ['invariants'])
        profile['invariants'] = self._parse_keywords(invariants_raw) or "None available"
        
        profile['resource_url'] = self._extract_field(row, ['resource_url']) or "None available"

        profile['category'] = self._extract_field(row, ['category']) or "category"

        profile['version'] = self._extract_field(row, ['version']) or "version"
        
        profile['resource_type'] = self._extract_field(row, ['resource_type']) or "Unknown"
        
        profile['use_contexts'] = self._parse_use_contexts(row)
        
        fhir_resource_raw = self._extract_field(row, ['fhir_resource'])
        profile['fhir_resource'] = self._parse_keywords(fhir_resource_raw)

        profile['fhir_searchable_text'] = self._extract_fhir_fields(fhir_resource_raw)

        return profile
    
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
            except (KeyError, AttributeError, TypeError) as e:
                continue
        return None
    
    def _parse_keywords(self, keywords_raw: Any) -> List[str]:
        """Parse keywords from various formats"""
        if not keywords_raw:
            return []
        
        keywords_str = str(keywords_raw).strip()
        if not keywords_str or keywords_str.lower() == 'nan':
            return []
        
        try:
            if keywords_str.startswith('['):
                return json.loads(keywords_str)
        except:
            pass

        for delimiter in [',', ';', '|', '\n']:
            if delimiter in keywords_str:
                keywords = [kw.strip() for kw in keywords_str.split(delimiter)]
                return [kw for kw in keywords if kw]
        

        return [keywords_str] if keywords_str else []
    
    def _parse_use_contexts(self, row: Dict) -> List[Dict]:
        """Parse use contexts if present"""

        scenarios = self._extract_field(row, ['scenarios', 'use_cases', 'use_contexts'])
        if not scenarios:
            return []
        
        try:

            if isinstance(scenarios, str) and scenarios.startswith('['):
                return json.loads(scenarios)
        except:
            pass
        
        return [{
            "scenario": str(scenarios),
            "keywords": self._parse_keywords(scenarios)
        }]
        
    def _parse_field_requirements(self, row: Dict) -> List[Dict]:

        scenarios = self._extract_field(row, ['1.', '2.','3.','4.','5.','6.','7.','8.','9.','10.'])
        if not scenarios:
            return []
        
        try:
            if isinstance(scenarios, str) and scenarios.startswith('['):
                return json.loads(scenarios)
        except:
            pass

        return [{
            "number": str(scenarios),
            "keywords": self._parse_keywords(scenarios)
        }]



    def _extract_fhir_fields(self, fhir_resource_data: Any) -> str:
        """Extract field names/keys from FHIR resource structure for searchable text"""
        if not fhir_resource_data:
            return ""
        
        field_names = set()  
        
        try:

            if isinstance(fhir_resource_data, str):
                fhir_data = json.loads(fhir_resource_data)
            elif isinstance(fhir_resource_data, dict):
                fhir_data = fhir_resource_data
            elif isinstance(fhir_resource_data, list):
                all_field_names = set()
                for resource in fhir_resource_data:
                    resource_fields = self._extract_fhir_fields(resource)
                    if resource_fields:
                        all_field_names.update(resource_fields.split())
                return " ".join(sorted(all_field_names))
            else:
                return ""

            self._extract_fhir_keys_recursive(fhir_data, field_names)
            
            return " ".join(sorted(field_names))
            
        except Exception as e:
            print(f"Error extracting FHIR field names: {e}")
            return ""

    def _extract_fhir_keys_recursive(self, data: Union[Dict, List, Any], field_names: set, max_depth: int = 5, current_depth: int = 0):
        """Recursively extract field names/keys from FHIR structure"""
        if current_depth >= max_depth:
            return
        
        if isinstance(data, dict):
            for key, value in data.items():

                field_names.add(key)
                

                if isinstance(value, (dict, list)):
                    self._extract_fhir_keys_recursive(value, field_names, max_depth, current_depth + 1)
        
        elif isinstance(data, list):
            for item in data[:10]:  
                if isinstance(item, (dict, list)):
                    self._extract_fhir_keys_recursive(item, field_names, max_depth, current_depth + 1)
    
    def _validate_profiles(self, profiles_data: List[Dict]) -> List[Dict]:
        """Validate profile data"""
        validated = []
        
        for i, profile in enumerate(profiles_data):
            try:
                if not profile.get('id'):
                    raise ValueError(f"Missing ID for profile {i}")
                if not profile.get('name'):
                    raise ValueError(f"Missing name for profile {i}")

                if not isinstance(profile.get('keywords', []), list):
                    profile['keywords'] = []

                if not isinstance(profile.get('use_contexts', []), list):
                    profile['use_contexts'] = []
                
                validated.append(profile)
                
            except Exception as e:
                print(f"Validation error for profile {i}: {e}")
                continue
        
        return validated
    
    def _load_profiles(self, profiles_data: List[Dict], dataset_id: str, db: Session) -> int:
        """Load validated profiles into database - UPDATED VERSION"""
        loaded_count = 0
        search_texts = []
        embeddings = []

        try:
            db.rollback()
        except:
            pass
        
        for profile_data in profiles_data:
            try:

                base_search_text = f"{profile_data['name']} {profile_data['description']} {' '.join(profile_data['keywords'])}"

                fhir_search_text = profile_data.get('fhir_searchable_text', '')
                if fhir_search_text:
                    base_search_text += f" {fhir_search_text}"

                embedding = self.model.encode([base_search_text])[0].tolist()

                profile = Profile(
                    id=profile_data['id'],
                    oid=profile_data.get('oid'),  
                    name=profile_data['name'],
                    description=profile_data['description'],
                    must_have=profile_data.get('must_have'),  
                    must_support=profile_data.get('must_support'),  
                    invariants=profile_data.get('invariants'),  
                    resource_url=profile_data.get('resource_url'),  
                    keywords=profile_data['keywords'],
                    category=profile_data['category'],
                    version=profile_data.get('version'),  
                    resource_type=profile_data['resource_type'],
                    use_contexts=profile_data['use_contexts'],
                    fhir_resource=profile_data.get('fhir_resource'),
                    fhir_searchable_text=profile_data.get('fhir_searchable_text', ''),
                    dataset_id=dataset_id,
                    search_text=base_search_text, 
                )
                

                existing = db.query(Profile).filter(Profile.id == profile_data['id']).first()
                if existing:
                    for key, value in profile_data.items():
                        if hasattr(existing, key):
                            setattr(existing, key, value)
                    existing.search_text = base_search_text
                    existing.embedding_vector = embedding
                    existing.dataset_id = dataset_id
                else:

                    db.add(profile)
                
                search_texts.append(base_search_text)
                embeddings.append(embedding)
                profile_data['dataset_id'] = dataset_id 
                
                loaded_count += 1
                
            except Exception as e:
                print(f"Error loading profile {profile_data.get('id', 'unknown')}: {e}")
 
                db.rollback()
                continue
 
        try:
            db.commit()
        except Exception as e:
            print(f"Commit failed: {e}")
            db.rollback()
            raise
        
        if self.collection and profiles_data:
            self._batch_add_to_chroma(profiles_data, search_texts, embeddings)
            
        return loaded_count
    
    def activate_dataset(self, dataset_id: str, db: Session) -> bool:

        try:
            dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
            if not dataset:
                raise ValueError(f"Dataset {dataset_id} not found")
            profiles_updated = db.query(Profile).filter(
                Profile.dataset_id == dataset_id
            ).update({"is_active": True})
            dataset.status = "active"
            dataset.activated_date = datetime.utcnow()
            db.commit()
            
            if self.collection:
                profile_ids = [p.id for p in db.query(Profile.id).filter(Profile.dataset_id == dataset_id).all()]
                for profile_id in profile_ids:
                    self._update_chroma_metadata(profile_id, True)


            print(f"Activated {profiles_updated} profiles from dataset {dataset.name}")
            return True
            
        except Exception as e:
            print(f"Error activating dataset {dataset_id}: {e}")
            db.rollback()
            return False
        
    def deactivate_dataset(self, dataset_id: str, db: Session) -> bool:
        """Deactivate a dataset (remove from search)"""
        try:
            dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
            if not dataset:
                raise ValueError(f"Dataset {dataset_id} not found")
            profiles_updated = db.query(Profile).filter(
                Profile.dataset_id == dataset_id
            ).update({"is_active": False})
            dataset.status = "inactive"
            dataset.deactivated_date = datetime.utcnow()
            db.commit()
            if self.collection:
                profile_ids = [p.id for p in db.query(Profile.id).filter(Profile.dataset_id == dataset_id).all()]
                for profile_id in profile_ids:
                    self._update_chroma_metadata(profile_id, False)

            print(f"Deactivated {profiles_updated} profiles from dataset {dataset.name}")
            return True
            
        except Exception as e:
            print(f"Error deactivating dataset {dataset_id}: {e}")
            db.rollback()
            return False
    
    
    #chroma methods    
    def _add_to_chroma(self, profile_data: Dict, search_text: str, embedding: List[float]) -> bool:
        """Add a single profile's embedding to Chroma"""
        if not self.collection:
            return False
        
        try:
            self.collection.upsert(
                ids=[profile_data['id']],
                embeddings=[embedding],
                documents=[search_text],
                metadatas=[{
                    'name': profile_data['name'],
                    'description': (profile_data.get('description') or '')[:1000],
                    'resource_type': profile_data.get('resource_type', 'Unknown'),
                    'category': profile_data.get('category', 'Unknown'),
                    'dataset_id': profile_data.get('dataset_id', ''),
                    'keywords': ','.join((profile_data.get('keywords') or [])[:20]),
                    'is_active': True
                }]
            )
            return True
        except Exception as e:
            print(f"Error adding to Chroma: {e}")
            return False
    
    def _batch_add_to_chroma(self, profiles_data: List[Dict], search_texts: List[str], 
                            embeddings: List[List[float]]) -> bool:
        """Batch add profiles to Chroma"""
        if not self.collection or not profiles_data:
            return False
        
        try:
            ids = [p['id'] for p in profiles_data]
            metadatas = []
            
            for profile_data in profiles_data:
                metadatas.append({
                    'name': profile_data['name'],
                    'description': (profile_data.get('description') or '')[:1000],
                    'resource_type': profile_data.get('resource_type', 'Unknown'),
                    'category': profile_data.get('category', 'Unknown'),
                    'dataset_id': profile_data.get('dataset_id', ''),
                    'keywords': ','.join((profile_data.get('keywords') or [])[:20]),
                    'is_active': True
                })
            
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=search_texts,
                metadatas=metadatas
            )
            
            print(f"Successfully added {len(profiles_data)} profiles to Chroma")
            return True
            
        except Exception as e:
            print(f"Error in batch add to Chroma: {e}")
            return False
    
    def _update_chroma_metadata(self, profile_id: str, is_active: bool) -> bool:
        """Update metadata for a profile in Chroma"""
        if not self.collection:
            return False
        
        try:
            # Get current document
            results = self.collection.get(ids=[profile_id], include=['metadatas'])
            if results['ids']:
                metadata = results['metadatas'][0]
                metadata['is_active'] = is_active
                
                self.collection.update(
                    ids=[profile_id],
                    metadatas=[metadata]
                )
                return True
            return False
            
        except Exception as e:
            print(f"Error updating Chroma metadata: {e}")
            return False
    
    def _delete_from_chroma(self, profile_ids: List[str]) -> bool:
        """Delete profiles from Chroma"""
        if not self.collection or not profile_ids:
            return False
        
        try:
            self.collection.delete(ids=profile_ids)
            print(f"Deleted {len(profile_ids)} profiles from Chroma")
            return True
        except Exception as e:
            print(f"Error deleting from Chroma: {e}")
            return False
    