# backend/services/etl_service.py
import pandas as pd
import json
import uuid
from typing import List, Dict, Any, Union, Optional
from sqlalchemy.orm import Session
from models.database.models import Dataset, Profile, ProcessingJob
from sentence_transformers import SentenceTransformer
import re
from datetime import datetime

class ETLService:
    """Extract, Transform, Load service for profile data"""
    
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def process_dataset(self, dataset_id: str, db: Session) -> bool:
        """Process an uploaded dataset"""
        try:
            # Get dataset
            dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
            if not dataset:
                raise ValueError(f"Dataset {dataset_id} not found")
            
            # Update status
            dataset.status = "processing"
            db.commit()
            
            # Load and parse file
            raw_data = self._load_file(dataset.file_path)
            
            # Transform data
            profiles_data = self._transform_data(raw_data, dataset.filename)
            
            # Validate data
            validated_data = self._validate_profiles(profiles_data)
            
            # Load into database
            profile_count = self._load_profiles(validated_data, dataset_id, db)
            
            # Update dataset
            dataset.status = "ready"
            dataset.processed_date = datetime.utcnow()
            dataset.record_count = profile_count
            db.commit()
            
            print(f"Successfully processed {profile_count} profiles for dataset {dataset.name}")
            return True
            
        except Exception as e:
            # Update dataset with error
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
            profile['id'] = f"generated_{uuid.uuid4().hex[:8]}_{row_index}"
        profile['name'] = self._extract_field(row, ['name'])
        if not profile['name']:
            raise ValueError(f"Missing required 'name' field in row {row_index}")
        profile['description'] = self._extract_field(row, [
            'description']) or ""

        keywords_raw = self._extract_field(row, [
            'keywords', 'Keywords', 'tags', 'Tags', 'terms', 'Terms'
        ])
        profile['keywords'] = self._parse_keywords(keywords_raw)
        
        # Category
        profile['category'] = self._extract_field(row, [
            'category', 'Category', 'type', 'Type', 'Class', 'class'
        ]) or "general"

        profile['version'] = self._extract_field(row, [
            'version'
        ]) or "version"
        
        # Resource type
        profile['resource_type'] = self._extract_field(row, [
            'resource_type', 'Resource Type', 'resourceType', 'resource', 'Resource'
        ]) or "Unknown"
        
        # Use contexts (optional advanced field)
        profile['use_contexts'] = self._parse_use_contexts(row)
        
        # FHIR Resource handling - ENHANCED
        fhir_resource_raw = self._extract_field(row, ['fhir_resource', 'fhir', 'resource', 'structure_definition'])
        profile['fhir_resource'] = self._parse_keywords(fhir_resource_raw)
        
        # NEW: Extract FHIR resource fields for search
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
            for item in data[:5]:  # Limit to avoid too much processing
                self._collect_fhir_field_names(item, field_names, max_depth, current_depth + 1)
    
    def _extract_field(self, row: Dict, possible_keys: List[str]) -> Optional[str]:
        """Extract field value trying multiple possible column names"""
        for key in possible_keys:
            if key in row and row[key] is not None:
                value = str(row[key]).strip()
                return value if value and value.lower() != 'nan' else None
        return None
    
    def _parse_keywords(self, keywords_raw: Any) -> List[str]:
        """Parse keywords from various formats"""
        if not keywords_raw:
            return []
        
        keywords_str = str(keywords_raw).strip()
        if not keywords_str or keywords_str.lower() == 'nan':
            return []
        
        # Try JSON format first
        try:
            if keywords_str.startswith('['):
                return json.loads(keywords_str)
        except:
            pass
        
        # Split by common delimiters
        for delimiter in [',', ';', '|', '\n']:
            if delimiter in keywords_str:
                keywords = [kw.strip() for kw in keywords_str.split(delimiter)]
                return [kw for kw in keywords if kw]
        
        # Single keyword
        return [keywords_str] if keywords_str else []
    
    def _parse_use_contexts(self, row: Dict) -> List[Dict]:
        """Parse use contexts if present"""
        # Look for use context fields
        scenarios = self._extract_field(row, ['scenarios', 'use_cases', 'use_contexts'])
        if not scenarios:
            return []
        
        try:
            # Try JSON format
            if isinstance(scenarios, str) and scenarios.startswith('['):
                return json.loads(scenarios)
        except:
            pass
        
        # Simple format - create basic use context
        return [{
            "scenario": str(scenarios),
            "keywords": self._parse_keywords(scenarios)
        }]
    
    def _extract_fhir_fields(self, fhir_resource_data: Any) -> str:
        """Extract field names/keys from FHIR resource structure for searchable text"""
        if not fhir_resource_data:
            return ""
        
        field_names = set()  # Use set to avoid duplicates
        
        try:
            # If it's a string, try to parse as JSON
            if isinstance(fhir_resource_data, str):
                fhir_data = json.loads(fhir_resource_data)
            elif isinstance(fhir_resource_data, dict):
                fhir_data = fhir_resource_data
            elif isinstance(fhir_resource_data, list):
                # Handle list of FHIR resources
                all_field_names = set()
                for resource in fhir_resource_data:
                    resource_fields = self._extract_fhir_fields(resource)
                    if resource_fields:
                        all_field_names.update(resource_fields.split())
                return " ".join(sorted(all_field_names))
            else:
                return ""
            
            # Extract field names recursively
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
                # Add all field names to make them searchable
                field_names.add(key)
                
                # Continue recursively for nested structures
                if isinstance(value, (dict, list)):
                    self._extract_fhir_keys_recursive(value, field_names, max_depth, current_depth + 1)
        
        elif isinstance(data, list):
            for item in data[:10]:  # Limit to first 10 items to avoid too much processing
                if isinstance(item, (dict, list)):
                    self._extract_fhir_keys_recursive(item, field_names, max_depth, current_depth + 1)
    
    def _validate_profiles(self, profiles_data: List[Dict]) -> List[Dict]:
        """Validate profile data"""
        validated = []
        
        for i, profile in enumerate(profiles_data):
            try:
                # Required fields
                if not profile.get('id'):
                    raise ValueError(f"Missing ID for profile {i}")
                if not profile.get('name'):
                    raise ValueError(f"Missing name for profile {i}")
                
                # Ensure keywords is a list
                if not isinstance(profile.get('keywords', []), list):
                    profile['keywords'] = []
                
                # Ensure use_contexts is a list
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
        
        # Session reset for transaction issues
        try:
            db.rollback()
        except:
            pass
        
        for profile_data in profiles_data:
            try:
                # Base search text from profile metadata
                base_search_text = f"{profile_data['name']} {profile_data['description']} {' '.join(profile_data['keywords'])}"
                
                # NEW: Add FHIR resource fields to searchable text
                fhir_search_text = profile_data.get('fhir_searchable_text', '')
                if fhir_search_text:
                    base_search_text += f" {fhir_search_text}"
                
                # Generate embedding from complete search text (now includes FHIR elements)
                embedding = self.model.encode([base_search_text])[0].tolist()
                
                # Create profile record
                profile = Profile(
                    id=profile_data['id'],
                    name=profile_data['name'],
                    description=profile_data['description'],
                    keywords=profile_data['keywords'],
                    category=profile_data['category'],
                    resource_type=profile_data['resource_type'],
                    use_contexts=profile_data['use_contexts'],
                    fhir_resource=profile_data.get('fhir_resource'),
                    dataset_id=dataset_id,
                    search_text=base_search_text,  # Now includes FHIR fields
                    embedding_vector=embedding
                )
                
                # Handle duplicates - update if exists
                existing = db.query(Profile).filter(Profile.id == profile_data['id']).first()
                if existing:
                    # Update existing profile
                    for key, value in profile_data.items():
                        if hasattr(existing, key):
                            setattr(existing, key, value)
                    existing.search_text = base_search_text
                    existing.embedding_vector = embedding
                    existing.dataset_id = dataset_id
                else:
                    # Add new profile
                    db.add(profile)
                
                loaded_count += 1
                
            except Exception as e:
                print(f"Error loading profile {profile_data.get('id', 'unknown')}: {e}")
                # Rollback on individual errors
                db.rollback()
                continue
        
        # Commit with error handling
        try:
            db.commit()
        except Exception as e:
            print(f"Commit failed: {e}")
            db.rollback()
            raise
            
        return loaded_count
    
    def activate_dataset(self, dataset_id: str, db: Session) -> bool:
        """Activate a dataset (make its profiles searchable)"""
        try:
            # Activate profiles from this dataset
            dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
            if not dataset:
                raise ValueError(f"Dataset {dataset_id} not found")
            
            # Update profiles
            profiles_updated = db.query(Profile).filter(
                Profile.dataset_id == dataset_id
            ).update({"is_active": True})
            
            # Update dataset status
            dataset.status = "active"
            dataset.activated_date = datetime.utcnow()
            
            db.commit()
            
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
            
            # deactivate profiles
            profiles_updated = db.query(Profile).filter(
                Profile.dataset_id == dataset_id
            ).update({"is_active": False})
            
            # Update dataset status
            dataset.status = "inactive"
            dataset.deactivated_date = datetime.utcnow()
            
            db.commit()
            
            print(f"Deactivated {profiles_updated} profiles from dataset {dataset.name}")
            return True
            
        except Exception as e:
            print(f"Error deactivating dataset {dataset_id}: {e}")
            db.rollback()
            return False