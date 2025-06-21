# backend/services/etl_service.py
import pandas as pd
import json
import uuid
from typing import List, Dict, Any, Optional
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
        """Transform a single row into profile format"""
        # Handle different column name formats
        profile = {}
        
        # ID field (required)
        profile['id'] = self._extract_field(row, ['id', 'profile_id', 'ID', 'Profile ID', 'Code'])
        if not profile['id']:
            profile['id'] = f"generated_{uuid.uuid4().hex[:8]}_{row_index}"
        
        # Name field (required)
        profile['name'] = self._extract_field(row, ['name', 'Name', 'title', 'Title', 'Profile Name', 'profile_name'])
        if not profile['name']:
            raise ValueError(f"Missing required 'name' field in row {row_index}")
        
        # Description field
        profile['description'] = self._extract_field(row, [
            'description', 'Description', 'desc', 'summary', 'Summary', 'Definition'
        ]) or ""
        
        # Keywords field
        keywords_raw = self._extract_field(row, [
            'keywords', 'Keywords', 'tags', 'Tags', 'terms', 'Terms'
        ])
        profile['keywords'] = self._parse_keywords(keywords_raw)
        
        # Category
        profile['category'] = self._extract_field(row, [
            'category', 'Category', 'type', 'Type', 'Class', 'class'
        ]) or "general"

        # version (optional, default to 1.0.0)
        profile['version'] = self._extract_field(row, [
            'version', 'Version', 'profile_version', 'Profile Version'
        ]) or "unknown version"
        
        # Resource type
        profile['resource_type'] = self._extract_field(row, [
            'resource_type', 'Resource Type', 'resourceType', 'resource', 'Resource'
        ]) or "Unknown"
        
        # Use contexts (optional advanced field)
        profile['use_contexts'] = self._parse_use_contexts(row)
        
        # FHIR Resource handling_-----------------------------------------------------------------------------------------------
        fhir_resource_raw = self._extract_field(row, [
            'fhir_resource', 'FHIR_Resource', 'fhir_data', 'resource_data', 'json_data'
        ])
        
        if fhir_resource_raw:
            try:
                # Try to parse as JSON to validate
                if fhir_resource_raw.startswith('{'):
                    fhir_data = json.loads(fhir_resource_raw)
                    profile['fhir_resource'] = fhir_data
                else:
                    # Might be base64 encoded
                    try:
                        import base64
                        decoded = base64.b64decode(fhir_resource_raw).decode('utf-8')
                        fhir_data = json.loads(decoded)
                        profile['fhir_resource'] = fhir_data
                    except:
                        # Store as string if can't decode
                        profile['fhir_resource'] = fhir_resource_raw
            except json.JSONDecodeError:
                print(f"Warning: Invalid JSON in FHIR resource for row {row_index}")
                profile['fhir_resource'] = None
        else:
            profile['fhir_resource'] = None
        
        return profile
    
    def _extract_fhir_data_elements(self, fhir_resource: Dict) -> str:
        """Universal FHIR data element extractor for any resource type"""
        if not fhir_resource or not isinstance(fhir_resource, dict):
            return ""
            
        data_elements = []
        element_paths = []
        semantic_tags = []
        
        def extract_json_structure(obj, path="", depth=0, max_depth=10):
            """Recursively extract structure from any JSON object"""
            if depth > max_depth:  # Prevent infinite recursion
                return
                
            if isinstance(obj, dict):
                for key, value in obj.items():
                    current_path = f"{path}.{key}" if path else key
                    
                    # Add the raw element path
                    element_paths.append(current_path)
                    
                    # Add semantic meaning based on key patterns (FHIR-agnostic)
                    semantic_tags.extend(self._get_semantic_tags_for_key(key, value, current_path))
                    
                    # Add data type information
                    data_elements.extend(self._get_data_type_tags(key, value, current_path))
                    
                    # Recurse into nested structures
                    if isinstance(value, (dict, list)):
                        extract_json_structure(value, current_path, depth + 1)
                        
            elif isinstance(obj, list) and obj:
                # Handle arrays
                data_elements.append(f"{path}_array")
                if len(obj) > 0:
                    data_elements.append(f"{path}_multiple_values")
                    # Analyze first item to understand array structure
                    extract_json_structure(obj[0], path, depth + 1)
        
        # Extract resource type and add base tags
        resource_type = fhir_resource.get('resourceType', fhir_resource.get('type', 'Unknown'))
        data_elements.extend([
            f"{resource_type}_resource",
            f"{resource_type.lower()}_data",
            "fhir_resource",
            "structured_data"
        ])
        
        # Extract all structural elements
        extract_json_structure(fhir_resource)
        
        # Add resource-specific semantic context
        if resource_type:
            scoped_paths = [f"{resource_type}.{path}" for path in element_paths[:50]]  # Limit to prevent explosion
            element_paths.extend(scoped_paths)
        
        # Combine all extracted elements
        all_elements = data_elements + element_paths + semantic_tags
        
        # Remove duplicates and clean
        unique_elements = list(set(all_elements))
        clean_elements = [elem.strip() for elem in unique_elements if elem and elem.strip() and len(elem) < 200]
        
        return ' '.join(clean_elements[:500])  # Limit total elements to prevent search text explosion
    
    def _get_semantic_tags_for_key(self, key: str, value: Any, path: str) -> List[str]:
        """Generate semantic tags based on key names (works for any JSON)"""
        tags = []
        key_lower = key.lower()
        
        # Identity and tracking patterns
        if any(pattern in key_lower for pattern in ['id', 'identifier', 'uuid', 'reference']):
            tags.extend(['identifiable', 'trackable', 'referenceable'])
            
        # Name and text patterns  
        if any(pattern in key_lower for pattern in ['name', 'title', 'label', 'display']):
            tags.extend(['named', 'textual', 'displayable'])
            
        # Contact and communication patterns
        if any(pattern in key_lower for pattern in ['phone', 'email', 'telecom', 'contact', 'communication']):
            tags.extend(['contactable', 'communicable', 'reachable'])
            
        # Location and address patterns
        if any(pattern in key_lower for pattern in ['address', 'location', 'city', 'country', 'postal', 'geographic']):
            tags.extend(['locatable', 'geographic', 'addressable'])
            
        # Time and date patterns
        if any(pattern in key_lower for pattern in ['date', 'time', 'period', 'instant', 'when', 'occurred']):
            tags.extend(['temporal', 'dated', 'time_bound'])
            
        # Status and state patterns
        if any(pattern in key_lower for pattern in ['status', 'state', 'active', 'enabled', 'valid']):
            tags.extend(['stateful', 'status_trackable', 'lifecycle_managed'])
            
        # Classification and coding patterns
        if any(pattern in key_lower for pattern in ['code', 'coding', 'category', 'class', 'type', 'system']):
            tags.extend(['coded', 'classified', 'categorized', 'systematic'])
            
        # Measurement and value patterns
        if any(pattern in key_lower for pattern in ['value', 'quantity', 'amount', 'measure', 'result', 'score']):
            tags.extend(['measurable', 'quantifiable', 'valued'])
            
        # Relationship patterns
        if any(pattern in key_lower for pattern in ['subject', 'patient', 'performer', 'author', 'source']):
            tags.extend(['relational', 'linked', 'associated'])
            
        # Clinical patterns
        if any(pattern in key_lower for pattern in ['clinical', 'medical', 'health', 'diagnosis', 'condition']):
            tags.extend(['clinical', 'medical', 'healthcare'])
            
        # Extension and customization patterns
        if any(pattern in key_lower for pattern in ['extension', 'custom', 'additional', 'extra']):
            tags.extend(['extensible', 'customizable', 'flexible'])
            
        # Text and narrative patterns
        if any(pattern in key_lower for pattern in ['text', 'narrative', 'description', 'note', 'comment']):
            tags.extend(['textual', 'narrative', 'descriptive'])
            
        return tags
    
    def _get_data_type_tags(self, key: str, value: Any, path: str) -> List[str]:
        """Generate data type tags based on value types"""
        tags = []
        
        if isinstance(value, str):
            tags.append(f"{path}_string")
            # Check for special string patterns
            if '@' in value:
                tags.append(f"{path}_email_format")
            elif value.isdigit():
                tags.append(f"{path}_numeric_string")
            elif any(date_pattern in value for date_pattern in ['-', '/', 'T', ':']):
                tags.append(f"{path}_date_format")
                
        elif isinstance(value, bool):
            tags.append(f"{path}_boolean")
            
        elif isinstance(value, (int, float)):
            tags.append(f"{path}_numeric")
            
        elif isinstance(value, list):
            tags.append(f"{path}_array")
            if value:
                if all(isinstance(item, dict) for item in value):
                    tags.append(f"{path}_object_array")
                elif all(isinstance(item, str) for item in value):
                    tags.append(f"{path}_string_array")
                    
        elif isinstance(value, dict):
            tags.append(f"{path}_object")
            # Check for common object patterns
            if 'system' in value and 'code' in value:
                tags.append(f"{path}_coding_object")
            elif 'reference' in value:
                tags.append(f"{path}_reference_object")
                
        return tags
    
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
        """Load validated profiles into database"""
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
                
                # Add FHIR schema/data elements if FHIR resource present
                if profile_data.get('fhir_resource'):
                    fhir_schema_text = self._extract_fhir_data_elements(profile_data['fhir_resource'])
                    search_text = f"{base_search_text} {fhir_schema_text}"
                else:
                    search_text = base_search_text
                
                # Generate embedding from complete search text (now includes FHIR elements)
                embedding = self.model.encode([search_text])[0].tolist()
                
                # Create profile record
                profile = Profile(
                    id=profile_data['id'],
                    name=profile_data['name'],
                    description=profile_data['description'],
                    keywords=profile_data['keywords'],
                    category=profile_data['category'],
                    version=profile_data['version'],  
                    resource_type=profile_data['resource_type'],
                    use_contexts=profile_data['use_contexts'],
                    fhir_resource=profile_data.get('fhir_resource'),
                    dataset_id=dataset_id,
                    search_text=search_text,  # Now includes FHIR data elements
                    embedding_vector=embedding
                )
                
                # Handle duplicates - update if exists
                existing = db.query(Profile).filter(Profile.id == profile_data['id']).first()
                if existing:
                    # Update existing profile
                    for key, value in profile_data.items():
                        if hasattr(existing, key):
                            setattr(existing, key, value)
                    existing.search_text = search_text
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