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
        
        # Resource type
        profile['resource_type'] = self._extract_field(row, [
            'resource_type', 'Resource Type', 'resourceType', 'resource', 'Resource'
        ]) or "Unknown"
        
        # Use contexts (optional advanced field)
        profile['use_contexts'] = self._parse_use_contexts(row)
        
        return profile
    
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
        
        for profile_data in profiles_data:
            try:
                # Create search text for full-text search
                search_text = f"{profile_data['name']} {profile_data['description']} {' '.join(profile_data['keywords'])}"
                
                # Generate embedding
                embedding = self.model.encode([search_text])[0].tolist()
                
                # Create profile record
                profile = Profile(
                    id=profile_data['id'],
                    name=profile_data['name'],
                    description=profile_data['description'],
                    keywords=profile_data['keywords'],
                    category=profile_data['category'],
                    resource_type=profile_data['resource_type'],
                    use_contexts=profile_data['use_contexts'],
                    dataset_id=dataset_id,
                    search_text=search_text,
                    embedding_vector=embedding
                )
                
                # Handle duplicates - update if exists
                existing = db.query(Profile).filter(Profile.id == profile_data['id']).first()
                if existing:
                    # Update existing profile
                    for key, value in profile_data.items():
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
                continue
        
        db.commit()
        return loaded_count
    
    def activate_dataset(self, dataset_id: str, db: Session) -> bool:
        """Activate a dataset (make its profiles searchable)"""
        try:
            # Deactivate all profiles first
            db.query(Profile).update({"is_active": False})
            
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