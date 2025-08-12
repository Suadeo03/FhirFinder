# backend/services/v2_fhir_etl_service.py
import pandas as pd
import json
import uuid
from typing import List, Dict, Any, Union, Optional
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from models.database.fhir_v2_model import V2FHIRdataset, V2FHIRdata, V2ProcessingJob
from config.chroma import get_chroma_instance, is_chroma_available
import os
from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import sessionmaker

import re
from datetime import datetime

class V2FHIRETLService:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Use singleton pattern
        try:
            self.chroma_config = get_chroma_instance()
            self.collection = self.chroma_config.get_collection()
            print("✅ V2FHIRETLService connected to Chroma singleton")
        except Exception as e:
            print(f"❌ V2FHIRETLService failed to connect to Chroma: {e}")
            self.chroma_config = None
            self.collection = None

    def process_dataset(self, dataset_id: str, db: Session) -> bool:
        """Process V2 FHIR mapping dataset"""
        try:
            # Get dataset
            dataset = db.query(V2FHIRdataset).filter(V2FHIRdataset.id == dataset_id).first()
            if not dataset:
                raise ValueError(f"Dataset {dataset_id} not found")
            
            dataset.status = "processing"
            db.commit()

            # Load the Excel/CSV file
            raw_data = self._load_file(dataset.file_path)

            # Transform data into V2 FHIR mapping format
            mapping_data = self._transform_data(raw_data, dataset.filename)

            # Validate the mapping data
            validated_data = self._validate_mappings(mapping_data)

            # Load mappings to database
            mapping_count = self._load_mappings(validated_data, dataset_id, db)
            
            # Update dataset status
            dataset.status = "ready"
            dataset.processed_date = datetime.utcnow()
            dataset.record_count = mapping_count
            db.commit()
            
            print(f"Successfully processed {mapping_count} V2-FHIR mappings for dataset {dataset.name}")
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
        """Transform raw data into V2 FHIR mapping format"""
        transformed = []
        
        for i, row in enumerate(raw_data):
            try:
                mapping = self._transform_single_mapping(row, i)
                if mapping:
                    transformed.append(mapping)
            except Exception as e:
                print(f"Error transforming row {i}: {e}")
                continue
        
        return transformed
    
    def _transform_single_mapping(self, row: Dict, row_index: int) -> Optional[Dict]:
        """
        Transform a single row into V2 FHIR mapping format
        Expected columns from your data:
        - id, local_id, resource, sub_detail, fhir_detail, fhir_version
        - hl7v2_field, hl7v2_field_detail, hl7v2_field_version
        """
        mapping = {}
        
        # Generate ID if not present
        mapping['id'] = self._extract_field(row, ['id'])
        if not mapping['id']:
            mapping['id'] = f"V2FHIR_{uuid.uuid4().hex[:8]}_{row_index}"

        # Local ID - try multiple possible column names
        mapping['local_id'] = self._extract_field(row, ['local_id', 'localId', 'local_identifier', 'Local ID'])
        if not mapping['local_id']:
            mapping['local_id'] = f"LOCAL_{row_index}"

        # FHIR Resource (required) - try multiple possible column names
        mapping['resource'] = self._extract_field(row, ['resource', 'fhir_resource', 'Resource', 'FHIR Resource'])
        if not mapping['resource']:
            # If no resource field, try to use a combination of other fields
            resource_fallback = self._extract_field(row, ['Patient', 'Observation', 'Encounter', 'Procedure'])
            if resource_fallback:
                mapping['resource'] = resource_fallback
            else:
                # Use row index as last resort
                mapping['resource'] = f"UnknownResource_{row_index}"
                print(f"Warning: No resource field found in row {row_index}, using fallback")

        # Sub detail
        mapping['sub_detail'] = self._extract_field(row, ['sub_detail', 'subDetail', 'sub_resource', 'Sub Detail']) or ""

        # FHIR details
        mapping['fhir_detail'] = self._extract_field(row, ['fhir_detail', 'fhirDetail', 'fhir_description', 'FHIR Detail']) or ""
        mapping['fhir_version'] = self._extract_field(row, ['fhir_version', 'fhirVersion', 'version', 'FHIR Version']) or "R4"

        # HL7 V2 field information
        mapping['hl7v2_field'] = self._extract_field(row, ['hl7v2_field', 'hl7_field', 'v2_field', 'HL7v2_field', 'HL7 V2 Field']) or ""
        mapping['hl7v2_field_detail'] = self._extract_field(row, ['hl7v2_field_detail', 'hl7_field_detail', 'v2_field_detail', 'HL7v2_field_detail', 'HL7 V2 Field Detail']) or ""
        mapping['hl7v2_field_version'] = self._extract_field(row, ['hl7v2_field_version', 'hl7_version', 'v2_version', 'HL7v2_version', 'HL7 V2 Version']) or "2.5"

        # Debug: Print the first few rows to see what we're getting
        if row_index < 3:
            print(f"Row {row_index} keys: {list(row.keys())}")
            print(f"Row {row_index} mapped values: {mapping}")

        return mapping
    
    def get_dataset_fields(self, raw_data: List[Dict]) -> str:
        """Get comma-separated list of all field names in dataset"""
        all_keys = set()
        for row in raw_data:
            if isinstance(row, dict):
                all_keys.update(row.keys())
        return ", ".join(sorted(all_keys))
    
    def _extract_field(self, row: Dict, possible_keys: List[str]) -> Optional[str]:
        """Extract field value from row using multiple possible key names"""
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
    
    def _validate_mappings(self, mapping_data: List[Dict]) -> List[Dict]:
        """Validate V2 FHIR mapping data"""
        validated = []
        
        for i, mapping in enumerate(mapping_data):
            try:
                if not mapping.get('id'):
                    raise ValueError(f"Missing ID for mapping {i}")
                if not mapping.get('local_id'):
                    raise ValueError(f"Missing local_id for mapping {i}")
                if not mapping.get('resource'):
                    raise ValueError(f"Missing resource for mapping {i}")
                
                validated.append(mapping)
                
            except Exception as e:
                print(f"Validation error for mapping {i}: {e}")
                continue
        
        return validated
    
    def _load_mappings(self, mapping_data: List[Dict], dataset_id: str, db: Session) -> int:
        """Load V2 FHIR mappings to database and Chroma"""
        loaded_count = 0
        search_texts = []
        embeddings = []
        chroma_mappings = []

        try:
            db.rollback()
        except:
            pass
        
        print(f"🔄 Loading {len(mapping_data)} V2-FHIR mappings to database and Chroma...")
        
        for mapping_data_item in mapping_data:
            try:
                # Create searchable text combining key fields
                search_text_parts = [
                    mapping_data_item.get('resource', ''),
                    mapping_data_item.get('fhir_detail', ''),
                    mapping_data_item.get('hl7v2_field', ''),
                    mapping_data_item.get('hl7v2_field_detail', ''),
                    mapping_data_item.get('sub_detail', '')
                ]
                search_text = ' '.join([part for part in search_text_parts if part])

                # Generate embedding
                embedding = self.model.encode([search_text])[0].tolist()
                
                # Create V2FHIRdata object
                v2_mapping = V2FHIRdata(
                    id=mapping_data_item['id'],
                    local_id=mapping_data_item['local_id'],
                    resource=mapping_data_item['resource'],
                    sub_detail=mapping_data_item.get('sub_detail'),
                    fhir_detail=mapping_data_item.get('fhir_detail'),
                    fhir_version=mapping_data_item.get('fhir_version'),
                    hl7v2_field=mapping_data_item.get('hl7v2_field'),
                    hl7v2_field_detail=mapping_data_item.get('hl7v2_field_detail'),
                    hl7v2_field_version=mapping_data_item.get('hl7v2_field_version'),
                    V2FHIRdataset_id=dataset_id,
                    is_active=False,  # Start as inactive
                    created_date=datetime.utcnow()
                )
                
                # Check if mapping already exists
                existing = db.query(V2FHIRdata).filter(V2FHIRdata.id == mapping_data_item['id']).first()
                if existing:
                    # Update existing mapping
                    for key, value in mapping_data_item.items():
                        if hasattr(existing, key):
                            setattr(existing, key, value)
                    existing.V2FHIRdataset_id = dataset_id
                else:
                    db.add(v2_mapping)
                
                # Prepare for Chroma
                chroma_mapping_data = mapping_data_item.copy()
                chroma_mapping_data['dataset_id'] = dataset_id
                chroma_mapping_data['search_text'] = search_text
                
                search_texts.append(search_text)
                embeddings.append(embedding)
                chroma_mappings.append(chroma_mapping_data)
                
                loaded_count += 1
                
            except Exception as e:
                print(f"❌ Error loading mapping {mapping_data_item.get('id', 'unknown')}: {e}")
                db.rollback()
                continue

        try:
            db.commit()
            print(f"✅ Database commit successful: {loaded_count} mappings")
        except Exception as e:
            print(f"❌ Database commit failed: {e}")
            db.rollback()
            raise

        # Add to Chroma if available
        if is_chroma_available() and chroma_mappings:
            print(f"🔄 Adding {len(chroma_mappings)} mappings to Chroma singleton...")
            chroma_success = self._batch_add_to_chroma(chroma_mappings, search_texts, embeddings)
            
            if not chroma_success:
                print(f"❌ Warning: Chroma addition failed, but database commit was successful")
        else:
            print(f"⚠️  Chroma singleton not available, skipping vector database addition")
            
        return loaded_count
    
    def activate_dataset(self, dataset_id: str, db: Session) -> bool:
        """Activate V2 FHIR dataset and all its mappings"""
        try:
            dataset = db.query(V2FHIRdataset).filter(V2FHIRdataset.id == dataset_id).first()
            if not dataset:
                raise ValueError(f"Dataset {dataset_id} not found")
            
            # Activate all mappings in this dataset
            mappings_updated = db.query(V2FHIRdata).filter(
                V2FHIRdata.V2FHIRdataset_id == dataset_id
            ).update({"is_active": True})
            
            dataset.status = "active"
            dataset.activated_date = datetime.utcnow()
            db.commit()
            
            # Update Chroma metadata if available
            if self.collection:
                mapping_ids = [m.id for m in db.query(V2FHIRdata.id).filter(V2FHIRdata.V2FHIRdataset_id == dataset_id).all()]
                for mapping_id in mapping_ids:
                    self._update_chroma_metadata(mapping_id, True)

            print(f"Activated {mappings_updated} V2-FHIR mappings from dataset {dataset.name}")
            return True
            
        except Exception as e:
            print(f"Error activating dataset {dataset_id}: {e}")
            db.rollback()
            return False
        
    def deactivate_dataset(self, dataset_id: str, db: Session) -> bool:
        """Deactivate V2 FHIR dataset and all its mappings"""
        try:
            dataset = db.query(V2FHIRdataset).filter(V2FHIRdataset.id == dataset_id).first()
            if not dataset:
                raise ValueError(f"Dataset {dataset_id} not found")
                
            # Deactivate all mappings in this dataset
            mappings_updated = db.query(V2FHIRdata).filter(
                V2FHIRdata.V2FHIRdataset_id == dataset_id
            ).update({"is_active": False})
            
            dataset.status = "inactive"
            dataset.deactivated_date = datetime.utcnow()
            db.commit()
            
            # Update Chroma metadata if available
            if self.collection:
                mapping_ids = [m.id for m in db.query(V2FHIRdata.id).filter(V2FHIRdata.V2FHIRdataset_id == dataset_id).all()]
                for mapping_id in mapping_ids:
                    self._update_chroma_metadata(mapping_id, False)

            # Clear cache if available
            try:
                from config.redis_cache import RedisQueryCache
                redis_client = RedisQueryCache()
                redis_client.clear_all_cache()
                print(f"✅ Cleared all cache after deactivating dataset")
            except Exception as e:
                print(f"⚠️  Warning: Could not clear cache: {e}")

            print(f"Deactivated {mappings_updated} V2-FHIR mappings from dataset {dataset.name}")
            return True
            
        except Exception as e:
            print(f"Error deactivating dataset {dataset_id}: {e}")
            db.rollback()
            return False
  
    def _add_to_chroma(self, mapping_data: Dict, search_text: str, embedding: List[float]) -> bool:
        """Add a single V2 FHIR mapping to Chroma"""
        if not self.collection:
            return False
        
        try:
            self.collection.upsert(
                ids=[mapping_data['id']],
                embeddings=[embedding],
                documents=[search_text],
                metadatas=[{
                    'local_id': mapping_data.get('local_id', ''),
                    'resource': mapping_data.get('resource', ''),
                    'sub_detail': (mapping_data.get('sub_detail') or '')[:500],
                    'fhir_detail': (mapping_data.get('fhir_detail') or '')[:1000],
                    'fhir_version': mapping_data.get('fhir_version', 'R4'),
                    'hl7v2_field': mapping_data.get('hl7v2_field', ''),
                    'hl7v2_field_detail': (mapping_data.get('hl7v2_field_detail') or '')[:1000],
                    'hl7v2_field_version': mapping_data.get('hl7v2_field_version', '2.5'),
                    'dataset_id': mapping_data.get('dataset_id', ''),
                    'is_active': True
                }]
            )
            return True
        except Exception as e:
            print(f"Error adding mapping to Chroma: {e}")
            return False

    def _batch_add_to_chroma(self, mappings_data: List[Dict], search_texts: List[str], embeddings: List[List[float]]) -> bool:
        """Add V2 FHIR mappings to Chroma in batch"""
        if not self.collection or not mappings_data:
            print(f"❌ Cannot add to Chroma: collection={bool(self.collection)}, mappings={len(mappings_data) if mappings_data else 0}")
            return False
    
        try:
            print(f"🔄 Starting batch add to Chroma: {len(mappings_data)} mappings")
            
            # Validate inputs
            if len(mappings_data) != len(search_texts) or len(mappings_data) != len(embeddings):
                print(f"❌ Input length mismatch: mappings={len(mappings_data)}, texts={len(search_texts)}, embeddings={len(embeddings)}")
                return False
            
            ids = [m['id'] for m in mappings_data]
            metadatas = []
            
            print(f"   Mapping IDs to add: {ids[:5]}{'...' if len(ids) > 5 else ''}")
            
            # Prepare metadata for each mapping
            for i, mapping_data in enumerate(mappings_data):
                try:
                    metadata = {
                        'local_id': mapping_data.get('local_id', ''),
                        'resource': mapping_data.get('resource', ''),
                        'sub_detail': (mapping_data.get('sub_detail') or '')[:500],
                        'fhir_detail': (mapping_data.get('fhir_detail') or '')[:1000],
                        'fhir_version': mapping_data.get('fhir_version', 'R4'),
                        'hl7v2_field': mapping_data.get('hl7v2_field', ''),
                        'hl7v2_field_detail': (mapping_data.get('hl7v2_field_detail') or '')[:1000],
                        'hl7v2_field_version': mapping_data.get('hl7v2_field_version', '2.5'),
                        'dataset_id': mapping_data.get('dataset_id', ''),
                        'is_active': True
                    }
                    metadatas.append(metadata)
                except Exception as e:
                    print(f"❌ Error preparing metadata for mapping {i}: {e}")
                    return False
            
            # Validate embeddings
            for i, embedding in enumerate(embeddings):
                if not isinstance(embedding, list) or len(embedding) == 0:
                    print(f"❌ Invalid embedding for mapping {ids[i]}: {type(embedding)}, length={len(embedding) if isinstance(embedding, list) else 'N/A'}")
                    return False
                if any(not isinstance(x, (int, float)) for x in embedding):
                    print(f"❌ Non-numeric values in embedding for mapping {ids[i]}")
                    return False
            
            print(f"✅ All inputs validated, performing upsert...")
            
            # Perform the upsert
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=search_texts,
                metadatas=metadatas
            )
            
            print(f"✅ Upsert completed, verifying...")
            
            # Verify the addition
            verification_results = self.collection.get(ids=ids)
            found_ids = verification_results.get('ids', [])
            
            if len(found_ids) == len(ids):
                print(f"✅ Successfully added {len(mappings_data)} mappings to Chroma (verified)")
                return True
            else:
                print(f"❌ Verification failed: expected {len(ids)}, found {len(found_ids)}")
                missing_ids = set(ids) - set(found_ids)
                print(f"   Missing IDs: {list(missing_ids)[:5]}")
                return False
                
        except Exception as e:
            print(f"❌ Error in batch add to Chroma: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _update_chroma_metadata(self, mapping_id: str, is_active: bool) -> bool:
        """Update metadata for a mapping in Chroma"""
        if not self.collection:
            return False
        
        try:
            # Get current document
            results = self.collection.get(ids=[mapping_id], include=['metadatas'])
            if results['ids']:
                metadata = results['metadatas'][0]
                metadata['is_active'] = is_active
                
                self.collection.update(
                    ids=[mapping_id],
                    metadatas=[metadata]
                )
                return True
            return False
            
        except Exception as e:
            print(f"Error updating Chroma metadata: {e}")
            return False
    
    def _delete_from_chroma(self, mapping_ids: List[str]) -> bool:
        """Delete mappings from Chroma"""
        if not self.collection or not mapping_ids:
            return False
        
        try:
            self.collection.delete(ids=mapping_ids)
            print(f"Deleted {len(mapping_ids)} mappings from Chroma")
            return True
        except Exception as e:
            print(f"Error deleting from Chroma: {e}")
            return False