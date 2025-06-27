# backend/models/database/models.py
from sqlalchemy import Column, String, Integer, DateTime, Boolean, Text, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSON
from datetime import datetime
import uuid

Base = declarative_base()

class Dataset(Base):
    """Track uploaded datasets and their processing status"""
    __tablename__ = "datasets"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False) #Name of Organization OID
    filename = Column(String(255), nullable=False)
    description = Column(Text) #Description of the dataset, verion, etc.
    status = Column(String(50), nullable=False, default="uploaded")  # uploaded, processing, ready, active, inactive, failed
    upload_date = Column(DateTime, default=datetime.utcnow)
    processed_date = Column(DateTime)
    activated_date = Column(DateTime)
    deactivated_date = Column(DateTime)
    record_count = Column(Integer, default=0)
    error_message = Column(Text)
    download_count = Column(Integer, default=0) #number of times dataset has been downloaded
    last_downloaded = Column(DateTime)
    file_size = Column(Integer)  # in bytes
    file_path = Column(String(500))  
    
    # Relationship to profiles
    profiles = relationship("Profile", back_populates="dataset", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Dataset(id='{self.id}', name='{self.name}', status='{self.status}')>"

class Profile(Base):
    """FHIR profiles from processed datasets"""
    __tablename__ = "profiles"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4())) 
    name = Column(String(500), nullable=False)
    description = Column(Text)
    keywords = Column(JSON)  # ["keyword1", "keyword2", ...]
    category = Column(String(100))
    version = Column(String(50))  # e.g. "1.0.0"
    resource_type = Column(String(100))
    use_contexts = Column(JSON)  # [{"scenario": "...", "keywords": [...]}]
    fhir_resource = Column(JSON)
    # Metadata
    dataset_id = Column(String, ForeignKey("datasets.id"), nullable=False)
    is_active = Column(Boolean, default=False)
    created_date = Column(DateTime, default=datetime.utcnow)
    
    # Search optimization
    embedding_vector = Column(JSON)  # Store precomputed embeddings
    search_text = Column(Text)  # Combined text for full-text search
    
    # Relationship to dataset
    dataset = relationship("Dataset", back_populates="profiles")
    
    def __repr__(self):
        return f"<Profile(id='{self.id}', name='{self.name[:50]}...', active={self.is_active})>"

class ProcessingJob(Base):
    """Track background processing jobs"""
    __tablename__ = "processing_jobs"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    dataset_id = Column(String, ForeignKey("datasets.id"), nullable=False)
    job_type = Column(String(50), nullable=False)  # upload, process, activate
    status = Column(String(50), nullable=False, default="pending")  # pending, running, completed, failed
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    progress_percent = Column(Integer, default=0)
    error_message = Column(Text)
    
    # Job-specific data
    job_data = Column(JSON)  # Store any job-specific parameters
    
    def __repr__(self):
        return f"<ProcessingJob(id='{self.id}', type='{self.job_type}', status='{self.status}')>"