from sqlalchemy import Column, String, Integer, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, configure_mappers
from sqlalchemy.dialects.postgresql import JSON
from datetime import datetime
import uuid

Base = declarative_base()


class V2FHIRdataset(Base):
    """Track uploaded datasets and their processing status"""
    __tablename__ = "V2fhirdatasets"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)  
    filename = Column(String(255), nullable=False)
    description = Column(Text)  
    status = Column(String(50), nullable=False, default="uploaded")
    upload_date = Column(DateTime, default=datetime.utcnow)
    processed_date = Column(DateTime)
    activated_date = Column(DateTime)
    deactivated_date = Column(DateTime)
    record_count = Column(Integer, default=0)
    error_message = Column(Text)
    download_count = Column(Integer, default=0)
    last_downloaded = Column(DateTime)
    file_size = Column(Integer)
    file_path = Column(String(500))
    
    # Relationships
    data_records = relationship("V2FHIRdata", back_populates="dataset", cascade="all, delete-orphan")
    processing_jobs = relationship("V2ProcessingJob", back_populates="dataset", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<V2FHIRdataset(id='{self.id}', name='{self.name}', status='{self.status}')>"


class V2FHIRdata(Base):
    """Individual V2-FHIR mapping records within a dataset"""
    __tablename__ = "V2fhirdata"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    local_id = Column(String(255), nullable=False)
    resource = Column(Text, nullable=False)  
    sub_detail = Column(String(100))  
    fhir_detail = Column(Text)  
    fhir_version = Column(String(100))  
    hl7v2_field = Column(String(100))  
    hl7v2_field_detail = Column(Text)
    hl7v2_field_version = Column(String(100))  
    V2FHIRdataset_id = Column(String, ForeignKey("V2fhirdatasets.id"), nullable=False)
    is_active = Column(Boolean, default=False)
    created_date = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    dataset = relationship("V2FHIRdataset", back_populates="data_records")
    
    def __repr__(self):
        return f"<V2FHIRdata(id='{self.id}', local_id='{self.local_id}', resource='{self.resource[:50]}...')>"


class V2ProcessingJob(Base):
    """Track background processing jobs for datasets"""
    __tablename__ = "V2fhir_processing_jobs"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    V2FHIRdataset_id = Column(String, ForeignKey("V2fhirdatasets.id"), nullable=False)
    job_type = Column(String(50), nullable=False)  # upload, process, activate
    status = Column(String(50), nullable=False, default="pending")  # pending, running, completed, failed
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    progress_percent = Column(Integer, default=0)
    error_message = Column(Text)
    job_data = Column(JSON)  # Store any job-specific parameters
    
    # Relationship
    dataset = relationship("V2FHIRdataset", back_populates="processing_jobs")
    
    def __repr__(self):
        return f"<V2ProcessingJob(id='{self.id}', type='{self.job_type}', status='{self.status}')>"


# Configure mappers after all models are defined
configure_mappers()