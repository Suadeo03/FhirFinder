from sqlalchemy import Column, String, Integer, DateTime, Boolean, Text, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, configure_mappers
from sqlalchemy.dialects.postgresql import JSON
from datetime import datetime
import uuid

Base = declarative_base()


class Formset(Base):
    """Track uploaded datasets and their processing status"""
    __tablename__ = "formsets"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)  # Name of Organization OID
    filename = Column(String(255), nullable=False)
    description = Column(Text)  # Description of the dataset, version, etc.
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
    forms = relationship("Form", back_populates="formset", cascade="all, delete-orphan")
    processing_jobs = relationship("FormProcessingJob", back_populates="formset", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Formset(id='{self.id}', name='{self.name}', status='{self.status}')>"


class Form(Base):
    """Individual forms within a formset"""
    __tablename__ = "forms"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    domain = Column(String(255), nullable=False)
    screening_tool = Column(Text, nullable=False)  # Changed from String(255) to Text - some are very long
    loinc_panel_code = Column(String(100))  # Increased from 50 to 100
    loinc_panel_name = Column(Text)  # Changed from String(255) to Text - some are very long
    question = Column(Text)  # Changed from String(255) to Text - some questions are long
    loinc_question_code = Column(String(100))  # Increased from 50 to 100
    loinc_question_name_long = Column(Text)  # Already Text, good
    answer_concept = Column(Text)  # Already Text, good
    loinc_answer = Column(Text)  # Changed from String(50) to Text - contains long lists
    loinc_concept = Column(Text)  # Already Text, good
    snomed_code_ct = Column(Text)  # Changed from String(50) to Text - can contain multiple codes
    formset_id = Column(String, ForeignKey("formsets.id"), nullable=False)
    is_active = Column(Boolean, default=False)
    created_date = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    formset = relationship("Formset", back_populates="forms")
    
    def __repr__(self):
        return f"<Form(id='{self.id}', domain='{self.domain}', screening_tool='{self.screening_tool[:50]}...')>"


class FormProcessingJob(Base):
    """Track background processing jobs for formsets"""
    __tablename__ = "form_processing_jobs"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    formset_id = Column(String, ForeignKey("formsets.id"), nullable=False)
    job_type = Column(String(50), nullable=False)  # upload, process, activate
    status = Column(String(50), nullable=False, default="pending")  # pending, running, completed, failed
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    progress_percent = Column(Integer, default=0)
    error_message = Column(Text)
    
    # Job-specific data
    job_data = Column(JSON)  # Store any job-specific parameters
    
    # Relationship
    formset = relationship("Formset", back_populates="processing_jobs")
    
    def __repr__(self):
        return f"<FormProcessingJob(id='{self.id}', type='{self.job_type}', status='{self.status}')>"


# Configure mappers after all models are defined
configure_mappers()  