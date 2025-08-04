# backend/config/database.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models.database.models import Base
from models.database.form_model import Base as FormBase
from models.database.feedback_models import Base as FeedbackBase
import os
from sqlalchemy import text

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://fhir_user:admin@localhost:5432/fhir_registry")

# Create engine
engine = create_engine(
    DATABASE_URL,
    echo=True if os.getenv("DEBUG") else False,  
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def create_tables():


    Base.metadata.create_all(bind=engine)
    FeedbackBase.metadata.create_all(bind=engine)
    FormBase.metadata.create_all(bind=engine)
    print("Database tables created successfully")

def get_db():
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Initialize database on import
def init_database():

    try:

        create_tables()
      
        print("Database initialized successfully")

    except Exception as e:
        print(f"Error initializing database: {e}")


def clear_postgres_data():
    """Clear all data from PostgreSQL tables"""
    

    
    try:
        with engine.connect() as connection:
            # Delete all feedback data
            connection.execute(text("DELETE FROM profiles CASCADE"))
            connection.execute(text("DELETE FROM datasets CASCADE"))
            connection.execute(text("DELETE FROM processing_jobs CASCADE"))
            connection.execute(text("DELETE FROM user_feedback CASCADE"))
            connection.execute(text("DELETE FROM search_quality_metrics CASCADE"))
            connection.execute(text("DELETE FROM formsets CASCADE"))
            connection.execute(text("DELETE FROM forms CASCADE"))
            connection.execute(text("DELETE FROM form_processing_jobs CASCADE"))


            connection.commit()
            print(f"✅ Deleted records from user_feedback")
            
            # Reset sequence
            connection.execute(text("ALTER SEQUENCE user_feedback_id_seq RESTART WITH 1"))
            connection.commit()
            print("✅ Reset user_feedback ID sequence")

            
    except Exception as e:
        print(f"❌ Error clearing PostgreSQL data: {e}")


