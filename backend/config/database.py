# backend/config/database.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models.database.models import Base
import os

# Database URL - start with SQLite for simplicity, can upgrade to PostgreSQL later
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./fhir_profiles.db")

# Create engine
engine = create_engine(
    DATABASE_URL,
    echo=True if os.getenv("DEBUG") else False,  # Log SQL queries in debug mode
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def create_tables():
    """Create all database tables"""
    Base.metadata.create_all(bind=engine)
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
    """Initialize database with tables"""
    try:
        create_tables()
        print("Database initialized successfully")
    except Exception as e:
        print(f"Error initializing database: {e}")

if __name__ == "__main__":
    init_database()