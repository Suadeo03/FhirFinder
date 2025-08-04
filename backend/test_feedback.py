#!/usr/bin/env python3
"""
Separate test file for debugging feedback issues
Run this file directly: python test_feedback_debug.py
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager
from sqlalchemy import create_engine # Adjust import as needed
from services.etl_service import ETLService
from services.training_service import FeedbackTraining
from models.database.models import Dataset, Profile

# Database session setup

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://fhir_user:admin@localhost:5432/fhir_registry")

# Create engine
engine = create_engine(
    DATABASE_URL,
    echo=True if os.getenv("DEBUG") else False,  
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
@contextmanager
def get_db_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def test_actual_feedback_process(dataset_id: str, db):
    """Fixed test function that handles NumPy arrays properly"""
    
    print("\n" + "="*60)
    print("🔍 TESTING ACTUAL FEEDBACK PROCESS")
    print("="*60)
    
    feedback_trainer = FeedbackTraining()
    
    # Get a real profile from your database
    test_profile = db.query(Profile).filter(
        Profile.dataset_id == dataset_id,
        Profile.is_active == True
    ).first()
    
    if not test_profile:
        print("❌ No active profiles found for testing")
        return
    
    print(f"🧪 Testing with profile: {test_profile.id} ({test_profile.name})")
    
    # Test 1: Check if profile exists in Chroma
    print(f"\n1. Checking profile in feedback trainer's Chroma connection...")
    
    try:
        current_results = feedback_trainer.collection.get(
            ids=[test_profile.id],
            include=['embeddings', 'metadatas']
        )
        
        print(f"   Results type: {type(current_results)}")
        print(f"   Keys: {list(current_results.keys()) if current_results else 'None'}")
        
        if current_results:
            ids_found = current_results.get('ids', [])
            embeddings = current_results.get('embeddings', [])
            metadatas = current_results.get('metadatas', [])
            
            print(f"   IDs returned: {ids_found}")
            
            # FIXED: Properly check embeddings without triggering NumPy error
            has_embeddings = embeddings is not None and len(embeddings) > 0
            print(f"   Has embeddings: {has_embeddings}")
            print(f"   Embeddings count: {len(embeddings) if embeddings else 0}")
            
            has_metadatas = metadatas is not None and len(metadatas) > 0
            print(f"   Has metadatas: {has_metadatas}")
            
            if has_embeddings and embeddings[0] is not None:
                print(f"   First embedding length: {len(embeddings[0])}")
                print(f"   First embedding type: {type(embeddings[0])}")
            else:
                print(f"   ❌ No valid embeddings")
        else:
            print(f"   ❌ No results returned from Chroma")
            return
    
    except Exception as e:
        print(f"   ❌ Error getting profile from Chroma: {e}")
        return
    
    # Test 2: Check the validation function
    print(f"\n2. Testing embedding validation function...")
    is_valid = feedback_trainer._is_valid_embedding_result(current_results)
    print(f"   Validation result: {is_valid}")
    
    # Test 3: Try actual feedback recording
    print(f"\n3. Testing actual feedback recording...")
    
    try:
        test_query = "patient uscore"
        feedback_trainer.record_user_feedback(
            query=test_query,
            profile_id=test_profile.id,
            feedback_type="positive",
            user_id="test_user",
            session_id="test_session",
            original_score=0.8,
            db=db
        )
        print(f"   ✅ Feedback recording successful!")
        
    except Exception as e:
        print(f"   ❌ Feedback recording failed: {e}")
        import traceback
        traceback.print_exc()

def test_specific_failing_profile_fixed(profile_id: str, db):
    """Fixed test for specific profile"""
    print(f"\n" + "="*60)
    print(f"🔍 TESTING SPECIFIC FAILING PROFILE: {profile_id}")
    print("="*60)
    
    feedback_trainer = FeedbackTraining()
    
    # Check if it exists in database
    failing_profile = db.query(Profile).filter(Profile.id == profile_id).first()
    if failing_profile:
        print(f"✅ Profile found in DB: {failing_profile.name}")
        print(f"   Dataset: {failing_profile.dataset_id}")
        print(f"   Is Active: {failing_profile.is_active}")
        
        # IMPORTANT: Check if this is the issue
        if not failing_profile.is_active:
            print(f"❌ FOUND THE PROBLEM: Profile is INACTIVE!")
            print(f"   Your search only returns active profiles, but feedback")
            print(f"   is being attempted on an inactive profile.")
            return
        
        # Check if it exists in Chroma
        try:
            results = feedback_trainer.collection.get(
                ids=[profile_id],
                include=['embeddings', 'metadatas']
            )
            
            if results and results.get('ids'):
                print(f"✅ Profile found in Chroma")
                embeddings = results.get('embeddings', [])
                # FIXED: Safe check for embeddings
                if embeddings and len(embeddings) > 0 and embeddings[0] is not None:
                    print(f"   Embedding length: {len(embeddings[0])}")
                else:
                    print(f"❌ No valid embedding in Chroma")
            else:
                print(f"❌ Profile NOT found in Chroma")
                
        except Exception as e:
            print(f"❌ Error checking Chroma: {e}")
    else:
        print(f"❌ Profile {profile_id} not found in database!")

# The REAL FIX for your feedback system
def fix_search_service_feedback_integration():
    """
    The real issue is that search returns active profiles, but feedback 
    can be given on inactive profiles that are still in the results cache.
    
    Here are the fixes needed:
    """
    
    print("\n" + "="*60)
    print("🔧 REAL FIXES NEEDED")
    print("="*60)
    
    print("1. ❌ PROBLEM IDENTIFIED:")
    print("   - Search returns only ACTIVE profiles")
    print("   - But cached results may contain profiles that became INACTIVE")
    print("   - Feedback tries to update inactive profiles")
    print("   - Inactive profiles may not be in current search scope")
    
    print("\n2. ✅ SOLUTIONS:")
    print("   A. Update feedback to handle inactive profiles")
    print("   B. Clear cache when profiles are deactivated")
    print("   C. Add validation in feedback recording")
    
    return {
        "problem": "Feedback attempted on inactive profile",
        "failing_profile_id": "HL7_95bbc7f5_15",
        "profile_status": "inactive",
        "solutions": [
            "Handle inactive profiles in feedback",
            "Clear cache on dataset deactivation", 
            "Add profile status validation"
        ]
    }
def compare_collections(dataset_id: str, db):
    """Compare ETL collection vs Feedback collection"""
    
    print("\n" + "="*60)
    print("🔍 COMPARING ETL vs FEEDBACK COLLECTIONS")
    print("="*60)
    
    etl_service = ETLService()
    feedback_trainer = FeedbackTraining()
    
    # Get a test profile
    test_profile = db.query(Profile).filter(
        Profile.dataset_id == dataset_id,
        Profile.is_active == True
    ).first()
    
    if not test_profile:
        print("❌ No test profile found")
        return
    
    print(f"Testing with profile: {test_profile.id}")
    
    # Test ETL collection
    print(f"\n1. Testing ETL Service collection...")
    try:
        etl_results = etl_service.collection.get(
            ids=[test_profile.id],
            include=['embeddings', 'metadatas']
        )
        print(f"   ETL Collection - IDs found: {etl_results.get('ids', [])}")
        print(f"   ETL Collection - Has embeddings: {bool(etl_results.get('embeddings'))}")
    except Exception as e:
        print(f"   ❌ ETL Collection error: {e}")
    
    # Test Feedback collection
    print(f"\n2. Testing Feedback Service collection...")
    try:
        feedback_results = feedback_trainer.collection.get(
            ids=[test_profile.id],
            include=['embeddings', 'metadatas']
        )
        print(f"   Feedback Collection - IDs found: {feedback_results.get('ids', [])}")
        print(f"   Feedback Collection - Has embeddings: {bool(feedback_results.get('embeddings'))}")
    except Exception as e:
        print(f"   ❌ Feedback Collection error: {e}")
    
    # Compare collection objects
    print(f"\n3. Comparing collection objects...")
    print(f"   ETL collection object: {etl_service.collection}")
    print(f"   Feedback collection object: {feedback_trainer.collection}")
    print(f"   Are they the same? {etl_service.collection is feedback_trainer.collection}")

def test_specific_failing_profile(profile_id: str, db):
    """Test a specific profile ID that's failing"""
    print(f"\n" + "="*60)
    print(f"🔍 TESTING SPECIFIC FAILING PROFILE: {profile_id}")
    print("="*60)
    
    feedback_trainer = FeedbackTraining()
    
    # Check if it exists in database
    failing_profile = db.query(Profile).filter(Profile.id == profile_id).first()
    if failing_profile:
        print(f"✅ Profile found in DB: {failing_profile.name}")
        print(f"   Dataset: {failing_profile.dataset_id}")
        print(f"   Is Active: {failing_profile.is_active}")
        
        # Check if it exists in Chroma
        try:
            results = feedback_trainer.collection.get(
                ids=[profile_id],
                include=['embeddings', 'metadatas']
            )
            
            if results and results.get('ids'):
                print(f"✅ Profile found in Chroma")
                embeddings = results.get('embeddings', [])
                if embeddings and embeddings[0]:
                    print(f"   Embedding length: {len(embeddings[0])}")
                else:
                    print(f"❌ No valid embedding in Chroma")
            else:
                print(f"❌ Profile NOT found in Chroma")
                
        except Exception as e:
            print(f"❌ Error checking Chroma: {e}")
    else:
        print(f"❌ Profile {profile_id} not found in database!")

def main():
    """Main test function"""
    print("🔍 STARTING FEEDBACK DEBUGGING...")
    
    with get_db_session() as db:
        # Test with your working dataset
        dataset_id = "4c4b43a0-de4c-442d-b39e-44137c52ce8e"
        
        # Test actual feedback process
        test_actual_feedback_process(dataset_id, db)
        
        # Compare collections
        compare_collections(dataset_id, db)
        
        # Test the specific failing profile if you have it
        failing_profile_id = "HL7_95bbc7f5_15"  # From your error message
        test_specific_failing_profile(failing_profile_id, db)

if __name__ == "__main__":
    main()