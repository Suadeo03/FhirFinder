# force_update_schema.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy import create_engine, text

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://fhir_user:admin@localhost:5432/fhir_registry")
def force_update_schema():
    """Force update the database schema"""
    try:
        engine = create_engine(DATABASE_URL)
        
        print("🔧 Force updating database schema...")
        
        with engine.connect() as conn:
            # First, let's see what we're working with
            result = conn.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = 'forms'
            """))
            
            if result.fetchone():
                print("📝 Forms table exists. Updating column types...")
                
                # Update all problematic columns
                updates = [
                    ("screening_tool", "TEXT"),
                    ("loinc_panel_code", "VARCHAR(100)"),
                    ("loinc_panel_name", "TEXT"),
                    ("question", "TEXT"),
                    ("loinc_question_code", "VARCHAR(100)"),
                    ("loinc_answer", "TEXT"),
                    ("snomed_code_ct", "TEXT"),
                    ("loinc_concept", "TEXT"),
                    ("answer_concept", "TEXT")
                ]
                
                for column_name, new_type in updates:
                    try:
                        # Check if column exists first
                        check_sql = text("""
                            SELECT column_name 
                            FROM information_schema.columns 
                            WHERE table_name = 'forms' 
                            AND column_name = :col_name
                        """)
                        exists = conn.execute(check_sql, {"col_name": column_name}).fetchone()
                        
                        if exists:
                            sql = f"ALTER TABLE forms ALTER COLUMN {column_name} TYPE {new_type}"
                            conn.execute(text(sql))
                            print(f"✅ Updated {column_name} to {new_type}")
                        else:
                            print(f"⚠️  Column {column_name} doesn't exist")
                            
                    except Exception as e:
                        print(f"❌ Failed to update {column_name}: {e}")
                
                # Commit all changes
                conn.commit()
                print("✅ Schema update completed!")
                
            else:
                print("❌ Forms table doesn't exist!")
                
    except Exception as e:
        print(f"❌ Error during schema update: {e}")

def verify_update():
    """Verify the schema was updated correctly"""
    try:
        engine = create_engine(DATABASE_URL)
        
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT column_name, data_type, character_maximum_length
                FROM information_schema.columns 
                WHERE table_name = 'forms' 
                AND column_name IN ('screening_tool', 'loinc_panel_name', 'question', 'loinc_answer', 'snomed_code_ct')
                ORDER BY column_name
            """))
            
            print("\n📊 Updated schema verification:")
            print("-" * 50)
            for row in result:
                col_name, data_type, max_length = row
                length_str = str(max_length) if max_length else "unlimited"
                print(f"{col_name:<20} | {data_type} ({length_str})")
            print("-" * 50)
            
    except Exception as e:
        print(f"Error verifying update: {e}")

if __name__ == "__main__":
    force_update_schema()
    verify_update()
    print("\n🎉 Database schema update process complete!")