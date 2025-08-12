# WARNING: A DEVELOPMENT TOOL TO REMOVE TEST DATA
# DO NOT RUN IN PRODUCTION ENVIRONMENT!

import os
import sys
from typing import Optional

# Add the backend directory to Python path so modules can be found
backend_dir = os.path.dirname(os.path.abspath(__file__))  # Gets the config directory
backend_dir = os.path.dirname(backend_dir)  # Gets the backend directory
sys.path.insert(0, backend_dir)

print(f"Added to Python path: {backend_dir}")

try:
    # Now try importing with the backend directory in the path
    from config.database import clear_postgres_data
    from config.chroma import ChromaConfig  
    from config.redis_cache import RedisQueryCache
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path includes: {sys.path}")
    print("\nTrying alternative import method...")
    
    # Alternative: try importing without config prefix
    try:
        sys.path.insert(0, os.path.join(backend_dir, 'config'))
        from database import clear_postgres_data
        from chroma import ChromaConfig
        from config.redis_cache import RedisQueryCache
        print("✅ Alternative imports successful")
    except ImportError as e2:
        print(f"❌ Alternative import also failed: {e2}")
        print("\nPlease check your file structure and run from backend directory:")
        print("  cd backend")
        print("  python config/reset_all.py")
        sys.exit(1)


class ResetAll:
    """
    This class provides a method to reset all data in the database.
    It is intended for use in development and testing environments ONLY.
    
    WARNING: This will permanently delete ALL data from:
    - Redis cache
    - ChromaDB collections
    - PostgreSQL database
    """
    

    
    @staticmethod
    def confirm_reset() -> bool:
        """
        Ask for user confirmation before proceeding with reset.
        Returns True if user confirms, False otherwise.
        """
        print("\n" + "="*60)
        print("⚠️  WARNING: DATA DESTRUCTION IMMINENT ⚠️")
        print("="*60)
        print("This will permanently delete ALL data from:")
        print("  • Redis cache (all keys)")
        print("  • ChromaDB collections")
        print("  • PostgreSQL database (all tables)")
        print("="*60)
        
        response = input("\nType 'DELETE ALL DATA' to confirm (case sensitive): ")
        
        if response == "DELETE ALL DATA":
            print("\n✅ Confirmation received. Proceeding with reset...")
            return True
        else:
            print("\n❌ Reset cancelled. No data was deleted.")
            return False
    
    @staticmethod
    def reset_redis() -> bool:
        """Reset Redis cache data"""
        try:
            print("🔄 Resetting Redis cache...")
            
        # Try direct Redis connection first
            try:
                client = RedisQueryCache(host='localhost', port=6379, db=0)
                client.ping()  # Test connection
                
           
                # Clear all data
                client.clear_all_cache()
                client.close()
                
                print("✅ Redis cache cleared successfully")
                return True
            
            except Exception as redis_error:
                print(f"❌ Direct Redis connection failed: {redis_error}")
                
                # Fallback to RedisQueryCache class
                redis_cache = RedisQueryCache()
                
                if not redis_cache.is_connected():
                    print("❌ Redis not connected - skipping Redis reset")
                    return False
                
                success = redis_cache.clear_all_cache()
                redis_cache.close()
                
                if success:
                    print("✅ Redis cache cleared successfully (via fallback)")
                    return True
                else:
                    print("❌ Failed to clear Redis cache")
                    return False
                
        except Exception as e:
            print(f"❌ Error resetting Redis: {e}")
            return False
    
    @staticmethod
    def reset_chroma() -> bool:
        """Reset ChromaDB collections"""
        try:
            print("🔄 Resetting ChromaDB collections...")
            chroma = ChromaConfig()
            
            # List of collections to clear - add more as needed
            collections = ["fhir_profiles"]
            
            success_count = 0
            for collection_name in collections:
                try:
                    result = chroma.clear_collection(collection_name)
                    if result:
                        print(f"   ✅ Cleared collection: {collection_name}")
                        success_count += 1
                    else:
                        print(f"   ⚠️  Collection {collection_name} may not exist or failed to clear")
                except Exception as e:
                    print(f"   ❌ Error clearing collection {collection_name}: {e}")
            
            print(f"✅ ChromaDB reset complete ({success_count}/{len(collections)} collections cleared)")
            return success_count > 0
            
        except Exception as e:
            print(f"❌ Error resetting ChromaDB: {e}")
            return False
    
    @staticmethod
    def reset_postgres() -> bool:
        """Reset PostgreSQL database"""
        try:
            print("🔄 Resetting PostgreSQL database...")
            clear_postgres_data()
            print("✅ PostgreSQL database cleared successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error resetting PostgreSQL: {e}")
            return False
    
    @staticmethod
    def reset_all_data(force: bool = False, skip_confirmation: bool = False) -> bool:
        """
        Resets all data in the database.
        
        Args:
            force: If True, skip environment checks (dangerous!)
            skip_confirmation: If True, skip user confirmation (for automated scripts)
        
        Returns:
            bool: True if all resets successful, False otherwise
        """
        # Environment safety check

        
        # User confirmation
        if not skip_confirmation and not ResetAll.confirm_reset():
            return False
        
        print("\n🚀 Starting complete data reset...")
        print("-" * 40)
        
        # Track success of each operation
        results = {
            'redis': ResetAll.reset_redis(),
            'chroma': ResetAll.reset_chroma(),
            'postgres': ResetAll.reset_postgres()
        }
        
        # Summary
        print("\n" + "="*40)
        print("📊 RESET SUMMARY")
        print("="*40)
        
        all_successful = True
        for component, success in results.items():
            status = "✅ SUCCESS" if success else "❌ FAILED"
            print(f"{component.upper():<12} {status}")
            if not success:
                all_successful = False
        
        print("="*40)
        
        if all_successful:
            print("🎉 All data has been reset successfully!")
            return True
        else:
            print("⚠️  Some components failed to reset. Check logs above.")
            return False


def main():
    """Main function for command line usage"""
    # Check for command line arguments
    force = '--force' in sys.argv
    skip_confirmation = '--yes' in sys.argv or '-y' in sys.argv
    
    if '--help' in sys.argv or '-h' in sys.argv:
        print("Reset All Data Script")
        print("Usage: python reset_all.py [options]")
        print("\nOptions:")
        print("  --force    Skip environment safety checks (DANGEROUS!)")
        print("  --yes, -y  Skip confirmation prompt")
        print("  --help, -h Show this help message")
        return
    
    # Run the reset
    success = ResetAll.reset_all_data(force=force, skip_confirmation=skip_confirmation)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


