# config/chroma.py
import threading
import os
from typing import Optional
import chromadb
from chromadb.config import Settings
import socket

class ChromaConfig:
    """Singleton Chroma configuration with multiple connection strategies"""
    
    _instance: Optional['ChromaConfig'] = None
    _lock = threading.Lock()
    _initialized = False
    
    def __new__(cls):
        """Ensure only one instance is created (thread-safe)"""
        if cls._instance is None:
            with cls._lock:
                # Double-check locking pattern
                if cls._instance is None:
                    cls._instance = super(ChromaConfig, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize Chroma client and collection only once"""
        
        
        
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    self.chroma_client = None
                    self.collection = None
                    self.connection_mode = None
                    try:
                        self.init_chroma()
                        ChromaConfig._initialized = True
                        print("✅ ChromaConfig singleton initialized successfully")
                    except Exception as e:
                        print(f"❌ Error initializing ChromaConfig singleton: {e}")
                        self.chroma_client = None
                        self.collection = None
                        self.connection_mode = "failed"

    def init_chroma(self):
        """Initialize Chroma with multiple fallback strategies"""
        collection_name = os.getenv('CHROMA_COLLECTION_NAME', 'fhir_profiles')

        # Try connection strategies in order of preference
        if self._try_http_connection(collection_name):
            return
        if self._try_persistent_client(collection_name):
            return
        if self._try_in_memory_client(collection_name):
            return

        print("❌ All Chroma initialization strategies failed")
        self.chroma_client = None
        self.collection = None
        self.connection_mode = "failed"

    def _try_http_connection(self, collection_name: str) -> bool:

        host = os.getenv('CHROMA_HOST', 'localhost')
        port = os.getenv('CHROMA_PORT', '8001')
 
        try:
            print(f"🔄 Attempting Chroma HTTP connection to {host}:{port}")
            
            # Create client with timeout settings
            self.chroma_client = chromadb.HttpClient(
                host=host,
                port=port,
                settings=Settings(
                    chroma_client_auth_provider="chromadb.auth.token.TokenAuthClientProvider",
                    chroma_client_auth_credentials=os.getenv('CHROMA_TOKEN', ''),
                    allow_reset=True
                ) if os.getenv('CHROMA_TOKEN') else Settings(allow_reset=True)
            )
            
            socket.setdefaulttimeout(5)  
            
            heartbeat = self.chroma_client.heartbeat()
            print(f"✅ Chroma server heartbeat: {heartbeat}")
            
            self.collection = self._get_or_create_collection(collection_name)
            if self.collection:
                self.connection_mode = f"http_server_{host}:{port}"
                print(f"✅ Connected to Chroma HTTP server at {host}:{port}")
                return True
                    
        except Exception as e:
            print(f"HTTP connection to {host}:{port} failed: {e}")
        return False

    def _try_persistent_client(self, collection_name: str) -> bool:
        """Try persistent local Chroma client"""
        try:
            persist_directory = os.getenv('CHROMA_PERSIST_DIR', './chroma_db')
            print(f"🔄 Attempting persistent Chroma client at {persist_directory}")
            
            os.makedirs(persist_directory, exist_ok=True)
            self.chroma_client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            self.collection = self._get_or_create_collection(collection_name)
            if self.collection:
                self.connection_mode = "persistent_local"
                print(f"Using persistent Chroma at {persist_directory}")
                return True
                
        except Exception as e:
            print(f"Persistent client failed: {e}")
            
        return False

    def _try_in_memory_client(self, collection_name: str) -> bool:
        """Try in-memory Chroma client"""
        try:
            print("🔄 Attempting in-memory Chroma client")
            
            self.chroma_client = chromadb.Client(
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            self.collection = self._get_or_create_collection(collection_name)
            if self.collection:
                self.connection_mode = "in_memory"
                print("⚠️  Using in-memory Chroma (data will not persist)")
                return True
                
        except Exception as e:
            print(f"❌ In-memory client failed: {e}")
            
        return False

    def _get_or_create_collection(self, collection_name: str):
        """Get existing collection or create new one"""
        try:
            # Try to get existing collection first
            collection = self.chroma_client.get_collection(name=collection_name)
            print(f"📁 Found existing collection: {collection_name}")
            return collection
            
        except Exception:
            try:

                collection = self.chroma_client.create_collection(
                    name=collection_name,
                    metadata={
                        "description": "FHIR Profile embeddings",
                        "hnsw:space": "cosine" 
                    }
                )
                print(f"Created new collection: {collection_name}")
                return collection
                
            except Exception as e:
                print(f"❌ Failed to create collection {collection_name}: {e}")
                return None

    def get_collection(self):
        """Get the singleton collection instance"""
        if not self._initialized or self.collection is None:
            raise RuntimeError("ChromaConfig not properly initialized")
        return self.collection
    
    def get_client(self):
        """Get the singleton client instance"""
        if not self._initialized or self.chroma_client is None:
            raise RuntimeError("ChromaConfig not properly initialized")
        return self.chroma_client
    
    def is_available(self) -> bool:
        """Check if Chroma is available and working"""
        return (self._initialized and 
                self.collection is not None and 
                self.chroma_client is not None and 
                self.connection_mode != "failed")

    def get_client_info(self) -> dict:
        """Get detailed client information"""
        return {
            "client_type": type(self.chroma_client).__name__ if self.chroma_client else None,
            "connection_mode": self.connection_mode,
            "collection_name": self.collection.name if self.collection else None,
            "collection_count": self.collection.count() if self.collection else 0,
            "is_available": self.is_available(),
            "singleton_initialized": self._initialized
        }

    def get_status(self) -> dict:

        if not self.is_available():
            return {
                "status": "unavailable",
                "initialized": self._initialized,
                "client": self.chroma_client is not None,
                "collection": self.collection is not None,
                "connection_mode": self.connection_mode,
                "count": 0
            }
        
        try:
            count = self.collection.count()
            return {
                "status": "available",
                "initialized": self._initialized,
                "client": True,
                "collection": True,
                "connection_mode": self.connection_mode,
                "count": count,
                "collection_name": self.collection.name
            }
        except Exception as e:
            return {
                "status": "error",
                "initialized": self._initialized,
                "client": self.chroma_client is not None,
                "collection": self.collection is not None,
                "connection_mode": self.connection_mode,
                "error": str(e),
                "count": 0
            }

    def test_connection(self) -> bool:
        try:
            if not self.collection:
                print("❌ No collection available for testing")
                return False
                
            # Try a simple operation
            count = self.collection.count()
            print(f"✅ Chroma test successful - collection has {count} items")
            return True
            
        except Exception as e:
            print(f"❌ Chroma test failed: {e}")
            return False

    def clear_collection(self, collection_name: str = None):
        try:
            if collection_name:
                collection = self.chroma_client.get_collection(collection_name)
            else:
                collection = self.collection
                
            if collection:
                all_items = collection.get()
                if all_items['ids']:
                    collection.delete(ids=all_items['ids'])
                    print(f"Cleared {len(all_items['ids'])} items from {collection.name}")
                else:
                    print(f"ℹ️  Collection {collection.name} was already empty")
            else:
                print("❌ No collection available to clear")
                
        except Exception as e:
            print(f"❌ Error clearing collection: {e}")

    def reset_collection(self):
        """Reset the collection (useful for testing or data refresh)"""
        if not self.is_available():
            print("❌ Cannot reset - Chroma not available")
            return False
        
        try:
            with self._lock:
                collection_name = self.collection.name
                
                # Delete existing collection
                self.chroma_client.delete_collection(collection_name)
                print(f"🗑️  Deleted existing collection: {collection_name}")
                
                # Recreate collection
                self.collection = self._get_or_create_collection(collection_name)
                if self.collection:
                    print(f"✅ Recreated collection: {collection_name}")
                    return True
                else:
                    print(f"❌ Failed to recreate collection: {collection_name}")
                    return False
                
        except Exception as e:
            print(f"❌ Error resetting collection: {e}")
            return False

    def switch_collection(self, collection_name: str):
        """Switch to a different collection"""
        if not self.chroma_client:
            print("❌ No Chroma client available")
            return False
        
        try:
            with self._lock:
                new_collection = self._get_or_create_collection(collection_name)
                if new_collection:
                    self.collection = new_collection
                    print(f"✅ Switched to collection: {collection_name}")
                    return True
                else:
                    print(f"❌ Failed to switch to collection: {collection_name}")
                    return False
        except Exception as e:
            print(f"❌ Error switching collection: {e}")
            return False
    
    @classmethod
    def reset_singleton(cls):
        with cls._lock:
            if cls._instance:
                # Clean up existing instance
                try:
                    if cls._instance.collection:
                        cls._instance.collection = None
                    if cls._instance.chroma_client:
                        cls._instance.chroma_client = None
                except:
                    pass
            
            cls._instance = None
            cls._initialized = False
        print("🔄 ChromaConfig singleton reset")

    def get_connection_diagnostics(self) -> dict:
        """Get detailed diagnostics for troubleshooting connection issues"""
        diagnostics = {
            "singleton_status": {
                "initialized": self._initialized,
                "instance_exists": self._instance is not None,
                "same_instance": self is ChromaConfig._instance
            },
            "environment_variables": {
                "CHROMA_HOST": os.getenv('CHROMA_HOST', 'localhost'),
                "CHROMA_PORT": os.getenv('CHROMA_PORT', '8001'),
                "CHROMA_COLLECTION_NAME": os.getenv('CHROMA_COLLECTION_NAME', 'fhir_profiles'),
                "CHROMA_PERSIST_DIR": os.getenv('CHROMA_PERSIST_DIR', './chroma_db'),
                "CHROMA_TOKEN": "***" if os.getenv('CHROMA_TOKEN') else None
            },
            "client_status": {
                "client_exists": self.chroma_client is not None,
                "client_type": type(self.chroma_client).__name__ if self.chroma_client else None,
                "connection_mode": self.connection_mode
            },
            "collection_status": {
                "collection_exists": self.collection is not None,
                "collection_name": self.collection.name if self.collection else None,
                "collection_count": self.collection.count() if self.collection else 0
            }
        }
        
        try:
            if self.chroma_client and hasattr(self.chroma_client, 'heartbeat'):
                diagnostics["connectivity"] = {
                    "heartbeat": str(self.chroma_client.heartbeat())
                }
        except Exception as e:
            diagnostics["connectivity"] = {
                "heartbeat_error": str(e)
            }
        
        return diagnostics


def get_chroma_instance() -> ChromaConfig:
    """Get the singleton ChromaConfig instance"""
    return ChromaConfig()

def get_chroma_collection():
    """Get the Chroma collection directly"""
    return get_chroma_instance().get_collection()

def get_chroma_client():
    """Get the Chroma client directly"""
    return get_chroma_instance().get_client()

def is_chroma_available() -> bool:
    """Check if Chroma is available"""
    try:
        return get_chroma_instance().is_available()
    except:
        return False

def get_chroma_info() -> dict:
    """Get Chroma client information"""
    try:
        return get_chroma_instance().get_client_info()
    except:
        return {"error": "ChromaConfig not available"}

def test_chroma_connection() -> bool:
    """Test Chroma connection"""
    try:
        return get_chroma_instance().test_connection()
    except:
        return False