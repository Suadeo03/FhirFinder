# config/chroma.py - FIXED VERSION
import threading
import os
from typing import Optional
import chromadb
from chromadb.config import Settings
import socket
import time

class ChromaConfig:
    """Singleton Chroma configuration with Docker-optimized connection strategies"""
    
    _instance: Optional['ChromaConfig'] = None
    _lock = threading.Lock()
    _initialized = False
    
    def __new__(cls):
        """Ensure only one instance is created (thread-safe)"""
        if cls._instance is None:
            with cls._lock:
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
        """Initialize Chroma with Docker-optimized strategies"""
        collection_name = os.getenv('CHROMA_COLLECTION_NAME', 'fhir_profiles')

        # In Docker environment, prioritize HTTP connection to persistent server
        # Remove in-memory fallback to prevent data loss
        if self._try_http_connection_with_retry(collection_name):
            return
        if self._try_persistent_client(collection_name):
            return

        # REMOVED: in-memory fallback to prevent data loss
        print("❌ All persistent Chroma initialization strategies failed")
        print("💡 Ensure ChromaDB server is running and accessible")
        self.chroma_client = None
        self.collection = None
        self.connection_mode = "failed"

    def _try_http_connection_with_retry(self, collection_name: str, max_retries: int = 5) -> bool:
        """Try HTTP connection with retries for Docker container startup timing"""
        host = os.getenv('CHROMA_HOST', 'chroma')  # Default to service name
        port = int(os.getenv('CHROMA_PORT', '8000'))  # Use internal port 8000
        
        for attempt in range(max_retries):
            try:
                print(f"🔄 Attempt {attempt + 1}/{max_retries}: Connecting to ChromaDB at {host}:{port}")
                
                # Create client with proper settings for server mode
                self.chroma_client = chromadb.HttpClient(
                    host=host,
                    port=port,
                    settings=Settings(
                        allow_reset=True,
                        anonymized_telemetry=False
                    )
                )
                
                # Test connection with timeout
                socket.setdefaulttimeout(10)
                heartbeat = self.chroma_client.heartbeat()
                print(f"✅ ChromaDB server heartbeat: {heartbeat}")
                
                # Get or create collection
                self.collection = self._get_or_create_collection(collection_name)
                if self.collection:
                    self.connection_mode = f"http_server_{host}:{port}"
                    print(f"✅ Connected to persistent ChromaDB server at {host}:{port}")
                    print(f"📊 Collection '{collection_name}' has {self.collection.count()} items")
                    return True
                    
            except Exception as e:
                print(f"❌ HTTP connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"⏳ Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                
        return False

    def _try_persistent_client(self, collection_name: str) -> bool:
        """Try persistent local Chroma client as fallback"""
        try:
            # In Docker, use a volume-mounted directory
            persist_directory = os.getenv('CHROMA_PERSIST_DIR', '/app/chroma_db')
            print(f"🔄 Attempting persistent Chroma client at {persist_directory}")
            
            # Ensure directory exists and is writable
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
                self.connection_mode = f"persistent_local_{persist_directory}"
                print(f"✅ Using persistent Chroma at {persist_directory}")
                print(f"📊 Collection '{collection_name}' has {self.collection.count()} items")
                return True
                
        except Exception as e:
            print(f"❌ Persistent client failed: {e}")
            
        return False

    def _get_or_create_collection(self, collection_name: str):
        """Get existing collection or create new one"""
        try:
            # Try to get existing collection first
            collection = self.chroma_client.get_collection(name=collection_name)
            print(f"📁 Found existing collection: {collection_name} with {collection.count()} items")
            return collection
            
        except Exception:
            try:
                # Create new collection with optimized settings
                collection = self.chroma_client.create_collection(
                    name=collection_name,
                    metadata={
                        "description": "FHIR Profile embeddings",
                        "hnsw:space": "cosine"
                    }
                )
                print(f"✨ Created new collection: {collection_name}")
                return collection
                
            except Exception as e:
                print(f"❌ Failed to create collection {collection_name}: {e}")
                return None

    def get_collection(self):
        """Get the singleton collection instance"""
        if not self._initialized or self.collection is None:
            # Try to reinitialize if connection was lost
            print("🔄 Collection not available, attempting to reinitialize...")
            self.init_chroma()
            if self.collection is None:
                raise RuntimeError(
                    "ChromaConfig not properly initialized. "
                    "Ensure ChromaDB server is running and accessible."
                )
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

    def test_connection(self) -> bool:
        """Test Chroma connection and data persistence"""
        try:
            if not self.collection:
                print("❌ No collection available for testing")
                return False
                
            # Test basic operations
            count = self.collection.count()
            print(f"✅ Connection test successful - collection has {count} items")
            
            # Test if we can query (even with empty results)
            if count > 0:
                sample_query = self.collection.peek(limit=1)
                print(f"✅ Sample data accessible: {len(sample_query.get('ids', []))} items")
            
            return True
            
        except Exception as e:
            print(f"❌ Connection test failed: {e}")
            return False

    def get_connection_info(self) -> dict:
        """Get detailed connection information for debugging"""
        return {
            "connection_mode": self.connection_mode,
            "is_available": self.is_available(),
            "collection_name": self.collection.name if self.collection else None,
            "collection_count": self.collection.count() if self.collection else 0,
            "client_type": type(self.chroma_client).__name__ if self.chroma_client else None,
            "environment": {
                "CHROMA_HOST": os.getenv('CHROMA_HOST', 'chroma'),
                "CHROMA_PORT": os.getenv('CHROMA_PORT', '8000'),
                "CHROMA_COLLECTION_NAME": os.getenv('CHROMA_COLLECTION_NAME', 'fhir_profiles'),
                "CHROMA_PERSIST_DIR": os.getenv('CHROMA_PERSIST_DIR', '/app/chroma_db')
            }
        }


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