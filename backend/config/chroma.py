import chromadb
import os
from chromadb.config import Settings

class ChromaConfig:
    
    def __init__(self):
        self.chroma_client = None
        self.collection = None
        self.connection_mode = None
        self.init_chroma()

    def init_chroma(self):

        collection_name = os.getenv('CHROMA_COLLECTION_NAME', 'fhir_profiles')
        
        if self._try_http_connection(collection_name):
            return
        if self._try_persistent_client(collection_name):
            return
        if self._try_in_memory_client(collection_name):
            return

        print("All Chroma initialization strategies failed")
        self.chroma_client = None
        self.collection = None
        self.connection_mode = "failed"

    def _try_http_connection(self, collection_name: str) -> bool:
        """Try to connect to Chroma HTTP server (Docker or remote)"""

        
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
            
            import socket
            socket.setdefaulttimeout(5)  
            
            heartbeat = self.chroma_client.heartbeat()
            print(f"✅ Chroma server heartbeat: {heartbeat}")
            
            self.collection = self._get_or_create_collection(collection_name)
            if self.collection:
                self.connection_mode = f"http_server_{host}:{port}"
                print(f"✅ Connected to Chroma HTTP server at {host}:{port}")
                return True
                    
        except Exception as e:
                print(f"❌ HTTP connection to {host}:{port} failed: {e}")
        return False

    def _try_persistent_client(self, collection_name: str) -> bool:
        """persistent local Chroma client"""
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
                print(f"✅ Using persistent Chroma at {persist_directory}")
                return True
                
        except Exception as e:
            print(f"Persistent client failed: {e}")
            
        return False

    def _try_in_memory_client(self, collection_name: str) -> bool:

        try:
            print("Attempting in-memory Chroma client")
            
            self.chroma_client = chromadb.Client(
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            

            self.collection = self._get_or_create_collection(collection_name)
            if self.collection:
                self.connection_mode = "in_memory"
                print("Using in-memory Chroma (data will not persist)")
                return True
                
        except Exception as e:
            print(f"In-memory client failed: {e}")
            
        return False

    def _get_or_create_collection(self, collection_name: str):
        try:

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
                print(f"Failed to create collection {collection_name}: {e}")
                return None

    def get_client_info(self) -> dict:

        return {
            "client_type": type(self.chroma_client).__name__ if self.chroma_client else None,
            "connection_mode": self.connection_mode,
            "collection_name": self.collection.name if self.collection else None,
            "collection_count": self.collection.count() if self.collection else 0,
            "is_available": self.collection is not None
        }

    def test_connection(self) -> bool:
        """Test if Chroma is properly initialized and working"""
        try:
            if not self.collection:
                return False
                
            # Try a simple operation
            count = self.collection.count()
            print(f"✅ Chroma test successful - collection has {count} items")
            return True
            
        except Exception as e:
            print(f"❌ Chroma test failed: {e}")
            return False

    def clear_collection(self, collection_name: str = None):
        """Clear a specific collection or the current collection"""
        try:
            if collection_name:
                collection = self.chroma_client.get_collection(collection_name)
            else:
                collection = self.collection
                
            if collection:
                all_items = collection.get()
                if all_items['ids']:
                    collection.delete(ids=all_items['ids'])
                    print(f"✅ Cleared {len(all_items['ids'])} items from {collection.name}")
                else:
                    print(f"ℹ️  Collection {collection.name} was already empty")
            else:
                print("❌ No collection available to clear")
                
        except Exception as e:
            print(f"❌ Error clearing collection: {e}")
