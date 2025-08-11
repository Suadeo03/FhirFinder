# backend/services/narrative/conversational_service.py
import requests
import json
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class FHIRConversationalService:
    """Summarize FHIR search results with the user's query"""
    
    def __init__(self, ollama_url: str = "http://localhost:11434", model: str = "tinyllama"):
        self.ollama_url = ollama_url
        self.model = model
        self.available = self._check_ollama_availability()
    
    def _check_ollama_availability(self) -> bool:
        """Check if Ollama is running"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=3)
            return response.status_code == 200
        except:
            return False
    
    def _query_ollama(self, prompt: str) -> str:
        """Send prompt to Ollama with fast settings"""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": 100,  
                        "temperature": 0.4   
                    }
                },
                timeout=20  
            )
            
            if response.status_code == 200:
                return response.json().get("response", "").strip()
            return ""
                
        except Exception as e:
            logger.warning(f"Ollama query failed: {e}")
            return ""
    
    def summarize_results(self, query: str, results: List[Dict]) -> str:
        """Generate a simple summary of search results"""
        
        if not results:
            return f"No FHIR profiles found for '{query}'. Try different search terms."
        
        if not self.available:
            first_result = results[0]
            resource_type = first_result.get('resource_type', 'FHIR profile')
            return f"Found {len(results)} {resource_type}(s) matching '{query}'."
        
        # Extract key info from results
        first_result = results[0]
        resource_type = first_result.get('resource_type', 'Unknown')
        name = first_result.get('name', 'Unnamed')[:60]  # Truncate long names
        description = first_result.get('description', '')[:100]  # Truncate
        
        # Create a simple, focused prompt
        prompt = f"""Summarize this search result in 1-2 sentences:

Query: "{query}"
Found: {len(results)} results
Best match: {resource_type} - {name}
Description: {description}

Write a brief, helpful summary."""

        summary = self._query_ollama(prompt)
        
        # Fallback if AI fails
        if not summary or len(summary.strip()) < 10:
            return f"Found {len(results)} {resource_type} profile(s) for '{query}'. Best match: {name}."
        
        return summary
    
    def get_status(self) -> Dict[str, Any]:
        """Get service status"""
        return {
            "available": self.available,
            "model": self.model
        }