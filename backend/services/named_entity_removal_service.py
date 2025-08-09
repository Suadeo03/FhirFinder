import spacy
import logging
from typing import List, Optional

class PHIQueryScrubber:
    """
    A lightweight service class to detect and remove names from search queries using only spaCy
    """
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize the PHI scrubber with spaCy
        
        Args:
            model_name: spaCy model to use for NER
        """
        try:
            self.nlp = spacy.load(model_name)
            logging.info("Simple PHI Scrubber initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize PHI Scrubber: {e}")
            self.nlp = None
    
    def scrub_query(self, query: str) -> str:
        """
        Remove person names from a search query
        
        Args:
            query: The original search query string
            
        Returns:
            str: The scrubbed query with names removed, or original query if no names found
        """
        if not query or not query.strip():
            return query
            
        if not self.nlp:
            logging.warning("PHI Scrubber not properly initialized, returning original query")
            return query
        
        try:
            doc = self.nlp(query)
            person_entities = [ent for ent in doc.ents if ent.label_ == "PERSON"]
            if not person_entities:
                return query
            

            scrubbed_text = query
            offset = 0
            
            for ent in sorted(person_entities, key=lambda x: x.start_char):
                start = ent.start_char - offset
                end = ent.end_char - offset
                

                scrubbed_text = scrubbed_text[:start] + scrubbed_text[end:]
                offset += (end - start)
            
            scrubbed_text = ' '.join(scrubbed_text.split()).strip()
            
            if not scrubbed_text:
                logging.warning("Scrubbing resulted in empty query, returning original")
                return query
                
            logging.info(f"Names detected and removed. Original: '{query}' -> Scrubbed: '{scrubbed_text}'")
            return scrubbed_text
            
        except Exception as e:
            logging.error(f"Error during PHI scrubbing: {e}")
            return query  # Return original query if scrubbing fails
    
    def has_names(self, query: str) -> bool:
        """
        Check if query contains person names
        
        Args:
            query: The search query string
            
        Returns:
            bool: True if person names are detected, False otherwise
        """
        if not query or not query.strip() or not self.nlp:
            return False
            
        try:
            doc = self.nlp(query)
            return any(ent.label_ == "PERSON" for ent in doc.ents)
        except Exception as e:
            logging.error(f"Error checking for names: {e}")
            return False
    
    def get_detected_names(self, query: str) -> List[dict]:
        """
        Get detailed information about detected person names
        
        Args:
            query: The search query string
            
        Returns:
            List[dict]: List of detected person names with details
        """
        if not query or not query.strip() or not self.nlp:
            return []
            
        try:
            doc = self.nlp(query)
            
            detected_names = []
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    detected_names.append({
                        "text": ent.text,
                        "start": ent.start_char,
                        "end": ent.end_char,
                        "confidence": getattr(ent, 'score', 0.95)  # spaCy doesn't always provide scores
                    })
            
            return detected_names
            
        except Exception as e:
            logging.error(f"Error getting name details: {e}")
            return []
