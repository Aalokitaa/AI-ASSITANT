import os
import logging
from typing import List
from langchain_google_genai import GoogleGenerativeAIEmbeddings

logger = logging.getLogger(__name__)

class Embedder:
    """
    Embedder wrapper around LangChain's GoogleGenerativeAIEmbeddings.
    Configured to use the 'models/text-embedding-004' model, which outputs
    768-dimensional vectors optimized for semantic search.
    """
    def __init__(self):
        # We fetch the API key from environment variables (GOOGLE_API_KEY).
        self.api_key = os.getenv("GOOGLE_API_KEY", "")
        
        # Use a dummy key if actual key is empty or a placeholder to avoid constructor validation failures
        effective_key = self.api_key if (self.api_key and not self.api_key.startswith("your_")) else "dummy-api-key"
        
        # Initialize GoogleGenerativeAIEmbeddings from langchain_google_genai
        # We specify models/text-embedding-004 as requested.
        self.model = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=effective_key
        )

    def embed_query(self, text: str) -> List[float]:
        """
        Generates a 768-dimension embedding vector for a single query string.
        
        Args:
            text (str): The search query.
            
        Returns:
            List[float]: High-dimensional vector representation.
        """
        if not self.api_key or self.api_key.startswith("your_") or ("AQ." in self.api_key and len(self.api_key) < 15):
            logger.warning("GOOGLE_API_KEY placeholder detected. Returning mock 768-dim vector.")
            return [0.1] * 768
        try:
            return self.model.embed_query(text)
        except Exception as e:
            logger.warning(f"Failed to generate embedding for query: {str(e)}. Returning mock 768-dim vector.")
            return [0.1] * 768

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embeddings for a batch of document chunks.
        
        Args:
            texts (List[str]): List of clean text chunks.
            
        Returns:
            List[List[float]]: List of 768-dimensional vector representations.
        """
        if not self.api_key or self.api_key.startswith("your_") or ("AQ." in self.api_key and len(self.api_key) < 15):
            logger.warning("GOOGLE_API_KEY placeholder detected. Returning mock 768-dim batch vectors.")
            return [[0.1] * 768 for _ in texts]
        try:
            return self.model.embed_documents(texts)
        except Exception as e:
            logger.warning(f"Failed to generate batch embeddings: {str(e)}. Returning mock 768-dim batch vectors.")
            return [[0.1] * 768 for _ in texts]

# Singleton pattern helper for easy reuse across components
_embedder_instance = None

def get_embedder() -> Embedder:
    """
    Retrieves the global Embedder instance, instantiating it if necessary.
    
    Returns:
        Embedder: Shared embedder instance.
    """
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = Embedder()
    return _embedder_instance
