import os
import logging
from typing import List, Dict, Any, Optional
from pinecone import Pinecone

from app.utils.embeddings import get_embedder

# Setup logger
logger = logging.getLogger(__name__)

# Try to import CrossEncoder from sentence_transformers for deep semantic reranking.
# We fall back gracefully to vector distance scores if imports fail or the machine lacks resources.
try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None
    logger.warning("sentence-transformers not installed. Reranking will fall back to Pinecone similarity scores.")

class DocumentRetriever:
    """
    Retrieves relevant text fragments for research queries using a two-stage retrieval pipeline:
    1. Dense Semantic Search: Embeds query and retrieves top-8 matching vectors from Pinecone.
    2. Deep Reranking: Scores the top-8 results using a ms-marco-MiniLM CrossEncoder, returning the top-3.
    """
    def __init__(self, index_name: Optional[str] = None):
        self.api_key = os.getenv("PINECONE_API_KEY", "")
        self.index_name = index_name or os.getenv("PINECONE_INDEX_NAME", "research-assistant")
        
        # Core embedder to generate query vector
        self.embedder = get_embedder()
        
        # Initialize Pinecone
        if self.api_key and not self.api_key.startswith("your_"):
            try:
                self.pc = Pinecone(api_key=self.api_key)
                self.index = self.pc.Index(self.index_name)
            except Exception as e:
                logger.error(f"Failed to initialize Pinecone Index in Retriever: {str(e)}")
                self.index = None
        else:
            logger.warning("PINECONE_API_KEY is not configured. Retrieval will be mocked.")
            self.index = None
            
        # Initialize Reranker Model
        self.reranker = None
        if CrossEncoder:
            try:
                logger.info("Loading CrossEncoder (ms-marco-MiniLM-L-6-v2) for second-stage reranking...")
                # Use CPU-only execution to remain stable across hosting platforms
                self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
                logger.info("CrossEncoder model loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load CrossEncoder model: {str(e)}. Falling back to base similarity.")
                self.reranker = None

    def retrieve(self, query: str, top_k: int = 8, final_top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Executes the two-stage RAG retrieval pipeline: semantic query -> rerank -> top chunks.
        
        Args:
            query (str): The research question/query.
            top_k (int): Number of dense vector chunks to retrieve from Pinecone.
            final_top_k (int): Number of top reranked chunks to return.
            
        Returns:
            List[Dict[str, Any]]: List of top-3 reranked chunks with text, metadata, and scores.
        """
        if not query or not query.strip():
            return []
            
        # Step 1: Generate Embedding Vector for the query
        try:
            query_vector = self.embedder.embed_query(query)
        except Exception as e:
            logger.error(f"Error embedding retrieval query: {str(e)}")
            return []
            
        # Step 2: Semantic search against Pinecone (First Stage)
        raw_chunks = []
        if self.index:
            try:
                search_response = self.index.query(
                    vector=query_vector,
                    top_k=top_k,
                    include_metadata=True
                )
                
                # Format matches
                for match in search_response.get("matches", []):
                    metadata = match.get("metadata", {})
                    # Retrieve the raw text stored during ingestion
                    chunk_text = metadata.pop("text", "")
                    
                    raw_chunks.append({
                        "text": chunk_text,
                        "metadata": metadata,
                        "base_score": match.get("score", 0.0)
                    })
            except Exception as e:
                logger.error(f"Error querying Pinecone index: {str(e)}")
                # Continue with empty list (or mock list in dev)
        else:
            logger.warning("Pinecone index is not initialized. Returning empty search results.")
            
        if not raw_chunks:
            return []
            
        # Step 3: Cross-Encoder Reranking (Second Stage)
        if self.reranker:
            try:
                # Prepare input pairs: (query, text)
                pairs = [(query, chunk["text"]) for chunk in raw_chunks]
                
                # Predict relevance scores (higher is better)
                rerank_scores = self.reranker.predict(pairs)
                
                # Map scores back to chunks
                for idx, score in enumerate(rerank_scores):
                    # Sigmoid scale the output logic for clean reading, or just use raw output
                    chunk = raw_chunks[idx]
                    chunk["rerank_score"] = float(score)
                    
                # Sort by rerank score descending
                raw_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
            except Exception as e:
                logger.error(f"Reranking execution failed: {str(e)}. Falling back to base similarity score.")
                raw_chunks.sort(key=lambda x: x["base_score"], reverse=True)
        else:
            # Fallback to sorting by Pinecone cosine similarity score
            raw_chunks.sort(key=lambda x: x["base_score"], reverse=True)
            for chunk in raw_chunks:
                chunk["rerank_score"] = chunk["base_score"]
                
        # Return only the requested top-n final chunks
        return raw_chunks[:final_top_k]
