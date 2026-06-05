import logging
from typing import List, Dict, Any

try:
    import tiktoken
except ImportError:
    tiktoken = None

logger = logging.getLogger(__name__)

class TokenChunker:
    """
    Splits text into chunks based on token count rather than raw character counts.
    Ensures that chunks fit within the target model's input context (512 tokens)
    while preserving context continuity via a sliding overlap (50 tokens).
    """
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # We use cl100k_base which is the standard encoding for text-embedding-3-small and GPT-4
        if tiktoken:
            try:
                self.encoding = tiktoken.get_encoding("cl100k_base")
            except Exception as e:
                logger.warning(f"Error loading cl100k_base encoding: {str(e)}. Falling back to character estimation.")
                self.encoding = None
        else:
            logger.warning("tiktoken library not found. Falling back to character approximation.")
            self.encoding = None

    def split_text(self, text: str) -> List[str]:
        """
        Splits a single body of text into 512-token chunks with 50-token overlap.
        If tiktoken is not available, falls back to a character-based approximation.
        
        Args:
            text (str): Raw string content of the document.
            
        Returns:
            List[str]: Clean text chunks.
        """
        if not text or not text.strip():
            return []

        # Fallback to character-based chunking if tiktoken encoding failed or is not available
        # 1 token is approximately 4 characters.
        if not self.encoding:
            char_size = self.chunk_size * 4
            char_overlap = self.chunk_overlap * 4
            
            chunks = []
            start = 0
            text_len = len(text)
            
            while start < text_len:
                end = start + char_size
                chunk = text[start:end].strip()
                if chunk:
                    chunks.append(chunk)
                start += (char_size - char_overlap)
            return chunks

        # Token-based chunking path
        tokens = self.encoding.encode(text)
        num_tokens = len(tokens)
        
        chunks = []
        start_idx = 0
        
        while start_idx < num_tokens:
            # Determine end token boundary
            end_idx = min(start_idx + self.chunk_size, num_tokens)
            chunk_tokens = tokens[start_idx:end_idx]
            
            # Decode back to clean string
            chunk_text = self.encoding.decode(chunk_tokens).strip()
            if chunk_text:
                chunks.append(chunk_text)
                
            # Move index forward by chunk size minus overlap
            start_idx += (self.chunk_size - self.chunk_overlap)
            
            # Prevent infinite loops in edge cases where chunk_size <= chunk_overlap
            if self.chunk_size <= self.chunk_overlap:
                break
                
        return chunks

    def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Splits a structured document dictionary into multiple chunk dictionaries,
        preserving the original metadata (e.g. source, title) while appending chunk-specific details.
        
        Args:
            document (Dict[str, Any]): Dict containing 'text' and 'metadata'.
            
        Returns:
            List[Dict[str, Any]]: List of chunks with mapped metadata.
        """
        text = document.get("text", "")
        metadata = document.get("metadata", {})
        
        text_chunks = self.split_text(text)
        chunks = []
        
        for idx, chunk_text in enumerate(text_chunks):
            # Shallow copy of metadata to avoid modifying the original dict
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_index"] = idx
            
            chunks.append({
                "text": chunk_text,
                "metadata": chunk_metadata
            })
            
        return chunks
