import os
import re
import logging
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse

import fitz  # PyMuPDF
from bs4 import BeautifulSoup
import requests
from pinecone import Pinecone, ServerlessSpec

from app.utils.chunker import TokenChunker
from app.utils.embeddings import get_embedder

logger = logging.getLogger(__name__)

class DocumentIngester:
    """
    Ingests research documents from various formats (PDFs, Web URLs, plain text files),
    chunks them using a token-sliding window, embeds each chunk using text-embedding-3-small,
    and indexes the resulting vectors into a persistent serverless Pinecone database.
    """
    def __init__(self, index_name: Optional[str] = None):
        self.api_key = os.getenv("PINECONE_API_KEY", "")
        self.index_name = index_name or os.getenv("PINECONE_INDEX_NAME", "research-assistant")
        
        # Initialize the chunker and embedder instances
        self.chunker = TokenChunker(chunk_size=512, chunk_overlap=50)
        self.embedder = get_embedder()
        
        # Initialize Pinecone Client
        if self.api_key and not self.api_key.startswith("your_"):
            try:
                self.pc = Pinecone(api_key=self.api_key)
                self._init_index()
            except Exception as e:
                logger.error(f"Failed to initialize Pinecone Client: {str(e)}")
                self.pc = None
        else:
            logger.warning("PINECONE_API_KEY is not configured. Index operations will be mocked or fail.")
            self.pc = None

    def _init_index(self) -> None:
        """
        Initializes the Pinecone serverless index.
        Creates it if it does not already exist.
        """
        if not self.pc:
            return
            
        try:
            # Check if index exists
            active_indexes = [idx.name for idx in self.pc.list_indexes()]
            if self.index_name not in active_indexes:
                logger.info(f"Pinecone index '{self.index_name}' does not exist. Creating serverless index...")
                
                # Standard dimension for Google text-embedding-004 is 768
                self.pc.create_index(
                    name=self.index_name,
                    dimension=768,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                logger.info(f"Index '{self.index_name}' created successfully.")
            else:
                logger.info(f"Pinecone index '{self.index_name}' already exists.")
        except Exception as e:
            logger.error(f"Error checking/creating Pinecone index: {str(e)}")

    def load_pdf(self, file_path: str) -> Dict[str, Any]:
        """
        Extracts raw text content from a local PDF file using PyMuPDF (fitz).
        
        Args:
            file_path (str): Path to the PDF file.
            
        Returns:
            Dict[str, Any]: Document containing raw text and source metadata.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found at: {file_path}")
            
        text = ""
        try:
            doc = fitz.open(file_path)
            for page in doc:
                text += page.get_text()
            doc.close()
        except Exception as e:
            raise RuntimeError(f"Error reading PDF file {file_path}: {str(e)}")
            
        title = os.path.basename(file_path)
        return {
            "text": text,
            "metadata": {
                "source": file_path,
                "title": title,
                "type": "pdf"
            }
        }

    def load_url(self, url: str) -> Dict[str, Any]:
        """
        Fetches web page content and extracts clean text using BeautifulSoup.
        Excludes script, style, head, and navigation sections.
        
        Args:
            url (str): Web URL string.
            
        Returns:
            Dict[str, Any]: Document containing parsed text and source metadata.
        """
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        try:
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
        except Exception as e:
            raise RuntimeError(f"Failed to fetch content from URL {url}: {str(e)}")
            
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Strip script, style, iframe, style, head, footer, nav elements
        for element in soup(["script", "style", "nav", "footer", "header", "noscript"]):
            element.decompose()
            
        # Extract plain text
        raw_text = soup.get_text(separator="\n")
        
        # Clean whitespace
        lines = (line.strip() for line in raw_text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        clean_text = "\n".join(chunk for chunk in chunks if chunk)
        
        parsed_url = urlparse(url)
        title = soup.title.string.strip() if soup.title else parsed_url.netloc
        
        return {
            "text": clean_text,
            "metadata": {
                "source": url,
                "title": title,
                "type": "url"
            }
        }

    def load_text(self, text: str, source_name: str = "raw_text") -> Dict[str, Any]:
        """
        Wraps raw string text in a standardized document dictionary.
        
        Args:
            text (str): Raw input text.
            source_name (str): Label for the source document.
            
        Returns:
            Dict[str, Any]: Standardized document dict.
        """
        return {
            "text": text,
            "metadata": {
                "source": source_name,
                "title": source_name,
                "type": "text"
            }
        }

    def ingest_document(self, doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunks, embeds, and indexes a standard document dict into Pinecone.
        
        Args:
            doc (Dict[str, Any]): Dict containing 'text' and 'metadata'.
            
        Returns:
            List[Dict[str, Any]]: The created document chunks.
        """
        if not doc.get("text", "").strip():
            logger.warning("Empty document text passed to ingestion.")
            return []
            
        # Split text into chunks
        chunks = self.chunker.chunk_document(doc)
        if not chunks:
            return []
            
        # Generate embeddings in batch
        texts_to_embed = [chunk["text"] for chunk in chunks]
        embeddings = self.embedder.embed_documents(texts_to_embed)
        
        # Format vectors for Pinecone upload
        vectors_to_upsert = []
        for idx, chunk in enumerate(chunks):
            chunk["embedding"] = embeddings[idx]
            
            # Create a unique ID for the chunk
            source_sanitized = re.sub(r'[^a-zA-Z0-9_\-]', '_', chunk["metadata"]["source"])
            chunk_id = f"{source_sanitized}_chunk_{chunk['metadata']['chunk_index']}"
            
            # Store raw text directly in metadata so we can read it on query
            metadata_payload = chunk["metadata"].copy()
            metadata_payload["text"] = chunk["text"]
            
            vectors_to_upsert.append((
                chunk_id,
                chunk["embedding"],
                metadata_payload
            ))
            
        # Upsert vectors to Pinecone index
        if self.pc:
            try:
                index = self.pc.Index(self.index_name)
                # Batch upsert in sizes of 100 to avoid Pinecone payload limits
                batch_size = 100
                for i in range(0, len(vectors_to_upsert), batch_size):
                    batch = vectors_to_upsert[i:i + batch_size]
                    index.upsert(vectors=batch)
                logger.info(f"Successfully upserted {len(vectors_to_upsert)} chunks to Pinecone index '{self.index_name}'.")
            except Exception as e:
                logger.warning(f"Error during Pinecone upsert: {str(e)}. Proceeding in demo fallback mode.")
        else:
            logger.warning("Pinecone client not initialized. Skipping database indexing step.")
            
        return chunks
