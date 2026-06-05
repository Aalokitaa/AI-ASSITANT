from typing import List, Optional
from pydantic import BaseModel, Field

# ==============================================================================
# INGESTION SCHEMAS
# ==============================================================================

class IngestURLRequest(BaseModel):
    """Request model for crawling and ingesting a web page URL."""
    url: str = Field(
        ..., 
        description="The web URL of the page to parse and ingest.",
        examples=["https://nature.com/articles/s41586-024-07487-w"]
    )

class IngestTextRequest(BaseModel):
    """Request model for submitting raw text content directly."""
    text: str = Field(
        ..., 
        description="The raw text body of the document to ingest."
    )
    source_name: str = Field(
        "custom_document", 
        description="A name or label representing this source text.",
        examples=["AlphaFold3_Summary"]
    )

class IngestResponse(BaseModel):
    """Standard response returned after successful document ingestion."""
    status: str = Field(..., description="Ingestion status (e.g., success).", examples=["success"])
    message: str = Field(..., description="Human-readable processing details.")
    chunks_count: int = Field(..., description="Number of 512-token chunks created and indexed.")

# ==============================================================================
# QUERY SCHEMAS
# ==============================================================================

class QueryRequest(BaseModel):
    """Request model for submitting research queries."""
    query: str = Field(
        ..., 
        description="The research question to answer.",
        examples=["What are the latest breakthroughs in protein folding prediction?"]
    )
    use_crew: bool = Field(
        False, 
        description="If True, uses a sequential multi-agent CrewAI crew. If False, runs the standard LangChain ReAct agent."
    )

class Citation(BaseModel):
    """Represents a structured source citation grounding a claim."""
    index: int = Field(..., description="1-indexed indicator corresponding to inline references.")
    title: str = Field(..., description="Document or source article title.")
    source: str = Field(..., description="URL or local path to the original source.")

class QueryResponse(BaseModel):
    """Structured research dossier containing answer, citations, and confidence score."""
    answer: str = Field(
        ..., 
        description="Markdown-formatted comprehensive answer grounded in retrieved/searched facts."
    )
    citations: List[Citation] = Field(
        default_factory=list, 
        description="List of structured citations mapped to inline reference tags."
    )
    confidence_score: float = Field(
        ..., 
        description="Overall source grounding ratio score (between 0.0 and 1.0).",
        examples=[0.92]
    )
