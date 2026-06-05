import os
import shutil
import tempfile
import logging
from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, status

from app.api.schemas import IngestURLRequest, IngestTextRequest, IngestResponse, QueryRequest, QueryResponse, Citation
from app.core.ingestion import DocumentIngester
from app.core.retrieval import DocumentRetriever
from app.core.agent import ResearchReActAgent
from app.core.crew import ResearchCrewOrchestrator
from app.tools.citation import CitationFormatter

logger = logging.getLogger(__name__)

router = APIRouter()

# Global ingester instance (will auto-create Pinecone index on first initialization if keys exist)
def get_ingester() -> DocumentIngester:
    return DocumentIngester()

# Global retriever instance
def get_retriever() -> DocumentRetriever:
    return DocumentRetriever()

# ==============================================================================
# INGESTION ENDPOINTS
# ==============================================================================

@router.post("/ingest/text", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_raw_text(request: IngestTextRequest):
    """
    Ingests and embeds raw text, upserting it into the vector database.
    """
    try:
        ingester = get_ingester()
        doc = ingester.load_text(request.text, request.source_name)
        chunks = ingester.ingest_document(doc)
        
        return IngestResponse(
            status="success",
            message=f"Text ingested successfully from source '{request.source_name}'",
            chunks_count=len(chunks)
        )
    except Exception as e:
        logger.error(f"Text ingestion route failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {str(e)}"
        )

@router.post("/ingest/url", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_url(request: IngestURLRequest):
    """
    Crawls the provided web page URL, parses the text body, embeds, and indexes it.
    """
    try:
        ingester = get_ingester()
        doc = ingester.load_url(request.url)
        chunks = ingester.ingest_document(doc)
        
        return IngestResponse(
            status="success",
            message=f"Web content from '{request.url}' crawled and indexed successfully.",
            chunks_count=len(chunks)
        )
    except Exception as e:
        logger.error(f"URL Ingestion route failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Crawling or indexing failed: {str(e)}"
        )

@router.post("/ingest/pdf", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_pdf(file: UploadFile = File(...)):
    """
    Accepts a binary PDF file, extracts text, chunks, embeds, and indexes it.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must be a PDF."
        )
        
    try:
        ingester = get_ingester()
        
        # Save file to a temporary workspace location to allow PyMuPDF reading
        # We create a temporary directory inside the system temp folder
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
            
        try:
            doc = ingester.load_pdf(tmp_path)
            # Override metadata source to reflect the uploaded filename instead of the temp file path
            doc["metadata"]["source"] = file.filename
            doc["metadata"]["title"] = file.filename
            
            chunks = ingester.ingest_document(doc)
        finally:
            # Assure disk cleanup happens even if ingestion fails
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                
        return IngestResponse(
            status="success",
            message=f"PDF file '{file.filename}' processed and indexed.",
            chunks_count=len(chunks)
        )
    except Exception as e:
        logger.error(f"PDF Ingestion route failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"PDF parsing or indexing failed: {str(e)}"
        )

# ==============================================================================
# QUERY ENDPOINTS
# ==============================================================================

@router.post("/query", response_model=QueryResponse, status_code=status.HTTP_200_OK)
async def query_research_assistant(request: QueryRequest):
    """
    Answers a research query by grounding in local documents (RAG) and web search.
    Supports either standard ReAct Agent or sequential CrewAI flow.
    """
    try:
        if request.use_crew:
            logger.info("Executing research query using multi-agent CrewAI orchestration...")
            
            # Step 1: Retrieve context to seed the crew's knowledge base
            retriever = get_retriever()
            chunks = retriever.retrieve(request.query)
            
            formatted_chunks = []
            combined_sources = []
            for i, chunk in enumerate(chunks):
                src = chunk["metadata"].get("source", "Unknown")
                title = chunk["metadata"].get("title", "Resource")
                text = chunk["text"]
                formatted_chunks.append(f"[Source {i+1}] Title: {title} | Path: {src}\nContent: {text}")
                
                combined_sources.append({
                    "text": text,
                    "metadata": chunk["metadata"]
                })
                
            context_str = "\n\n---\n\n".join(formatted_chunks)
            
            # Step 2: Initialize and execute crew
            orchestrator = ResearchCrewOrchestrator()
            answer_text = orchestrator.run_crew(request.query, context_str)
            
            # Step 3: Compute citations and confidence score post-run
            confidence = CitationFormatter.calculate_confidence_score(answer_text, combined_sources)
            referenced_indices = CitationFormatter.parse_inline_citations(answer_text)
            
            citations = []
            for idx in referenced_indices:
                if 0 <= idx - 1 < len(combined_sources):
                    meta = combined_sources[idx - 1]["metadata"]
                    citations.append(Citation(
                        index=idx,
                        title=meta.get("title", "Source Document"),
                        source=meta.get("source", "N/A")
                    ))
                    
            return QueryResponse(
                answer=answer_text,
                citations=citations,
                confidence_score=confidence
            )
            
        else:
            logger.info("Executing research query using standard LangChain ReAct Agent...")
            agent = ResearchReActAgent()
            result = agent.run(request.query)
            
            citations_payload = []
            for cit in result.get("citations", []):
                citations_payload.append(Citation(
                    index=cit["index"],
                    title=cit["title"],
                    source=cit["source"]
                ))
                
            return QueryResponse(
                answer=result["answer"],
                citations=citations_payload,
                confidence_score=result["confidence_score"]
            )
            
    except Exception as e:
        logger.error(f"Query route failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Research processing encountered an error: {str(e)}"
        )
