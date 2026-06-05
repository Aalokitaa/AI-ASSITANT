import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment variables from .env file before anything else runs
load_dotenv()

# Configure global system logging
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("app.main")

# Import routers
from app.api.routes import router as api_router

def create_app() -> FastAPI:
    """
    Application factory pattern. Sets up FastAPI, logs, middleware, and routers.
    """
    app = FastAPI(
        title="AI Research Assistant API",
        description=(
            "An enterprise-ready RAG API designed to ingest research documents, "
            "synthesize comparative analyses with Claude 3.5 Sonnet, fact-check claims, "
            "and output cited academic dossiers."
        ),
        version="1.0.0"
    )
    
    # Enable CORS (Cross-Origin Resource Sharing)
    # Required if building a web application or fronting the service with a client dashboard
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Mount routing layers
    app.include_router(api_router, prefix="/api/v1", tags=["Research System"])
    
    # Simple Health Check route for deployment environments (Docker/Kubernetes/Railway)
    @app.get("/health", tags=["Monitoring"])
    def health_check():
        """
        Retrieves status of API components for automated uptime checks.
        """
        # Check Pinecone & API keys initialization status
        pinecone_configured = bool(os.getenv("PINECONE_API_KEY"))
        openai_configured = bool(os.getenv("OPENAI_API_KEY"))
        anthropic_configured = bool(os.getenv("ANTHROPIC_API_KEY"))
        
        return {
            "status": "healthy",
            "environment": {
                "pinecone": "configured" if pinecone_configured else "missing",
                "openai": "configured" if openai_configured else "missing",
                "anthropic": "configured" if anthropic_configured else "missing",
            }
        }
        
    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    # Read environment configs
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    
    logger.info(f"Starting uvicorn server on http://{host}:{port}")
    uvicorn.run("app.main:app", host=host, port=port, reload=False)
