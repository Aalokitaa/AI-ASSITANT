# AI Research Assistant: Autonomous RAG & Multi-Agent Co-Pilot

> An enterprise-grade, high-performance RAG pipeline and multi-agent system designed to automate academic synthesis, facts validation, and citation parsing.

---

## System Architecture

The project features a **two-stage retrieval pipeline** integrated with a **multi-agent orchestration team** powered by Claude 3.5 Sonnet and GPT-4o.

```
+-------------------------------------------------------------+
|                     USER INTERFACE LAYER                    |
|                        FastAPI / REST                       |
+------------------------------+------------------------------+
                               |
            ┌──────────────────┴──────────────────┐
            ▼ (Ingestion Path)                    ▼ (Query Path)
+------------------------------+  +---------------------------+
|      INGESTION PIPELINE      |  |      RETRIEVAL LAYER      |
|  - URL Loader (BS4)          |  |  - Query Embedder         |
|  - PDF Loader (PyMuPDF)      |  |  - Pinecone Search (k=8)  |
|  - Token Chunker (512/50)    |  |  - MiniLM Reranking (k=3) |
|  - Pinecone Serverless       |  +-------------+-------------+
+------------------------------+                |
                                                ▼ (Context Chunks)
+-------------------------------------------------------------+
|                  AGENT ORCHESTRATION LAYER                  |
|          LangChain ReAct / CrewAI Sequential Flow           |
|                                                             |
|   +-----------------------+     +-----------------------+   |
|   |     ResearchAgent     | --> |   FactCheckerAgent    |   |
|   |  (Claude 3.5 Sonnet)  |     |       (GPT-4o)        |   |
|   +-----------+-----------+     +-----------+-----------+   |
|               |                             |               |
|               ▼ (Web/KB Tools)              ▼               |
|   +-----------+-----------+     +-----------+-----------+   |
|   |      Tool Layer       |     |     CitationAgent     |   |
|   |  - Web Search (Tavily)|     |       (GPT-4o)        |   |
|   +-----------------------+     +-----------+-----------+   |
|                                             |               |
|                                             ▼               |
|   +-----------------------------------------------------+   |
|   |                    Output Layer                     |   |
|   |     Structured JSON with Citations & Confidence     |   |
|   +-----------------------------------------------------+   |
+-------------------------------------------------------------+
```

---

## Directory Structure

```
research_assistant/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── api/
│   │   ├── routes.py
│   │   └── schemas.py
│   ├── core/
│   │   ├── ingestion.py
│   │   ├── retrieval.py
│   │   ├── agent.py
│   │   └── crew.py
│   ├── tools/
│   │   ├── web_search.py
│   │   └── citation.py
│   └── utils/
│       ├── embeddings.py
│       └── chunker.py
├── tests/
│   ├── test_ingestion.py
│   ├── test_retrieval.py
│   └── test_agent.py
├── notebooks/
│   └── demo.ipynb
├── requirements.txt
├── .env.example
├── Dockerfile
└── README.md
```

---

## Configuration Reference

Create a `.env` file in the root directory:

```env
# Core API Keys
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here

# Vector DB Configuration
PINECONE_API_KEY=your_pinecone_key_here
PINECONE_INDEX_NAME=research-assistant

# Web Search Integration
TAVILY_API_KEY=your_tavily_key_here

# Server Settings
PORT=8000
HOST=0.0.0.0
LOG_LEVEL=info
```

---

## Quick Start Guide

### 1. Local Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Start FastAPI locally
python -m app.main
```
The server will start at `http://localhost:8000`. Documentation will be available at `http://localhost:8000/docs`.

### 2. Run Tests
```bash
pytest tests/
```

### 3. Interactive Notebook
Launch Jupyter Notebook and open [demo.ipynb](file:///c:/Users/chibb/OneDrive/Desktop/My%20Projects/AI%20ASSISTANT/research_assistant/notebooks/demo.ipynb) to walk through the pipeline step-by-step.

---

## API Reference

### 1. Ingest URL
* **Endpoint**: `POST /api/v1/ingest/url`
* **Request**:
```bash
curl -X POST "http://localhost:8000/api/v1/ingest/url" \
     -H "Content-Type: application/json" \
     -d '{"url": "https://nature.com/articles/s41586-024-07487-w"}'
```
* **Response**:
```json
{
  "status": "success",
  "message": "Web content from 'https://nature.com/...' crawled and indexed successfully.",
  "chunks_count": 14
}
```

### 2. Ingest PDF Document
* **Endpoint**: `POST /api/v1/ingest/pdf`
* **Request**:
```bash
curl -X POST "http://localhost:8000/api/v1/ingest/pdf" \
     -F "file=@/path/to/paper.pdf"
```

### 3. Submit Research Query
* **Endpoint**: `POST /api/v1/query`
* **Request**:
```bash
curl -X POST "http://localhost:8000/api/v1/query" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "What are the latest breakthroughs in protein folding prediction and how do they compare to AlphaFold2?",
       "use_crew": false
     }'
```
* **Response**:
```json
{
  "answer": "Protein folding prediction breakthroughs are highlighted by AlphaFold 3 (2024)... [1]. Comparing AF3 and AF2 [2]...",
  "citations": [
    {
      "index": 1,
      "title": "Accurate structure prediction of biomolecular interactions with AlphaFold 3",
      "source": "https://nature.com/articles/s41586-024-07487-w"
    }
  ],
  "confidence_score": 0.94
}
```

---

## Deployment Guide

### Deploying with Docker

1. **Build container**:
```bash
docker build -t research-assistant .
```

2. **Run container locally**:
```bash
docker run -p 8000:8000 --env-file .env research-assistant
```

### Deploying on Railway

1. Install Railway CLI and login (`railway login`).
2. Initialize project (`railway init`).
3. Set your variables in Railway console corresponding to `.env`.
4. Deploy:
```bash
railway up
```
Railway will automatically detect the `Dockerfile`, spin up a container, expose port 8000, and execute the health check.

---

## License

Distributed under the MIT License. See `LICENSE` for details.
