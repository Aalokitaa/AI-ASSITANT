import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from app.core.retrieval import DocumentRetriever

@patch('app.core.retrieval.get_embedder')
@patch('app.core.retrieval.Pinecone')
def test_retriever_pipeline(mock_pinecone, mock_get_embedder):
    """Verify retriever calls Pinecone query and executes reranking."""
    
    # 1. Mock embedder to return standard mock vector
    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = [0.1] * 768
    mock_get_embedder.return_value = mock_embedder
    
    # 2. Mock Pinecone client query return payload
    mock_index = MagicMock()
    mock_index.query.return_value = {
        "matches": [
            {
                "id": "chunk_1",
                "score": 0.85,
                "metadata": {
                    "text": "AlphaFold 3 utilizes diffusion module architecture.",
                    "source": "nature_af3.pdf",
                    "title": "AF3 Bio"
                }
            },
            {
                "id": "chunk_2",
                "score": 0.70,
                "metadata": {
                    "text": "AlphaFold 2 used Evoformer blocks to process Multiple Sequence Alignments.",
                    "source": "nature_af2.pdf",
                    "title": "AF2 Bio"
                }
            }
        ]
    }
    
    mock_pinecone_instance = MagicMock()
    mock_pinecone_instance.Index.return_value = mock_index
    mock_pinecone.return_value = mock_pinecone_instance
    
    # 3. Initialize retriever (with mocked environment variables)
    with patch.dict(os.environ, {"PINECONE_API_KEY": "test-key-1234"}):
        retriever = DocumentRetriever(index_name="test-index")
    
    # Mock reranker model to avoid downloading MiniLM in test environment
    retriever.reranker = MagicMock()
    # Mock predict method to return custom rerank scores:
    # Let's say chunk_2 is reranked as highly relevant (score 1.5) and chunk_1 is lower (score 0.5)
    retriever.reranker.predict.return_value = [0.5, 1.5]
    
    # 4. Execute retrieval
    results = retriever.retrieve("AlphaFold2 Evoformer", top_k=2, final_top_k=2)
    
    # Assertions
    assert len(results) == 2
    # Verify reranking reordered results based on prediction scores
    # Since chunk_2 has score 1.5, it should be first in results
    assert "Evoformer" in results[0]["text"]
    assert results[0]["rerank_score"] == 1.5
    assert results[1]["rerank_score"] == 0.5
