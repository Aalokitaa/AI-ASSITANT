import os
import sys
from unittest.mock import MagicMock, patch

# Ensure the root project directory is in PYTHONPATH so we can run tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from app.utils.chunker import TokenChunker
from app.core.ingestion import DocumentIngester

def test_token_chunker_basic_split():
    """Verify sliding window chunker splits short and long text correctly."""
    chunker = TokenChunker(chunk_size=10, chunk_overlap=2)
    text = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen"
    
    chunks = chunker.split_text(text)
    assert len(chunks) > 1
    # Check that text is preserved
    assert "one two three" in chunks[0]

def test_token_chunker_empty_input():
    """Verify chunker outputs empty list for empty/null strings."""
    chunker = TokenChunker()
    assert chunker.split_text("") == []
    assert chunker.split_text("   ") == []

@patch('app.core.ingestion.requests.get')
def test_ingester_load_url(mock_get):
    """Test URL parsing removes script tags and extracts clean content."""
    # Mock response object returning standard HTML markup
    mock_response = MagicMock()
    mock_response.content = b"""
    <html>
        <head><title>Test Article Title</title></head>
        <body>
            <nav>Header Nav Link</nav>
            <h1>Key breakthrough in research</h1>
            <p>This is the actual grounding text body.</p>
            <script>console.log("ignore me");</script>
        </body>
    </html>
    """
    mock_get.return_value = mock_response
    
    ingester = DocumentIngester()
    doc = ingester.load_url("https://example-article.com/bio")
    
    assert doc["metadata"]["title"] == "Test Article Title"
    assert "Key breakthrough" in doc["text"]
    assert "This is the actual grounding text body" in doc["text"]
    # Check that navigation links and scripts were stripped
    assert "Header Nav Link" not in doc["text"]
    assert "console.log" not in doc["text"]

@patch('app.core.ingestion.fitz.open')
def test_ingester_load_pdf(mock_fitz_open):
    """Test PDF text extractor reads file pages correctly."""
    # Mock PDF document and page iterator
    mock_doc = MagicMock()
    mock_page1 = MagicMock()
    mock_page1.get_text.return_value = "Protein folding structural coordinates."
    mock_page2 = MagicMock()
    mock_page2.get_text.return_value = "Calculated via diffusion network model."
    
    mock_doc.__iter__.return_value = [mock_page1, mock_page2]
    mock_fitz_open.return_value = mock_doc
    
    ingester = DocumentIngester()
    
    # We patch os.path.exists to simulate that file exists
    with patch('app.core.ingestion.os.path.exists', return_value=True):
        doc = ingester.load_pdf("/fake/path/paper.pdf")
        
    assert doc["metadata"]["title"] == "paper.pdf"
    assert "coordinates" in doc["text"]
    assert "diffusion network" in doc["text"]
