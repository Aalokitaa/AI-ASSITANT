import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from app.tools.citation import CitationFormatter
from app.core.agent import ResearchReActAgent

def test_citation_parser():
    """Verify regex correctly parses multiple variations of inline citations."""
    parser = CitationFormatter()
    
    text1 = "AlphaFold 3 makes predictions for all biomolecules [1]. It achieves low root-mean-square deviation [2]."
    assert parser.parse_inline_citations(text1) == [1, 2]
    
    text2 = "Models ESM3 and AlphaFold3 demonstrate state of the art results [1, 3]."
    assert parser.parse_inline_citations(text2) == [1, 3]
    
    text3 = "Comprehensive coverage of proteins and nucleic acids [1-3]."
    assert parser.parse_inline_citations(text3) == [1, 2, 3]

def test_calculate_confidence_score():
    """Verify confidence score algorithm aligns with citation density and source overlap."""
    parser = CitationFormatter()
    
    # Grounding context
    sources = [
        {"text": "AlphaFold 3 is a diffusion model that predicts structural complexes.", "metadata": {"source": "af3.txt"}},
        {"text": "ESM3 generates novel proteins from evolutionary scale models.", "metadata": {"source": "esm3.txt"}}
    ]
    
    # 1. Good grounding (contains citation references and matches word content)
    good_response = "AlphaFold 3 uses a diffusion model [1] to predict structural complexes while ESM3 designs novel proteins [2]."
    score_good = parser.calculate_confidence_score(good_response, sources)
    assert score_good > 0.8
    
    # 2. Hallucinated/uncited response (contains no citations)
    bad_response = "ChatGPT is a chatbot created by OpenAI to answer general text questions."
    score_bad = parser.calculate_confidence_score(bad_response, sources)
    assert score_bad < 0.5

@patch('app.core.agent.ChatGoogleGenerativeAI')
def test_agent_tools_configuration(mock_gemini):
    """Verify that ReAct agent configures the 3 required tools correctly."""
    # Prevent initialization of remote APIs by mocking LLM objects
    mock_llm = MagicMock()
    mock_gemini.return_value = mock_llm
    
    # Initialize agent (forces tools configuration)
    agent_instance = ResearchReActAgent()
    
    tool_names = [tool.name for tool in agent_instance.tools]
    assert "knowledge_base_search" in tool_names
    assert "web_search" in tool_names
    assert "format_citation" in tool_names
    
    assert len(agent_instance.tools) == 3
