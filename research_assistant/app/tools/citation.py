import re
import logging
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

class CitationFormatter:
    """
    Utility tool to format sources and parse citations from synthesized text.
    Provides methods to clean titles, format URLs, structure source logs,
    and calculate text-to-source alignment/grounding confidence scores.
    """
    
    @staticmethod
    def format_source(index: int, title: str, source_uri: str) -> str:
        """
        Formats a single reference source into a standardized string.
        
        Args:
            index (int): 1-indexed citation number.
            title (str): Clean title of the document or URL.
            source_uri (str): Local path or web URL.
            
        Returns:
            str: E.g., "[1] 'AlphaFold 3 Breakthroughs', https://nature.com/..."
        """
        return f"[{index}] {title} - ({source_uri})"

    @staticmethod
    def parse_inline_citations(text: str) -> List[int]:
        """
        Scans a response body for numeric inline citations like [1], [2], [1, 3] etc.
        
        Args:
            text (str): Synthesized response text.
            
        Returns:
            List[int]: List of unique citation indices referenced in the text.
        """
        if not text:
            return []
            
        # Match pattern [1], [12], [1, 2], [1,2,3]
        matches = re.findall(r'\[([\d\s,\-]+)\]', text)
        indices = set()
        
        for match in matches:
            # Split by comma or hyphen to support [1, 2] or range formats (e.g., [1-3])
            parts = re.split(r'[,]', match)
            for part in parts:
                part = part.strip()
                if part.isdigit():
                    indices.add(int(part))
                elif '-' in part:
                    # Support hyphen range (e.g. 1-3 -> 1, 2, 3)
                    subparts = part.split('-')
                    if len(subparts) == 2 and subparts[0].strip().isdigit() and subparts[1].strip().isdigit():
                        start = int(subparts[0].strip())
                        end = int(subparts[1].strip())
                        for i in range(start, end + 1):
                            indices.add(i)
                            
        return sorted(list(indices))

    @staticmethod
    def calculate_confidence_score(
        text: str, 
        sources: List[Dict[str, Any]]
    ) -> float:
        """
        Calculates a grounding/confidence score (between 0.0 and 1.0)
        based on the proportion of cited statements and semantic overlap.
        
        Formula elements:
        - Presence of citation indices (does the output actually cite sources?)
        - Keyword/N-gram overlap between cited chunks and response.
        - Claims coverage.
        
        Args:
            text (str): The final synthesized response.
            sources (List[Dict[str, Any]]): The retrieved source documents with keys 'text', 'metadata'.
            
        Returns:
            float: Score indicating grounding confidence (0.0 = low, 1.0 = high).
        """
        if not text or not text.strip():
            return 0.0
            
        if not sources:
            return 0.0
            
        # 1. Base Citation Count Check
        cited_indices = CitationFormatter.parse_inline_citations(text)
        citation_penalty = 1.0
        
        # If there are sources but the text doesn't cite any, penalize heavily
        if not cited_indices:
            citation_penalty = 0.3
        elif len(cited_indices) > len(sources):
            # Citing indices that don't exist
            citation_penalty = 0.7
            
        # 2. Text Overlap Verification
        # We check how much token/word vocabulary from the cited sources is present in the response
        source_words = set()
        for idx in cited_indices:
            # Handle 1-indexing to list element mapping
            src_list_idx = idx - 1
            if 0 <= src_list_idx < len(sources):
                src_text = sources[src_list_idx].get("text", "").lower()
                # Simple word tokenization
                words = re.findall(r'\b[a-z]{3,}\b', src_text)
                source_words.update(words)
                
        if not source_words:
            # If nothing was actually cited or source text is empty, fall back to check all sources
            for src in sources:
                src_text = src.get("text", "").lower()
                words = re.findall(r'\b[a-z]{3,}\b', src_text)
                source_words.update(words)
                
        text_words = set(re.findall(r'\b[a-z]{3,}\b', text.lower()))
        
        if not text_words or not source_words:
            return 0.5 * citation_penalty
            
        overlap = text_words.intersection(source_words)
        
        # We calculate the ratio of words in the response that came from the sources
        # We filter out very common words (stop words) to get a more accurate semantic signal
        stop_words = {
            'the', 'and', 'for', 'that', 'with', 'this', 'from', 'have', 'are', 'was', 'were', 'will',
            'should', 'would', 'could', 'about', 'their', 'there', 'they', 'what', 'which', 'who', 'how'
        }
        clean_overlap = overlap - stop_words
        clean_text_words = text_words - stop_words
        
        if not clean_text_words:
            overlap_ratio = 1.0
        else:
            overlap_ratio = len(clean_overlap) / len(clean_text_words)
            
        # Cap overlap ratio at reasonable levels; text is synthesized so it won't be 100% copy-paste
        # Let's scale it so that an overlap ratio of 0.4 corresponds to high grounding
        grounding_score = min(overlap_ratio / 0.4, 1.0)
        
        # Final combined score: 70% grounding/overlap + 30% citation penalty
        final_score = (0.7 * grounding_score) + (0.3 * citation_penalty)
        
        # Bound between 0.1 and 0.99 (rarely 1.0 due to synthesis variance)
        return max(0.1, min(0.99, round(final_score, 2)))
