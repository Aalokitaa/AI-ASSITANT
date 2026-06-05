# Preemptively import sentence_transformers on Windows to avoid DLL collision / Access Violation (0xC0000005) crashes with pyarrow/torch/fitz load orders.
try:
    import sentence_transformers
except Exception:
    pass

import os
import sys
import logging
from dotenv import load_dotenv

# Ensure app directories are in the system path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Load environmental variables
load_dotenv()

# Configure simple stdout logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("run_pipeline")

from app.core.ingestion import DocumentIngester
from app.core.agent import ResearchReActAgent

def run_end_to_end_demo():
    print("=" * 80)
    print("      AI RESEARCH ASSISTANT: END-TO-END PIPELINE RUNNER (ALL-IN-ONE)")
    print("=" * 80)

    # 1. Ingestion Phase
    print("\n[Step 1/3] Initializing Ingester and Parsing Source Documents...")
    ingester = DocumentIngester()
    
    sample_paper_abstract = (
        "AlphaFold 3 predicts the structure and interactions of proteins, nucleic acids, small molecules, "
        "ions, and chemical modifications. By employing a diffusion-based architecture rather than standard "
        "Multiple Sequence Alignments (MSAs), it predicts coordinates directly for all atoms in a complex. "
        "This results in a 50% improvement in predicting protein-ligand structural interactions compared to "
        "classical molecular docking techniques. It sets a new standard for biological complex simulation."
    )
    
    print("-> Loading and chunking sample paper abstract...")
    doc = ingester.load_text(sample_paper_abstract, source_name="alphafold3_nature_2024")
    
    # Ingest document (generates mock embeddings and handles Pinecone gracefully if keys are invalid)
    chunks = ingester.ingest_document(doc)
    print(f"-> Ingestion Complete. Chunks created: {len(chunks)}")
    
    # 2. Agent Orchestration Phase
    print("\n[Step 2/3] Initializing ReAct Agent and Running Research Reasoning...")
    agent = ResearchReActAgent()
    
    query = "What are the latest breakthroughs in protein folding prediction and how do they compare to AlphaFold2?"
    print(f"-> Executing Query: '{query}'")
    
    # Run agent loop
    result = agent.run(query)
    
    # 3. Output Generation Phase
    print("\n[Step 3/3] Research Dossier Successfully Synthesized!")
    print("=" * 80)
    print("\n=== RESEARCH REPORT (MARKDOWN) ===")
    print(result["answer"])
    
    print("\n" + "=" * 50)
    print(f"Confidence Score: {result['confidence_score'] * 100:.1f}%")
    print("=" * 50)
    
    print("\n=== SOURCES & CITATIONS ===")
    for cit in result["citations"]:
        print(f"[{cit['index']}] {cit['title']} | Source: {cit['source']}")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    run_end_to_end_demo()
