import json
import sys
from pathlib import Path
from typing import List, Dict, Any
from tabulate import tabulate
import re

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.document_loader import DocumentLoader
from src.ingestion.chunker import DocumentChunker
from src.ingestion.providers.huggingface import HuggingFaceEmbeddingProvider
from src.retrieval.vector_store import VectorStore
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# --- SCRIPT CONFIGURATION ---

# 1. DOCUMENT TO TEST
PDF_FILE_PATH = "/home/dell-linux-dev3/Projects/techdoc-genie/data/raw/postgresql/postgresql-16-A4.pdf"

# 2. CHUNKING STRATEGIES TO COMPARE
CHUNKING_CONFIGS = [
    {
        "strategy_name": "Recursive_512_Overlap_50",
        "chunker": DocumentChunker(chunk_size=512, chunk_overlap=50),
        "chunking_method": "chunk_recursive"
    },
    {
        "strategy_name": "Recursive_1024_Overlap_100",
        "chunker": DocumentChunker(chunk_size=1024, chunk_overlap=100),
        "chunking_method": "chunk_recursive"
    },
    {
        "strategy_name": "Recursive_2048_Overlap_200",
        "chunker": DocumentChunker(chunk_size=2048, chunk_overlap=200),
        "chunking_method": "chunk_recursive"
    },
    {
        "strategy_name": "Recursive_4096_Overlap_400",
        "chunker": DocumentChunker(chunk_size=4096, chunk_overlap=400),
        "chunking_method": "chunk_recursive"
    },
    {
        "strategy_name": "Semantic_512_Overlap_50",
        "chunker": DocumentChunker(chunk_size=512, chunk_overlap=50),
        "chunking_method": "chunk_semantic"
    },
]

# 3. TEST QUERIES
TEST_QUERIES_PATH = "evals/test_queries.json"
RETRIEVAL_K = 5 # How many documents to retrieve for checking

# --- MAIN EVALUATION SCRIPT ---

def check_keywords(content: str, keywords: List[str]) -> bool:
    """Checks if all keywords are present in the content, case-insensitively."""
    content_lower = content.lower()
    for keyword in keywords:
        # Use regex to find whole words to avoid partial matches (e.g., 'gin' in 'beginning')
        if not re.search(r'\b' + re.escape(keyword.lower()) + r'\b', content_lower):
            return False
    return True


def run_retrieval_evaluation():
    """
    Evaluates the retrieval effectiveness of different chunking strategies
    without involving an LLM.
    """
    logger.info("--- Starting Retrieval-Only Evaluation for Chunking Strategies ---")

    # --- 1. Load Source Document and Queries ---
    logger.info(f"Loading source document: {PDF_FILE_PATH}")
    pdf_path = Path(PDF_FILE_PATH)
    if not pdf_path.exists():
        logger.error(f"Document not found at {PDF_FILE_PATH}. Please update the path.")
        sys.exit(1)
        
    loader = DocumentLoader(data_dir=str(pdf_path.parent), doc_format='pdf')
    loaded_docs = loader.load_and_prepare()
    source_documents = [doc for doc in loaded_docs if doc.metadata.get('filename') == pdf_path.name]
    
    if not source_documents:
        logger.error(f"Could not load the specified PDF: {pdf_path.name}.")
        sys.exit(1)
        
    logger.info("Successfully loaded source document(s).")
    
    try:
        with open(TEST_QUERIES_PATH, 'r') as f:
            test_queries = json.load(f)['queries']
        logger.info(f"Loaded {len(test_queries)} test queries from {TEST_QUERIES_PATH}")
    except FileNotFoundError:
        logger.error(f"Test queries file not found at {TEST_QUERIES_PATH}. Cannot proceed.")
        sys.exit(1)

    # --- 2. Initialize Embedder ---
    logger.info("Initializing HuggingFace embedder...")
    embedder = HuggingFaceEmbeddingProvider()

    # --- 3. Iterate Through Strategies and Evaluate ---
    strategy_scores = []

    for config in CHUNKING_CONFIGS:
        strategy_name = config['strategy_name']
        chunker = config['chunker']
        chunking_method_name = config['chunking_method']
        
        logger.info(f"\n--- Evaluating Strategy: {strategy_name} ---")

        # a) Chunk the document
        chunking_method = getattr(chunker, chunking_method_name)
        chunks = chunking_method(source_documents)
        logger.info(f"Created {len(chunks)} chunks.")

        # b) Create an in-memory vector store
        logger.info("Creating in-memory vector store...")
        vector_store = VectorStore(embedder=embedder)
        vector_store.create_from_documents(chunks)
        
        # c) Run test queries and calculate metrics
        hits = 0
        total_reciprocal_rank = 0.0
        
        for item in test_queries:
            query = item['query']
            keywords = item['context_keywords']
            
            retrieved_docs = vector_store.similarity_search(query, k=RETRIEVAL_K)
            
            found = False
            for i, doc in enumerate(retrieved_docs):
                if check_keywords(doc.page_content, keywords):
                    hits += 1
                    total_reciprocal_rank += 1 / (i + 1)
                    found = True
                    break # Stop at the first relevant document found
        
        # d) Calculate final scores for the strategy
        hit_rate = (hits / len(test_queries)) * 100
        mrr = total_reciprocal_rank / len(test_queries)
        
        strategy_scores.append({
            "Strategy": strategy_name,
            "Num Chunks": len(chunks),
            "Hit Rate (%)": f"{hit_rate:.2f}",
            "MRR": f"{mrr:.3f}"
        })
        logger.info(f"Finished evaluation for {strategy_name}: Hit Rate={hit_rate:.2f}%, MRR={mrr:.3f}")

    # --- 4. Display Results ---
    print("\n\n" + "="*100)
    print("--- RETRIEVAL EVALUATION SUMMARY ---")
    print("="*100)
    
    headers = strategy_scores[0].keys()
    table_data = [s.values() for s in strategy_scores]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    print("\n- Hit Rate: Percentage of queries that returned at least one relevant document.")
    print("- MRR (Mean Reciprocal Rank): A score that considers the rank of the first relevant document. Higher is better.")


if __name__ == "__main__":
    run_retrieval_evaluation()
