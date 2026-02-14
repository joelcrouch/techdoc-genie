import json
import sys
from pathlib import Path
from typing import List, Dict, Any
from tabulate import tabulate
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.document_loader import DocumentLoader
from src.ingestion.chunker import DocumentChunker
from src.ingestion.providers.huggingface import HuggingFaceEmbeddingProvider
from src.retrieval.vector_store import VectorStore
from src.agent.rag_chain import RAGChain
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# --- SCRIPT CONFIGURATION ---

# 1. DOCUMENT TO TEST AND VECTOR STORE CONFIG
# PDF_PATH = "/home/dell-linux-dev3/Projects/techdoc-genie/data/raw/ubuntu_docs/ubuntu.pdf"
PDF_PATH = "/home/dell-linux-dev3/Projects/techdoc-genie/data/raw/postgresql/postgresql-16-A4.pdf"
CHUNKING_CONFIG = {
    "strategy_name": "Recursive_512_Overlap_50",
    "chunker": DocumentChunker(chunk_size=512, chunk_overlap=50),
    "chunking_method": "chunk_recursive"
}

# 2. TEST QUERIES
TEST_QUERIES_PATH = "evals/test_queries.json"

# 3. SEMANTIC SIMILARITY MODEL
SIMILARITY_MODEL = 'all-MiniLM-L6-v2'


# --- MAIN EVALUATION SCRIPT ---

def run_rag_evaluation():
    """
    Loads a PDF, builds a vector store using a single chunking strategy,
    runs test queries through the RAG chain, and evaluates the quality of
    the generated answers against a ground truth using semantic similarity.
    """
    logger.info("--- Starting RAG Answer Quality Evaluation ---")

    # --- 1. Load Source Document and Queries ---
    logger.info(f"Loading source document: {PDF_PATH}")
    pdf_path = Path(PDF_PATH)
    if not pdf_path.exists():
        logger.error(f"Document not found at {PDF_PATH}.")
        sys.exit(1)

    loader = DocumentLoader(data_dir=str(pdf_path.parent), doc_format='pdf')
    source_documents = [doc for doc in loader.load_and_prepare() if doc.metadata.get('filename') == pdf_path.name]
    
    if not source_documents:
        logger.error(f"Could not load the specified PDF: {pdf_path.name}.")
        sys.exit(1)

    try:
        with open(TEST_QUERIES_PATH, 'r') as f:
            test_queries = json.load(f)['queries']
        logger.info(f"Loaded {len(test_queries)} test queries from {TEST_QUERIES_PATH}")
    except FileNotFoundError:
        logger.error(f"Test queries file not found at {TEST_QUERIES_PATH}.")
        sys.exit(1)

    # --- 2. Initialize Embedders ---
    logger.info("Initializing HuggingFace embedder for vector store...")
    doc_embedder = HuggingFaceEmbeddingProvider()
    logger.info(f"Initializing SentenceTransformer for similarity scoring: {SIMILARITY_MODEL}...")
    similarity_model = SentenceTransformer(SIMILARITY_MODEL)

    # --- 3. Build Vector Store ---
    strategy_name = CHUNKING_CONFIG['strategy_name']
    chunker = CHUNKING_CONFIG['chunker']
    chunking_method = getattr(chunker, CHUNKING_CONFIG['chunking_method'])
    chunks = chunking_method(source_documents)
    logger.info(f"Created {len(chunks)} chunks using strategy: {strategy_name}")

    logger.info("Creating in-memory vector store...")
    vector_store = VectorStore(embedder=doc_embedder)
    vector_store.create_from_documents(chunks)
    logger.info("In-memory vector store created successfully.")

    # --- 4. Initialize RAG chain ---
    rag_chain = RAGChain(
        vector_store=vector_store,
        llm_provider_type="ollama",
        model_id="phi3:mini"
    )

    # --- 5. Run Queries and Evaluate Answers ---
    evaluation_results = []
    for query_item in test_queries:
        query = query_item['query']
        ground_truth = query_item.get('ground_truth', '')
        
        logger.info(f"Running query: '{query}'")
        
        if not ground_truth:
            logger.warning(f"No ground_truth found for query '{query}'. Skipping similarity score.")
            semantic_similarity = None
        else:
            try:
                # Get RAG answer
                result = rag_chain.query_with_citations(query)
                llm_answer = result.get("answer", "")

                # Calculate semantic similarity
                if llm_answer:
                    embeddings = similarity_model.encode([llm_answer, ground_truth])
                    similarity_score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
                    semantic_similarity = float(similarity_score)
                else:
                    semantic_similarity = 0.0

                evaluation_results.append({
                    "query_id": query_item['id'],
                    "query": query,
                    "ground_truth": ground_truth,
                    "llm_answer": llm_answer,
                    "semantic_similarity_score": semantic_similarity,
                    "num_sources": len(result.get("citations", [])),
                    # Placeholders for manual scoring
                    "manual_relevance_score": None,
                    "manual_groundedness_score": None,
                    "manual_completeness_score": None,
                    "manual_citation_quality_score": None,
                })
                
            except Exception as e:
                logger.error(f"Error during RAG query for '{query}': {e}", exc_info=True)
                semantic_similarity = None

    # --- 6. Display and Save Results ---
    display_and_save_results(evaluation_results)


def display_and_save_results(results: List[Dict[str, Any]]):
    """
    Prints a summary table of the evaluation and saves the full results to a JSON file.
    """
    if not results:
        logger.warning("No evaluation results to display or save.")
        return

    print("\n\n" + "="*120)
    print("--- RAG ANSWER QUALITY EVALUATION SUMMARY ---")
    print("="*120)

    # Display summary table
    headers = ["Query ID", "Query", "Semantic Similarity", "Num Sources"]
    table_data = []
    for res in results:
        table_data.append([
            res['query_id'],
            res['query'][:60] + '...' if len(res['query']) > 60 else res['query'],
            f"{res['semantic_similarity_score']:.4f}" if res['semantic_similarity_score'] is not None else "N/A",
            res['num_sources']
        ])
    
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

    # Save full results to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"evals/results/rag_evaluation_results_{timestamp}.json"
    
    # Create results directory if it doesn't exist
    Path(output_filename).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_filename, 'w') as f:
        json.dump(results, f, indent=4)
        
    logger.info(f"Full evaluation results saved to: {output_filename}")


if __name__ == "__main__":
    run_rag_evaluation()

