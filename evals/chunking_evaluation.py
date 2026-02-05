import json
import sys
from pathlib import Path
from typing import List, Dict, Any
from tabulate import tabulate

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

# 1. DOCUMENT TO TEST
# You can change this to the path of any monolithic PDF you want to evaluate.
# PDF_PATH = "/home/dell-linux-dev3/Projects/techdoc-genie/data/raw/ubuntu_docs/ubuntu.pdf"
PDF_PATH="/home/dell-linux-dev3/Projects/techdoc-genie/data/raw/postgresql/postgresql-16-A4.pdf"
# 2. CHUNKING STRATEGIES TO COMPARE
# A list of dictionaries, each defining a chunking configuration.
CHUNKING_CONFIGS = [
    {
        "strategy_name": "Recursive_256_Overlap_30",
        "chunker": DocumentChunker(chunk_size=256, chunk_overlap=30),
        "chunking_method": "chunk_recursive"
    },
    {
        "strategy_name": "Recursive_512_Overlap_50",
        "chunker": DocumentChunker(chunk_size=512, chunk_overlap=50),
        "chunking_method": "chunk_recursive"
    },
    {
        "strategy_name": "Semantic_512_Overlap_50",
        "chunker": DocumentChunker(chunk_size=512, chunk_overlap=50),
        "chunking_method": "chunk_semantic"
    },
]

# 3. TEST QUERIES
# Path to the JSON file containing evaluation questions.
TEST_QUERIES_PATH = "evals/test_queries.json"

# --- MAIN EVALUATION SCRIPT ---

def run_chunking_evaluation():
    """
    Loads a single PDF, processes it with different chunking strategies,
    and evaluates the quality of RAG responses for each strategy.
    """
    logger.info("--- Starting Chunking Strategy Evaluation ---")

    # --- 1. Load Source Document and Queries ---
    logger.info(f"Loading source document: {PDF_PATH}")
    pdf_path = Path(PDF_PATH)
    if not pdf_path.exists():
        logger.error(f"Document not found at {PDF_PATH}. Please update the PDF_PATH variable in this script.")
        sys.exit(1)
        
    # Use DocumentLoader to load just one file
    loader = DocumentLoader(data_dir=str(pdf_path.parent), doc_format='pdf')
    # We need to manually filter to our specific document if there are others
    source_documents = [doc for doc in loader.load_and_prepare() if doc.metadata.get('filename') == pdf_path.name]
    
    if not source_documents:
        logger.error(f"Could not load the specified PDF: {pdf_path.name}. Check the filename and path.")
        sys.exit(1)
        
    logger.info(f"Successfully loaded {len(source_documents)} source document(s).")
    
    try:
        with open(TEST_QUERIES_PATH, 'r') as f:
            test_queries = [q['query'] for q in json.load(f)['queries']]
        logger.info(f"Loaded {len(test_queries)} test queries from {TEST_QUERIES_PATH}")
    except FileNotFoundError:
        logger.error(f"Test queries file not found at {TEST_QUERIES_PATH}. Cannot proceed.")
        sys.exit(1)

    # --- 2. Initialize Embedder (once) ---
    logger.info("Initializing HuggingFace embedder...")
    embedder = HuggingFaceEmbeddingProvider()

    # --- 3. Iterate Through Strategies and Evaluate ---
    evaluation_results = []

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
        try:
            # We don't specify `persist_path` to keep it in memory
            vector_store = VectorStore(embedder=embedder)
            vector_store.create_from_documents(chunks)
            logger.info("In-memory vector store created successfully.")
        except Exception as e:
            logger.error(f"Failed to create vector store for strategy '{strategy_name}': {e}", exc_info=True)
            continue # Skip to next strategy

        # c) Initialize RAG chain
        rag_chain = RAGChain(
            vector_store=vector_store,
            llm_provider_type="ollama",
            model_id="phi3:mini"
        )
        
        # d) Run test queries
        for query in test_queries:
            logger.info(f"Running query: '{query}'")
            try:
                result = rag_chain.query_with_citations(query)
                evaluation_results.append({
                    "strategy": strategy_name,
                    "num_chunks": len(chunks),
                    "query": query,
                    "answer": result.get("answer", "NO ANSWER"),
                    "num_sources": len(result.get("citations", []))
                })
            except Exception as e:
                logger.error(f"Error during query for strategy '{strategy_name}': {e}", exc_info=True)
                evaluation_results.append({
                    "strategy": strategy_name,
                    "num_chunks": len(chunks),
                    "query": query,
                    "answer": f"ERROR: {e}",
                    "num_sources": 0
                })

    # --- 4. Display Results ---
    display_results(evaluation_results)


def display_results(results: List[Dict[str, Any]]):
    """
    Prints the evaluation results in a readable format, grouped by query.
    """
    print("\n\n" + "="*100)
    print("--- CHUNKING STRATEGY EVALUATION RESULTS ---")
    print("="*100)

    # Group results by query
    results_by_query = {}
    for res in results:
        if res['query'] not in results_by_query:
            results_by_query[res['query']] = []
        results_by_query[res['query']].append(res)
        
    # Print comparison for each query
    for query, query_results in results_by_query.items():
        print("\n\n" + "-"*100)
        print(f"QUERY: {query}")
        print("-"*100)
        
        for res in sorted(query_results, key=lambda x: x['strategy']):
            print(f"\n>> STRATEGY: {res['strategy']} (Chunks: {res['num_chunks']}, Sources: {res['num_sources']})")
            print("   ANSWER: " + res['answer'].strip().replace('\n', ' '))
            
    # Print summary table
    print("\n\n" + "="*100)
    print("SUMMARY TABLE")
    print("="*100)
    
    headers = ["Strategy", "Num Chunks", "Avg Sources Retrieved"]
    table_data = []
    for config in CHUNKING_CONFIGS:
        strategy_name = config['strategy_name']
        strategy_results = [res for res in results if res['strategy'] == strategy_name]
        if not strategy_results:
            continue
            
        num_chunks = strategy_results[0]['num_chunks']
        avg_sources = sum(res['num_sources'] for res in strategy_results) / len(strategy_results)
        table_data.append([strategy_name, num_chunks, f"{avg_sources:.1f}"])
        
    print(tabulate(table_data, headers=headers, tablefmt="grid"))


if __name__ == "__main__":
    run_chunking_evaluation()
