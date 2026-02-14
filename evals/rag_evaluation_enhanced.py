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
from src.utils.config import get_settings # Import get_settings to access ollama_timeout

logger = setup_logger(__name__)

# --- SCRIPT CONFIGURATION ---
EVALUATION_CONFIGS_PATH = "evals/evaluation_configs.json"
SIMILARITY_MODEL = 'all-MiniLM-L6-v2'


# --- HELPER FUNCTIONS ---
def get_file_name_from_path(file_path: str) -> str:
    """Extracts a clean file name from a given path."""
    return Path(file_path).stem.replace('.', '_').replace('-', '_')

# --- MAIN EVALUATION SCRIPT ---

def run_rag_evaluation_enhanced():
    """
    Loads configuration from evaluation_configs.json, iterates through
    document paths, LLM configurations, and chunking strategies to
    evaluate the quality of RAG responses against ground truth.
    """
    logger.info("--- Starting Enhanced RAG Answer Quality Evaluation ---")

    # --- 1. Load Evaluation Configurations ---
    try:
        with open(EVALUATION_CONFIGS_PATH, 'r') as f:
            eval_configs = json.load(f)
        logger.info(f"Loaded evaluation configurations from {EVALUATION_CONFIGS_PATH}")
    except FileNotFoundError:
        logger.error(f"Evaluation configurations file not found at {EVALUATION_CONFIGS_PATH}.")
        sys.exit(1)

    document_paths = eval_configs.get("document_paths", {})
    test_queries_path = eval_configs.get("test_queries_path", "evals/test_queries.json")
    llm_configs = eval_configs.get("llm_configs", [])
    embedder_config = eval_configs.get("embedder_config", {})
    chunking_strategies = eval_configs.get("chunking_strategies", [])

    if not document_paths or not llm_configs or not chunking_strategies:
        logger.error("Missing essential configurations in evaluation_configs.json (document_paths, llm_configs, or chunking_strategies).")
        sys.exit(1)

    # --- 2. Load Test Queries ---
    try:
        with open(test_queries_path, 'r') as f:
            test_queries = json.load(f)['queries']
        logger.info(f"Loaded {len(test_queries)} test queries from {test_queries_path}")
    except FileNotFoundError:
        logger.error(f"Test queries file not found at {test_queries_path}.")
        sys.exit(1)

    # --- 3. Initialize Embedders ---
    logger.info("Initializing HuggingFace embedder for vector store...")
    # Dynamically create embedder based on config
    if embedder_config.get("provider") == "huggingface":
        doc_embedder = HuggingFaceEmbeddingProvider(model_name=embedder_config.get("model"))
    else:
        logger.error(f"Unsupported embedder provider: {embedder_config.get('provider')}")
        sys.exit(1)

    logger.info(f"Initializing SentenceTransformer for similarity scoring: {SIMILARITY_MODEL}...")
    similarity_model = SentenceTransformer(SIMILARITY_MODEL)

    all_evaluation_results = []

    # --- 4. Iterate Through Documents, LLMs, and Chunking Strategies ---
    for doc_name, doc_path in document_paths.items():
        logger.info(f"\n--- Processing Document: {doc_name} ---")
        pdf_path = Path(doc_path)
        if not pdf_path.exists():
            logger.error(f"Document not found at {doc_path}. Skipping this document.")
            continue

        loader = DocumentLoader(data_dir=str(pdf_path.parent), doc_format='pdf')
        source_documents = [doc for doc in loader.load_and_prepare() if doc.metadata.get('filename') == pdf_path.name]
        
        if not source_documents:
            logger.error(f"Could not load the specified PDF: {pdf_path.name}. Skipping this document.")
            continue

        for llm_config in llm_configs:
            llm_provider = llm_config.get("provider")
            llm_model_id = llm_config.get("model_id")
            logger.info(f"\n--- Testing LLM: {llm_provider}/{llm_model_id} ---")

            for chunk_strategy in chunking_strategies:
                strategy_name = chunk_strategy['name']
                chunk_size = chunk_strategy['chunk_size']
                chunk_overlap = chunk_strategy['chunk_overlap']
                chunk_method_name = chunk_strategy['method']

                logger.info(f"\n--- Evaluating Strategy: {strategy_name} ---")

                # a) Chunk the document
                chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                chunking_method = getattr(chunker, chunk_method_name)
                chunks = chunking_method(source_documents)
                logger.info(f"Created {len(chunks)} chunks using strategy: {strategy_name}")

                # b) Create an in-memory vector store
                logger.info("Creating in-memory vector store...")
                try:
                    vector_store = VectorStore(embedder=doc_embedder)
                    vector_store.create_from_documents(chunks)
                    logger.info("In-memory vector store created successfully.")
                except Exception as e:
                    logger.error(f"Failed to create vector store for strategy '{strategy_name}': {e}", exc_info=True)
                    continue # Skip to next chunking strategy

                # c) Initialize RAG chain
                # Access ollama_timeout from get_settings() if provider is ollama
                llm_settings = get_settings()
                rag_chain = RAGChain(
                    vector_store=vector_store,
                    llm_provider_type=llm_provider,
                    model_id=llm_model_id
                )
                logger.info(f"RAG chain initialized with {llm_provider}/{llm_model_id}")

                # d) Run Queries and Evaluate Answers ---
                experiment_results = []
                for query_item in test_queries:
                    query = query_item['query']
                    ground_truth = query_item.get('ground_truth', '')
                    
                    logger.info(f"Running query: '{query}'")
                    
                    if not ground_truth:
                        logger.warning(f"No ground_truth found for query '{query}'. Skipping similarity score.")
                        semantic_similarity = None
                        llm_answer = "N/A - No ground truth"
                        num_sources = 0
                    else:
                        try:
                            result = rag_chain.query_with_citations(query)
                            llm_answer = result.get("answer", "")

                            if llm_answer:
                                embeddings = similarity_model.encode([llm_answer, ground_truth])
                                similarity_score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
                                semantic_similarity = float(similarity_score)
                            else:
                                semantic_similarity = 0.0 # If no answer, similarity is 0

                            num_sources = len(result.get("citations", []))
                            
                        except Exception as e:
                            logger.error(f"Error during RAG query for '{query}' with {llm_provider}/{llm_model_id} and {strategy_name}: {e}", exc_info=True)
                            llm_answer = f"ERROR: {e}"
                            semantic_similarity = None
                            num_sources = 0

                    experiment_results.append({
                        "document": doc_name,
                        "llm_provider": llm_provider,
                        "llm_model_id": llm_model_id,
                        "chunking_strategy_name": strategy_name,
                        "chunk_size": chunk_size,
                        "chunk_overlap": chunk_overlap,
                        "chunking_method": chunk_method_name,
                        "query_id": query_item['id'],
                        "query": query,
                        "ground_truth": ground_truth,
                        "llm_answer": llm_answer,
                        "semantic_similarity_score": semantic_similarity,
                        "num_sources": num_sources,
                        # Placeholders for manual scoring
                        "manual_relevance_score": None,
                        "manual_groundedness_score": None,
                        "manual_completeness_score": None,
                        "manual_citation_quality_score": None,
                    })
                all_evaluation_results.extend(experiment_results)

    # --- 5. Display and Save Results ---
    display_and_save_all_results(all_evaluation_results, embedder_config)


def display_and_save_all_results(all_results: List[Dict[str, Any]], embedder_config: Dict[str, Any]):
    """
    Prints a summary table of the evaluation and saves the full results to individual JSON files
    for each unique experiment combination.
    """
    if not all_results:
        logger.warning("No evaluation results to display or save.")
        return

    # Group results by experiment (doc, llm, chunking strategy)
    grouped_results: Dict[str, List[Dict[str, Any]]] = {}
    for res in all_results:
        key = f"{res['document']}_{res['llm_provider']}_{res['llm_model_id']}_{res['chunking_strategy_name']}"
        if key not in grouped_results:
            grouped_results[key] = []
        grouped_results[key].append(res)

    print("\n\n" + "="*120)
    print("--- ENHANCED RAG ANSWER QUALITY EVALUATION SUMMARY ---")
    print("="*120)

    overall_summary_table_data = []

    # Display and save results for each experiment
    for key, experiment_results in grouped_results.items():
        doc_name = experiment_results[0]['document']
        llm_provider = experiment_results[0]['llm_provider']
        llm_model_id = experiment_results[0]['llm_model_id']
        strategy_name = experiment_results[0]['chunking_strategy_name']

        logger.info(f"\n--- Results for: Doc={doc_name}, LLM={llm_provider}/{llm_model_id}, Strategy={strategy_name} ---")

        # Calculate average semantic similarity for the experiment
        valid_scores = [r['semantic_similarity_score'] for r in experiment_results if r['semantic_similarity_score'] is not None]
        avg_semantic_similarity = np.mean(valid_scores) if valid_scores else 0.0

        num_queries_with_ground_truth = sum(1 for r in experiment_results if r['ground_truth'])
        num_queries_answered = sum(1 for r in experiment_results if r['llm_answer'] and not r['llm_answer'].startswith("ERROR"))
        
        answer_rate = (num_queries_answered / num_queries_with_ground_truth * 100) if num_queries_with_ground_truth > 0 else 0.0

        overall_summary_table_data.append([
            doc_name,
            f"{llm_provider}/{llm_model_id}",
            strategy_name,
            f"{avg_semantic_similarity:.4f}",
            f"{answer_rate:.2f}%"
        ])

        # Print detailed table for this experiment
        headers = ["Query ID", "Query", "Semantic Similarity", "Num Sources", "Answer"]
        table_data = []
        for res in experiment_results:
            table_data.append([
                res['query_id'],
                res['query'][:50] + '...' if len(res['query']) > 50 else res['query'],
                f"{res['semantic_similarity_score']:.4f}" if res['semantic_similarity_score'] is not None else "N/A",
                res['num_sources'],
                res['llm_answer'][:70] + '...' if len(res['llm_answer']) > 70 else res['llm_answer']
            ])
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        # Save full results for this experiment to JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Ensure 'vectorizer' part of filename is derived, using 'huggingface' for now
        embedder_name_for_filename = embedder_config.get("provider", "unknown_embedder")
        output_filename = f"evals/results/{get_file_name_from_path(doc_name)}_{llm_provider}_{llm_model_id}_{strategy_name}_{embedder_name_for_filename}_results_{timestamp}.json"
        
        Path(output_filename).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_filename, 'w') as f:
            json.dump(experiment_results, f, indent=4)
            
        logger.info(f"Detailed results for experiment '{key}' saved to: {output_filename}")

    print("\n\n" + "="*120)
    print("--- OVERALL EXPERIMENT SUMMARY ---")
    print("="*120)
    overall_headers = ["Document", "LLM", "Chunking Strategy", "Avg Semantic Sim.", "Answer Rate"]
    print(tabulate(overall_summary_table_data, headers=overall_headers, tablefmt="grid"))


if __name__ == "__main__":
    run_rag_evaluation_enhanced()
