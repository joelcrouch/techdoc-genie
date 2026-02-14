import json
import sys
from pathlib import Path
from typing import List, Dict, Any
from tabulate import tabulate
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from tqdm import tqdm
import pickle

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

EVALUATION_CONFIGS_PATH = "evals/evaluation_configs.json" # New: Configuration file path
SIMILARITY_MODEL = 'all-MiniLM-L6-v2'

# 4. PARALLELIZATION CONFIG
MAX_WORKERS = 10  # Adjust based on your CPU cores - start conservative since Ollama is already using resources

# 5. CHECKPOINT CONFIG
CHECKPOINT_DIR = Path("evals/checkpoints")
CHECKPOINT_INTERVAL = 5  # Save checkpoint every N queries


# --- CHECKPOINT MANAGEMENT ---

def save_checkpoint(results: List[Dict], completed_ids: set, checkpoint_file: Path):
    """Save intermediate results to checkpoint file."""
    checkpoint_data = {
        "results": results,
        "completed_ids": list(completed_ids),
        "timestamp": datetime.now().isoformat()
    }
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    logger.info(f"Checkpoint saved: {len(completed_ids)} queries completed")


def load_checkpoint(checkpoint_file: Path) -> tuple:
    """Load results from checkpoint file."""
    if not checkpoint_file.exists():
        return [], set()
    
    with open(checkpoint_file, 'rb') as f:
        checkpoint_data = pickle.load(f)
    
    results = checkpoint_data["results"]
    completed_ids = set(checkpoint_data["completed_ids"])
    logger.info(f"Loaded checkpoint: {len(completed_ids)} queries already completed")
    return results, completed_ids


# --- HELPER FUNCTIONS ---
def get_file_name_from_path(file_path: str) -> str:
    """Extracts a clean file name from a given path."""
    return Path(file_path).stem.replace('.', '_').replace('-', '_')

# --- MAIN EVALUATION SCRIPT ---

def run_rag_evaluation_parallel():
    """
    Loads evaluation configurations, iterates through documents, LLM configs,
    and chunking strategies, running test queries in parallel and evaluating
    answer quality against ground truth with checkpointing.
    """
    logger.info("--- Starting Parallel RAG Answer Quality Evaluation ---")

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
    results_lock = threading.Lock() # For thread-safe appending to all_evaluation_results

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

                # --- a) Build or Load Vector Store (cached per strategy) ---
                vector_store_cache_path = Path("evals/cache") / f"vector_store_{doc_name}_{strategy_name}.pkl"
                vector_store = None
                if vector_store_cache_path.exists():
                    logger.info(f"Loading cached vector store for {strategy_name} from {vector_store_cache_path}")
                    with open(vector_store_cache_path, 'rb') as f:
                        vector_store = pickle.load(f)
                else:
                    chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                    if chunk_method_name == "chunk_semantic":
                        logger.warning("Semantic chunking not yet fully implemented, falling back to recursive.")
                        chunking_method = getattr(chunker, "chunk_recursive") # Fallback for now
                    else:
                        chunking_method = getattr(chunker, chunk_method_name)
                    
                    chunks = chunking_method(source_documents)
                    logger.info(f"Created {len(chunks)} chunks.")

                    logger.info("Creating in-memory vector store...")
                    try:
                        vector_store = VectorStore(embedder=doc_embedder)
                        vector_store.create_from_documents(chunks)
                        logger.info("In-memory vector store created successfully.")
                        # Cache the vector store
                        vector_store_cache_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(vector_store_cache_path, 'wb') as f:
                            pickle.dump(vector_store, f)
                        logger.info(f"Vector store cached to {vector_store_cache_path}")
                    except Exception as e:
                        logger.error(f"Failed to create vector store for strategy '{strategy_name}': {e}", exc_info=True)
                        continue # Skip to next chunking strategy if vector store creation fails

                # --- c) Initialize RAG chain ---
                # Access ollama_timeout from get_settings() if provider is ollama
                rag_chain = RAGChain(
                    vector_store=vector_store,
                    llm_provider_type=llm_provider,
                    model_id=llm_model_id
                )
                logger.info(f"RAG chain initialized with {llm_provider}/{llm_model_id}")

                # --- d) Checkpoint Management for current experiment ---
                experiment_id = f"{doc_name}_{llm_provider}_{llm_model_id}_{strategy_name}"
                checkpoint_file = CHECKPOINT_DIR / f"checkpoint_{experiment_id}.pkl"
                current_experiment_results, completed_ids = load_checkpoint(checkpoint_file)
                
                remaining_queries = [q for q in test_queries if q['id'] not in completed_ids]
                logger.info(f"Queries to process for this experiment: {len(remaining_queries)} (already completed: {len(completed_ids)})")

                if not remaining_queries and current_experiment_results:
                    logger.info("All queries for this experiment already completed. Appending previous results.")
                    all_evaluation_results.extend(current_experiment_results)
                    continue
                elif not remaining_queries:
                     logger.warning("No queries to process for this experiment.")
                     continue
                
                # --- e) Process queries in parallel with progress bar ---
                logger.info(f"Processing {len(remaining_queries)} queries in parallel with {MAX_WORKERS} workers...")
                
                queries_processed_count = [len(completed_ids)] # Mutable counter for checkpointing

                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    futures = {executor.submit(process_single_query, 
                                                query_item, 
                                                rag_chain, 
                                                similarity_model,
                                                doc_name, llm_provider, llm_model_id, strategy_name, chunk_size, chunk_overlap, chunk_method_name
                                                ): query_item for query_item in remaining_queries}
                    
                    with tqdm(total=len(remaining_queries), desc=f"Experiment '{experiment_id}'", unit="query") as pbar:
                        for future in as_completed(futures):
                            result = future.result()
                            
                            with results_lock: # Protect shared list and checkpoint counter
                                all_evaluation_results.append(result)
                                current_experiment_results.append(result)
                                completed_ids.add(result['query_id'])
                                queries_processed_count[0] += 1
                                
                                # Save checkpoint periodically for current experiment
                                if queries_processed_count[0] % CHECKPOINT_INTERVAL == 0:
                                    save_checkpoint(current_experiment_results, completed_ids, checkpoint_file)
                            
                            pbar.update(1)
                
                # Final checkpoint save for the current experiment
                save_checkpoint(current_experiment_results, completed_ids, checkpoint_file)
                # Clean up experiment checkpoint after successful completion
                if checkpoint_file.exists():
                    checkpoint_file.unlink()
                    logger.info(f"Checkpoint file for experiment '{experiment_id}' removed after successful completion.")

    # --- 5. Display and Save All Results ---
    display_and_save_all_results(all_evaluation_results, embedder_config)


def process_single_query(
    query_item: Dict[str, Any], 
    rag_chain: RAGChain, 
    similarity_model: SentenceTransformer, 
    doc_name: str,
    llm_provider: str,
    llm_model_id: str,
    strategy_name: str,
    chunk_size: int,
    chunk_overlap: int,
    chunk_method_name: str
) -> Dict[str, Any]:
    """Processes a single query through the RAG chain and evaluates it."""
    query = query_item['query']
    ground_truth = query_item.get('ground_truth', '')
    
    semantic_similarity = None
    llm_answer = ""
    num_sources = 0

    try:
        if not ground_truth:
            logger.warning(f"No ground_truth found for query '{query_item['id']}'. Skipping similarity score.")
            llm_answer = "N/A - No ground truth"
        else:
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
        logger.error(f"Error during RAG query for '{query}' with {llm_provider}/{llm_model_id} and {strategy_name}: {e}")
        llm_answer = f"ERROR: {str(e)}"
        semantic_similarity = None
        num_sources = 0
        
    return {
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
    }


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
        key = (res['document'], res['llm_provider'], res['llm_model_id'], res['chunking_strategy_name'])
        if key not in grouped_results:
            grouped_results[key] = []
        grouped_results[key].append(res)

    print("\n\n" + "="*120)
    print("--- OVERALL ENHANCED RAG ANSWER QUALITY EVALUATION SUMMARY ---")
    print("="*120)

    overall_summary_table_data = []

    # Display and save results for each experiment
    for key_tuple, experiment_results in grouped_results.items():
        doc_name, llm_provider, llm_model_id, strategy_name = key_tuple

        logger.info(f"\n--- Results for: Doc={doc_name}, LLM={llm_provider}/{llm_model_id}, Strategy={strategy_name} ---")

        # Calculate average semantic similarity for the experiment
        valid_scores = [r['semantic_similarity_score'] for r in experiment_results 
                        if r['semantic_similarity_score'] is not None and not str(r['llm_answer']).startswith("ERROR")] # Filter out errors
        avg_semantic_similarity = np.mean(valid_scores) if valid_scores else 0.0

        num_queries_with_ground_truth = sum(1 for r in experiment_results if r['ground_truth'])
        num_queries_answered = sum(1 for r in experiment_results if r['llm_answer'] and not str(r['llm_answer']).startswith("ERROR"))
        
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
        embedder_name_for_filename = embedder_config.get("provider", "unknown_embedder")
        output_filename = f"evals/results/{get_file_name_from_path(doc_name)}_{llm_provider.replace('/', '_')}_{llm_model_id.replace('/', '_')}_{strategy_name}_{embedder_name_for_filename}_results_{timestamp}.json"
        
        Path(output_filename).parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare output data for JSON
        output_data = {
            "metadata": {
                "timestamp": timestamp,
                "document": doc_name,
                "llm_provider": llm_provider,
                "llm_model_id": llm_model_id,
                "chunking_strategy_name": strategy_name,
                "chunk_size": experiment_results[0]['chunk_size'],
                "chunk_overlap": experiment_results[0]['chunk_overlap'],
                "chunking_method": experiment_results[0]['chunking_method'],
                "embedder_provider": embedder_config.get("provider"),
                "embedder_model": embedder_config.get("model"),
                "total_queries_evaluated": len(experiment_results),
                "average_semantic_similarity": float(avg_semantic_similarity),
                "answer_rate": float(answer_rate)
            },
            "results": experiment_results
        }

        with open(output_filename, 'w') as f:
            json.dump(output_data, f, indent=4)
            
        logger.info(f"Detailed results for experiment 'Doc={doc_name}, LLM={llm_provider}/{llm_model_id}, Strategy={strategy_name}' saved to: {output_filename}")

    print("\n\n" + "="*120)
    print("--- OVERALL EXPERIMENT SUMMARY ---")
    print("="*120)
    overall_headers = ["Document", "LLM", "Chunking Strategy", "Avg Semantic Sim.", "Answer Rate"]
    print(tabulate(overall_summary_table_data, headers=overall_headers, tablefmt="grid"))


if __name__ == "__main__":
    run_rag_evaluation_parallel()
