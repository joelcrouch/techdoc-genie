import json
import sys
from pathlib import Path
from typing import List, Dict, Any
import pickle

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.document_loader import DocumentLoader
from src.ingestion.chunker import DocumentChunker
from src.ingestion.providers.huggingface import HuggingFaceEmbeddingProvider
from src.retrieval.vector_store import VectorStore
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# --- SCRIPT CONFIGURATION ---
EVALUATION_CONFIGS_PATH = "evals/evaluation_configs.json"
CHECKPOINT_DIR = Path("evals/checkpoints")
CACHE_DIR = Path("evals/cache")

# --- HELPER FUNCTIONS ---
def get_file_name_from_path(file_path: str) -> str:
    """Extracts a clean file name from a given path."""
    return Path(file_path).stem.replace('.', '_').replace('-', '_')

def build_all_vector_stores_from_config():
    """
    Reads evaluation configurations and builds/caches all specified vector stores.
    """
    logger.info("--- Starting Vector Store Pre-building ---")

    # --- 1. Load Evaluation Configurations ---
    try:
        with open(EVALUATION_CONFIGS_PATH, 'r') as f:
            eval_configs = json.load(f)
        logger.info(f"Loaded evaluation configurations from {EVALUATION_CONFIGS_PATH}")
    except FileNotFoundError:
        logger.error(f"Evaluation configurations file not found at {EVALUATION_CONFIGS_PATH}.")
        sys.exit(1)

    document_paths = eval_configs.get("document_paths", {})
    embedder_config = eval_configs.get("embedder_config", {})
    chunking_strategies = eval_configs.get("chunking_strategies", [])

    if not document_paths or not chunking_strategies:
        logger.error("Missing essential configurations in evaluation_configs.json (document_paths or chunking_strategies).")
        sys.exit(1)

    # --- 2. Initialize Embedder ---
    logger.info("Initializing HuggingFace embedder for vector store...")
    if embedder_config.get("provider") == "huggingface":
        doc_embedder_provider = HuggingFaceEmbeddingProvider(model_name=embedder_config.get("model"))
        doc_embedder = doc_embedder_provider.model # Get the actual embedder instance
    else:
        logger.error(f"Unsupported embedder provider: {embedder_config.get('provider')}")
        sys.exit(1)
    
    CACHE_DIR.mkdir(parents=True, exist_ok=True) # Ensure cache directory exists

    # --- 3. Iterate Through Documents and Chunking Strategies ---
    for doc_name, doc_path in document_paths.items():
        logger.info(f"--- Processing Document: {doc_name} ---")
        pdf_path = Path(doc_path)
        if not pdf_path.exists():
            logger.error(f"Document not found at {doc_path}. Skipping this document.")
            continue

        loader = DocumentLoader(data_dir=str(pdf_path.parent), doc_format='pdf')
        source_documents = [doc for doc in loader.load_and_prepare() if doc.metadata.get('filename') == pdf_path.name]
        
        if not source_documents:
            logger.error(f"Could not load the specified PDF: {pdf_path.name}. Skipping this document.")
            continue

        for chunk_strategy in chunking_strategies:
            strategy_name = chunk_strategy['name']
            chunk_size = chunk_strategy['chunk_size']
            chunk_overlap = chunk_strategy['chunk_overlap']
            chunk_method_name = chunk_strategy['method']

            logger.info(f"--- Building Vector Store for Strategy: {strategy_name} (Document: {doc_name}) ---")

            vector_store_cache_path = CACHE_DIR / f"vector_store_{get_file_name_from_path(doc_name)}_{strategy_name}.pkl"
            
            if vector_store_cache_path.exists():
                logger.info(f"Cached vector store already exists for {strategy_name}. Skipping creation.")
                continue

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
                vector_store = VectorStore(embedder=doc_embedder) # Use doc_embedder here
                vector_store.create_from_documents(chunks)
                logger.info("In-memory vector store created successfully.")
                
                with open(vector_store_cache_path, 'wb') as f:
                    pickle.dump(vector_store, f)
                logger.info(f"Vector store cached to {vector_store_cache_path}")
            except Exception as e:
                logger.error(f"Failed to create vector store for strategy '{strategy_name}' and document '{doc_name}': {e}", exc_info=True)
                # Do NOT sys.exit(1) here, allow other vector stores to be built.
                continue 
    
    logger.info("--- Vector Store Pre-building Completed ---")


if __name__ == "__main__":
    build_all_vector_stores_from_config()