"""Simple interactive interface for querying the RAG system."""
import sys
import argparse
from pathlib import Path

# Add the project root to the Python path  kinda hackish-maybe think about
#  possibly a better way that is more elegant?
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.retrieval.vector_store import VectorStore
from src.agent.rag_chain import RAGChain
from src.agent.citation_manager import ResponseFormatter
from src.utils.logger import setup_logger
from src.utils.config import get_settings  # Import get_settings
from src.ingestion.embedder import DocumentEmbedder # Import DocumentEmbedder

logger = setup_logger(__name__)

def interactive_query(provider: str, model_id: str, prompt_type: str, vector_store_name: str | None):
    """Starts an interactive query session with the specified RAG configuration."""
    logger.info("Loading vector store...")
    
    try:
        # 1. Get settings and create the embedder
        settings = get_settings()
        embedder = DocumentEmbedder(
            provider=settings.embedding_provider,
            model=settings.embedding_model
        )

        # 2. Determine vector store name and path
        if vector_store_name is None:
            # If no name is provided, derive it from the default chunking settings
            vector_store_name = f"vectorstore_chunk{settings.chunk_size}_overlap{settings.chunk_overlap}"
        
        vector_store_dir = f"data/vector_store/{vector_store_name}"

        # 3. Pass the correct path and embedder to the VectorStore
        vector_store = VectorStore(embedder=embedder.embedder, persist_path=vector_store_dir)
        
        # 4. Load the store using its name
        vector_store.load(name=vector_store_name)
        
        logger.info(f"Vector store '{vector_store_name}' loaded successfully from {vector_store_dir}.")
    except Exception as e:
        logger.error(f"Failed to load vector store: {e}", exc_info=True)
        logger.error("Please ensure the vector store exists and its name is correct.")
        return

    logger.info(f"Initializing RAG chain with provider '{provider}' and model '{model_id}'...")
    rag_chain = RAGChain(
        vector_store,
        llm_provider_type=provider,
        model_id=model_id,
        prompt_type=prompt_type,
        retrieval_k=5
    )
    
    print("\n" + "="*80)
    print("Technical Documentation Assistant")
    print(f"Provider: {provider}, Model: {model_id}, Prompt: {prompt_type}")
    print(f"Vector Store: {vector_store_name}")
    print("Type 'exit' or 'quit' to end.")
    print("="*80 + "\n")
    
    while True:
        question = input("\nYour question: ").strip() # noqa
        
        if question.lower() in ['exit', 'quit', 'q']:
            print("Goodbye!")
            break
        
        if not question:
            continue
        
        print("\n🔍 Searching documentation and generating answer...")
        try:
            result = rag_chain.query_with_citations(question)
            formatted_output = ResponseFormatter.format_for_cli(result)
            print(formatted_output)
        except Exception as e:
            logger.error(f"An error occurred during query processing: {e}", exc_info=True)
            print("\n" + "="*80)
            print("Sorry, an error occurred. Please check the logs for more details.")
            print("="*80)

def cli_main():
    """Main function for the interactive CLI."""
    settings = get_settings() # Import and get settings here

    parser = argparse.ArgumentParser(description="Interactive RAG Query Interface")
    parser.add_argument(
        "--provider", 
        type=str, 
        default=settings.llm_default_provider, # Use configurable default
        choices=["ollama", "openai", "claude", "gemini"],
        help="The LLM provider to use."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=settings.llm_default_model_id, # Use configurable default
        help="The model ID to use (e.g., 'phi3:mini', 'gpt-4-turbo-preview', 'claude-3-opus-20240229', 'gemini-pro')."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="base",
        help="The prompt type to use (e.g., 'base', 'detailed', 'troubleshooting')."
    )
    parser.add_argument(
        "--vector-store",
        type=str,
        default=None,
        help="Name of the vector store to load (e.g., 'ubuntu_docs_pdf')."
    )
    args = parser.parse_args()
    
    interactive_query(args.provider, args.model, args.prompt, args.vector_store)

if __name__ == "__main__":
    cli_main()
