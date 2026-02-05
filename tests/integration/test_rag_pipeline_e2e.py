import pytest
import sys
from pathlib import Path

# Add the project root to the Python path to allow for absolute imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.agent.rag_chain import RAGChain
from src.retrieval.vector_store import VectorStore
from src.ingestion.embedder import DocumentEmbedder
from src.utils.config import get_settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# --- Test Configuration ---
# The name of the vector store created by the default build script
POSTGRES_VECTOR_STORE_NAME = "vectorstore_chunk512_overlap50" 
TEST_QUESTION = "How do I create an index in PostgreSQL?"


@pytest.fixture(scope="module")
def rag_chain_local_llm():
    """
    Pytest fixture to set up the RAG chain with a local LLM.
    This runs only once per test module.
    """
    try:
        logger.info("Setting up RAG chain for integration test...")
        
        settings = get_settings()
        embedder = DocumentEmbedder(
            provider=settings.embedding_provider, 
            model=settings.embedding_model
        )
        
        # Construct path and name for the postgres vector store
        vector_store_dir = f"data/vector_store/{POSTGRES_VECTOR_STORE_NAME}"
        
        vector_store = VectorStore(
            embedder=embedder.embedder, 
            persist_path=vector_store_dir
        )
        vector_store.load(name=POSTGRES_VECTOR_STORE_NAME)
        
        logger.info(f"Successfully loaded vector store: {POSTGRES_VECTOR_STORE_NAME}")
        
        # Initialize RAG chain with local Ollama provider
        chain = RAGChain(
            vector_store=vector_store,
            llm_provider_type="ollama",
            model_id="phi3:mini"
        )
        logger.info("RAG chain setup complete.")
        return chain
    except Exception as e:
        logger.error(f"Failed to set up RAG chain fixture: {e}", exc_info=True)
        # Skip all tests in this module if the setup fails
        pytest.fail(f"Failed to initialize RAG chain fixture: {e}")


def test_e2e_query_local_llm(rag_chain_local_llm: RAGChain):
    """
    Tests a full end-to-end query to the RAG pipeline using a local LLM.
    
    This test:
    1. Uses the pre-configured RAG chain.
    2. Asks a question about creating an index.
    3. Asserts that the response is valid and contains expected content.
    """
    logger.info(f"Running end-to-end test with question: '{TEST_QUESTION}'")
    
    # 1. Ask the question
    result = rag_chain_local_llm.query_with_citations(TEST_QUESTION)
    
    # 2. Log the result for debugging
    logger.info(f"Received answer: {result.get('answer')}")
    logger.info(f"Received {len(result.get('citations', []))} citations.")

    # 3. Assert the response is valid
    assert result is not None, "The result from the RAG chain should not be None."
    
    answer = result.get("answer")
    assert answer and isinstance(answer, str), "The answer should be a non-empty string."
    
    # 4. Assert the content of the response is relevant
    # Check for keywords that should appear in an answer about creating an index
    expected_keywords = ["CREATE", "INDEX"]
    for keyword in expected_keywords:
        assert keyword.lower() in answer.lower(), f"Answer should contain the keyword '{keyword}'"
        
    # 5. Assert that citations are present
    citations = result.get("citations")
    assert citations is not None, "Citations should be present in the result."
    assert isinstance(citations, list), "Citations should be a list."
    assert len(citations) > 0, "There should be at least one citation."
    
    logger.info("End-to-end test passed successfully.")

