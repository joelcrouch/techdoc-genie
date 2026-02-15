"""Test basic RAG functionality."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.vector_store import VectorStore
from src.agent.rag_chain import RAGChain
from src.utils.logger import setup_logger
from src.utils.config import get_settings
from src.ingestion.embedder import DocumentEmbedder # Ensure this is imported

logger = setup_logger(__name__)

def test_basic_query():
    """Test a simple query."""
    settings = get_settings() # Get settings
    embedder = DocumentEmbedder( # Create embedder
        provider=settings.embedding_provider,
        model=settings.embedding_model
    )
    # Load vector store
    vector_store = VectorStore(embedder=embedder.embedder, persist_path=settings.vector_store_path) # Pass embedder and configurable persist_path
    vector_store.load(name="ubuntu_docs_pdf") # Loading ubuntu_docs_pdf as per initial intent
    
    # Create RAG chain
    rag_chain = RAGChain(vector_store)
    
    # Test query
    question = "How do I create an index in PostgreSQL?"
    result = rag_chain.query_with_citations(question)
    
    print(f"\nQuestion: {question}")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nNumber of sources: {result['num_sources']}")
    
    for citation in result['citations']:
        print(f"\nSource {citation['id']}:")
        print(f"  File: {citation['metadata'].get('filename')}")
        print(f"  Content: {citation['content']}")

if __name__ == "__main__":
    test_basic_query()

