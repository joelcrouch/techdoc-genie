from src.ingestion.document_loader import DocumentLoader
from src.ingestion.chunker import DocumentChunker
from src.ingestion.providers.huggingface import HuggingFaceEmbeddingProvider
from src.retrieval.vector_store import VectorStore
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def test_ingestion_pipeline():
    # Step 1: Load documents
    test_file_dir = "data/raw/postgresql"
    loader = DocumentLoader(data_dir=test_file_dir, doc_format="html")
    documents = loader.load_html()
    documents = [doc for doc in documents if "tutorial-sql.html" in doc.metadata["filename"]]
    assert len(documents) > 0, "No documents loaded"
    assert "tutorial-sql.html" in documents[0].metadata["filename"]

    # Step 2: Chunk documents
    chunker = DocumentChunker(chunk_size=512, chunk_overlap=50)
    chunks = chunker.chunk_documents(documents, strategy="recursive")
    assert len(chunks) > 0, "No chunks created"

    # Step 3: Initialize embedder
    embedder = HuggingFaceEmbeddingProvider()
    assert embedder is not None, "Embedder not initialized"

    # Step 4: Create vector store in memory (no saving)
    vector_store = VectorStore(embedder=embedder, persist_path=None)
    vector_store.create_from_documents(chunks)
    assert vector_store is not None, "Vector store not created"

    # Step 5: Test similarity search
    query = "How do I create a table in PostgreSQL?"
    results = vector_store.similarity_search(query, k=3)
    assert len(results) > 0, "Similarity search returned no results"
    assert all(isinstance(r.page_content, str) for r in results), "Similarity search results contain non-string content"

