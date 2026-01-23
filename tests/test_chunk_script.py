import pytest
from scripts.chunk_test import main  # import the main function
from src.ingestion.document_loader import DocumentLoader

@pytest.fixture
def sample_docs():
    """Load a small set of documents for testing."""
    loader = DocumentLoader(data_dir="tests/test_data", doc_format="html")
    docs = loader.load_html()
    assert docs, "No test documents loaded"
    return docs

def test_chunk_script_runs(monkeypatch, sample_docs):
    """Test the chunk_test script logic on sample documents."""

    # We'll monkeypatch DocumentLoader.load_and_prepare to return our test docs
    from scripts.chunk_test import DocumentLoader as ChunkLoader
    monkeypatch.setattr(ChunkLoader, "load_and_prepare", lambda self: sample_docs)

    # Test with a few chunk sizes
    chunk_sizes = [128, 256]
    
    # Call main
    main(chunk_sizes, chunk_overlap=20, strategy="recursive")
    
    # Now validate chunks
    from src.ingestion.chunker import DocumentChunker
    chunker = DocumentChunker(chunk_size=128, chunk_overlap=20)
    chunks = chunker.chunk_documents(sample_docs, strategy="recursive")
    
    # Check we got chunks
    assert chunks, "No chunks produced"
    
    # Check metadata
    for i, chunk in enumerate(chunks):
        assert "chunk_id" in chunk.metadata
        assert chunk.metadata["chunk_id"] == i
        assert "chunk_size" in chunk.metadata
        assert chunk.metadata["chunk_size"] == len(chunk.page_content)
    
    # Check that chunk lengths are reasonable
    lengths = [len(c.page_content) for c in chunks]
    assert all(l > 0 for l in lengths), "Some chunks have zero length"
