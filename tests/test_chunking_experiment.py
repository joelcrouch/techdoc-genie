# this si a example for when you actually want to test the llm once its oadedd.
import pytest
from src.ingestion.document_loader import DocumentLoader
from src.ingestion.chunker import DocumentChunker

@pytest.fixture
def all_documents():
    """Load all documents using DocumentLoader."""
    loader = DocumentLoader()
    documents = loader.load_and_prepare()
    assert documents, "No documents loaded"
    return documents

@pytest.mark.parametrize("chunk_size", [256, 512, 1024])
def test_chunking_statistics(all_documents, chunk_size):
    """
    Test recursive chunking for multiple chunk sizes.
    Checks that chunks are produced and lengths make sense.
    """
    chunk_overlap = 50
    chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = chunker.chunk_documents(all_documents, strategy="recursive")

    # Check that some chunks are produced
    assert chunks, f"No chunks produced for chunk_size={chunk_size}"

    # Collect lengths
    chunk_lengths = [len(c.page_content) for c in chunks]
    avg_length = sum(chunk_lengths) / len(chunks)

    # Assertions
    assert all(l > 0 for l in chunk_lengths), "Some chunks are empty"
    assert min(chunk_lengths) <= chunk_size, "Chunk length exceeds chunk_size"
    assert max(chunk_lengths) <= chunk_size + chunk_overlap, "Chunk exceeds allowed overlap"
    assert avg_length > 0, "Average chunk length should be positive"

    # Optional: sanity output
    print(f"\nChunk size: {chunk_size}")
    print(f"  Total chunks: {len(chunks)}")
    print(f"  Avg chunk length: {avg_length:.0f}")
    print(f"  Min/Max chunk length: {min(chunk_lengths)}/{max(chunk_lengths)}")
    print(f"  Sample chunk preview:\n  {chunks[0].page_content[:200]}...")
