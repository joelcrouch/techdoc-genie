import pytest
from pathlib import Path
from src.ingestion.document_loader import DocumentLoader
from src.ingestion.chunker import DocumentChunker


TEST_FILE = "tests/test_data/admin.html"

@pytest.fixture
def admin_doc():
    """Load just the admin.html document."""
    loader = DocumentLoader(data_dir=Path(TEST_FILE).parent, doc_format="html")
    docs = loader.load_html()
    # Only return the admin.html document
    admin_docs = [doc for doc in docs if doc.metadata.get("filename") == Path(TEST_FILE).name]
    assert admin_docs, "admin.html not loaded"
    return admin_docs

@pytest.mark.parametrize("strategy", ["recursive", "token", "semantic"])
def test_chunking_strategies(admin_doc, strategy):
    """Test all chunking strategies individually on admin.html"""
    chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)  # adjust size as needed
    chunks = chunker.chunk_documents(admin_doc, strategy=strategy)

    # Basic assertions
    assert chunks, f"No chunks produced for strategy {strategy}"
    for i, chunk in enumerate(chunks):
        assert "chunk_id" in chunk.metadata
        assert "chunk_size" in chunk.metadata
        assert chunk.metadata["chunk_id"] == i
        assert chunk.metadata["chunk_size"] == len(chunk.page_content)
    
    # Optional: print a summary
    print(f"\nStrategy: {strategy}, Chunks: {len(chunks)}")
    print(f"First chunk preview:\n{chunks[0].page_content[:200]}...\n")

# @pytest.mark.parametrize("strategy", ["recursive", "token", "semantic"])
# def test_chunk_metadata_and_content(admin_doc, strategy):
#     """Ensure chunks have metadata and combined text equals original document."""
#     chunker = DocumentChunker(chunk_size=200, chunk_overlap=50)
#     chunks = chunker.chunk_documents(admin_doc, strategy=strategy)

#     # Check that we got some chunks
#     assert chunks, f"No chunks produced with {strategy} strategy"

#     # Check metadata
#     for i, chunk in enumerate(chunks):
#         assert "chunk_id" in chunk.metadata
#         assert chunk.metadata["chunk_id"] == i
#         assert "chunk_size" in chunk.metadata
#         assert chunk.metadata["chunk_size"] == len(chunk.page_content)

#     # Check that the concatenated text matches the original (ignoring minor overlaps)
#     combined_text = "".join(chunk.page_content for chunk in chunks)
#     original_text = admin_doc[0].page_content
#     # Allow some overlap; check that all original text is included
#     assert original_text in combined_text or combined_text in original_text