import pytest
from pathlib import Path
from langchain.schema import Document
from src.ingestion.chunker import DocumentChunker
from src.ingestion.document_loader import DocumentLoader

TEST_DATA_DIR = Path(__file__).parent / "test_data"

def load_test_docs() -> list[Document]:
    """Load HTML test documents using the updated loader."""
    loader = DocumentLoader(data_dir=TEST_DATA_DIR, doc_format="html")
    return loader.load_html()


def test_html_loader_produces_text():
    """Verify HTML loader strips tags but keeps headings/paragraphs."""
    documents = load_test_docs()
    assert documents, "No documents loaded from HTML"

    # Pick the first document for checks
    doc = documents[0]
    content = doc.page_content
    assert isinstance(content, str)
    assert len(content) > 50, "Document content is suspiciously short"

    # Check that heading markers exist
    assert "#" in content or "##" in content, "Headings not preserved in text"

    print(content[:500])  # Optional: inspect first 500 chars


def test_chunk_recursive():
    """Test recursive character chunking on loaded HTML documents."""
    documents = load_test_docs()
    chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)  # reasonable small size for test

    chunks = chunker.chunk_recursive(documents)

    assert chunks, "No chunks produced"
    for i, chunk in enumerate(chunks):
        assert isinstance(chunk.page_content, str)
        assert len(chunk.page_content) <= 550, "Chunk exceeds expected size"

    print(f"Created {len(chunks)} chunks")

