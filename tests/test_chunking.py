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



# from pathlib import Path
# from langchain.schema import Document
# from src.ingestion.chunker import DocumentChunker

# def load_test_docs(test_dir="tests/test_data"):
#     """Load test documents into LangChain Document objects."""
#     docs = []
#     for file_path in Path(test_dir).glob("*.txt"):
#         content = file_path.read_text()
#         docs.append(Document(page_content=content, metadata={"source": str(file_path)}))
#     return docs

# def test_chunk_recursive():
#     """Test recursive character chunking."""
#     documents = load_test_docs()
#     chunker = DocumentChunker(chunk_size=50, chunk_overlap=10)  # small for testing
    
#     chunks = chunker.chunk_recursive(documents)
    
#     assert chunks, "No chunks produced"
    
#     # Check metadata not added yet
#     assert all(hasattr(c, "page_content") for c in chunks)
#     assert all(isinstance(c.page_content, str) for c in chunks)
    
#     # Optional: check that chunk length does not exceed chunk_size + overlap
#     for c in chunks:
#         assert len(c.page_content) <= 60, f"Chunk too long: {len(c.page_content)}"