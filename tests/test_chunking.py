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

# A small section of a dummy HTML file for testing
DUMMY_HTML_CONTENT_CHUNK = """
<!DOCTYPE html>
<html>
<head>
    <title>Dummy Page for Chunking</title>
    <style>body { font-family: sans-serif; }</style>
</head>
<body>
    <h1>Main Title for Chunking Test</h1>
    <p>This is a paragraph for the chunking test. It should be long enough to pass the length check.</p>
    <h2>Subtitle for Chunking Test</h2>
    <ul>
        <li>Item 1 for chunking</li>
        <li>Item 2 for chunking</li>
        <li>Item 3 for chunking, making sure the content is substantial.</li>
    </ul>
    <p>Another paragraph to ensure sufficient content length. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.</p>
</body>
</html>
"""

# The expected plain text after conversion
EXPECTED_TEXT_OUTPUT_CHUNK = """# Main Title for Chunking Test

This is a paragraph for the chunking test. It should be long enough to pass the length check.

## Subtitle for Chunking Test

Item 1 for chunking

Item 2 for chunking

Item 3 for chunking, making sure the content is substantial.

Another paragraph to ensure sufficient content length. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."""

def test_html_loader_produces_text(tmp_path):
    """Verify HTML loader strips tags but keeps headings/paragraphs using controlled content."""
    # Create a temporary HTML file with known good content
    temp_html_file = tmp_path / "test_chunking_doc.html"
    temp_html_file.write_text(DUMMY_HTML_CONTENT_CHUNK)

    loader = DocumentLoader(data_dir=str(tmp_path), doc_format="html")
    documents = loader.load_html()
    
    assert documents, "No documents loaded from HTML"

    # Pick the first (and only) document for checks
    doc = documents[0]
    content = doc.page_content
    assert isinstance(content, str)
    assert len(content) > 50, f"Document content is suspiciously short: {len(content)}"
    assert content.strip() == EXPECTED_TEXT_OUTPUT_CHUNK.strip(), "Processed HTML content does not match expected output."

    # Check that heading markers exist
    assert "#" in content or "##" in content, "Headings not preserved in text"

    # print(content[:500])  # Optional: inspect first 500 chars

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

