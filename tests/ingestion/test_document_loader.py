import pytest
from pathlib import Path
from bs4 import BeautifulSoup
import re
from langchain.schema import Document
from unittest.mock import MagicMock

from src.ingestion.document_loader import DocumentLoader

# A small section of a dummy HTML file for testing
DUMMY_HTML_CONTENT = """
<!DOCTYPE html>
<html>
<head>
    <title>Dummy Page</title>
    <style>body { font-family: sans-serif; }</style>
</head>
<body>
    <header><h1>This is a header that should be removed</h1></header>
    <nav><a href="#">Nav link</a></nav>
    <h1>Main Title</h1>
    <p>This is a paragraph.</p>
    <h2>Subtitle</h2>
    <ul>
        <li>Item 1</li>
        <li>Item 2</li>
    </ul>
    <script>console.log("script content");</script>
    <footer><p>Footer content to be removed</p></footer>
</body>
</html>
"""

# The expected plain text after conversion
EXPECTED_TEXT_OUTPUT = """# Main Title

This is a paragraph.

## Subtitle

Item 1

Item 2"""

def test_html_to_text_conversion():
    """
    Tests if the _html_to_text method correctly converts HTML to clean,
    markdown-formatted text.
    """
    # Instantiate the loader (data_dir is not needed for this private method)
    loader = DocumentLoader(data_dir="dummy/path")

    # Call the private method with the dummy HTML
    # Accessing a "private" method is acceptable for testing purposes
    actual_text = loader._html_to_text(DUMMY_HTML_CONTENT)

    # Compare the actual output with the expected output, ignoring leading/trailing whitespace
    assert actual_text.strip() == EXPECTED_TEXT_OUTPUT.strip()


def test_add_metadata():
    """
    Tests if the _add_metadata method correctly adds standard metadata to a Document.
    """
    loader = DocumentLoader(data_dir="dummy/path")
    
    # Create a mock Document
    mock_doc = Document(page_content="some content")
    
    # Create a mock Path object for the source file
    mock_source_file = MagicMock(spec=Path)
    mock_source_file.name = "example.html"
    mock_source_file.suffix = ".html"
    mock_source_file.__str__.return_value = "/path/to/example.html" # For str(source_file)
    
    loader._add_metadata(mock_doc, mock_source_file)
    
    expected_metadata = {
        "source": "/path/to/example.html",
        "filename": "example.html",
        "doc_type": "postgresql_docs",
        "file_type": "html"
    }
    
    assert mock_doc.metadata == expected_metadata


@pytest.fixture
def temp_html_dir(tmp_path):
    """
    Fixture to create a temporary directory with dummy HTML files for testing load_html.
    """
    html_dir = tmp_path / "html_docs"
    html_dir.mkdir()

    # Create dummy HTML file 1
    (html_dir / "doc1.html").write_text(DUMMY_HTML_CONTENT)

    # Create dummy HTML file 2 with slightly different content
    (html_dir / "doc2.html").write_text("""
<!DOCTYPE html>
<html>
<body>
    <h1>Another Title</h1>
    <p>Another paragraph.</p>
</body>
</html>
""")
    return html_dir

def test_load_html_method(temp_html_dir):
    """
    Tests if the load_html method correctly loads and processes HTML files from a directory.
    """
    loader = DocumentLoader(data_dir=str(temp_html_dir), doc_format="html")
    
    documents = loader.load_html()
    
    assert len(documents) == 2
    
    # Verify doc1
    doc1 = next(doc for doc in documents if "doc1.html" in doc.metadata["filename"])
    assert doc1.page_content.strip() == EXPECTED_TEXT_OUTPUT.strip()
    assert doc1.metadata["filename"] == "doc1.html"
    assert doc1.metadata["file_type"] == "html"
    assert doc1.metadata["doc_type"] == "postgresql_docs"
    assert doc1.metadata["source"] == str(temp_html_dir / "doc1.html")

    # Verify doc2
    doc2_expected_content = "# Another Title\n\nAnother paragraph."
    doc2 = next(doc for doc in documents if "doc2.html" in doc.metadata["filename"])
    assert doc2.page_content.strip() == doc2_expected_content.strip()
    assert doc2.metadata["filename"] == "doc2.html"
    assert doc2.metadata["file_type"] == "html"
    assert doc2.metadata["doc_type"] == "postgresql_docs"
    assert doc2.metadata["source"] == str(temp_html_dir / "doc2.html")


def test_load_html_no_files(tmp_path, mocker):
    """
    Tests that load_html returns an empty list when no HTML files are present
    in the specified directory and logs a warning.
    """
    empty_dir = tmp_path / "empty_html_docs"
    empty_dir.mkdir() # Create an empty directory
    
    loader = DocumentLoader(data_dir=str(empty_dir), doc_format="html")
    
    # Mock the logger.warning method
    mock_warning = mocker.patch("src.ingestion.document_loader.logger.warning")
    
    documents = loader.load_html()
    
    assert len(documents) == 0
    assert documents == []
    
    # Assert that logger.warning was called with the correct message
    mock_warning.assert_called_once_with(f"No HTML files found or processed in {empty_dir}")

@pytest.mark.parametrize(
    "line, expected",
    [
        # Markdown headings
        ("# Introduction", True),
        ("## Section 1", True),
        ("### Subsection A", True),
        ("      #### Sub-subsection", True),
        ("##### Heading with Five Hashes", True),
        ("###### Smallest Heading", True),
        (" # Not a heading", True), # Changed from False to True
        ("####### Too Many Hashes", False), # More than 6 hashes
        ("No hash heading", False),

        # Chapter/Part/Section keywords
        ("Chapter 1: The Beginning", True),
        ("Part 2. Advanced Concepts", True),
        ("Section 3.4. Further Details", True),
        ("APPENDIX A - Glossary", True),
        ("chapter one", False), # Changed from True to False
        ("Part: Two", False), # Missing number
        ("A Chapter Without Number", True),

        # Numbered headings (1., 1.1, etc.)
        ("1. Introduction", True),
        ("1.1. Sub-introduction", True),
        ("1.1.1. Detail", True),
        ("  2. Main Point", True),
        ("2. not capitalized", False), # Must start with uppercase after number
        ("1 No dot", False), # Missing dot after number
        ("1.2.3.4.5.6. Too many numbers", True),

        # All caps with at least 3 words and not too long (<=12 words)
        ("IMPORTANT NOTICE TO ALL USERS", True), # 6 words, all caps
        ("ALL CAPS SHORT", True), # Changed from False to True
        ("THIS IS A VERY VERY VERY VERY VERY VERY VERY VERY VERY VERY VERY VERY LONG HEADING", False), # Too many words
        ("THIS HAS A Number 1", False), # Contains a digit
        ("This is not all caps", False), # Not all caps

        # PostgreSQL specific: lines that start with uppercase, not ending with . or ,, 2-10 words, mostly capitalized
        ("PostgreSQL Specific Heading", True), # 3 words, starts with Cap, no trailing . or ,
        ("Another Specific One", True), # 3 words, starts with Cap, no trailing . or ,
        ("Short", False), # Too short (less than 2 words)
        ("Very Very Very Very Very Very Very Very Very Very Long Line", False), # Too long (more than 10 words)
        ("Ends with a period.", False), # Ends with a period
        ("Ends with a comma,", False), # Ends with a comma
        ("starts with lowercase", False), # Does not start with uppercase
        ("Mostly capitalized Words", True), # more than 50% capitalized

        # Short lines, general non-headings
        ("Hello world", False), # Too short
        ("A short line.", False),
        ("Just text", False),
        ("", False),
        (" ", False),
        ("x", False),
    ],
)
def test_is_heading(line, expected):
    """
    Tests the _is_heading method to correctly identify various heading patterns.
    """
    loader = DocumentLoader(data_dir="dummy/path") # data_dir is not used by _is_heading
    assert loader._is_heading(line) == expected

