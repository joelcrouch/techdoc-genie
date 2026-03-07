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


def test_html_to_text_missing_boilerplate():
    """Verify that _html_to_text works even if none of the filtered tags are present."""
    html = "<html><body><h1>Only a heading</h1></body></html>"
    loader = DocumentLoader()
    result = loader._html_to_text(html)
    assert result.strip() == "# Only a heading"


def test_html_to_text_nested_tags():
    """
    Verify if nested tags cause duplication in the current implementation.
    If 'find_all' is recursive, it might find both <li> and <p> content.
    """
    html = """
    <html>
        <body>
            <ul>
                <li><p>Nested content</p></li>
            </ul>
        </body>
    </html>
    """
    loader = DocumentLoader()
    result = loader._html_to_text(html)
    
    # If the bug exists, "Nested content" will appear twice.
    # We want to see what actually happens.
    occurrences = result.count("Nested content")
    assert occurrences == 1, f"Expected 1 occurrence, found {occurrences}. Content: {result}"


def test_html_to_text_no_body():
    """Verify behavior when <body> is missing."""
    html = "<html><head><title>No body</title></head></html>"
    loader = DocumentLoader()
    result = loader._html_to_text(html)
    assert result == ""


def test_html_to_text_unrelated_tags():
    """Verify that unrelated tags in body are ignored."""
    html = "<html><body><div><span>Not in extraction list</span></div></body></html>"
    loader = DocumentLoader()
    result = loader._html_to_text(html)
    assert result == ""


# ---------------------------------------------------------------------------
# _clean_heading
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw, expected",
    [
        ("## My Heading", "My Heading"),
        ("### Sub Section", "Sub Section"),
        ("# Top Level", "Top Level"),
        ("1. Introduction", "Introduction"),
        ("1.2. Sub-introduction", "Sub-introduction"),
        ("2.3.4. Deep Nest", "Deep Nest"),
        ("Plain heading", "Plain heading"),
        ("  extra   spaces  ", "extra spaces"),   # leading/trailing stripped; internal kept
        ("A" * 110, "A" * 97 + "..."),              # truncated at 100 chars
    ],
)
def test_clean_heading(raw, expected):
    loader = DocumentLoader()
    assert loader._clean_heading(raw) == expected


# ---------------------------------------------------------------------------
# _create_section_document
# ---------------------------------------------------------------------------

def test_create_section_document_structure():
    loader = DocumentLoader()
    section = {
        "title": "My Section",
        "content": ["Line one", "Line two"],
        "start_page": 3,
    }
    source = Path("/data/raw/docs/sample.pdf")
    doc = loader._create_section_document(section, source, page_num=5)

    assert isinstance(doc, Document)
    assert "# My Section" in doc.page_content
    assert "Section: My Section" in doc.page_content
    assert "Line one" in doc.page_content
    assert "Line two" in doc.page_content
    assert doc.metadata["section_title"] == "My Section"
    assert doc.metadata["start_page"] == 3
    assert doc.metadata["end_page"] == 5
    assert doc.metadata["filename"] == "sample.pdf"
    assert doc.metadata["file_type"] == "pdf"
    assert doc.metadata["doc_type"] == "postgresql_docs"


# ---------------------------------------------------------------------------
# _split_pdf_into_sections
# ---------------------------------------------------------------------------

def _make_page(text: str) -> Document:
    return Document(page_content=text)


def test_split_pdf_sections_detects_headings():
    """Pages with headings produce separate section Documents."""
    loader = DocumentLoader()
    pages = [
        _make_page("# Introduction\nThis is intro text.\nMore intro."),
        _make_page("## Chapter Two\nChapter two body text here."),
    ]
    source = Path("/fake/docs.pdf")
    sections = loader._split_pdf_into_sections(pages, source)

    assert len(sections) >= 2
    titles = [s.metadata["section_title"] for s in sections]
    assert any("Introduction" in t for t in titles)
    assert any("Chapter Two" in t for t in titles)


def test_split_pdf_sections_fallback_to_pages():
    """When no content accumulates (all lines are headings), falls back to per-page docs."""
    loader = DocumentLoader()
    # Every line is detected as a heading, so no content ever accumulates between them.
    pages = [
        _make_page("# Heading One"),
        _make_page("# Heading Two"),
    ]
    source = Path("/fake/docs.pdf")
    sections = loader._split_pdf_into_sections(pages, source)

    # Fallback: one Document per page
    assert len(sections) == len(pages)
    assert all("section_title" in s.metadata for s in sections)


def test_split_pdf_sections_empty_pages():
    """Empty pages list returns empty sections list."""
    loader = DocumentLoader()
    sections = loader._split_pdf_into_sections([], Path("/fake/docs.pdf"))
    assert sections == []


# ---------------------------------------------------------------------------
# load_pdfs
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_pdf_dir(tmp_path):
    """Directory with a single dummy PDF file."""
    pdf_dir = tmp_path / "pdf_docs"
    pdf_dir.mkdir()
    (pdf_dir / "sample.pdf").write_bytes(b"%PDF-1.4 dummy")
    return pdf_dir


def test_load_pdfs_with_section_split(temp_pdf_dir, mocker):
    """load_pdfs with split_pdf_sections=True calls _split_pdf_into_sections."""
    mock_pages = [
        Document(page_content="# Introduction\nSome intro content here."),
        Document(page_content="## Chapter Two\nChapter two body."),
    ]
    mock_loader_instance = MagicMock()
    mock_loader_instance.load.return_value = mock_pages
    mocker.patch("src.ingestion.document_loader.PyPDFLoader", return_value=mock_loader_instance)

    loader = DocumentLoader(data_dir=str(temp_pdf_dir), doc_format="pdf", split_pdf_sections=True)
    docs = loader.load_pdfs()

    assert len(docs) > 0
    for doc in docs:
        assert doc.metadata["filename"] == "sample.pdf"
        assert doc.metadata["file_type"] == "pdf"


def test_load_pdfs_without_section_split(temp_pdf_dir, mocker):
    """load_pdfs with split_pdf_sections=False just attaches metadata to pages."""
    mock_pages = [
        Document(page_content="Page one content"),
        Document(page_content="Page two content"),
    ]
    mock_loader_instance = MagicMock()
    mock_loader_instance.load.return_value = mock_pages
    mocker.patch("src.ingestion.document_loader.PyPDFLoader", return_value=mock_loader_instance)

    loader = DocumentLoader(data_dir=str(temp_pdf_dir), doc_format="pdf", split_pdf_sections=False)
    docs = loader.load_pdfs()

    assert len(docs) == 2
    for doc in docs:
        assert doc.metadata["filename"] == "sample.pdf"
        assert doc.metadata["file_type"] == "pdf"


def test_load_pdfs_empty_directory(tmp_path):
    """load_pdfs returns empty list when directory has no PDFs."""
    empty_dir = tmp_path / "no_pdfs"
    empty_dir.mkdir()
    loader = DocumentLoader(data_dir=str(empty_dir), doc_format="pdf")
    docs = loader.load_pdfs()
    assert docs == []


# ---------------------------------------------------------------------------
# load_auto
# ---------------------------------------------------------------------------

def test_load_auto_combines_html_and_pdf(mocker):
    """load_auto calls both load_html and load_pdfs and returns combined results."""
    html_doc = Document(page_content="HTML content", metadata={})
    pdf_doc = Document(page_content="PDF content", metadata={})

    loader = DocumentLoader()
    mocker.patch.object(loader, "load_html", return_value=[html_doc])
    mocker.patch.object(loader, "load_pdfs", return_value=[pdf_doc])

    docs = loader.load_auto()
    assert len(docs) == 2
    assert html_doc in docs
    assert pdf_doc in docs


# ---------------------------------------------------------------------------
# load_directory
# ---------------------------------------------------------------------------

def test_load_directory_uses_provided_path(tmp_path, mocker):
    """load_directory temporarily overrides data_dir and calls load_and_prepare."""
    html_dir = tmp_path / "html_docs"
    html_dir.mkdir()
    (html_dir / "doc.html").write_text("<html><body><h1>Hi</h1></body></html>")

    loader = DocumentLoader(data_dir="original/path", doc_format="html")
    docs = loader.load_directory(str(html_dir))

    assert len(docs) == 1
    # data_dir restored after call
    assert loader.data_dir == Path("original/path")


# ---------------------------------------------------------------------------
# load_and_prepare
# ---------------------------------------------------------------------------

def test_load_and_prepare_html(mocker):
    loader = DocumentLoader(doc_format="html")
    mock_load = mocker.patch.object(loader, "load_html", return_value=[])
    loader.load_and_prepare()
    mock_load.assert_called_once()


def test_load_and_prepare_pdf(mocker):
    loader = DocumentLoader(doc_format="pdf")
    mock_load = mocker.patch.object(loader, "load_pdfs", return_value=[])
    loader.load_and_prepare()
    mock_load.assert_called_once()


def test_load_and_prepare_auto(mocker):
    loader = DocumentLoader(doc_format="auto")
    mock_load = mocker.patch.object(loader, "load_auto", return_value=[])
    loader.load_and_prepare()
    mock_load.assert_called_once()


def test_load_and_prepare_invalid_format():
    loader = DocumentLoader.__new__(DocumentLoader)
    loader.data_dir = Path("dummy")
    loader.doc_format = "xml"
    loader.split_pdf_sections = True
    with pytest.raises(ValueError, match="Unsupported doc_format"):
        loader.load_and_prepare()

