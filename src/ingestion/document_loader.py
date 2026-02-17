# src/ingestion/document_loader.py
from pathlib import Path
from typing import List, Literal
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from bs4 import BeautifulSoup, Tag, NavigableString
import re

from ..utils.logger import setup_logger

logger = setup_logger(__name__)

DocFormat = Literal["html", "pdf", "auto"]


class DocumentLoader:
    """Load and prepare documentation files (HTML and PDF)."""
    
    def __init__(
        self,
        data_dir: str = "data/raw/postgresql",
        doc_format: DocFormat = "html",
        split_pdf_sections: bool = True,  # NEW: Enable intelligent PDF splitting
    ):
        self.data_dir = Path(data_dir)
        self.doc_format = doc_format
        self.split_pdf_sections = split_pdf_sections
    
    def load_html(self) -> List[Document]:
        """Load and parse HTML files into clean text Documents."""
        logger.info(f"Loading HTML files from {self.data_dir}")
        
        documents: List[Document] = []
        
        for html_file in self.data_dir.glob("*.html"):
            with open(html_file, "r", encoding="utf-8") as f:
                html_content = f.read()
            
            text = self._html_to_text(html_content)
            
            if text.strip():
                doc = Document(page_content=text)
                self._add_metadata(doc, html_file)
                documents.append(doc)
        
        if not documents:
            logger.warning(f"No HTML files found or processed in {self.data_dir}")
        else:
            logger.info(f"Loaded {len(documents)} HTML documents")
        return documents
    
    def _html_to_text(self, html_content: str) -> str:
        """Convert HTML to plain text while preserving headings as Markdown."""
        soup = BeautifulSoup(html_content, "html.parser")
        
        # Remove boilerplate elements
        for tag in soup(["script", "style", "nav", "footer", "header", "form", "input"]):
            if tag.parent: # Ensure tag has a parent before decomposing
                tag.decompose()
        
        lines = []
        
        if soup.body:
            # Iterate through the direct children of the body
            for element in soup.body.children:
                if isinstance(element, Tag):
                    if element.name.startswith("h") and element.get_text(strip=True):
                        level = int(element.name[1])
                        heading_text = element.get_text(strip=True)
                        lines.append(f"{'#' * level} {heading_text}")
                    elif element.name == "p" and element.get_text(strip=True):
                        lines.append(element.get_text(strip=True))
                    elif element.name in ["ul", "ol"]:
                        for li in element.find_all("li"):
                            if li.get_text(strip=True):
                                lines.append(li.get_text(strip=True))
                    elif element.name in ["dt", "dd"] and element.get_text(strip=True):
                        lines.append(element.get_text(strip=True))
                elif isinstance(element, NavigableString) and element.strip(): # Handle direct text nodes under body
                    lines.append(element.strip())


        cleaned_lines = [line for line in lines if line.strip()]
        return "\n\n".join(cleaned_lines)
    
    def load_pdfs(self) -> List[Document]:
        """Load PDFs, optionally splitting into logical sections."""
        logger.info(f"Loading PDF files from {self.data_dir}")
        documents = []
        
        for pdf_file in self.data_dir.glob("*.pdf"):
            logger.info(f"Processing: {pdf_file.name}")
            
            loader = PyPDFLoader(str(pdf_file))
            pages = loader.load()
            
            if self.split_pdf_sections:
                # Split into sections intelligently
                section_docs = self._split_pdf_into_sections(pages, pdf_file)
                documents.extend(section_docs)
                logger.info(f"  → Created {len(section_docs)} sections from {pdf_file.name}")
            else:
                # Just add metadata to pages
                for doc in pages:
                    self._add_metadata(doc, pdf_file)
                documents.extend(pages)
                logger.info(f"  → Loaded {len(pages)} pages from {pdf_file.name}")
        
        logger.info(f"Total: {len(documents)} PDF documents")
        return documents
    
    def _split_pdf_into_sections(
        self, 
        pages: List[Document], 
        source_file: Path
    ) -> List[Document]:
        """
        Split PDF pages into logical sections based on headings.
        
        This looks for patterns like:
        - "Chapter 1: ..."
        - "1. Introduction"
        - "## Heading"
        - Lines in ALL CAPS
        
        Args:
            pages: List of page Documents from PyPDFLoader
            source_file: Source PDF file path
            
        Returns:
            List of section Documents
        """
        sections = []
        current_section = {
            "title": "Introduction",
            "content": [],
            "start_page": 0
        }
        
        for page_num, page in enumerate(pages):
            content = page.page_content
            lines = content.split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check if this line looks like a heading
                if self._is_heading(line):
                    # Save previous section if it has content
                    if current_section["content"]:
                        sections.append(self._create_section_document(
                            current_section, 
                            source_file,
                            page_num
                        ))
                    
                    # Start new section
                    current_section = {
                        "title": self._clean_heading(line),
                        "content": [],
                        "start_page": page_num
                    }
                else:
                    # Add to current section
                    current_section["content"].append(line)
        
        # Don't forget the last section
        if current_section["content"]:
            sections.append(self._create_section_document(
                current_section, 
                source_file,
                len(pages) - 1
            ))
        
        # If no sections were found, treat each page as a section
        if not sections:
            logger.warning(f"No sections detected in {source_file.name}, using pages as sections")
            sections = []
            for page_num, page in enumerate(pages):
                doc = Document(
                    page_content=page.page_content,
                    metadata={
                        "source": str(source_file),
                        "filename": source_file.name,
                        "doc_type": "postgresql_docs",
                        "file_type": "pdf",
                        "page": page_num,
                        "section_title": f"Page {page_num + 1}"
                    }
                )
                sections.append(doc)
        
        return sections
    
    def _is_heading(self, line: str) -> bool:
        """
        Detect if a line is likely a heading.
        
        Patterns:
        - Starts with "Chapter", "Part", "Section"
        - Starts with numbers like "1.", "1.1", "Chapter 1"
        - All caps (at least 3 words)
        - Starts with markdown heading markers (##, ###)
        """
        line = line.strip()
        
        # Too short to be a heading
        if len(line) < 5:
            return False
        
        # Markdown headings
        if re.match(r'^#{1,6}\s+', line):
            return True
        
        # Chapter/Part/Section keywords
        if re.match(r'^(Chapter|Part|Section|Appendix)\s+\d+', line, re.IGNORECASE):
            return True
        
        # Numbered headings (1., 1.1, 1.1.1, etc.)
        if re.match(r'^\d+(\.\d+)*\.\s+[A-Z]', line):
            return True
        
        # All caps with at least 3 words and not too long
        words = line.split()
        if (len(words) >= 3 and 
            len(words) <= 12 and 
            line.isupper() and 
            not any(char.isdigit() for char in line[:10])):
            return True
        
        # PostgreSQL specific: lines that start with uppercase and end with common heading patterns
        if (line[0].isupper() and 
            not line.endswith('.') and 
            not line.endswith(',') and
            len(words) <= 10 and
            len(words) >= 3 and # Changed from >=2 to >=3
            not any(char.isdigit() for char in line)): # Added: must not contain digits
            # Check if it looks like a title (mostly capitalized words)
            capitalized_count = sum(1 for word in words if word and word[0].isupper())
            if capitalized_count / len(words) > 0.5:
                return True
        
        return False
    
    def _clean_heading(self, heading: str) -> str:
        """Clean and normalize heading text."""
        # Remove markdown markers
        heading = re.sub(r'^#{1,6}\s+', '', heading)
        
        # Remove leading numbers if present
        heading = re.sub(r'^\d+(\.\d+)*\.?\s+', '', heading)
        
        # Normalize whitespace
        heading = ' '.join(heading.split())
        
        # Limit length
        if len(heading) > 100:
            heading = heading[:97] + "..."
        
        return heading
    
    def _create_section_document(
        self, 
        section: dict, 
        source_file: Path,
        page_num: int
    ) -> Document:
        """Create a Document from a section dictionary."""
        # Combine title and content
        # content_parts = [f"# {section['title']}", ""] + section['content']
        # Skip very small sections (noise)
        # if len(" ".join(section["content"])) < 300:
        #     continue

        content_parts = [
            f"# {section['title']}",
            f"Section: {section['title']}",  # reinforce for embeddings
            ""
        ] + section['content']

        content = "\n".join(content_parts)

        doc = Document(
            page_content=content,
            metadata={
                "source": str(source_file),
                "filename": source_file.name,
                "doc_type": "postgresql_docs",
                "file_type": "pdf",
                "section_title": section['title'],
                "start_page": section['start_page'],
                "end_page": page_num,
            }
        )
        
        return doc
    
    def load_auto(self) -> List[Document]:
        """Load all supported formats."""
        documents: List[Document] = []
        documents.extend(self.load_html())
        documents.extend(self.load_pdfs())
        return documents
    
    def _add_metadata(self, doc: Document, source_file: Path) -> None:
        """Add standard metadata to a document."""
        doc.metadata["source"] = str(source_file)
        doc.metadata["filename"] = source_file.name
        doc.metadata["doc_type"] = "postgresql_docs"
        doc.metadata["file_type"] = source_file.suffix.lstrip(".")
    
    def load_directory(self, directory: str) -> List[Document]:
        """
        Load documents from a specific directory.
        
        This is a convenience method that temporarily overrides self.data_dir.
        """
        original_dir = self.data_dir
        self.data_dir = Path(directory)
        
        try:
            documents = self.load_and_prepare()
        finally:
            self.data_dir = original_dir
        
        return documents
    
    def load_and_prepare(self) -> List[Document]:
        """Public interface for loading documents."""
        if self.doc_format == "html":
            return self.load_html()
        elif self.doc_format == "pdf":
            return self.load_pdfs()
        elif self.doc_format == "auto":
            return self.load_auto()
        else:
            raise ValueError(f"Unsupported doc_format: {self.doc_format}")



