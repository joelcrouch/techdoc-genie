from pathlib import Path
from typing import List, Literal
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from bs4 import BeautifulSoup
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

DocFormat = Literal["html", "pdf", "auto"]


class DocumentLoader:
    """Load and prepare documentation files (HTML and PDF)."""

    def __init__(
        self,
        data_dir: str = "data/raw/postgresql",
        doc_format: DocFormat = "html",
    ):
        self.data_dir = Path(data_dir)
        self.doc_format = doc_format

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

        logger.info(f"Loaded {len(documents)} HTML documents")
        return documents

    def _html_to_text(self, html_content: str) -> str:
        """Convert HTML to plain text while preserving headings as Markdown."""
        soup = BeautifulSoup(html_content, "html.parser")

        # Remove boilerplate elements
        for tag in soup(["script", "style", "nav", "footer", "header", "form", "input"]):
            tag.decompose()

        lines = []
        if soup.body:
            for elem in soup.body.descendants:
                if elem.name and elem.name.startswith("h") and elem.get_text(strip=True):
                    # Convert headings to Markdown style
                    level = int(elem.name[1])
                    heading_text = elem.get_text(strip=True)
                    lines.append(f"{'#' * level} {heading_text}")
                elif elem.name == "p" and elem.get_text(strip=True):
                    lines.append(elem.get_text(strip=True))
                elif elem.name in ["li", "dt", "dd"] and elem.get_text(strip=True):
                    lines.append(elem.get_text(strip=True))
                elif elem.string and not elem.name:
                    text = elem.string.strip()
                    if text:
                        lines.append(text)

        return "\n\n".join(lines)

    def load_pdfs(self) -> List[Document]:
        """Load PDFs using PyPDFLoader (unchanged)."""
        logger.info(f"Loading PDF files from {self.data_dir}")
        documents: List[Document] = []

        for pdf_file in self.data_dir.glob("*.pdf"):
            loader = PyPDFLoader(str(pdf_file))
            docs = loader.load()

            for doc in docs:
                self._add_metadata(doc, pdf_file)

            documents.extend(docs)

        logger.info(f"Loaded {len(documents)} PDF pages")
        return documents

    def load_auto(self) -> List[Document]:
        """Load all supported formats."""
        documents: List[Document] = []
        documents.extend(self.load_html())
        documents.extend(self.load_pdfs())
        return documents

    def _add_metadata(self, doc: Document, source_file: Path) -> None:
        doc.metadata["source"] = str(source_file)
        doc.metadata["filename"] = source_file.name
        doc.metadata["doc_type"] = "postgresql_docs"
        doc.metadata["file_type"] = source_file.suffix.lstrip(".")

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

