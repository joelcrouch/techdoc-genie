#src/ingestion/chunker.py`:

from typing import List, Dict, Literal
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    MarkdownHeaderTextSplitter
)
from langchain.schema import Document
from ..utils.logger import setup_logger
from ..utils.config import get_settings

logger = setup_logger(__name__)

class DocumentChunker:
    """Split documents into chunks using various strategies."""
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None
    ):
        settings = get_settings()
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
    
    def chunk_recursive(
        self,
        documents: List[Document]
    ) -> List[Document]:
        """Recursive character-based chunking (default strategy)."""
        logger.info(f"Chunking with recursive strategy: size={self.chunk_size}, overlap={self.chunk_overlap}")
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks")
        
        return chunks
    
    def chunk_by_tokens(
        self,
        documents: List[Document]
    ) -> List[Document]:
        """Token-based chunking (for precise token limits)."""
        logger.info(f"Chunking with token strategy")
        
        splitter = TokenTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        chunks = splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks")
        
        return chunks
    
    def chunk_semantic(
        self,
        documents: List[Document],
        breakpoint_threshold: float = 0.5
    ) -> List[Document]:
        """Semantic chunking (placeholder for future enhancement)."""
        # For Sprint 0, we'll use recursive as fallback
        logger.warning("Semantic chunking not yet implemented, using recursive")
        return self.chunk_recursive(documents)
    
    def add_chunk_metadata(
        self,
        chunks: List[Document]
    ) -> List[Document]:
        """Add chunk-specific metadata."""
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_id'] = i
            chunk.metadata['chunk_size'] = len(chunk.page_content)
        
        return chunks
    
    def chunk_documents(
        self,
        documents: List[Document],
        strategy: Literal["recursive", "token", "semantic"] = "recursive"
    ) -> List[Document]:
        """Chunk documents using specified strategy."""
        
        if strategy == "recursive":
            chunks = self.chunk_recursive(documents)
        elif strategy == "token":
            chunks = self.chunk_by_tokens(documents)
        elif strategy == "semantic":
            chunks = self.chunk_semantic(documents)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
        
        chunks = self.add_chunk_metadata(chunks)
        
        return chunks