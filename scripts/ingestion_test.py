#!/usr/bin/env python3
"""
Complete ingestion pipeline: Load → Chunk → Embed → Store

Standalone MVP script to test:
- Document loading
- Chunking
- Embedding
- Optional in-memory vector store (no persistence)
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.document_loader import DocumentLoader
from src.ingestion.chunker import DocumentChunker
from src.ingestion.providers.huggingface import HuggingFaceEmbeddingProvider
from src.retrieval.vector_store import VectorStore
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def main():
    """Run the ingestion pipeline on a single file or small subset."""
    logger.info("Starting document ingestion pipeline (test mode)")
    logger.info("=" * 80)

   # Step 1: Load documents
    logger.info("Step 1: Loading documents...")
    # Use loader pointing at the folder containing the single file
    test_file_dir = "data/raw/postgresql"
    loader = DocumentLoader(data_dir=test_file_dir, doc_format="html")
    documents = loader.load_html()  # or loader.load_and_prepare()
    # Optionally filter only the single file
    documents = [doc for doc in documents if "tutorial-sql.html" in doc.metadata["filename"]]

    logger.info(f"✅ Loaded {len(documents)} document(s): tutorial-sql.html")

    # Step 2: Chunk documents
    logger.info("\nStep 2: Chunking documents...")
    chunker = DocumentChunker(chunk_size=512, chunk_overlap=50)
    chunks = chunker.chunk_documents(documents, strategy="recursive")
    logger.info(f"✅ Created {len(chunks)} chunks")
    for i, chunk in enumerate(chunks[:3]):
        logger.info(f"Chunk {i} preview: {chunk.page_content[:60]}...")

    # Step 3: Initialize embedder
    logger.info("\nStep 3: Initializing HuggingFace embedder...")
    embedder = HuggingFaceEmbeddingProvider()
    logger.info("✅ Embedder ready")

    # Step 4: Create vector store in memory (no saving)
    logger.info("\nStep 4: Creating vector store (in-memory)...")
    vector_store = VectorStore(embedder=embedder, persist_path=None)
    vector_store.create_from_documents(chunks)
    logger.info("✅ Vector store created in memory")

    # Step 5: Test similarity search
    logger.info("\nStep 5: Testing similarity search...")
    query = "How do I create a table in PostgreSQL?"
    results = vector_store.similarity_search(query, k=3)
    logger.info("Top 3 similarity search results:")
    for r in results:
        logger.info(f"- {r.metadata.get('source')} | {r.page_content[:60]}...")

    logger.info("\n" + "=" * 80)
    logger.info("✅ Test pipeline completed successfully!")


if __name__ == "__main__":
    main()
