#!/usr/bin/env python3
"""
Build a disk-backed vector store for all PostgreSQL docs (HTML and PDF),
with directory names reflecting the chunking strategy.

Now with intelligent PDF section splitting!

Usage:
    python scripts/build_postgres_vectorstore.py
    
    # Disable PDF section splitting
    python scripts/build_postgres_vectorstore.py --no-split-sections
"""
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.document_loader import DocumentLoader
from src.ingestion.chunker import DocumentChunker
from src.ingestion.providers.huggingface import HuggingFaceEmbeddingProvider
from src.retrieval.vector_store import VectorStore
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


# --- Configurable chunking strategy ---
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
CHUNK_STRATEGY = "recursive"  # recursive / token / semantic


def main():
    parser = argparse.ArgumentParser(
        description="Build PostgreSQL documentation vector store"
    )
    parser.add_argument(
        '--no-split-sections',
        action='store_true',
        help='Disable intelligent PDF section splitting (use pages instead)'
    )
    parser.add_argument(
        '--data-dir',
        default='data/raw/postgresql',
        help='Directory containing documents (default: data/raw/postgresql)'
    )
    parser.add_argument(
        '--doc-format',
        choices=['html', 'pdf', 'auto'],
        default='auto',
        help='Document format to load (default: auto - loads both HTML and PDF)'
    )
    
    args = parser.parse_args()
    
    logger.info("Starting PostgreSQL docs vectorization pipeline")
    logger.info("=" * 80)
    logger.info(f"Chunk size: {CHUNK_SIZE}")
    logger.info(f"Chunk overlap: {CHUNK_OVERLAP}")
    logger.info(f"Chunk strategy: {CHUNK_STRATEGY}")
    logger.info(f"Document format: {args.doc_format}")
    logger.info(f"PDF section splitting: {'Disabled' if args.no_split_sections else 'Enabled'}")
    logger.info("=" * 80)
    
    # Step 1: Load PostgreSQL documents (HTML and/or PDF)
    logger.info(f"\nStep 1: Loading documents from {args.data_dir}...")
    loader = DocumentLoader(
        data_dir=args.data_dir,
        doc_format=args.doc_format,
        split_pdf_sections=not args.no_split_sections  # NEW: Enable intelligent PDF splitting
    )
    documents = loader.load_and_prepare()
    logger.info(f"✅ Loaded {len(documents)} documents")
    
    if not documents:
        logger.error(f"No documents found in {args.data_dir}")
        logger.info("Make sure you have .html or .pdf files in the directory")
        sys.exit(1)
    
    # Show document breakdown
    html_docs = sum(1 for d in documents if d.metadata.get('file_type') == 'html')
    pdf_docs = sum(1 for d in documents if d.metadata.get('file_type') == 'pdf')
    logger.info(f"   - HTML documents: {html_docs}")
    logger.info(f"   - PDF documents: {pdf_docs}")
    
    # Step 2: Chunk documents
    logger.info(f"\nStep 2: Chunking documents...")
    chunker = DocumentChunker(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    
    if CHUNK_STRATEGY == "recursive":
        chunks = chunker.chunk_recursive(documents)
    elif CHUNK_STRATEGY == "token":
        chunks = chunker.chunk_by_tokens(documents)
    elif CHUNK_STRATEGY == "semantic":
        chunks = chunker.chunk_semantic(documents)
    else:
        logger.error(f"Unknown chunking strategy: {CHUNK_STRATEGY}")
        sys.exit(1)
    
    logger.info(f"✅ Created {len(chunks)} chunks with strategy '{CHUNK_STRATEGY}'")
    logger.info(f"   - Average chunks per document: {len(chunks) / len(documents):.1f}")
    
    # Step 3: Initialize HuggingFace embedder
    logger.info(f"\nStep 3: Initializing embedder...")
    embedder = HuggingFaceEmbeddingProvider()
    logger.info("✅ Embedder ready")
    
    # Step 4: Create vector store (disk-backed) with name reflecting chunking
    vector_store_dir = f"./data/vector_store/vectorstore_chunk{CHUNK_SIZE}_overlap{CHUNK_OVERLAP}"
    vector_store_name = f"vectorstore_chunk{CHUNK_SIZE}_overlap{CHUNK_OVERLAP}"
    
    logger.info(f"\nStep 4: Creating vector store...")
    vector_store = VectorStore(embedder=embedder, persist_path=vector_store_dir)
    vector_store.create_from_documents(chunks)
    logger.info(f"✅ Vector store created in memory ({vector_store_name})")
    
    # Step 5: Save to disk
    logger.info(f"\nStep 5: Saving vector store...")
    vector_store.save(name=vector_store_name)
    logger.info(f"✅ Vector store saved to: {vector_store_dir}/{vector_store_name}")
    
    # Optional: test similarity search
    logger.info(f"\nStep 6: Testing similarity search...")
    sample_query = "How do I create a table in PostgreSQL?"
    results = vector_store.similarity_search(sample_query, k=3)
    
    logger.info(f"\nTop 3 results for: '{sample_query}'")
    logger.info("-" * 80)
    for i, doc in enumerate(results, 1):
        filename = doc.metadata.get('filename', 'unknown')
        section = doc.metadata.get('section_title', 'N/A')
        preview = doc.page_content[:100].replace('\n', ' ')
        
        logger.info(f"\n#{i}")
        logger.info(f"  File: {filename}")
        if section != 'N/A':
            logger.info(f"  Section: {section}")
        logger.info(f"  Preview: {preview}...")
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("✅ PostgreSQL vector database creation completed!")
    logger.info("=" * 80)
    logger.info(f"Documents loaded: {len(documents)}")
    logger.info(f"Chunks created: {len(chunks)}")
    logger.info(f"Vector store location: {vector_store_dir}/{vector_store_name}")
    logger.info("\nNext steps:")
    logger.info("  1. Query: python scripts/query_docs.py 'your question'")
    logger.info("  2. Evaluate: Try different chunk sizes by editing CHUNK_SIZE")
    logger.info("  3. Compare: Create multiple stores with different configs")


if __name__ == "__main__":
    main()
