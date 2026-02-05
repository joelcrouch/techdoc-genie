#!/usr/bin/env python3
"""
Build a disk-backed vector store from a specified directory of documents.

Usage:
    python scripts/build_generic_vectorstore.py -d <data_directory> -f <doc_format> -n <vectorstore_name>
    
Example:
    # Build a vector store from Ubuntu PDF docs
    python scripts/build_generic_vectorstore.py \
        -d /home/dell-linux-dev3/Projects/techdoc-genie/data/raw/ubuntu_docs/ \
        -f pdf \
        -n ubuntu_docs_pdf
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


# --- Configurable chunking strategy (can be made args if needed) ---
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
CHUNK_STRATEGY = "recursive"  # recursive / token / semantic


def main():
    parser = argparse.ArgumentParser(
        description="Build a generic vector store from documentation"
    )
    parser.add_argument(
        '--data-dir',
        '-d',
        required=True,
        help='Directory containing documents (e.g., data/raw/my_docs/)'
    )
    parser.add_argument(
        '--doc-format',
        '-f',
        choices=['html', 'pdf', 'auto'],
        default='auto',
        help='Document format to load (default: auto - loads both HTML and PDF)'
    )
    parser.add_argument(
        '--name',
        '-n',
        type=str,
        required=True,
        help='Name for the output vector store (e.g., "my_new_vectorstore")'
    )
    parser.add_argument(
        '--no-split-sections',
        action='store_true',
        help='Disable intelligent PDF section splitting (use pages instead)'
    )
    
    args = parser.parse_args()
    
    logger.info("Starting generic docs vectorization pipeline")
    logger.info("=" * 80)
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Document format: {args.doc_format}")
    logger.info(f"Vector store name: {args.name}")
    logger.info(f"Chunk size: {CHUNK_SIZE}")
    logger.info(f"Chunk overlap: {CHUNK_OVERLAP}")
    logger.info(f"Chunk strategy: {CHUNK_STRATEGY}")
    logger.info(f"PDF section splitting: {'Disabled' if args.no_split_sections else 'Enabled'}")
    logger.info("=" * 80)
    
    # Step 1: Load documents
    logger.info(f"\nStep 1: Loading documents from {args.data_dir}...")
    loader = DocumentLoader(
        data_dir=args.data_dir,
        doc_format=args.doc_format,
        split_pdf_sections=not args.no_split_sections
    )
    documents = loader.load_and_prepare()
    logger.info(f"✅ Loaded {len(documents)} documents")
    
    if not documents:
        logger.error(f"No documents found in {args.data_dir} with format {args.doc_format}")
        logger.info("Please ensure your data directory contains the specified document types.")
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
    
    # Step 4: Create vector store (disk-backed)
    vector_store_dir = Path("data/vector_store") / args.name # Output to data/vector_store/<name>
    
    logger.info(f"\nStep 4: Creating vector store...")
    vector_store = VectorStore(embedder=embedder, persist_path=str(vector_store_dir))
    vector_store.create_from_documents(chunks)
    logger.info(f"✅ Vector store created in memory")
    
    # Step 5: Save to disk
    logger.info(f"\nStep 5: Saving vector store...")
    vector_store.save(name=args.name) # Save inside the named directory
    logger.info(f"✅ Vector store saved to: {vector_store_dir}")
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("✅ Vector store build completed!")
    logger.info("=" * 80)
    logger.info(f"Documents loaded: {len(documents)}")
    logger.info(f"Chunks created: {len(chunks)}")
    logger.info(f"Vector store location: {vector_store_dir}")
    logger.info("\nNext steps:")
    logger.info("  1. Query with RAG: python src/agent/query_interface.py --vector-store " + args.name)
    logger.info("  2. Evaluate: Analyze chunking and retrieval effectiveness.")


if __name__ == "__main__":
    main()
