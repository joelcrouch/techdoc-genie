#!/usr/bin/env python3
"""
Script to load documents and test chunking with various chunk sizes.
Usage: python chunk_test.py --chunk_sizes 256 512 1024
"""

import argparse
from statistics import median
from src.ingestion.document_loader import DocumentLoader
from src.ingestion.chunker import DocumentChunker

def main(chunk_sizes, chunk_overlap=50, strategy="recursive"):
    # Load documents
    loader = DocumentLoader()
    documents = loader.load_and_prepare()
    
    total_chars = sum(len(doc.page_content) for doc in documents)
    print(f"Loaded {len(documents)} documents")
    print(f"Total characters: {total_chars:,}")

    for size in chunk_sizes:
        chunker = DocumentChunker(chunk_size=size, chunk_overlap=chunk_overlap)
        chunks = chunker.chunk_documents(documents, strategy=strategy)

        chunk_lengths = [len(c.page_content) for c in chunks]
        avg_length = sum(chunk_lengths) / len(chunks) if chunks else 0

        print(f"\nChunk size: {size}")
        print(f"  Total chunks: {len(chunks)}")
        print(f"  Avg chunk length: {avg_length:.0f}")
        print(f"  Min/Max chunk length: {min(chunk_lengths) if chunks else 0}/{max(chunk_lengths) if chunks else 0}")
        print(f"  Median chunk length: {median(chunk_lengths) if chunks else 0}")
        print(f"  Sample chunk preview:\n  {chunks[0].page_content[:200]}..." if chunks else "  No chunks produced")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test document chunking with different chunk sizes.")
    parser.add_argument(
        "--chunk_sizes",
        nargs="+",
        type=int,
        default=[256, 512, 1024],
        help="List of chunk sizes to test"
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=50,
        help="Number of overlapping characters between chunks"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["recursive", "token", "semantic"],
        default="recursive",
        help="Chunking strategy to use"
    )
    args = parser.parse_args()

    main(args.chunk_sizes, chunk_overlap=args.chunk_overlap, strategy=args.strategy)
