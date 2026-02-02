

#!/usr/bin/env python3
"""
Test script for document chunking that can be run as both:
1. A pytest test: pytest tests/test_chunk_script.py
2. A CLI tool: python tests/test_chunk_script.py --chunk_sizes 256 512 1024

Usage: python tests/test_chunking.py --chunk_sizes 256 512 1024
"""
import argparse
import sys
from statistics import median
import pytest
from src.ingestion.document_loader import DocumentLoader
from src.ingestion.chunker import DocumentChunker


def analyze_chunks(chunk_sizes, chunk_overlap=50, strategy="recursive", documents=None):
    """
    Analyze chunking with various chunk sizes.
    
    Args:
        chunk_sizes: List of chunk sizes to test
        chunk_overlap: Number of overlapping characters
        strategy: Chunking strategy ("recursive", "token", or "semantic")
        documents: Optional list of documents to use (if None, loads from DocumentLoader)
    
    Returns:
        dict: Results of the analysis
    """
    # Load documents if not provided
    if documents is None:
        loader = DocumentLoader()
        documents = loader.load_and_prepare()
    
    total_chars = sum(len(doc.page_content) for doc in documents)
    print(f"Loaded {len(documents)} documents")
    print(f"Total characters: {total_chars:,}")
    
    results = {}
    
    for size in chunk_sizes:
        chunker = DocumentChunker(chunk_size=size, chunk_overlap=chunk_overlap)
        chunks = chunker.chunk_documents(documents, strategy=strategy)
        chunk_lengths = [len(c.page_content) for c in chunks]
        avg_length = sum(chunk_lengths) / len(chunks) if chunks else 0
        
        results[size] = {
            'total_chunks': len(chunks),
            'avg_length': avg_length,
            'min_length': min(chunk_lengths) if chunks else 0,
            'max_length': max(chunk_lengths) if chunks else 0,
            'median_length': median(chunk_lengths) if chunks else 0,
            'chunks': chunks
        }
        
        print(f"\nChunk size: {size}")
        print(f"  Total chunks: {len(chunks)}")
        print(f"  Avg chunk length: {avg_length:.0f}")
        print(f"  Min/Max chunk length: {results[size]['min_length']}/{results[size]['max_length']}")
        print(f"  Median chunk length: {results[size]['median_length']}")
        print(f"  Sample chunk preview:\n  {chunks[0].page_content[:200]}..." if chunks else "  No chunks produced")
    
    return results


# ===== PYTEST TESTS =====

@pytest.fixture
def sample_docs():
    """Load a small set of documents for testing."""
    loader = DocumentLoader(data_dir="tests/test_data", doc_format="html")
    docs = loader.load_html()
    assert docs, "No test documents loaded"
    return docs


def test_chunk_analysis_runs(sample_docs):
    """Test the chunking analysis on sample documents."""
    chunk_sizes = [128, 256]
    
    # Run analysis
    results = analyze_chunks(
        chunk_sizes, 
        chunk_overlap=20, 
        strategy="recursive",
        documents=sample_docs
    )
    
    # Validate results
    assert len(results) == len(chunk_sizes), "Should have results for each chunk size"
    
    for size in chunk_sizes:
        assert size in results, f"Missing results for chunk size {size}"
        assert results[size]['total_chunks'] > 0, f"No chunks produced for size {size}"
        assert results[size]['avg_length'] > 0, f"Invalid avg length for size {size}"


def test_chunk_metadata(sample_docs):
    """Test that chunks have proper metadata."""
    chunker = DocumentChunker(chunk_size=128, chunk_overlap=20)
    chunks = chunker.chunk_documents(sample_docs, strategy="recursive")
    
    assert chunks, "No chunks produced"
    
    # Check metadata
    for i, chunk in enumerate(chunks):
        assert "chunk_id" in chunk.metadata, f"Missing chunk_id in chunk {i}"
        assert chunk.metadata["chunk_id"] == i, f"Incorrect chunk_id in chunk {i}"
        assert "chunk_size" in chunk.metadata, f"Missing chunk_size in chunk {i}"
        assert chunk.metadata["chunk_size"] == len(chunk.page_content), f"Incorrect chunk_size in chunk {i}"


def test_chunk_lengths(sample_docs):
    """Test that chunk lengths are reasonable."""
    chunker = DocumentChunker(chunk_size=128, chunk_overlap=20)
    chunks = chunker.chunk_documents(sample_docs, strategy="recursive")
    
    lengths = [len(c.page_content) for c in chunks]
    assert all(l > 0 for l in lengths), "Some chunks have zero length"
    assert all(l <= 128 + 100 for l in lengths), "Some chunks are unreasonably large (>128+tolerance)"


# ===== CLI INTERFACE =====   just thinking about trying this out, maybe 

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Test document chunking with different chunk sizes."
    )
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
    
    analyze_chunks(
        args.chunk_sizes, 
        chunk_overlap=args.chunk_overlap, 
        strategy=args.strategy
    )


if __name__ == "__main__":
    main()