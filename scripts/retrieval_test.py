#!/usr/bin/env python3
"""
Test vector search with sample queries.

This tests the quality of your retrieval system with various PostgreSQL queries.

Usage:
    python scripts/test_retrieval.py
    
    # Use different vector store
    python scripts/test_retrieval.py --store-dir data/vector_store/vectorstore_chunk1024_overlap100
    
    # More results per query
    python scripts/test_retrieval.py --k 5
"""
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.vector_store import VectorStore
from src.ingestion.providers.huggingface import HuggingFaceEmbeddingProvider
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


# Test queries for PostgreSQL documentation
TEST_QUERIES = [
    "How do I create an index in PostgreSQL?",
    "What are the differences between INNER JOIN and LEFT JOIN?",
    "How do I configure connection pooling?",
    "What is MVCC and how does it work?",
    "How do I back up a PostgreSQL database?",
    "What is a B-tree index?",
    "How do I optimize query performance?",
    "What are the different types of indexes?",
    "How do I create a table in PostgreSQL?",
    "What is the difference between VACUUM and ANALYZE?",
]


def test_retrieval(store_dir: str, store_name: str, k: int = 3, verbose: bool = False):
    """
    Test retrieval with sample queries.
    
    Args:
        store_dir: Directory containing the vector store
        store_name: Name of the vector store index
        k: Number of results to retrieve per query
        verbose: Show full content instead of preview
    """
    logger.info("=" * 80)
    logger.info("PostgreSQL Documentation Retrieval Test")
    logger.info("=" * 80)
    logger.info(f"Vector store: {store_dir}/{store_name}")
    logger.info(f"Results per query: {k}")
    logger.info("")
    
    # Initialize embedder (must match the one used during ingestion)
    logger.info("Initializing embedder...")
    embedder = HuggingFaceEmbeddingProvider()
    
    # Load vector store
    logger.info(f"Loading vector store from {store_dir}...")
    vector_store = VectorStore(embedder=embedder, persist_path=store_dir)
    
    try:
        vector_store.load(name=store_name)
        logger.info("✅ Vector store loaded successfully\n")
    except FileNotFoundError as e:
        logger.error(f"Vector store not found: {e}")
        logger.info("\nMake sure you've run the ingestion pipeline first:")
        logger.info("  python scripts/ingestion_pipeline_real.py")
        sys.exit(1)
    
    logger.info(f"Testing with {len(TEST_QUERIES)} queries\n")
    logger.info("=" * 80)
    
    # Track scores for analysis
    all_scores = []
    
    for i, query in enumerate(TEST_QUERIES, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"Query {i}/{len(TEST_QUERIES)}: {query}")
        logger.info("=" * 80)
        
        # Get top k results with scores
        results = vector_store.similarity_search_with_score(query, k=k)
        
        for j, (doc, score) in enumerate(results, 1):
            all_scores.append(score)
            
            logger.info(f"\nResult #{j} | Score: {score:.4f}")
            logger.info("-" * 80)
            
            # Show metadata
            filename = doc.metadata.get('filename', 'unknown')
            section = doc.metadata.get('section_title', 'N/A')
            page = doc.metadata.get('page', doc.metadata.get('start_page', 'N/A'))
            
            logger.info(f"File: {filename}")
            if section != 'N/A':
                logger.info(f"Section: {section}")
            if page != 'N/A':
                logger.info(f"Page: {page}")
            
            # Show content
            content = doc.page_content
            if verbose:
                logger.info(f"\nContent:\n{content}")
            else:
                # Show preview (first 300 chars)
                preview = content[:300].replace('\n', ' ')
                if len(content) > 300:
                    preview += "..."
                logger.info(f"\nPreview: {preview}")
    
    # Summary statistics
    logger.info("\n" + "=" * 80)
    logger.info("Test Summary")
    logger.info("=" * 80)
    logger.info(f"Total queries: {len(TEST_QUERIES)}")
    logger.info(f"Results per query: {k}")
    logger.info(f"Total results: {len(all_scores)}")
    
    if all_scores:
        logger.info(f"\nScore Statistics:")
        logger.info(f"  Min score: {min(all_scores):.4f}")
        logger.info(f"  Max score: {max(all_scores):.4f}")
        logger.info(f"  Avg score: {sum(all_scores)/len(all_scores):.4f}")
        
        # Score distribution
        excellent = sum(1 for s in all_scores if s < 0.5)
        good = sum(1 for s in all_scores if 0.5 <= s < 1.0)
        fair = sum(1 for s in all_scores if 1.0 <= s < 2.0)
        poor = sum(1 for s in all_scores if s >= 2.0)
        
        logger.info(f"\nScore Distribution (lower is better):")
        logger.info(f"  Excellent (< 0.5): {excellent} ({excellent/len(all_scores)*100:.1f}%)")
        logger.info(f"  Good (0.5-1.0): {good} ({good/len(all_scores)*100:.1f}%)")
        logger.info(f"  Fair (1.0-2.0): {fair} ({fair/len(all_scores)*100:.1f}%)")
        logger.info(f"  Poor (≥ 2.0): {poor} ({poor/len(all_scores)*100:.1f}%)")
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ Retrieval test completed!")
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Test PostgreSQL documentation retrieval quality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with default vector store
  %(prog)s
  
  # Test with different store
  %(prog)s --store-dir data/vector_store/vectorstore_chunk1024_overlap100
  
  # Get more results per query
  %(prog)s --k 5
  
  # Show full content instead of preview
  %(prog)s --verbose
        """
    )
    
    parser.add_argument(
        '--store-dir',
        default='./data/vector_store/vectorstore_chunk512_overlap50',
        help='Vector store directory (default: ./data/vector_store/vectorstore_chunk512_overlap50)'
    )
    parser.add_argument(
        '--store-name',
        default='vectorstore_chunk512_overlap50',
        help='Vector store index name (default: vectorstore_chunk512_overlap50)'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=3,
        help='Number of results per query (default: 3)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show full content instead of preview'
    )
    
    args = parser.parse_args()
    
    test_retrieval(
        store_dir=args.store_dir,
        store_name=args.store_name,
        k=args.k,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()