import pytest
from src.retrieval.vector_store import VectorStore
from src.ingestion.providers.huggingface import HuggingFaceEmbeddingProvider
from src.utils.logger import setup_logger
import numpy as np 

logger = setup_logger(__name__)

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


def test_retrieval_pipeline():
    store_dir = './data/vector_store/vectorstore_chunk512_overlap50'
    store_name = 'vectorstore_chunk512_overlap50'
    k = 3

    embedder = HuggingFaceEmbeddingProvider()
    assert embedder is not None, "Embedder not initialized"

    vector_store = VectorStore(embedder=embedder, persist_path=store_dir)

    try:
        vector_store.load(name=store_name)
    except FileNotFoundError:
        pytest.fail(f"Vector store not found at {store_dir}/{store_name}. Make sure ingestion pipeline has been run.")
    assert vector_store.vectorstore is not None, "Vector store not loaded"

    assert len(TEST_QUERIES) > 0, "No test queries defined"

    all_scores = []

    for i, query in enumerate(TEST_QUERIES, 1):
        results = vector_store.similarity_search_with_score(query, k=k)
        assert len(results) > 0, f"No results for query '{query}'"

        for doc, score in results:
            all_scores.append(score)
            assert isinstance(doc.page_content, str), "Document content is not string"
            assert doc.page_content != "", "Document content is empty"
            assert isinstance(score, (float, int, np.floating)), "Score is not a number"
            # assert isinstance(score, float), "Score is not a float"
            # Assuming lower score is better and some reasonable range
            assert score >= 0.0, f"Score {score} is negative for query '{query}'"

    assert len(all_scores) > 0, "No scores collected"
    min_score = min(all_scores)
    max_score = max(all_scores)
    avg_score = sum(all_scores) / len(all_scores)

    # Basic assertion to ensure some "good" results are likely
    # This threshold might need adjustment based on actual data and desired performance
    assert min_score < 1.0, f"Minimum score {min_score} is too high, indicating poor retrieval" # Assuming lower is better

    logger.info(f"Retrieval Test Summary:")
    logger.info(f"  Total queries: {len(TEST_QUERIES)}")
    logger.info(f"  Results per query: {k}")
    logger.info(f"  Total results: {len(all_scores)}")
    logger.info(f"  Min score: {min_score:.4f}")
    logger.info(f"  Max score: {max_score:.4f}")
    logger.info(f"  Avg score: {avg_score:.4f}")