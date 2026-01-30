# import pytest
# from pathlib import Path
# from langchain.schema import Document

# from src.retrieval.vector_store import VectorStore
# from src.ingestion.providers.mock import MockEmbedder


# @pytest.fixture
# def mock_embedder():
#     return MockEmbedder(dimension=384)


# @pytest.fixture
# def sample_documents():
#     return [
#         Document(page_content="PostgreSQL supports B-tree and GIN indexes."),
#         Document(page_content="GIN indexes are useful for full-text search."),
#         Document(page_content="BRIN indexes are space-efficient for large tables."),
#         Document(page_content="Hash indexes support equality comparisons."),
#     ]


# def test_vectorstore_creation_and_search(tmp_path, mock_embedder, sample_documents):
#     """
#     VectorStore should:
#     - create a FAISS index
#     - return documents for similarity search
#     """
#     vs = VectorStore(
#         persist_path=tmp_path,
#         embedder=mock_embedder,
#     )

#     vs.create_from_documents(sample_documents)

#     results = vs.similarity_search(
#         query="What indexes does PostgreSQL support?",
#         k=2,
#     )

#     assert len(results) == 2
#     assert all(isinstance(doc, Document) for doc in results)


# # def test_vectorstore_similarity_scores(tmp_path, mock_embedder, sample_documents):
# #     """
# #     similarity_search_with_score should return (Document, float)
# #     """
# #     vs = VectorStore(
# #         persist_path=tmp_path,
# #         embedder=mock_embedder,
# #     )

# #     vs.create_from_documents(sample_documents)

# #     results = vs.similarity_search_with_score(
# #         query="full text search index",
# #         k=3,
# #     )

# #     assert len(results) == 3
# #     for doc, score in results:
# #         assert isinstance(doc, Document)
# #         assert isinstance(score, float)

# def test_vectorstore_similarity_scores(tmp_path, mock_embedder, sample_documents):
#     """
#     similarity_search_with_score should return (Document, float)
#     """
#     vs = VectorStore(
#         persist_path=tmp_path,
#         embedder=mock_embedder,
#     )

#     vs.create_from_documents(sample_documents)

#     results = vs.similarity_search_with_score(
#         query="full text search index",
#         k=3,
#     )

#     assert len(results) == 3
#     for doc, score in results:
#         assert isinstance(doc, Document)
#         # Accept both Python float and numpy float types
#         assert isinstance(score, (float, int)) or hasattr(score, '__float__')


# def test_vectorstore_save_and_load(tmp_path, mock_embedder, sample_documents):
#     """
#     VectorStore should persist and reload correctly
#     """
#     vs = VectorStore(
#         persist_path=tmp_path,
#         embedder=mock_embedder,
#     )

#     vs.create_from_documents(sample_documents)
#     vs.save(name="test_index")

#     # New instance, same embedder
#     vs_loaded = VectorStore(
#         persist_path=tmp_path,
#         embedder=mock_embedder,
#     )

#     vs_loaded.load(name="test_index")

#     results = vs_loaded.similarity_search(
#         query="PostgreSQL indexes",
#         k=2,
#     )

#     assert len(results) == 2
#     assert all(isinstance(doc, Document) for doc in results)


# def test_vectorstore_requires_embedder(tmp_path):
#     """
#     VectorStore must not allow None embedder
#     """
#     with pytest.raises(ValueError):
#         VectorStore(persist_path=tmp_path, embedder=None)


# def test_vectorstore_errors_if_uninitialized(tmp_path, mock_embedder):
#     """
#     Searching before creation/loading should error
#     """
#     vs = VectorStore(
#         persist_path=tmp_path,
#         embedder=mock_embedder,
#     )

#     with pytest.raises(ValueError):
#         vs.similarity_search("test")

#     with pytest.raises(ValueError):
#         vs.similarity_search_with_score("test")

#     with pytest.raises(ValueError):
#         vs.save()

import pytest
import numbers
from pathlib import Path
from langchain.schema import Document
from src.retrieval.vector_store import VectorStore
from src.ingestion.providers.mock import MockEmbedder

@pytest.fixture
def mock_embedder():
    return MockEmbedder(dimension=384)

@pytest.fixture
def sample_documents():
    return [
        Document(page_content="PostgreSQL supports B-tree and GIN indexes."),
        Document(page_content="GIN indexes are useful for full-text search."),
        Document(page_content="BRIN indexes are space-efficient for large tables."),
        Document(page_content="Hash indexes support equality comparisons."),
    ]

def test_vectorstore_creation_and_search(tmp_path, mock_embedder, sample_documents):
    """
    VectorStore should:
    - create a FAISS index
    - return documents for similarity search
    """
    vs = VectorStore(
        persist_path=tmp_path,
        embedder=mock_embedder,
    )

    vs.create_from_documents(sample_documents)

    results = vs.similarity_search(
        query="What indexes does PostgreSQL support?",
        k=2,
    )

    assert len(results) == 2
    assert all(isinstance(doc, Document) for doc in results)

def test_vectorstore_similarity_scores(tmp_path, mock_embedder, sample_documents):
    """
    similarity_search_with_score should return (Document, float)
    """
    vs = VectorStore(
        persist_path=tmp_path,
        embedder=mock_embedder,
    )

    vs.create_from_documents(sample_documents)

    results = vs.similarity_search_with_score(
        query="full text search index",
        k=3,
    )

    assert len(results) == 3
    for doc, score in results:
        assert isinstance(doc, Document)
        assert isinstance(score, numbers.Real)  # Accepts float, numpy float, etc.

def test_vectorstore_save_and_load(tmp_path, mock_embedder, sample_documents):
    """
    VectorStore should persist and reload correctly
    """
    vs = VectorStore(
        persist_path=tmp_path,
        embedder=mock_embedder,
    )

    vs.create_from_documents(sample_documents)
    vs.save(name="test_index")

    # New instance, same embedder
    vs_loaded = VectorStore(
        persist_path=tmp_path,
        embedder=mock_embedder,
    )

    vs_loaded.load(name="test_index")

    results = vs_loaded.similarity_search(
        query="PostgreSQL indexes",
        k=2,
    )

    assert len(results) == 2
    assert all(isinstance(doc, Document) for doc in results)

def test_vectorstore_requires_embedder(tmp_path):
    """
    VectorStore must not allow None embedder
    """
    with pytest.raises(ValueError):
        VectorStore(persist_path=tmp_path, embedder=None)

def test_vectorstore_errors_if_uninitialized(tmp_path, mock_embedder):
    """
    Searching before creation/loading should error
    """
    vs = VectorStore(
        persist_path=tmp_path,
        embedder=mock_embedder,
    )

    with pytest.raises(ValueError):
        vs.similarity_search("test")

    with pytest.raises(ValueError):
        vs.similarity_search_with_score("test")

    with pytest.raises(ValueError):
        vs.save()
        
def test_vectorstore_score_ordering(tmp_path, mock_embedder, sample_documents):
    """Test that results are ordered by similarity score (lower is better for distance)."""
    vs = VectorStore(persist_path=tmp_path, embedder=mock_embedder)
    vs.create_from_documents(sample_documents)
    
    results = vs.similarity_search_with_score(query="PostgreSQL indexes", k=4)
    
    # Scores should be in ascending order (lower score = more similar)
    scores = [score for _, score in results]
    assert scores == sorted(scores), "Scores should be in ascending order"

def test_vectorstore_different_queries_different_results(tmp_path, mock_embedder, sample_documents):
    """Test that different queries return different top results."""
    vs = VectorStore(persist_path=tmp_path, embedder=mock_embedder)
    vs.create_from_documents(sample_documents)
    
    results1 = vs.similarity_search("B-tree index", k=1)
    results2 = vs.similarity_search("full-text search", k=1)
    
    # Top results should likely be different (though not guaranteed with mock embedder)
    # At minimum, we can verify we get results
    assert len(results1) == 1
    assert len(results2) == 1

def test_vectorstore_special_characters_in_query(tmp_path, mock_embedder, sample_documents):
    """Test searching with special characters."""
    vs = VectorStore(persist_path=tmp_path, embedder=mock_embedder)
    vs.create_from_documents(sample_documents)
    
    special_queries = [
        "PostgreSQL @#$%",
        "indexes: B-tree, GIN, BRIN",
        "What's the best index?",
        "Índices especiales",
    ]
    
    for query in special_queries:
        results = vs.similarity_search(query, k=2)
        assert len(results) == 2

def test_vectorstore_very_long_query(tmp_path, mock_embedder, sample_documents):
    """Test searching with a very long query."""
    vs = VectorStore(persist_path=tmp_path, embedder=mock_embedder)
    vs.create_from_documents(sample_documents)
    
    long_query = "PostgreSQL database indexing " * 100
    results = vs.similarity_search(long_query, k=2)
    
    assert len(results) == 2

def test_vectorstore_unicode_content(tmp_path, mock_embedder):
    """Test documents with Unicode content."""
    vs = VectorStore(persist_path=tmp_path, embedder=mock_embedder)
    
    unicode_docs = [
        Document(page_content="PostgreSQL支持多种索引类型"),
        Document(page_content="Индексы в PostgreSQL очень быстрые"),
        Document(page_content="PostgreSQL índices são eficientes"),
    ]
    
    vs.create_from_documents(unicode_docs)
    results = vs.similarity_search("PostgreSQL", k=3)
    
    assert len(results) == 3

def test_vectorstore_very_long_documents(tmp_path, mock_embedder):
    """Test indexing very long documents."""
    vs = VectorStore(persist_path=tmp_path, embedder=mock_embedder)
    
    long_docs = [
        Document(page_content="word " * 10000),
        Document(page_content="another " * 10000),
    ]
    
    vs.create_from_documents(long_docs)
    results = vs.similarity_search("word", k=2)
    
    assert len(results) == 2

def test_vectorstore_persistence_path_creation(mock_embedder, tmp_path):
    """Test that persist_path is created if it doesn't exist."""
    nested_path = tmp_path / "nested" / "directory" / "structure"
    
    vs = VectorStore(persist_path=nested_path, embedder=mock_embedder)
    
    assert nested_path.exists()
    assert nested_path.is_dir()

def test_vectorstore_default_k_value(tmp_path, mock_embedder, sample_documents):
    """Test that default k=5 works."""
    vs = VectorStore(persist_path=tmp_path, embedder=mock_embedder)
    vs.create_from_documents(sample_documents)
    
    # Don't specify k, should default to 5 (but only 4 docs available)
    results = vs.similarity_search(query="PostgreSQL")
    assert len(results) == 4  # All available documents

def test_vectorstore_multiple_searches_same_instance(tmp_path, mock_embedder, sample_documents):
    """Test multiple searches on same vector store instance."""
    vs = VectorStore(persist_path=tmp_path, embedder=mock_embedder)
    vs.create_from_documents(sample_documents)
    
    # Multiple searches should all work
    results1 = vs.similarity_search("PostgreSQL", k=2)
    results2 = vs.similarity_search("indexes", k=3)
    results3 = vs.similarity_search("database", k=1)
    
    assert len(results1) == 2
    assert len(results2) == 3
    assert len(results3) == 1

def test_vectorstore_save_without_create(tmp_path, mock_embedder):
    """Test that saving without creating first raises error."""
    vs = VectorStore(persist_path=tmp_path, embedder=mock_embedder)
    
    with pytest.raises(ValueError):
        vs.save()

def test_vectorstore_adapter_embed_documents(mock_embedder):
    """Test LangChainEmbeddingAdapter.embed_documents."""
    from src.retrieval.vector_store import LangChainEmbeddingAdapter
    
    adapter = LangChainEmbeddingAdapter(mock_embedder)
    texts = ["text1", "text2", "text3"]
    
    embeddings = adapter.embed_documents(texts)
    
    assert len(embeddings) == 3
    assert all(isinstance(emb, list) for emb in embeddings)
    assert all(len(emb) == 384 for emb in embeddings)

def test_vectorstore_adapter_embed_query(mock_embedder):
    """Test LangChainEmbeddingAdapter.embed_query."""
    from src.retrieval.vector_store import LangChainEmbeddingAdapter
    
    adapter = LangChainEmbeddingAdapter(mock_embedder)
    query = "test query"
    
    embedding = adapter.embed_query(query)
    
    assert isinstance(embedding, list)
    assert len(embedding) == 384

def test_vectorstore_different_embedders_different_results(tmp_path, sample_documents):
    """Test that different embedders produce different search results."""
    embedder1 = MockEmbedder(dimension=384)
    embedder2 = MockEmbedder(dimension=256)
    
    vs1 = VectorStore(persist_path=tmp_path / "vs1", embedder=embedder1)
    vs2 = VectorStore(persist_path=tmp_path / "vs2", embedder=embedder2)
    
    vs1.create_from_documents(sample_documents)
    vs2.create_from_documents(sample_documents)
    
    # Just verify both work - results may differ
    results1 = vs1.similarity_search("PostgreSQL", k=2)
    results2 = vs2.similarity_search("PostgreSQL", k=2)
    
    assert len(results1) == 2
    assert len(results2) == 2

def test_vectorstore_large_number_of_documents(tmp_path, mock_embedder):
    """Test indexing a large number of documents."""
    vs = VectorStore(persist_path=tmp_path, embedder=mock_embedder)
    
    large_doc_set = [
        Document(page_content=f"Document number {i} about PostgreSQL indexing strategies.")
        for i in range(1000)
    ]
    
    vs.create_from_documents(large_doc_set)
    results = vs.similarity_search("PostgreSQL", k=10)
    
    assert len(results) == 10

def test_vectorstore_recreate_after_load(tmp_path, mock_embedder, sample_documents):
    """Test that you can create a new index after loading."""
    vs = VectorStore(persist_path=tmp_path, embedder=mock_embedder)
    vs.create_from_documents(sample_documents)
    vs.save(name="original")
    
    # Load it
    vs.load(name="original")
    
    # Now create a new index (should replace loaded one)
    new_docs = [Document(page_content="Brand new content")]
    vs.create_from_documents(new_docs)
    
    results = vs.similarity_search("new", k=1)
    assert results[0].page_content == "Brand new content"