from src.ingestion.providers.huggingface import HuggingFaceEmbeddingProvider
import numpy as np

def test_huggingface_embed_query():
    embedder = HuggingFaceEmbeddingProvider()

    vec = embedder.embed_query("PostgreSQL indexing strategies")

    assert isinstance(vec, list)
    assert len(vec) > 0
    assert all(isinstance(v, float) for v in vec)


def test_huggingface_embed_documents_with_batch_size():
    embedder = HuggingFaceEmbeddingProvider()

    texts = [
        "PostgreSQL supports B-tree indexes.",
        "GIN indexes are useful for full-text search.",
    ]

    vectors = embedder.embed_documents(texts, batch_size=2)

    assert len(vectors) == len(texts)
    assert all(len(v) == len(vectors[0]) for v in vectors)


def test_huggingface_empty_list():
    """Test that empty list returns empty embeddings."""
    embedder = HuggingFaceEmbeddingProvider()
    vectors = embedder.embed_documents([])
    assert vectors == []

def test_huggingface_single_document():
    """Test embedding a single document."""
    embedder = HuggingFaceEmbeddingProvider()
    vectors = embedder.embed_documents(["Single text"])
    assert len(vectors) == 1
    assert len(vectors[0]) > 0

def test_huggingface_embedding_dimension_consistency():
    """Test that all embeddings have the same dimension."""
    embedder = HuggingFaceEmbeddingProvider()
    query_vec = embedder.embed_query("test query")
    doc_vecs = embedder.embed_documents(["doc1", "doc2", "doc3"])
    
    # All should have same dimension
    assert len(query_vec) == len(doc_vecs[0])
    assert all(len(v) == len(query_vec) for v in doc_vecs)

def test_huggingface_deterministic_embeddings():
    """Test that same text produces same embedding."""
    embedder = HuggingFaceEmbeddingProvider()
    text = "PostgreSQL database"
    
    vec1 = embedder.embed_query(text)
    vec2 = embedder.embed_query(text)
    
    assert vec1 == vec2

def test_huggingface_different_texts_different_embeddings():
    """Test that different texts produce different embeddings."""
    embedder = HuggingFaceEmbeddingProvider()
    
    vec1 = embedder.embed_query("PostgreSQL")
    vec2 = embedder.embed_query("MongoDB")
    
    assert vec1 != vec2

def test_huggingface_custom_model():
    """Test initialization with custom model name."""
    embedder = HuggingFaceEmbeddingProvider(model_name="all-MiniLM-L6-v2")
    vec = embedder.embed_query("test")
    assert isinstance(vec, list)
    assert len(vec) > 0

def test_huggingface_special_characters():
    """Test embedding text with special characters."""
    embedder = HuggingFaceEmbeddingProvider()
    texts = [
        "Text with @#$% special chars!",
        "Unicode: 你好 мир",
        "Newlines\nand\ttabs",
    ]
    vectors = embedder.embed_documents(texts)
    assert len(vectors) == 3
    assert all(len(v) > 0 for v in vectors)

def test_huggingface_long_text():
    """Test embedding very long text."""
    embedder = HuggingFaceEmbeddingProvider()
    long_text = "PostgreSQL " * 500  # Very long text
    vec = embedder.embed_query(long_text)
    assert isinstance(vec, list)
    assert len(vec) > 0

#fix floationg point discrepacny i thinkk
def test_huggingface_batch_sizes():
    """Test different batch sizes produce same results."""
    embedder = HuggingFaceEmbeddingProvider()
    texts = [f"Text {i}" for i in range(10)]
    
    vecs_batch_2 = embedder.embed_documents(texts, batch_size=2)
    vecs_batch_5 = embedder.embed_documents(texts, batch_size=5)
    
    assert len(vecs_batch_2) == len(vecs_batch_5)
    # Results should be nearly identical regardless of batch size
    for v1, v2 in zip(vecs_batch_2, vecs_batch_5):
        assert np.allclose(v1, v2, rtol=1e-5, atol=1e-7)

def test_huggingface_embedding_values_range():
    """Test that embedding values are reasonable floats."""
    embedder = HuggingFaceEmbeddingProvider()
    vec = embedder.embed_query("test")
    
    # Check values are finite floats (not NaN or inf)
    assert all(isinstance(v, float) for v in vec)
    assert all(not (v != v) for v in vec)  # Not NaN
    assert all(abs(v) < 1e10 for v in vec)  # Not infinity

def test_huggingface_semantic_similarity():
    """Test that similar texts have similar embeddings."""
    embedder = HuggingFaceEmbeddingProvider()
    
    # Similar texts
    vec1 = embedder.embed_query("database indexing")
    vec2 = embedder.embed_query("database indexes")
    
    # Very different text
    vec3 = embedder.embed_query("cooking recipes")
    
    # Calculate cosine similarity (simple dot product for normalized vectors)
    def cosine_similarity(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = sum(x * x for x in a) ** 0.5
        mag_b = sum(y * y for y in b) ** 0.5
        return dot / (mag_a * mag_b)
    
    sim_similar = cosine_similarity(vec1, vec2)
    sim_different = cosine_similarity(vec1, vec3)
    
    # Similar texts should be more similar than different texts
    assert sim_similar > sim_different

def test_huggingface_empty_string():
    """Test embedding an empty string."""
    embedder = HuggingFaceEmbeddingProvider()
    vec = embedder.embed_query("")
    assert isinstance(vec, list)
    assert len(vec) > 0  # Should still return embedding