import pytest
from src.ingestion.embedder import DocumentEmbedder

def test_embedder_mock():
    # Initialize embedder with mock provider
    embedder = DocumentEmbedder(provider="mock")

    # Test embed_documents with a small list
    texts = ["Hello world", "Test document", "Another chunk"]
    embeddings = embedder.embed_documents(texts, batch_size=2)

    # Check we got embeddings for each text
    assert len(embeddings) == len(texts), "Should return one embedding per text"

    # Check each embedding is a list of floats
    for emb in embeddings:
        assert isinstance(emb, list), "Each embedding should be a list"
        assert all(isinstance(x, float) for x in emb), "Embedding elements should be floats"

    # Test embed_query
    query_emb = embedder.embed_query("Query text")
    assert isinstance(query_emb, list), "Query embedding should be a list"
    assert all(isinstance(x, float) for x in query_emb), "Query embedding elements should be floats"

    # Test batch processing with batch_size smaller than number of texts
    texts_long = ["Doc " + str(i) for i in range(10)]
    embeddings_long = embedder.embed_documents(texts_long, batch_size=3)
    assert len(embeddings_long) == len(texts_long), "Batching should still return all embeddings"

def test_mock_embedder_deterministic():
    """Test that mock embedder produces identical embeddings for same text."""
    embedder = DocumentEmbedder(provider="mock")
    text = "PostgreSQL database indexing"
    
    emb1 = embedder.embed_query(text)
    emb2 = embedder.embed_query(text)
    
    assert emb1 == emb2, "Same text should produce identical embeddings"

def test_mock_embedder_different_texts():
    """Test that different texts produce different embeddings."""
    embedder = DocumentEmbedder(provider="mock")
    
    emb1 = embedder.embed_query("First text")
    emb2 = embedder.embed_query("Second text")
    
    assert emb1 != emb2, "Different texts should produce different embeddings"

def test_mock_embedder_default_dimension():
    """Test that default dimension is 384."""
    embedder = DocumentEmbedder(provider="mock")
    emb = embedder.embed_query("test")
    
    assert len(emb) == 384, "Default dimension should be 384"

def test_mock_embedder_custom_dimension():
    """Test that custom dimension works."""
    # Note: This assumes DocumentEmbedder can pass dimension to MockEmbedder
    # If not, you might need to test MockEmbedder directly
    from src.ingestion.providers.mock import MockEmbedder
    
    embedder = MockEmbedder(dimension=128)
    emb = embedder.embed_query("test")
    
    assert len(emb) == 128, "Custom dimension should be respected"

def test_mock_embedder_embedding_consistency():
    """Test that embeddings are consistent across documents and queries."""
    embedder = DocumentEmbedder(provider="mock")
    text = "test text"
    
    query_emb = embedder.embed_query(text)
    doc_emb = embedder.embed_documents([text])[0]
    
    assert query_emb == doc_emb, "Same text should have same embedding via query or documents"

def test_mock_embedder_empty_list():
    """Test that empty list returns empty embeddings."""
    embedder = DocumentEmbedder(provider="mock")
    embeddings = embedder.embed_documents([])
    
    assert embeddings == [], "Empty list should return empty embeddings"

def test_mock_embedder_single_document():
    """Test embedding a single document."""
    embedder = DocumentEmbedder(provider="mock")
    embeddings = embedder.embed_documents(["Single text"])
    
    assert len(embeddings) == 1
    assert len(embeddings[0]) == 384

def test_mock_embedder_empty_string():
    """Test embedding an empty string."""
    embedder = DocumentEmbedder(provider="mock")
    emb = embedder.embed_query("")
    
    assert isinstance(emb, list)
    assert len(emb) == 384
    assert all(isinstance(x, float) for x in emb)

def test_mock_embedder_whitespace_sensitivity():
    """Test that whitespace changes affect embeddings."""
    embedder = DocumentEmbedder(provider="mock")
    
    emb1 = embedder.embed_query("test")
    emb2 = embedder.embed_query("test ")
    emb3 = embedder.embed_query(" test")
    
    # Whitespace should produce different hashes, thus different embeddings
    assert emb1 != emb2
    assert emb1 != emb3

def test_mock_embedder_case_sensitivity():
    """Test that case changes affect embeddings."""
    embedder = DocumentEmbedder(provider="mock")
    
    emb1 = embedder.embed_query("Test")
    emb2 = embedder.embed_query("test")
    
    assert emb1 != emb2, "Case should affect embeddings"

def test_mock_embedder_special_characters():
    """Test that special characters are handled."""
    embedder = DocumentEmbedder(provider="mock")
    texts = [
        "Text with @#$% special chars!",
        "Unicode: 你好 мир",
        "Newlines\nand\ttabs",
        "Quotes 'single' and \"double\"",
    ]
    
    embeddings = embedder.embed_documents(texts)
    assert len(embeddings) == len(texts)
    assert all(len(emb) == 384 for emb in embeddings)
    # All should be different
    assert len(set(tuple(emb) for emb in embeddings)) == len(texts)

def test_mock_embedder_value_range():
    """Test that embedding values are in expected range [-1.0, 1.0]."""
    embedder = DocumentEmbedder(provider="mock")
    emb = embedder.embed_query("test text")
    
    assert all(-1.0 <= x <= 1.0 for x in emb), "Values should be in range [-1, 1]"

def test_mock_embedder_no_nans_or_infs():
    """Test that embeddings don't contain NaN or infinity values."""
    embedder = DocumentEmbedder(provider="mock")
    emb = embedder.embed_query("test")
    
    assert all(not (x != x) for x in emb), "Should not contain NaN"
    assert all(abs(x) < float('inf') for x in emb), "Should not contain infinity"

def test_mock_embedder_batch_size_ignored():
    """Test that batch_size parameter doesn't affect results."""
    embedder = DocumentEmbedder(provider="mock")
    texts = [f"Text {i}" for i in range(10)]
    
    emb_batch_1 = embedder.embed_documents(texts, batch_size=1)
    emb_batch_5 = embedder.embed_documents(texts, batch_size=5)
    emb_batch_100 = embedder.embed_documents(texts, batch_size=100)
    
    assert emb_batch_1 == emb_batch_5 == emb_batch_100

def test_mock_embedder_large_batch():
    """Test handling of large number of documents."""
    embedder = DocumentEmbedder(provider="mock")
    texts = [f"Document {i}" for i in range(1000)]
    
    embeddings = embedder.embed_documents(texts)
    assert len(embeddings) == 1000
    assert all(len(emb) == 384 for emb in embeddings)

def test_mock_embedder_very_long_text():
    """Test embedding very long text."""
    embedder = DocumentEmbedder(provider="mock")
    long_text = "word " * 10000  # Very long text
    
    emb = embedder.embed_query(long_text)
    assert len(emb) == 384
    assert all(isinstance(x, float) for x in emb)

def test_mock_embedder_reproducibility_across_instances():
    """Test that different instances produce same embeddings for same text."""
    embedder1 = DocumentEmbedder(provider="mock")
    embedder2 = DocumentEmbedder(provider="mock")
    
    text = "reproducibility test"
    emb1 = embedder1.embed_query(text)
    emb2 = embedder2.embed_query(text)
    
    assert emb1 == emb2, "Different instances should produce identical embeddings"

def test_mock_embedder_hash_collision_unlikely():
    """Test that similar texts produce different embeddings (no hash collision)."""
    embedder = DocumentEmbedder(provider="mock")
    
    texts = [
        "test1",
        "test2",
        "test3",
        "test4",
        "test5",
    ]
    
    embeddings = embedder.embed_documents(texts)
    # All embeddings should be unique
    unique_embeddings = len(set(tuple(emb) for emb in embeddings))
    assert unique_embeddings == len(texts), "Each text should have unique embedding"

def test_mock_embedder_numeric_strings():
    """Test handling of numeric strings."""
    embedder = DocumentEmbedder(provider="mock")
    
    texts = ["123", "456.789", "0", "-100", "1e10"]
    embeddings = embedder.embed_documents(texts)
    
    assert len(embeddings) == len(texts)
    assert all(len(emb) == 384 for emb in embeddings)
    # All should be different
    assert len(set(tuple(emb) for emb in embeddings)) == len(texts)