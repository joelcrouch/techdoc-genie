import pytest
import numpy as np
import torch # Added for torch.Tensor type checking
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

def test_sentence_transformer_embedding():
    # Option 1: Fast and efficient (384 dimensions)
    model_name = 'all-MiniLM-L6-v2'
    
    model = SentenceTransformer(model_name)
    assert model is not None, f"Failed to load model: {model_name}"
    
    text = """
    PostgreSQL supports advanced indexing techniques including
    B-tree, Hash, GiST, SP-GiST, GIN, and BRIN indexes.
    """
    
    embedding = model.encode(text)
    assert isinstance(embedding, np.ndarray), "Embedding is not a numpy array"
    assert len(embedding) == 384, f"Expected vector length 384, got {len(embedding)}"
    assert all(isinstance(v, (float, np.floating)) for v in embedding), "Embedding values are not floats"
    assert embedding.shape == (384,), f"Expected shape (384,), got {embedding.shape}"
    assert embedding.dtype == np.float32, f"Expected dtype float32, got {embedding.dtype}"
    
    batch_texts = [
        "PostgreSQL is a powerful database system.",
        "B-tree indexes are the default in PostgreSQL.",
        "GIN indexes are useful for full-text search."
    ]
    
    batch_embeddings = model.encode(batch_texts, show_progress_bar=False) # Disable progress bar for tests
    assert isinstance(batch_embeddings, np.ndarray), "Batch embeddings are not a numpy array"
    assert batch_embeddings.shape == (len(batch_texts), 384), f"Expected batch embeddings shape ({len(batch_texts)}, 384), got {batch_embeddings.shape}"
    
    query = "What indexing methods does PostgreSQL support?"
    query_embedding = model.encode(query)
    doc_embedding = model.encode(text)
    
    similarity = cos_sim(query_embedding, doc_embedding)
    assert similarity.shape == (1, 1), f"Expected similarity shape (1, 1), got {similarity.shape}"
    
    scalar_similarity = similarity.item()
    assert isinstance(scalar_similarity, float), "Scalar similarity is not a float"
    assert -1.0 <= scalar_similarity <= 1.0, f"Similarity score {scalar_similarity} out of range [-1.0, 1.0]"