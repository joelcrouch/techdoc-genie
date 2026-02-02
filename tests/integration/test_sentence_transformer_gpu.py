import pytest
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

def test_sentence_transformer_gpu_embedding():
    # Check GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🔧 Using device: {device}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")

    if device != "cuda": # Use != "cuda" to correctly skip if not CUDA
        pytest.skip("CUDA not available, skipping GPU test.")
    
    print("\n🔄 Loading model...")
    # Load model and specify device explicitly
    model_name = 'all-MiniLM-L6-v2'
    model = SentenceTransformer(model_name, device=device)
    model.to(device) # Explicitly move model to device
    
    assert model is not None, f"Failed to load model: {model_name}"
    assert next(model.parameters()).is_cuda, "Model is not on GPU"
    print(f"✅ Model loaded on {device}")
    print(f"   Model is on GPU: {next(model.parameters()).is_cuda}")
    
    # Same test text as before
    text = """
    PostgreSQL supports advanced indexing techniques including
    B-tree, Hash, GiST, SP-GiST, GIN, and BRIN indexes.
    """
    
    print("\n🔄 Generating embedding...")
    embedding = model.encode(text)
    assert isinstance(embedding, np.ndarray), "Embedding is not a numpy array"
    assert len(embedding) == 384, f"Expected vector length 384, got {len(embedding)}"
    assert all(isinstance(v, (float, np.floating)) for v in embedding), "Embedding values are not floats"
    assert embedding.shape == (384,), f"Expected shape (384,), got {embedding.shape}"
    assert embedding.dtype == np.float32, f"Expected dtype float32, got {embedding.dtype}"
    print("\n✅ Embedding generated")
    print(f"Vector length: {len(embedding)}")
    print(f"First 10 values: {embedding[:10]}")
    print(f"Value types OK: {all(isinstance(v, (float, np.floating)) for v in embedding)}")
    print(f"Vector shape: {embedding.shape}")
    print(f"Data type: {embedding.dtype}")
    
    # Test batch encoding (this is where GPU really shines!)
    batch_texts = [
        "PostgreSQL is a powerful database system.",
        "B-tree indexes are the default in PostgreSQL.",
        "GIN indexes are useful for full-text search.",
        "BRIN indexes are block range indexes.",
        "Hash indexes are useful for equality comparisons.",
        "GiST indexes support geometric data types.",
        "SP-GiST indexes are space-partitioned GiST.",
        "Partial indexes index only subset of rows.",
    ] * 4  # 32 texts total to show GPU benefit
    
    print(f"\n🔄 Testing batch encoding (GPU advantage here)...")
    print(f"   Encoding {len(batch_texts)} texts...")
    batch_embeddings = model.encode(
        batch_texts, 
        show_progress_bar=False, # progress bar interferes with test output
        batch_size=32  # Larger batches benefit more from GPU
    )
    assert isinstance(batch_embeddings, np.ndarray), "Batch embeddings are not a numpy array"
    assert batch_embeddings.shape == (len(batch_texts), 384), f"Expected batch embeddings shape ({len(batch_texts)}, 384), got {batch_embeddings.shape}"
    print(f"✅ Batch embeddings generated: {batch_embeddings.shape}")
    
    print("\n🔄 Testing similarity calculation...")
    query = "What indexing methods does PostgreSQL support?"
    query_embedding = model.encode(query)
    doc_embedding = model.encode(text)
    
    similarity = cos_sim(query_embedding, doc_embedding)
    scalar_similarity = similarity.item()
    assert isinstance(scalar_similarity, float), "Scalar similarity is not a float"
    assert -1.0 <= scalar_similarity <= 1.0, f"Similarity score {scalar_similarity} out of range [-1.0, 1.0]"
    print(f"✅ Similarity score: {scalar_similarity:.4f}")
    print("   (Higher = more similar, range: -1 to 1)")
    
    # Show GPU memory usage
    if device == "cuda":
        print(f"\n📊 GPU Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"   GPU Memory reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")