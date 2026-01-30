#!/usr/bin/env python3
"""
Script to test sentence transformers with GPU acceleration
"""
from sentence_transformers import SentenceTransformer
import numpy as np
import torch

def main():
    # Check GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 Using device: {device}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")
    
    print("\n🔄 Loading model...")
    
    # Load model and specify device explicitly
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    
    print(f"✅ Model loaded on {device}")
    print(f"   Model is on GPU: {next(model.parameters()).is_cuda}")
    
    # Same test text as before
    text = """
    PostgreSQL supports advanced indexing techniques including
    B-tree, Hash, GiST, SP-GiST, GIN, and BRIN indexes.
    """
    
    print("\n🔄 Generating embedding...")
    embedding = model.encode(text)
    
    print("\n✅ Embedding generated")
    print(f"Vector length: {len(embedding)}")
    print(f"First 10 values: {embedding[:10]}")
    print(f"Value types OK: {all(isinstance(v, (float, np.floating)) for v in embedding)}")
    print(f"Vector shape: {embedding.shape}")
    print(f"Data type: {embedding.dtype}")
    
    # Test batch encoding (this is where GPU really shines!)
    print("\n🔄 Testing batch encoding (GPU advantage here)...")
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
    
    print(f"   Encoding {len(batch_texts)} texts...")
    batch_embeddings = model.encode(
        batch_texts, 
        show_progress_bar=True,
        batch_size=32  # Larger batches benefit more from GPU
    )
    print(f"✅ Batch embeddings generated: {batch_embeddings.shape}")
    
    # Test similarity (bonus - shows how retrieval would work)
    print("\n🔄 Testing similarity calculation...")
    from sentence_transformers.util import cos_sim
    
    query = "What indexing methods does PostgreSQL support?"
    query_embedding = model.encode(query)
    doc_embedding = model.encode(text)
    
    similarity = cos_sim(query_embedding, doc_embedding)
    print(f"✅ Similarity score: {similarity.item():.4f}")
    print("   (Higher = more similar, range: -1 to 1)")
    
    # Show GPU memory usage
    if device == "cuda":
        print(f"\n📊 GPU Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"   GPU Memory reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

if __name__ == "__main__":
    main()