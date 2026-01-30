#!/usr/bin/env python3
"""
Script to test if sentence transformers work and we are getting back a vector to put into the db
"""
from sentence_transformers import SentenceTransformer
import numpy as np

def main():
    print("🔄 Loading model...")
    # Choose your model - uncomment the one you want to test
    
    # Option 1: Fast and efficient (384 dimensions)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Option 2: Better quality (768 dimensions)
    # model = SentenceTransformer('all-mpnet-base-v2')
    
    # Option 3: Good for retrieval (384 dimensions)
    # model = SentenceTransformer('thenlper/gte-small')
    
    print(f"✅ Model loaded: {model}")
    
    # Same test text as Gemini version
    text = """
    PostgreSQL supports advanced indexing techniques including
    B-tree, Hash, GiST, SP-GiST, GIN, and BRIN indexes.
    """
    
    print("🔄 Generating embedding...")
    embedding = model.encode(text)
    
    print("\n✅ Embedding generated")
    print(f"Vector length: {len(embedding)}")
    print(f"First 10 values: {embedding[:10]}")
    print(f"Value types OK: {all(isinstance(v, (float, np.floating)) for v in embedding)}")
    print(f"Vector shape: {embedding.shape}")
    print(f"Data type: {embedding.dtype}")
    
    # Test batch encoding (useful for your actual use case)
    print("\n🔄 Testing batch encoding...")
    batch_texts = [
        "PostgreSQL is a powerful database system.",
        "B-tree indexes are the default in PostgreSQL.",
        "GIN indexes are useful for full-text search."
    ]
    
    batch_embeddings = model.encode(batch_texts, show_progress_bar=True)
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

if __name__ == "__main__":
    main()