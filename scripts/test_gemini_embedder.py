#!/usr/bin/env python3
"""
Script to test if the api key work s and we are getting back a vector ot put inot the db
"""

import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

def main():
    load_dotenv()
    api_key=os.getenv("GEMINI_EMBEDDING_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_EMBEDDING_KEY not found in environment")

    client = genai.Client(api_key=api_key)

    text = """
    PostgreSQL supports advanced indexing techniques including
    B-tree, Hash, GiST, SP-GiST, GIN, and BRIN indexes.
    """

    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config=types.EmbedContentConfig(
            task_type="retrieval_document"
        ),
    )

    embedding = response.embeddings[0].values

    print("✅ Embedding generated")
    print(f"Vector length: {len(embedding)}")
    print(f"First 10 values: {embedding[:10]}")
    print(f"Value types OK:", all(isinstance(v, float) for v in embedding))


if __name__ == "__main__":
    main()
