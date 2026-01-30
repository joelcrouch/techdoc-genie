import argparse
from pathlib import Path

from src.ingestion.document_loader import DocumentLoader
from src.ingestion.chunker import DocumentChunker
from src.ingestion.providers.huggingface import HuggingFaceEmbeddingProvider
from src.retrieval.vector_store import VectorStore
from scripts.chunking_configs import CHUNKING_CONFIGS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--docs", required=True)
    args = parser.parse_args()

    config = CHUNKING_CONFIGS[args.config]

    # 1️⃣ Load documents
    loader = DocumentLoader(args.docs)
    documents = loader.load()

    # 2️⃣ Chunk
    chunker = DocumentChunker(
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap,
)
    chunks = chunker.chunk_documents(
    documents,
    strategy="recursive",  # or token / semantic
)
    # 3️⃣ Embedder
    embedder = HuggingFaceEmbeddingProvider(
        model_name="all-MiniLM-L6-v2",
        device="cuda",
    )

    # 4️⃣ Vector store
    vs = VectorStore(embedder=embedder)
    vs.create_from_documents(chunks)
    vs.save(name="index")

    print(f"✅ Vector store built for {args.config}")


if __name__ == "__main__":
    main()
