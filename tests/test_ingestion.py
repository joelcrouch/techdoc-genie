import pytest
from pathlib import Path
from src.ingestion.document_loader import DocumentLoader

def test_document_loader():
    """Test basic document loading."""
    loader = DocumentLoader(data_dir="data/raw/postgresql")
    
    # This test will only pass if documents exist
    documents = loader.load_and_prepare()
    assert documents, "No documents loaded — check data/raw contents"
    assert len(documents) > 0, "Should load at least one document"
    assert all(hasattr(doc, 'page_content') for doc in documents)
    assert all(hasattr(doc, 'metadata') for doc in documents)

def test_metadata_enrichment():
    """Test metadata is properly added."""
    loader = DocumentLoader(data_dir="data/raw")
    documents = loader.load_and_prepare()
    
    if documents:
        assert 'filename' in documents[0].metadata
        assert 'doc_type' in documents[0].metadata