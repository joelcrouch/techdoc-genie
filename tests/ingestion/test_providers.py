"""Tests for ingestion embedding providers and chunking_configs."""
import pytest
from unittest.mock import MagicMock

from src.ingestion.chunking_configs import CHUNKING_CONFIGS
from src.ingestion.providers.gemini import GeminiEmbeddingProvider


# ---------------------------------------------------------------------------
# chunking_configs
# ---------------------------------------------------------------------------

def test_chunking_configs_has_expected_keys():
    assert "chunk512_overlap50" in CHUNKING_CONFIGS
    assert "chunk1024_overlap100" in CHUNKING_CONFIGS
    assert "semantic" in CHUNKING_CONFIGS


def test_chunking_configs_values():
    cfg = CHUNKING_CONFIGS["chunk512_overlap50"]
    assert cfg["chunk_size"] == 512
    assert cfg["chunk_overlap"] == 50
    assert cfg["strategy"] == "recursive"

    cfg2 = CHUNKING_CONFIGS["chunk1024_overlap100"]
    assert cfg2["chunk_size"] == 1024
    assert cfg2["chunk_overlap"] == 100

    assert CHUNKING_CONFIGS["semantic"]["strategy"] == "semantic"


# ---------------------------------------------------------------------------
# GeminiEmbeddingProvider
# ---------------------------------------------------------------------------

@pytest.fixture
def gemini_provider(mocker):
    """GeminiEmbeddingProvider with GoogleGenerativeAIEmbeddings mocked out."""
    mock_embeddings = MagicMock()
    mocker.patch(
        "src.ingestion.providers.gemini.GoogleGenerativeAIEmbeddings",
        return_value=mock_embeddings,
    )
    provider = GeminiEmbeddingProvider(model="models/embedding-001", api_key="fake-key")
    provider._mock_embeddings = mock_embeddings
    return provider


def test_gemini_provider_embed_documents(gemini_provider):
    expected = [[0.1, 0.2], [0.3, 0.4]]
    gemini_provider._mock_embeddings.embed_documents.return_value = expected

    result = gemini_provider.embed_documents(["text one", "text two"])

    gemini_provider._mock_embeddings.embed_documents.assert_called_once_with(["text one", "text two"])
    assert result == expected


def test_gemini_provider_embed_query(gemini_provider):
    expected = [0.5, 0.6, 0.7]
    gemini_provider._mock_embeddings.embed_query.return_value = expected

    result = gemini_provider.embed_query("what is postgres?")

    gemini_provider._mock_embeddings.embed_query.assert_called_once_with("what is postgres?")
    assert result == expected
