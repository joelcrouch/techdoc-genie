import pytest
import re
from src.agent.providers.llm_interface import LLMInterface

def test_llm_interface_mock():
    # Only mock provider should succeed
    llm = LLMInterface(provider="mock", model="test-model", api_key="dummy")

    output = llm.generate_text("Hello world")
    assert isinstance(output, str)
    assert "Hello world" in output

    texts = ["foo", "bar"]
    embeddings = llm.embed_texts(texts)
    assert len(embeddings) == len(texts)
    assert all(isinstance(vec, list) for vec in embeddings)
    assert all(all(isinstance(v, (int, float)) for v in vec) for vec in embeddings)


def test_llm_interface_not_implemented():
    # All other providers should raise NotImplementedError
    for provider in ["openai", "claude", "gemini", "local"]:
        with pytest.raises(
            NotImplementedError,
            match=re.compile(f"{provider} provider not implemented yet", re.IGNORECASE)
        ):
            LLMInterface(provider=provider, model="test-model", api_key="dummy")
