import random

class MockLLMProvider:
    def generate_text(self, prompt:str, **kwargs) -> str:
        return f"Mock Response: {prompt}"
    
    def embed_texts(self, texts):
        return [[random.random() for _ in range(8)] for _ in texts]